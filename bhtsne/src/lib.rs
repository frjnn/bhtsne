//! # bhtsne
//!
//! `bhtsne` contains the implementations of both a parallel, exact, version of the t-SNE algorithm
//! and a parallel, approximate, version leveraging the Barnes-Hut algorithm.
//!
//! The implementation supports custom data types and custom defined metrics. See [`tSNE`] for more
//! details.
//!
//! This crate also includes [`load_csv`], a commodity function to parse data, record by record,
//! from a csv file.
//!
//! # Example
//!
//! ```ignore
//! # use std::error::Error;
//! use bhtsne;
//!
//! const N: usize = 150;         // Number of vectors to embed.
//! const D: usize = 4;           // The dimensionality of the
//!                               // original space.
//! const THETA: f32 = 0.5;       // Parameter used by the Barnes-Hut algorithm.
//!                               // Small values improve accuracy but increase complexity.
//!    
//! const PERPLEXITY: f32 = 10.0; // Perplexity of the conditional distribution.
//! const EPOCHS: usize = 2000;   // Number of fitting iterations.
//!
//! // Loads the data from a csv file skipping the first row,
//! // treating it as headers and skipping the 5th column,
//! // treating it as a class label.
//! // Do note that you can also switch to f64s for higher precision.
//! let data: Vec<f32> = bhtsne::load_csv("iris.csv", true, Some(&[4]), |float| {
//!     float.parse().unwrap()
//! })?;
//! let samples: Vec<&[f32]> = data.chunks(D).collect();
//!
//! // Executes the Barnes-Hut approximation of the algorithm and writes the embedding to the
//! // specified csv file.
//! bhtsne::tSNE::<f32, &[f32], 2>::new(&samples)
//!     .perplexity(PERPLEXITY)
//!     .epochs(EPOCHS)
//!     .barnes_hut(THETA, |sample_a, sample_b| {
//!         sample_a
//!             .iter()
//!             .zip(sample_b.iter())
//!             .map(|(a, b)| (a - b).powi(2))
//!             .sum::<f32>()
//!             .sqrt()
//!     })
//!     .write_csv("iris_embedding.csv")?;
//! # Ok::<(), Box<dyn Error>>(())
//! ```
mod repulsion;
mod tsne;

#[cfg(feature = "csv")]
mod csv;

use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use num_traits::{Float, cast::AsPrimitive};

use repulsion::{BarnesHutRepulsion, InterpolatedRepulsion, Repulsion};

#[cfg(feature = "csv")]
pub use csv::load_csv;

/// Public re-exports.
pub use {
    barnes_hut_tree::{Dim, Morton},
    rustfft::FftNum,
    tsne::interpolation::FftDim,
    tsne::spectral::{SpectralBlock, SpectralParams},
};

/// Monomorphized spectral solver captured by [`tSNE::spectral_init_with`], where the
/// [`SpectralBlock`] bound is available, and invoked by the seeding step of the fit,
/// where it is not.
type SpectralSeeder<T> = fn(&[usize], &[u32], &[T], SpectralParams) -> Vec<T>;

use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::{ParallelSlice, ParallelSliceMut},
};

/// Minimum sample count before the Barnes-Hut per-epoch light passes (the fused gradient update and
/// the zero-mean recentering) fan out across the thread pool.
const PARALLEL_CODE_THRESHOLD: usize = 4096;

/// Boxed closure invoked at the end of each fitting epoch with the epoch index and a
/// snapshot of the current embedding. See [`tSNE::epoch_callback`].
///
/// The callback is only ever invoked sequentially from the fitting thread, so it
/// needs neither `Send` nor `Sync`. A [`tSNE`] holding a non-`Send`/non-`Sync`
/// callback is itself non-`Send`/non-`Sync` while that callback is set.
pub type EpochCallback<'data, T> = Box<dyn FnMut(usize, &[T]) + 'data>;

/// Records which fitting routine last ran, so [`tSNE::kl_divergence`] can pick
/// the matching cost evaluation.
enum Fit<T> {
    Exact,
    BarnesHut { theta: T },
    Interpolated,
}

/// A sample's nearest neighbor for [`tSNE::barnes_hut_with_neighbors`]: its index
/// and the distance (not a similarity) to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Neighbor<T> {
    pub index: usize,
    pub distance: T,
}

/// A precomputed sparse, symmetric affinity graph in the CSR-like layout
/// `barnes_hut` uses internally. Obtain via [`tSNE::affinities`] and inject
/// with [`tSNE::with_affinities`].
#[derive(Clone)]
pub struct SparseAffinities<T> {
    rows: Vec<usize>,
    columns: Vec<u32>,
    values: Vec<T>,
    perplexity: T,
}

impl<T> SparseAffinities<T> {
    /// Returns the row indices of the sparse affinity matrix.
    pub fn rows(&self) -> &[usize] {
        &self.rows
    }

    /// Returns the column indices of the sparse affinity matrix.
    pub fn columns(&self) -> &[u32] {
        &self.columns
    }

    /// Returns the affinity values.
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Returns the perplexity used when building the affinities.
    pub fn perplexity(&self) -> &T {
        &self.perplexity
    }
}

/// t-distributed stochastic neighbor embedding. Provides a parallel implementation of both the
/// exact version of the algorithm and the tree accelerated one leveraging space partitioning trees.
#[allow(non_camel_case_types)]
pub struct tSNE<'data, T, U, const D: usize = 2>
where
    T: Send + Sync + Float + Sum + DivAssign + MulAssign + AddAssign + SubAssign,
    U: Send + Sync,
{
    data: &'data [U],
    learning_rate: Option<T>,
    epochs: usize,
    momentum: T,
    final_momentum: T,
    momentum_switch_epoch: usize,
    stop_lying_epoch: usize,
    early_exaggeration: T,
    perplexity: T,
    p_values: Vec<T>,
    p_rows: Vec<usize>,
    p_columns: Vec<u32>,
    q_values: Vec<T>,
    y: Vec<T>,
    dy: Vec<T>,
    uy: Vec<T>,
    gains: Vec<T>,
    epoch_callback: Option<EpochCallback<'data, T>>,
    initial_embedding: Option<Vec<T>>,
    spectral_init: Option<(SpectralParams, SpectralSeeder<T>)>,
    stop_lying_fired: bool,
    cached_perplexity: Option<T>,
    fit: Option<Fit<T>>,
}

impl<'data, T, U, const D: usize> tSNE<'data, T, U, D>
where
    T: Float
        + Send
        + Sync
        + AsPrimitive<usize>
        + Sum
        + DivAssign
        + AddAssign
        + MulAssign
        + SubAssign,
    U: Send + Sync,
{
    /// Creates a new t-SNE instance.
    ///
    /// # Arguments
    ///
    /// `data` - dataset to execute the t-SNE algorithm on.
    ///
    /// According to the original implementation, the following configuration is provided by
    /// default:
    ///
    /// * `learning_rate = auto` (`max(n_samples / early_exaggeration / 4, 50)`)
    /// * `epochs = 1000`
    /// * `momentum = 0.5`
    /// * `final_momentum = 0.8`
    /// * `stop_lying_epoch = 250`
    /// * `early_exaggeration = 12.0`
    /// * embedding space dimensionality `D = 2` (the trailing const generic of [`tSNE`])
    /// * `perplexity = 20.0`
    /// * `random_init = false`
    ///
    /// Such parameters can be set to different values with the provided methods.
    ///
    /// # Examples
    ///
    /// The dataset in input needs to be formed by singular entities. For instance, general vector
    /// data can be handled in the following way:
    ///
    /// ```
    /// use bhtsne::tSNE;
    ///
    /// const N: usize = 1000; // Supposedly 1000 25-dimensional points.
    /// const D: usize = 25;
    ///
    /// let data: Vec<f32> = vec![0.0_f32; N * D];
    /// let vectors: Vec<&[f32]> = data.chunks(D).collect();
    ///
    /// let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&vectors); // Will compute using f32s.
    /// let mut tsne: tSNE<f64, &[f32]> = tSNE::new(&vectors); // Will compute using f64s.
    /// let mut tsne: tSNE<f32, &[f32], 3> = tSNE::new(&vectors); // Three dimensional embedding.
    /// ```
    ///
    /// One can also use `&str`, [`String`] or custom data types:
    ///
    /// ```
    /// use bhtsne::tSNE;
    ///
    /// const N: usize = 1000; // Supposedly 1000 strings.
    /// let strings: Vec<&str> = vec!["Hello World!"; N];
    ///
    /// let mut tsne: tSNE<f32, &str> = tSNE::new(&strings);
    /// ```
    pub fn new(data: &'data [U]) -> Self {
        Self {
            data,
            learning_rate: None,
            epochs: 1000,
            momentum: T::from(0.5).unwrap(),
            final_momentum: T::from(0.8).unwrap(),
            momentum_switch_epoch: 250,
            stop_lying_epoch: 250,
            early_exaggeration: T::from(12.0).unwrap(),
            perplexity: T::from(20.0).unwrap(),
            p_values: Vec::new(),
            p_rows: Vec::new(),
            p_columns: Vec::new(),
            q_values: Vec::new(),
            y: Vec::new(),
            dy: Vec::new(),
            uy: Vec::new(),
            gains: Vec::new(),
            epoch_callback: None,
            initial_embedding: None,
            spectral_init: None,
            stop_lying_fired: false,
            cached_perplexity: None,
            fit: None,
        }
    }

    /// Sets an explicit learning rate, overriding the size-scaled default.
    ///
    /// When left unset the learning rate defaults to
    /// `max(n_samples / early_exaggeration / 4, 50)`, following scikit-learn and
    /// openTSNE, which adapts the step size to the dataset.
    ///
    /// # Arguments
    ///
    /// `learning_rate` - new value for the learning rate.
    pub fn learning_rate(&mut self, learning_rate: T) -> &mut Self {
        self.learning_rate = Some(learning_rate);

        self
    }

    /// Sets new epochs, i.e the maximum number of fitting iterations.
    ///
    /// # Arguments
    ///
    /// `epochs` - new value for the epochs.
    pub fn epochs(&mut self, epochs: usize) -> &mut Self {
        self.epochs = epochs;

        self
    }

    /// Sets a new momentum.
    ///
    /// # Arguments
    ///
    /// `momentum` - new value for the momentum.
    pub fn momentum(&mut self, momentum: T) -> &mut Self {
        self.momentum = momentum;

        self
    }

    /// Sets a new final momentum.
    ///
    /// # Arguments
    ///
    /// `final_momentum` - new value for the final momentum.
    pub fn final_momentum(&mut self, final_momentum: T) -> &mut Self {
        self.final_momentum = final_momentum;

        self
    }

    /// Sets a new momentum switch epoch, i.e. the epoch after which the algorithm switches to
    /// `final_momentum` for the map update.
    ///
    /// # Arguments
    ///
    /// `momentum_switch_epoch` - new value for the momentum switch epoch.
    pub fn momentum_switch_epoch(&mut self, momentum_switch_epoch: usize) -> &mut Self {
        self.momentum_switch_epoch = momentum_switch_epoch;

        self
    }

    /// Sets a new stop lying epoch, i.e. the epoch after which the P distribution values become
    /// true, as defined in the original implementation. For epochs < `stop_lying_epoch` the values
    /// of the P distribution are multiplied by the `early_exaggeration` factor (`12.0` by default).
    ///
    /// A value of `0` disables the early exaggeration entirely, useful when warm starting from an
    /// already converged embedding.
    ///
    /// # Arguments
    ///
    /// `stop_lying_epoch` - new value for the stop lying epoch.
    pub fn stop_lying_epoch(&mut self, stop_lying_epoch: usize) -> &mut Self {
        self.stop_lying_epoch = stop_lying_epoch;

        self
    }

    /// Sets the early exaggeration factor, the multiplier applied to the `P`
    /// distribution for epochs before `stop_lying_epoch`. Larger values push
    /// clusters further apart early on. The original recipe uses `12.0`. A value
    /// of `1.0` disables exaggeration (equivalent to `stop_lying_epoch(0)`).
    ///
    /// # Arguments
    ///
    /// `early_exaggeration` - new value for the early exaggeration factor.
    pub fn early_exaggeration(&mut self, early_exaggeration: T) -> &mut Self {
        self.early_exaggeration = early_exaggeration;

        self
    }

    /// Sets a new perplexity value.
    ///
    /// # Arguments
    ///
    /// `perplexity` - new value for the perplexity. It's used so that the bandwidth of the Gaussian
    ///  kernels, is set in such a way that the perplexity of each the conditional distribution *Pi*
    ///  equals a predefined perplexity *u*.
    ///
    /// A good value for perplexity lies between 5.0 and 50.0.
    pub fn perplexity(&mut self, perplexity: T) -> &mut Self {
        self.perplexity = perplexity;

        self
    }

    /// Sets a callback invoked at the end of each fitting epoch by both [`exact`]
    /// and [`barnes_hut`].
    ///
    /// # Arguments
    ///
    /// `callback` - closure called with the zero-based epoch index and a snapshot of
    /// the current embedding, laid out as in the result of [`embedding`]. It can be
    /// used to monitor convergence or to animate intermediate embeddings.
    ///
    /// The callback is invoked sequentially from the calling thread once per epoch.
    /// To observe progress more sparsely simply return early from the closure for
    /// the epochs to skip.
    ///
    /// The callback is never sent to or shared with another thread, so it needs
    /// neither `Send` nor `Sync`. This allows single threaded targets, such as
    /// wasm, to attach a callback over non-`Send` resources (for example one that
    /// posts progress to a worker scope) directly, with no wrapper. Setting such a
    /// callback makes the [`tSNE`] itself non-`Send`/non-`Sync` while it is set,
    /// which is harmless because the fit runs on the owning thread.
    ///
    /// [`exact`]: tSNE::exact
    /// [`barnes_hut`]: tSNE::barnes_hut
    /// [`embedding`]: tSNE::embedding
    ///
    /// # Examples
    ///
    /// ```
    /// use bhtsne::tSNE;
    ///
    /// const N: usize = 100;
    /// const D: usize = 25;
    ///
    /// let data: Vec<f32> = vec![0.0_f32; N * D];
    /// let vectors: Vec<&[f32]> = data.chunks(D).collect();
    ///
    /// let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&vectors);
    /// tsne.epoch_callback(|epoch, embedding| {
    ///     if epoch % 10 == 0 {
    ///         println!("epoch {}: {} coordinates", epoch, embedding.len());
    ///     }
    /// });
    /// ```
    pub fn epoch_callback<C>(&mut self, callback: C) -> &mut Self
    where
        C: FnMut(usize, &[T]) + 'data,
    {
        self.epoch_callback = Some(Box::new(callback));

        self
    }

    /// Seeds the embedding with the given coordinates instead of initializing it
    /// randomly, for warm starts. The seed is consumed by the next fit, which
    /// panics if its length is not `n_samples * D`.
    ///
    /// # Arguments
    ///
    /// `embedding` - row-major initial coordinates.
    pub fn initial_embedding(&mut self, embedding: impl Into<Vec<T>>) -> &mut Self {
        self.initial_embedding = Some(embedding.into());

        self
    }

    /// Use spectral embedding initialization instead of random, with the default
    /// [`SpectralParams`].
    ///
    /// Computes a low-dimensional embedding from the affinity graph's normalized
    /// Laplacian via Chebyshev-filtered subspace iteration with Rayleigh-Ritz
    /// projections. This typically produces better-separated clusters than random
    /// initialization, especially for well-structured data.
    ///
    /// Use [`spectral_init_with`] to tune the solver parameters.
    ///
    /// If an explicit [`initial_embedding`] is also set, it takes precedence and
    /// this flag is ignored.
    ///
    /// [`initial_embedding`]: tSNE::initial_embedding
    /// [`spectral_init_with`]: tSNE::spectral_init_with
    pub fn spectral_init(&mut self) -> &mut Self
    where
        T: Default,
        Dim<D>: SpectralBlock,
    {
        self.spectral_init_with(SpectralParams::default())
    }

    /// Use spectral embedding initialization instead of random, with custom
    /// [`SpectralParams`].
    ///
    /// ```
    /// use bhtsne::{SpectralParams, tSNE};
    ///
    /// let data: Vec<f32> = vec![0.0; 100 * 4];
    /// let samples: Vec<&[f32]> = data.chunks(4).collect();
    /// let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
    /// tsne.spectral_init_with(SpectralParams::new().rounds(3).degree(12));
    /// ```
    ///
    /// If an explicit [`initial_embedding`] is also set, it takes precedence and
    /// this setting is ignored. The solver is monomorphized here, where the
    /// [`SpectralBlock`] bound is available, so the seeding inside the fit needs no
    /// bound of its own.
    ///
    /// [`initial_embedding`]: tSNE::initial_embedding
    pub fn spectral_init_with(&mut self, params: SpectralParams) -> &mut Self
    where
        T: Default,
        Dim<D>: SpectralBlock,
    {
        self.spectral_init = Some((params, tsne::spectral::spectral_embedding::<T, D>));

        self
    }

    /// Returns the computed embedding.
    pub fn embedding(&self) -> Vec<T> {
        self.y.clone()
    }

    /// Returns the Kullback-Leibler divergence of the current embedding, the cost
    /// t-SNE minimizes, or `None` before a fit. Exact after [`exact`], a tree
    /// approximation after [`barnes_hut`]. Recomputed on each call.
    ///
    /// [`exact`]: tSNE::exact
    /// [`barnes_hut`]: tSNE::barnes_hut
    pub fn kl_divergence(&self) -> Option<T>
    where
        T: FftNum,
        Dim<D>: Morton<D>,
    {
        let n_samples = self.data.len();
        match self.fit.as_ref()? {
            Fit::Exact => Some(tsne::evaluate_error::<T, D>(
                &self.p_values,
                &self.y,
                n_samples,
            )),
            Fit::BarnesHut { theta } => Some(
                BarnesHutRepulsion::<T, <Dim<D> as Morton<D>>::Word, D>::new(*theta).error(
                    &self.p_rows,
                    &self.p_columns,
                    &self.p_values,
                    &self.y,
                    n_samples,
                ),
            ),
            Fit::Interpolated => Some(InterpolatedRepulsion::<T, D>::new().error(
                &self.p_rows,
                &self.p_columns,
                &self.p_values,
                &self.y,
                n_samples,
            )),
        }
    }

    /// Performs a parallel exact version of the t-SNE algorithm. Pairwise distances between samples
    /// in the input space will be computed accordingly to the supplied function `distance_f`.
    ///
    /// # Arguments
    ///
    /// `distance_f` - distance function.
    ///
    /// **Do note** that such a distance function needs not to be a metric distance, i.e. it is not
    /// necessary for it so satisfy the triangle inequality. Consequently, the squared euclidean
    /// distance, and many other, can be used.
    pub fn exact<F: Fn(&U, &U) -> T + Send + Sync>(&mut self, distance_f: F) -> &mut Self {
        let data = self.data;
        let n_samples = self.data.len(); // Number of samples in data.

        // Checks that the supplied perplexity is suitable for the number of samples at hand.
        tsne::check_perplexity(&self.perplexity, &n_samples);

        // Number of entries in gradient and gains matrices.
        let grad_entries = n_samples * D;
        // Number of entries in pairwise measures matrices.
        let pairwise_entries = n_samples * n_samples;

        // Prepares the buffers.
        tsne::prepare_buffers(
            &mut self.y,
            &mut self.dy,
            &mut self.uy,
            &mut self.gains,
            grad_entries,
        );
        // Prepare distributions matrices.
        self.p_values.resize(pairwise_entries, T::zero()); // P.
        self.q_values.resize(pairwise_entries, T::zero()); // Q.

        // Alignment prevents false sharing.
        let mut distances: Vec<T> = vec![T::zero(); pairwise_entries];
        // Zeroes the diagonal entries. The distances vector is recycled but the elements
        // corresponding to the diagonal entries of the distance matrix are always kept to 0. and
        // never written on. This hold as an invariant through all the algorithm.
        distances
            .iter_mut()
            .step_by(n_samples + 1)
            .for_each(|d| *d = T::zero());

        // Compute pairwise distances in parallel with the user supplied function.
        // Only upper triangular entries, excluding the diagonal are computed: flat indexes are
        // unraveled to pick such entries.
        tsne::compute_pairwise_distance_matrix(
            &mut distances,
            distance_f,
            |index| &data[*index],
            n_samples,
        );

        // Compute gaussian perplexity in parallel. First, the conditional distribution is computed
        // for each element. Each row of the P matrix is independent from the others, thus, this
        // computation is accordingly parallelized.
        {
            let perplexity = &self.perplexity;
            self.p_values
                .par_chunks_mut(n_samples)
                .zip(distances.par_chunks(n_samples))
                .for_each(|(p_values_row, distances_row)| {
                    tsne::search_beta(p_values_row, distances_row, perplexity);
                });
        }

        // Symmetrize pairwise input similarities. Conditional probabilities must be summed to
        // obtain the joint P distribution.
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let symmetric = self.p_values[j * n_samples + i];
                self.p_values[i * n_samples + j] += symmetric;
                self.p_values[j * n_samples + i] = self.p_values[i * n_samples + j];
            }
        }

        // Normalize P, disable the early exaggeration if requested, and seed the embedding.
        self.finalize_p_and_seed(grad_entries);

        // Resolve the learning rate once: explicit if set, otherwise size-scaled.
        let learning_rate = self.resolve_learning_rate(n_samples);

        // The callback is detached from self for the fit so the epoch loop is free
        // to borrow the other fields mutably; it is put back once the loop ends.
        let (mut epoch_callback, mut snapshot) = self.take_callback_and_snapshot(grad_entries);

        // Main fitting loop.
        for epoch in 0..self.epochs {
            // Compute pairwise squared euclidean distances between embeddings in parallel.
            let (y_chunks, _) = self.y.as_chunks::<D>();
            tsne::compute_pairwise_distance_matrix(
                &mut distances,
                |ith: &[T; D], jth: &[T; D]| {
                    ith.iter()
                        .zip(jth.iter())
                        .map(|(&i, &j)| (i - j).powi(2))
                        .sum()
                },
                |index| &y_chunks[*index],
                n_samples,
            );

            // Computes Q.
            self.q_values
                .par_iter_mut()
                .zip(distances.par_iter())
                .for_each(|(q, d)| *q = (T::one() + *d).recip());

            // Computes the exact gradient in parallel.
            let q_values_sum: T = self.q_values.par_iter().copied().sum::<T>();
            // Precompute the reciprocal so the n^2 inner gradient terms multiply by it instead of
            // each dividing by the same sum, mirroring the Barnes-Hut path's `inverse_q_sum`.
            let inverse_q_sum = q_values_sum.recip();

            // Immutable borrow to self must happen outside of the inner sequential
            // loop. The outer parallel loop already has a mutable borrow.
            let (dy_chunks, _) = self.dy.as_chunks_mut::<D>();
            dy_chunks
                .par_iter_mut()
                .zip(y_chunks.par_iter())
                .zip(self.p_values.par_chunks(n_samples))
                .zip(self.q_values.par_chunks(n_samples))
                .for_each(
                    |(((dy_sample, y_sample), p_values_sample), q_values_sample)| {
                        p_values_sample
                            .iter()
                            .zip(q_values_sample.iter())
                            .zip(y_chunks.iter())
                            .for_each(|((&p, &q), other_sample)| {
                                let m = (p - q * inverse_q_sum) * q;
                                dy_sample
                                    .iter_mut()
                                    .zip(y_sample.iter())
                                    .zip(other_sample.iter())
                                    .for_each(|((dy_el, &y_el), &other_el)| {
                                        *dy_el += (y_el - other_el) * m
                                    });
                            });
                    },
                );

            // Updates the embedding in parallel with gradient descent.
            tsne::update_solution(
                &mut self.y,
                &self.dy,
                &mut self.uy,
                &mut self.gains,
                &learning_rate,
                &self.momentum,
            );

            // Zeroes the gradient.
            self.dy.fill(T::zero());

            // Make solution zero mean.
            tsne::zero_mean::<T, D>(&mut self.y, n_samples);

            self.epoch_tail(epoch, &mut epoch_callback, &mut snapshot);
        }
        // Puts the callback back in place.
        self.epoch_callback = epoch_callback;
        // Clears buffers used for fitting.
        tsne::clear_buffers(&mut self.dy, &mut self.uy, &mut self.gains);
        self.fit = Some(Fit::Exact);

        self
    }

    /// Resolves the learning rate: the explicit value if set, otherwise the
    /// size-scaled `max(n_samples / early_exaggeration / 4, 50)` default, as used
    /// by scikit-learn and openTSNE.
    #[inline]
    fn resolve_learning_rate(&self, n_samples: usize) -> T {
        self.learning_rate.unwrap_or_else(|| {
            let auto =
                T::from(n_samples).unwrap() / self.early_exaggeration / T::from(4.0).unwrap();

            auto.max(T::from(50.0).unwrap())
        })
    }

    /// Normalizes P, undoes the early exaggeration if disabled, and seeds the
    /// embedding. Shared by `exact` and `approximate_fit`.
    fn finalize_p_and_seed(&mut self, grad_entries: usize) {
        // Normalize P values.
        tsne::normalize_p_values(&mut self.p_values, self.early_exaggeration);
        // With no early exaggeration phase, undo the lying immediately.
        if self.stop_lying_epoch == 0 {
            tsne::stop_lying(&mut self.p_values, self.early_exaggeration);
        }

        // Seed: explicit embedding > spectral init > random.
        match self.initial_embedding.take() {
            Some(init) => {
                assert_eq!(
                    init.len(),
                    grad_entries,
                    "error: initial embedding has {} values, expected n_samples * D = {}",
                    init.len(),
                    grad_entries
                );
                self.y.iter_mut().zip(&init).for_each(|(y, &v)| *y = v);
            }
            None => match self.spectral_init {
                Some((params, seeder)) => {
                    let seed = seeder(&self.p_rows, &self.p_columns, &self.p_values, params);
                    self.y.iter_mut().zip(&seed).for_each(|(y, &v)| *y = v);
                }
                None => tsne::random_init(&mut self.y),
            },
        }
    }

    /// Performs a parallel Barnes-Hut approximation of the t-SNE algorithm.
    ///
    /// # Arguments
    ///
    /// * `theta` - determines the accuracy of the approximation. Must be **strictly greater than 0.0**.
    ///   Large values for θ increase the speed of the algorithm but decrease its accuracy.
    ///   For small values of θ it is less probable that a cell in the space partitioning tree will
    ///   be treated as a single point. For θ equal to 0.0 the method degenerates in the exact
    ///   version.
    ///
    /// * `metric_f` - metric function.
    ///
    /// **Do note that** `metric_f` **must be a metric distance**, i.e. it must
    /// satisfy the [triangle inequality](https://en.wikipedia.org/wiki/Triangle_inequality).
    pub fn barnes_hut<F>(&mut self, theta: T, metric_f: F) -> &mut Self
    where
        F: Fn(&U, &U) -> T + Send + Sync,
        Dim<D>: Morton<D>,
    {
        // Validate before doing any work.
        self.validate_fit_params(theta);

        let n_samples = self.data.len();
        if self.has_cached_affinities(n_samples) {
            return self.run_cached(
                n_samples,
                BarnesHutRepulsion::<T, <Dim<D> as Morton<D>>::Word, D>::new(theta),
                Fit::BarnesHut { theta },
            );
        }

        // No cached affinities or mismatched dataset, rebuild.
        let data = self.data;
        // Number of points to consider when approximating the conditional distribution P.
        let n_neighbors: usize = (T::from(3.0).unwrap() * self.perplexity).as_();
        // Build ball tree on the data set.
        let tree = tsne::vptree::VPTree::new(data, &metric_f);

        // The `move` closure owns the tree so `build_affinities` can drop it before
        // the training loop. The `+ 1` is the sample itself, excluded by the search.
        self.approximate_fit(
            n_neighbors,
            move |scratch, index, p_columns_row, distances_row| {
                tree.search(
                    &data[index],
                    index,
                    n_neighbors + 1,
                    (p_columns_row, distances_row),
                    scratch,
                    &metric_f,
                );
            },
            BarnesHutRepulsion::<T, <Dim<D> as Morton<D>>::Word, D>::new(theta),
            Fit::BarnesHut { theta },
        )
    }

    /// Like [`barnes_hut`], but uses caller-supplied nearest neighbors instead of a
    /// vantage point tree, doing no metric evaluations. `neighbors[i]` are sample
    /// `i`'s neighbors by ascending distance, excluding `i`, every row of equal
    /// length `k` (used as `n_neighbors`, pick `k` near `3 * perplexity`). Indices
    /// within a row must be distinct and distances finite and non-negative; these
    /// are not checked.
    ///
    /// # Panics
    ///
    /// If `theta <= 0.0`, the rows are not one per sample, differ in length, are
    /// empty, or hold an out-of-range index.
    ///
    /// [`barnes_hut`]: tSNE::barnes_hut
    pub fn barnes_hut_with_neighbors(
        &mut self,
        theta: T,
        neighbors: &[Vec<Neighbor<T>>],
    ) -> &mut Self
    where
        Dim<D>: Morton<D>,
    {
        // Reject a bad theta up front, on the cached path too, matching `barnes_hut`.
        self.validate_fit_params(theta);
        self.approximate_fit_with_neighbors(
            neighbors,
            BarnesHutRepulsion::<T, <Dim<D> as Morton<D>>::Word, D>::new(theta),
            Fit::BarnesHut { theta },
        )
    }

    /// Validates the shape of a caller-supplied neighbor table and returns the
    /// common per-sample neighbor count. Shared by the Barnes-Hut and FIt-SNE
    /// `*_with_neighbors` entry points.
    ///
    /// # Panics
    ///
    /// If the rows are not one per sample, differ in length, or are empty.
    fn check_neighbors(&self, neighbors: &[Vec<Neighbor<T>>]) -> usize {
        let n_samples = self.data.len();
        assert_eq!(
            neighbors.len(),
            n_samples,
            "error: neighbors has {} rows, expected one per sample = {}",
            neighbors.len(),
            n_samples
        );

        // Rows share one dense n_samples * k block, so all must be the same length.
        let n_neighbors = neighbors.first().map_or(0, Vec::len);
        assert!(
            n_neighbors > 0 && neighbors.iter().all(|row| row.len() == n_neighbors),
            "error: every neighbors row must have the same length, greater than zero."
        );
        n_neighbors
    }

    /// Asserts every neighbor index is a valid sample index. Only needed on a cache
    /// miss, where the indices are actually dereferenced.
    ///
    /// # Panics
    ///
    /// If any neighbor index is `>= n_samples`.
    fn assert_neighbor_indices_in_range(&self, neighbors: &[Vec<Neighbor<T>>]) {
        let n_samples = self.data.len();
        assert!(
            neighbors
                .iter()
                .flatten()
                .all(|neighbor| neighbor.index < n_samples),
            "error: a neighbor index is out of range, every index must be < n_samples = {n_samples}."
        );
    }

    /// Performs a parallel FIt-SNE fit: the FFT-accelerated, interpolation-based
    /// flavor of the algorithm (Linderman et al., 2019). Like [`barnes_hut`] it
    /// builds the sparse affinity graph from a vantage point tree over `metric_f`,
    /// but it approximates the repulsive forces with an `O(n)` interpolation on an
    /// equal-spaced grid rather than a space-partitioning tree, which is typically
    /// faster at large sample counts and carries no `theta` accuracy knob.
    ///
    /// Restricted to embedding dimensionalities `D in {1, 2}`, the range over which
    /// the interpolation grid stays tractable (the `Dim<D>: FftDim` bound), with
    /// `D = 2` the usual choice for visualization.
    ///
    /// # Arguments
    ///
    /// `metric_f` - metric function. As with [`barnes_hut`] it **must be a metric
    /// distance**, i.e. it must satisfy the triangle inequality, since it feeds the
    /// vantage point tree.
    ///
    /// [`barnes_hut`]: tSNE::barnes_hut
    pub fn fit_sne<F>(&mut self, metric_f: F) -> &mut Self
    where
        F: Fn(&U, &U) -> T + Send + Sync,
        T: FftNum,
        Dim<D>: FftDim,
    {
        // Perplexity is the only fit parameter to validate here: there is no theta.
        tsne::check_perplexity(&self.perplexity, &self.data.len());

        let n_samples = self.data.len();
        if self.has_cached_affinities(n_samples) {
            return self.run_cached(n_samples, InterpolatedRepulsion::new(), Fit::Interpolated);
        }

        // No cached affinities or mismatched dataset, rebuild.
        let data = self.data;
        // Number of points to consider when approximating the conditional distribution P.
        let n_neighbors: usize = (T::from(3.0).unwrap() * self.perplexity).as_();
        // Build the vantage point tree on the data set.
        let tree = tsne::vptree::VPTree::new(data, &metric_f);

        self.approximate_fit(
            n_neighbors,
            move |scratch, index, p_columns_row, distances_row| {
                tree.search(
                    &data[index],
                    index,
                    n_neighbors + 1,
                    (p_columns_row, distances_row),
                    scratch,
                    &metric_f,
                );
            },
            InterpolatedRepulsion::new(),
            Fit::Interpolated,
        )
    }

    /// Like [`fit_sne`], but uses caller-supplied nearest neighbors instead of a
    /// vantage point tree, doing no metric evaluations. The `neighbors` table has the
    /// same shape and guarantees as in [`barnes_hut_with_neighbors`].
    ///
    /// # Panics
    ///
    /// If the rows are not one per sample, differ in length, are empty, or hold an
    /// out-of-range index.
    ///
    /// [`fit_sne`]: tSNE::fit_sne
    /// [`barnes_hut_with_neighbors`]: tSNE::barnes_hut_with_neighbors
    pub fn fit_sne_with_neighbors(&mut self, neighbors: &[Vec<Neighbor<T>>]) -> &mut Self
    where
        T: FftNum,
        Dim<D>: FftDim,
    {
        self.approximate_fit_with_neighbors(
            neighbors,
            InterpolatedRepulsion::new(),
            Fit::Interpolated,
        )
    }

    /// Builds the affinity graph through `fill_neighbors`, normalizes and seeds, then
    /// runs the optimization loop with `strategy`. The shared body of every approximate
    /// fit from a freshly built graph, vantage-point-tree or caller-supplied neighbors,
    /// Barnes-Hut or FIt-SNE.
    fn approximate_fit<R, F>(
        &mut self,
        n_neighbors: usize,
        fill_neighbors: F,
        strategy: R,
        fit: Fit<T>,
    ) -> &mut Self
    where
        R: Repulsion<T, D>,
        F: Fn(&mut tsne::vptree::SearchScratch<T>, usize, &mut [u32], &mut [T]) + Send + Sync,
    {
        let grad_entries = self.build_affinities(n_neighbors, fill_neighbors);
        // Normalize P, disable the early exaggeration if requested, and seed the embedding.
        self.finalize_p_and_seed(grad_entries);
        self.run_loop(grad_entries, strategy, fit)
    }

    /// Shared body of the two `*_with_neighbors` entry points: validates the neighbor
    /// table's shape, reuses cached affinities when they encode the same neighbors, and
    /// otherwise builds the graph from the caller-supplied neighbors and fits with
    /// `strategy`. The repulsion strategy and the [`Fit`] marker are the only
    /// differences between the Barnes-Hut and FIt-SNE variants.
    fn approximate_fit_with_neighbors<R>(
        &mut self,
        neighbors: &[Vec<Neighbor<T>>],
        strategy: R,
        fit: Fit<T>,
    ) -> &mut Self
    where
        R: Repulsion<T, D>,
    {
        let n_samples = self.data.len();
        let n_neighbors = self.check_neighbors(neighbors);

        // If cached affinities encode the same neighbors, reuse them. Their indices are
        // provably valid from the run that produced the cache.
        if self.cached_affinities_match_neighbors(neighbors) {
            return self.run_cached(n_samples, strategy, fit);
        }

        // Cache miss: indices will be dereferenced, validate them.
        self.assert_neighbor_indices_in_range(neighbors);

        self.approximate_fit(
            n_neighbors,
            |_, index, p_columns_row, distances_row| {
                copy_neighbor_row(neighbors, index, p_columns_row, distances_row);
            },
            strategy,
            fit,
        )
    }

    /// Runs the optimization epoch loop with a pluggable repulsion `strategy`. Called
    /// after the affinity graph is built, normalized, and the embedding seeded.
    ///
    /// `fit` is the marker recorded on completion so [`kl_divergence`] can pick the
    /// matching cost evaluation.
    ///
    /// [`gradient_descent_step`]: tsne::gradient_descent_step
    /// [`kl_divergence`]: tSNE::kl_divergence
    fn run_loop<R>(&mut self, grad_entries: usize, mut strategy: R, fit: Fit<T>) -> &mut Self
    where
        R: Repulsion<T, D>,
    {
        let n_samples = self.data.len();
        // Attractive and repulsive force buffers, reused across epochs.
        let mut positive_forces: Vec<T> = vec![T::zero(); grad_entries];
        let mut negative_forces: Vec<T> = vec![T::zero(); grad_entries];

        // The callback is detached from self for the fit so the epoch loop is free
        // to borrow the other fields mutably; it is put back once the loop ends.
        let (mut epoch_callback, mut snapshot) = self.take_callback_and_snapshot(grad_entries);

        // Loop-invariant.
        let learning_rate = self.resolve_learning_rate(n_samples);

        for epoch in 0..self.epochs {
            // Forces and the reciprocal of the Q normalizer Z from the repulsion strategy.
            let inverse_norm = strategy.step(
                &self.y,
                &self.p_rows,
                &self.p_columns,
                &self.p_values,
                &mut positive_forces,
                &mut negative_forces,
            );

            // Fuse the gradient (`positive - negative * inverse_norm`) into the update.
            tsne::gradient_descent_step::<T, D>(
                &mut self.y,
                &positive_forces,
                &negative_forces,
                &mut self.uy,
                &mut self.gains,
                tsne::GradientStep {
                    learning_rate,
                    momentum: self.momentum,
                    inverse_norm,
                },
            );

            // Make the solution zero mean.
            tsne::zero_mean::<T, D>(&mut self.y, n_samples);

            self.epoch_tail(epoch, &mut epoch_callback, &mut snapshot);
        }
        // Puts the callback back in place.
        self.epoch_callback = epoch_callback;
        // Clears buffers used for fitting.
        tsne::clear_buffers(&mut self.dy, &mut self.uy, &mut self.gains);
        self.fit = Some(fit);

        self
    }

    /// Returns the affinity graph from the last `barnes_hut` fit in pristine
    /// (pre-exaggeration) form, or `None` if no Barnes-Hut fit has run.
    /// Its values sum to approximately 1.
    pub fn affinities(&self) -> Option<SparseAffinities<T>> {
        if self.p_rows.is_empty() {
            return None;
        }
        // Undo exaggeration unless stop_lying has already undone it.
        let values = if self.stop_lying_fired {
            self.p_values.clone()
        } else {
            let scale = self.early_exaggeration.recip();
            self.p_values.iter().map(|v| *v * scale).collect()
        };

        Some(SparseAffinities {
            rows: self.p_rows.clone(),
            columns: self.p_columns.clone(),
            values,
            perplexity: self.perplexity,
        })
    }

    /// Computes the spectral embedding of the affinity graph and returns it as a
    /// row-major matrix of `n * D` values, without touching the fitting state. To
    /// seed a fit with it, use [`spectral_init`] instead, or pass the result to
    /// [`initial_embedding`].
    ///
    /// Runs a Chebyshev-filtered subspace iteration on the shifted similarity
    /// matrix `M = (I + D^{-1/2} P D^{-1/2}) / 2` with a fixed budget of sparse
    /// matvecs, extracting the leading nontrivial eigenvector estimates with
    /// Rayleigh-Ritz projections. The result separates well-connected components
    /// and is suitable as a t-SNE seed.
    ///
    /// Uses the affinity graph already built by [`barnes_hut`] or provided via
    /// [`with_affinities`]. Available for the dimensionalities carrying a
    /// [`SpectralBlock`] impl.
    ///
    /// [`spectral_init`]: tSNE::spectral_init
    /// [`initial_embedding`]: tSNE::initial_embedding
    /// [`barnes_hut`]: tSNE::barnes_hut
    /// [`with_affinities`]: tSNE::with_affinities
    pub fn spectral_embedding(&self) -> Vec<T>
    where
        T: Default,
        Dim<D>: SpectralBlock,
    {
        self.spectral_embedding_with(SpectralParams::default())
    }

    /// Same as [`spectral_embedding`], with custom [`SpectralParams`].
    ///
    /// [`spectral_embedding`]: tSNE::spectral_embedding
    pub fn spectral_embedding_with(&self, params: SpectralParams) -> Vec<T>
    where
        T: Default,
        Dim<D>: SpectralBlock,
    {
        tsne::spectral::spectral_embedding::<T, D>(
            &self.p_rows,
            &self.p_columns,
            &self.p_values,
            params,
        )
    }

    /// Injects a previously extracted affinity graph. The next `barnes_hut`
    /// call will reuse it and skip the neighbor search.
    ///
    /// [`barnes_hut`]: tSNE::barnes_hut
    pub fn with_affinities(&mut self, affinities: SparseAffinities<T>) -> &mut Self {
        self.perplexity = affinities.perplexity;
        self.p_rows = affinities.rows;
        self.p_columns = affinities.columns;
        self.p_values = affinities.values;
        self.cached_perplexity = Some(self.perplexity);

        self
    }

    /// Builds the sparse, symmetric affinity graph `P` shared by every approximate
    /// fitting strategy.
    ///
    /// # Arguments
    ///
    /// * `n_neighbors` - number of nearest neighbors per sample.
    ///
    /// * `fill_neighbors` - writes sample `index`'s neighbor indices and distances.
    fn build_affinities<F>(&mut self, n_neighbors: usize, fill_neighbors: F) -> usize
    where
        F: Fn(&mut tsne::vptree::SearchScratch<T>, usize, &mut [u32], &mut [T]) + Send + Sync,
    {
        let n_samples = self.data.len(); // Number of samples in data.
        // Number of entries in gradient and gains matrices.
        let grad_entries = n_samples * D;
        // Number of entries in pairwise measures matrices.
        let pairwise_entries = n_samples * n_neighbors;

        // Prepare buffers. The approximate update fuses the gradient into the gradient-descent step,
        // so unlike `exact` it never materializes a gradient buffer: allocate only the embedding,
        // momentum buffer, and gains, leaving `dy` empty.
        self.y.resize(grad_entries, T::zero());
        self.uy.resize(grad_entries, T::zero());
        self.gains.resize(grad_entries, T::one());
        // The P distribution values are restricted to a subset of size n_neighbors for each input
        // sample.
        self.p_values.resize(pairwise_entries, T::zero());

        // This vector is used to keep track of the indexes for each nearest neighbors of each
        // sample. There's a one to one correspondence between the elements of p_columns
        // an the elements of p_values: for each row i of length n_neighbors of such matrices it
        // holds that p_columns[i][j] corresponds to the index sample which contributes
        // to p_values[i][j]. This vector is freed inside symmetrize_sparse_matrix.
        let mut p_columns: Vec<u32> = vec![0u32; pairwise_entries];

        // Fill the neighbor rows, then fit the per-point Gaussian bandwidth.
        {
            // Distances buffer.
            let mut distances: Vec<T> = vec![T::zero(); pairwise_entries];

            let perplexity = &self.perplexity; // Immutable borrow must be outside.
            self.p_values
                .par_chunks_mut(n_neighbors)
                .zip(distances.par_chunks_mut(n_neighbors))
                .zip(p_columns.par_chunks_mut(n_neighbors))
                .enumerate()
                .for_each_init(
                    tsne::vptree::SearchScratch::default,
                    |scratch, (index, ((p_values_row, distances_row), p_columns_row))| {
                        // Writes the indices and the distances of the nearest neighbors of the sample.
                        fill_neighbors(scratch, index, p_columns_row, distances_row);
                        debug_assert!(!p_columns_row.contains(&(index as u32)));
                        tsne::search_beta(p_values_row, distances_row, perplexity);
                    },
                );
        }

        // Free whatever the filler owns (the vantage point tree) before training.
        drop(fill_neighbors);

        // Symmetrize sparse P matrix.
        tsne::symmetrize_sparse_matrix(
            &mut self.p_rows,
            &mut self.p_columns,
            p_columns,
            &mut self.p_values,
            n_samples,
            &n_neighbors,
        );

        // Record the perplexity used so a later change invalidates the cache.
        self.cached_perplexity = Some(self.perplexity);

        grad_entries
    }

    /// Detaches the epoch callback from `self` for the duration of a fit, so the
    /// epoch loop is free to borrow the other fields mutably. Returns it together
    /// with the scratch buffer its snapshots are copied into, empty when there is
    /// no callback. The caller restores the callback once the loop ends.
    ///
    /// # Arguments
    ///
    /// `grad_entries` - length of the embedding, i.e. the size of the snapshot buffer.
    fn take_callback_and_snapshot(
        &mut self,
        grad_entries: usize,
    ) -> (Option<EpochCallback<'data, T>>, Vec<T>) {
        let epoch_callback = self.epoch_callback.take();
        let snapshot = match epoch_callback {
            Some(_) => vec![T::zero(); grad_entries],
            None => Vec::new(),
        };

        (epoch_callback, snapshot)
    }

    /// Shared tail of each training epoch: stops the early exaggeration at the
    /// right epoch, switches momentum, and fires the epoch callback.
    ///
    /// # Arguments
    ///
    /// * `epoch` - zero-based index of the epoch that just completed.
    ///
    /// * `epoch_callback` - callback detached from `self` for the fit, invoked
    ///   here with a snapshot of the current embedding.
    ///
    /// * `snapshot` - scratch buffer the embedding is copied into before being
    ///   handed to the callback. Unused when there is no callback.
    fn epoch_tail(
        &mut self,
        epoch: usize,
        epoch_callback: &mut Option<EpochCallback<'data, T>>,
        snapshot: &mut [T],
    ) {
        if epoch == self.stop_lying_epoch && epoch != 0 {
            tsne::stop_lying(&mut self.p_values, self.early_exaggeration);
            self.stop_lying_fired = true;
        }
        if epoch == self.momentum_switch_epoch {
            self.momentum = self.final_momentum;
        }
        if let Some(callback) = epoch_callback.as_mut() {
            snapshot.copy_from_slice(&self.y);
            callback(epoch, snapshot);
        }
    }

    /// Validates `theta` and the perplexity before any expensive setup.
    fn validate_fit_params(&self, theta: T) {
        assert!(
            theta > T::zero(),
            "error: theta value must be greater than 0.0.
            A value of 0.0 corresponds to using the exact version of the algorithm."
        );
        tsne::check_perplexity(&self.perplexity, &self.data.len());
    }

    /// Returns true if cached affinities match the current dataset.
    fn has_cached_affinities(&self, n_samples: usize) -> bool {
        !self.p_rows.is_empty()
            && self.p_rows.len() == n_samples + 1
            && self.cached_perplexity == Some(self.perplexity)
    }

    /// Returns true if cached affinities encode the same neighbor indices as
    /// the provided neighbors slice. Returns false if affinities are absent
    /// or neighbors differ.
    fn cached_affinities_match_neighbors(&self, neighbors: &[Vec<Neighbor<T>>]) -> bool {
        if self.p_rows.is_empty()
            || self.p_rows.len() != neighbors.len() + 1
            || self.cached_perplexity != Some(self.perplexity)
        {
            return false;
        }

        let p_rows = &self.p_rows;
        let p_columns = &self.p_columns;
        neighbors.iter().enumerate().all(|(i, row)| {
            let start = p_rows[i];
            let end = p_rows[i + 1];
            (end - start) == row.len()
                && row
                    .iter()
                    .enumerate()
                    .all(|(j, neighbor)| p_columns[start + j] == neighbor.index as u32)
        })
    }

    /// Prepares the optimization buffers and seeds the embedding when reusing a
    /// cached affinity graph, returning the embedding length `n_samples * D`.
    ///
    /// Shared by the Barnes-Hut and FIt-SNE cached paths, which differ only in the loop that
    /// follows.
    fn prepare_cached(&mut self, n_samples: usize) -> usize {
        self.stop_lying_fired = false;
        let grad_entries = n_samples * D;
        self.y.resize(grad_entries, T::zero());
        self.uy.resize(grad_entries, T::zero());
        self.gains.resize(grad_entries, T::one());
        self.finalize_p_and_seed(grad_entries);

        grad_entries
    }

    /// Reuses the cached affinity graph: prepares the optimization buffers, seeds the
    /// embedding, and runs the loop with `strategy`. Shared by the Barnes-Hut and
    /// FIt-SNE cached paths, which differ only in the strategy and the [`Fit`] marker.
    fn run_cached<R>(&mut self, n_samples: usize, strategy: R, fit: Fit<T>) -> &mut Self
    where
        R: Repulsion<T, D>,
    {
        let grad_entries = self.prepare_cached(n_samples);

        self.run_loop(grad_entries, strategy, fit)
    }
}

#[inline]
fn copy_neighbor_row<T: Copy>(
    neighbors: &[Vec<Neighbor<T>>],
    index: usize,
    p_columns_row: &mut [u32],
    distances_row: &mut [T],
) {
    p_columns_row
        .iter_mut()
        .zip(distances_row.iter_mut())
        .zip(neighbors[index].iter())
        .for_each(|((column, distance), neighbor)| {
            *column = neighbor.index as u32;
            *distance = neighbor.distance;
        });
}

#[cfg(test)]
mod test;
