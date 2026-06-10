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
//! ```no_run
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
//! const NO_DIMS: u8 = 2;        // Dimensionality of the embedded space.
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
//! bhtsne::tSNE::new(&samples)
//!     .embedding_dim(NO_DIMS)
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
mod tsne;

#[cfg(test)]
mod test;

use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

#[cfg(feature = "csv")]
use std::{error::Error, fs::File};

use num_traits::{Float, cast::AsPrimitive};

use crossbeam::utils::CachePadded;

use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::{ParallelSlice, ParallelSliceMut},
};

/// Boxed closure invoked at the end of each fitting epoch with the epoch index and a
/// snapshot of the current embedding. See [`tSNE::epoch_callback`].
///
/// The `Send + Sync` bounds only serve to keep [`tSNE`] itself `Send + Sync`, the
/// callback is only ever invoked sequentially from the fitting thread.
pub type EpochCallback<'data, T> = Box<dyn FnMut(usize, &[T]) + Send + Sync + 'data>;

/// Records which fitting routine last ran, so [`tSNE::kl_divergence`] can pick
/// the matching cost evaluation.
enum Fit<T> {
    Exact,
    BarnesHut { theta: T },
}

/// A sample's nearest neighbor for [`tSNE::barnes_hut_with_neighbors`]: its index
/// and the distance (not a similarity) to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Neighbor<T> {
    pub index: usize,
    pub distance: T,
}

/// t-distributed stochastic neighbor embedding. Provides a parallel implementation of both the
/// exact version of the algorithm and the tree accelerated one leveraging space partitioning trees.
#[allow(non_camel_case_types)]
pub struct tSNE<'data, T, U>
where
    T: Send + Sync + Float + Sum + DivAssign + MulAssign + AddAssign + SubAssign,
    U: Send + Sync,
{
    data: &'data [U],
    learning_rate: T,
    epochs: usize,
    momentum: T,
    final_momentum: T,
    momentum_switch_epoch: usize,
    stop_lying_epoch: usize,
    embedding_dim: u8,
    perplexity: T,
    p_values: Vec<CachePadded<T>>,
    p_rows: Vec<usize>,
    p_columns: Vec<usize>,
    q_values: Vec<CachePadded<T>>,
    y: Vec<CachePadded<T>>,
    dy: Vec<CachePadded<T>>,
    uy: Vec<CachePadded<T>>,
    gains: Vec<CachePadded<T>>,
    epoch_callback: Option<EpochCallback<'data, T>>,
    initial_embedding: Option<Vec<T>>,
    fit: Option<Fit<T>>,
}

impl<'data, T, U> tSNE<'data, T, U>
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
    /// * `learning_rate = 200`
    /// * `epochs = 1000`
    /// * `momentum = 0.5`
    /// * `final_momentum = 0.8`
    /// * `stop_lying_epoch = 250`
    /// * `embedding_dim = 2`
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
            learning_rate: T::from(200.0).unwrap(),
            epochs: 1000,
            momentum: T::from(0.5).unwrap(),
            final_momentum: T::from(0.8).unwrap(),
            momentum_switch_epoch: 250,
            stop_lying_epoch: 250,
            embedding_dim: 2,
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
            fit: None,
        }
    }

    /// Sets a new learning rate.
    ///
    /// # Arguments
    ///
    /// `learning_rate` - new value for the learning rate.
    pub fn learning_rate(&mut self, learning_rate: T) -> &mut Self {
        self.learning_rate = learning_rate;

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
    /// of the P distribution are multiplied by a factor equal to `12.0`.
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

    /// Sets a new value for the embedding dimension.
    ///
    /// # Arguments
    ///
    /// `embedding_dim` - new value for the embedding space dimensionality.
    pub fn embedding_dim(&mut self, embedding_dim: u8) -> &mut Self {
        self.embedding_dim = embedding_dim;

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
    /// The `Send + Sync` bounds only serve to keep [`tSNE`] itself `Send + Sync`.
    /// On single threaded targets, such as wasm, closures over non `Send` resources
    /// can be made compatible with a wrapper like the `send_wrapper` crate.
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
        C: FnMut(usize, &[T]) + Send + Sync + 'data,
    {
        self.epoch_callback = Some(Box::new(callback));

        self
    }

    /// Seeds the embedding with the given coordinates instead of initializing it
    /// randomly, for warm starts. The seed is consumed by the next fit, which
    /// panics if its length is not `n_samples * embedding_dim`.
    ///
    /// # Arguments
    ///
    /// `embedding` - row-major initial coordinates.
    pub fn initial_embedding(&mut self, embedding: impl Into<Vec<T>>) -> &mut Self {
        self.initial_embedding = Some(embedding.into());

        self
    }

    /// Returns the computed embedding.
    pub fn embedding(&self) -> Vec<T> {
        self.y.iter().map(|x| **x).collect()
    }

    /// Returns the Kullback-Leibler divergence of the current embedding, the cost
    /// t-SNE minimizes, or `None` before a fit. Exact after [`exact`], a tree
    /// approximation after [`barnes_hut`]. Recomputed on each call.
    ///
    /// [`exact`]: tSNE::exact
    /// [`barnes_hut`]: tSNE::barnes_hut
    pub fn kl_divergence(&self) -> Option<T> {
        let n_samples = self.data.len();
        let embedding_dim = self.embedding_dim as usize;
        match self.fit.as_ref()? {
            Fit::Exact => Some(tsne::evaluate_error(
                &self.p_values,
                &self.y,
                n_samples,
                embedding_dim,
            )),
            Fit::BarnesHut { theta } => Some(tsne::evaluate_error_approximately(
                &self.p_rows,
                &self.p_columns,
                &self.p_values,
                &self.y,
                n_samples,
                embedding_dim,
                *theta,
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

        let embedding_dim = self.embedding_dim as usize;
        // NUmber of entries in gradient and gains matrices.
        let grad_entries = n_samples * embedding_dim;
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
        self.p_values.resize(pairwise_entries, T::zero().into()); // P.
        self.q_values.resize(pairwise_entries, T::zero().into()); // Q.

        // Alignment prevents false sharing.
        let mut distances: Vec<CachePadded<T>> = vec![T::zero().into(); pairwise_entries];
        // Zeroes the diagonal entries. The distances vector is recycled but the elements
        // corresponding to the diagonal entries of the distance matrix are always kept to 0. and
        // never written on. This hold as an invariant through all the algorithm.
        for i in 0..n_samples {
            distances[i * n_samples + i] = T::zero().into();
        }

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
                let symmetric = *self.p_values[j * n_samples + i];
                *self.p_values[i * n_samples + j] += symmetric;
                *self.p_values[j * n_samples + i] = *self.p_values[i * n_samples + j];
            }
        }

        // Normalize P, disable the early exaggeration if requested, and seed the embedding.
        self.finalize_p_and_seed(grad_entries);

        // Vector used to store the mean values for each embedding dimension. It's used
        // to make the solution zero mean.
        let mut means: Vec<T> = vec![T::zero(); embedding_dim];

        // The callback is moved out of self so that the epoch loop is free to borrow
        // the other fields mutably. It is put back at the end of the fitting.
        let mut epoch_callback = self.epoch_callback.take();
        // Scratch buffer for the embedding snapshots passed to the callback.
        let mut snapshot: Vec<T> = match epoch_callback {
            Some(_) => vec![T::zero(); grad_entries],
            None => Vec::new(),
        };

        // Main fitting loop.
        for epoch in 0..self.epochs {
            // Compute pairwise squared euclidean distances between embeddings in parallel.
            tsne::compute_pairwise_distance_matrix(
                &mut distances,
                |ith: &[CachePadded<T>], jth: &[CachePadded<T>]| {
                    ith.iter()
                        .zip(jth.iter())
                        .map(|(&i, &j)| (*i - *j).powi(2))
                        .sum()
                },
                |index| &self.y[index * embedding_dim..index * embedding_dim + embedding_dim],
                n_samples,
            );

            // Computes Q.
            self.q_values
                .par_iter_mut()
                .zip(distances.par_iter())
                .for_each(|(q, d)| **q = T::one() / (T::one() + **d));

            // Computes the exact gradient in parallel.
            let q_values_sum: T = self.q_values.par_iter().map(|&q| *q).sum();

            // Immutable borrow to self must happen outside of the inner sequential
            // loop. The outer parallel loop already has a mutable borrow.
            let y = &self.y;
            self.dy
                .par_chunks_mut(embedding_dim)
                .zip(self.y.par_chunks(embedding_dim))
                .zip(self.p_values.par_chunks(n_samples))
                .zip(self.q_values.par_chunks(n_samples))
                .for_each(
                    |(((dy_sample, y_sample), p_values_sample), q_values_sample)| {
                        p_values_sample
                            .iter()
                            .zip(q_values_sample.iter())
                            .zip(y.chunks(embedding_dim))
                            .for_each(|((&p, &q), other_sample)| {
                                let m: T = (*p - *q / q_values_sum) * *q;
                                dy_sample
                                    .iter_mut()
                                    .zip(y_sample.iter())
                                    .zip(other_sample.iter())
                                    .for_each(|((dy_el, &y_el), &other_el)| {
                                        **dy_el += (*y_el - *other_el) * m
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
                &self.learning_rate,
                &self.momentum,
            );

            // Zeroes the gradient.
            self.dy.iter_mut().for_each(|el| **el = T::zero());

            // Make solution zero mean.
            tsne::zero_mean(&mut means, &mut self.y, n_samples, embedding_dim);

            // Stop lying about the P-values if the time is right. Epoch 0 is
            // handled before the loop, skip it here to avoid dividing twice.
            if epoch == self.stop_lying_epoch && epoch != 0 {
                tsne::stop_lying(&mut self.p_values);
            }

            // Switches momentum if the time is right.
            if epoch == self.momentum_switch_epoch {
                self.momentum = self.final_momentum;
            }

            // Reports the embedding at the end of the epoch.
            if let Some(callback) = epoch_callback.as_mut() {
                snapshot
                    .iter_mut()
                    .zip(self.y.iter())
                    .for_each(|(dst, src)| *dst = **src);
                callback(epoch, &snapshot);
            }
        }
        // Puts the callback back in place.
        self.epoch_callback = epoch_callback;
        // Clears buffers used for fitting.
        tsne::clear_buffers(&mut self.dy, &mut self.uy, &mut self.gains);
        self.fit = Some(Fit::Exact);

        self
    }

    /// Normalizes P, undoes the early exaggeration if disabled, and seeds the
    /// embedding. Shared by `exact` and `barnes_hut_fit`.
    fn finalize_p_and_seed(&mut self, grad_entries: usize) {
        // Normalize P values.
        tsne::normalize_p_values(&mut self.p_values);
        // With no early exaggeration phase, undo the lying immediately.
        if self.stop_lying_epoch == 0 {
            tsne::stop_lying(&mut self.p_values);
        }

        // Seed from the supplied embedding if any, otherwise initialize randomly.
        match self.initial_embedding.take() {
            Some(init) => {
                assert_eq!(
                    init.len(),
                    grad_entries,
                    "error: initial embedding has {} values, expected n_samples * embedding_dim = {}",
                    init.len(),
                    grad_entries
                );
                self.y.iter_mut().zip(&init).for_each(|(y, &v)| **y = v);
            }
            None => tsne::random_init(&mut self.y),
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
    {
        // Validate before building the tree so misuse does not pay for it first.
        self.validate_fit_params(theta);

        let data = self.data;
        // Number of points to consider when approximating the conditional distribution P.
        let n_neighbors: usize = (T::from(3.0).unwrap() * self.perplexity).as_();

        // Build ball tree on the data set.
        let tree = tsne::vptree::VPTree::new(data, &metric_f);

        // The `move` closure owns the tree so `barnes_hut_fit` can drop it before
        // the training loop. The `+ 1` is the sample itself, excluded by the search.
        self.barnes_hut_fit(
            theta,
            n_neighbors,
            move |index, p_columns_row, distances_row| {
                tree.search(
                    &data[index],
                    index,
                    n_neighbors + 1,
                    p_columns_row,
                    distances_row,
                    &metric_f,
                );
            },
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
    ) -> &mut Self {
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

        // These indices later index P rows in symmetrize_sparse_matrix; reject
        // out-of-range here instead of panicking cryptically there.
        assert!(
            neighbors
                .iter()
                .flatten()
                .all(|neighbor| neighbor.index < n_samples),
            "error: a neighbor index is out of range, every index must be < n_samples = {n_samples}."
        );

        self.barnes_hut_fit(theta, n_neighbors, |index, p_columns_row, distances_row| {
            p_columns_row
                .iter_mut()
                .zip(distances_row.iter_mut())
                .zip(neighbors[index].iter())
                .for_each(|((column, distance), neighbor)| {
                    **column = neighbor.index;
                    **distance = neighbor.distance;
                });
        })
    }

    /// Shared Barnes-Hut fit. `fill_neighbors(index, p_columns_row, distances_row)`
    /// writes sample `index`'s neighbor indices and distances; the rest is common.
    fn barnes_hut_fit<F>(&mut self, theta: T, n_neighbors: usize, fill_neighbors: F) -> &mut Self
    where
        F: Fn(usize, &mut [CachePadded<usize>], &mut [CachePadded<T>]) + Send + Sync,
    {
        // Idempotent: `barnes_hut` already validated before building its tree.
        self.validate_fit_params(theta);

        let n_samples = self.data.len(); // Number of samples in data.
        let embedding_dim = self.embedding_dim as usize;
        // NUmber of entries in gradient and gains matrices.
        let grad_entries = n_samples * embedding_dim;
        // Number of entries in pairwise measures matrices.
        let pairwise_entries = n_samples * n_neighbors;

        // Prepare buffers
        tsne::prepare_buffers(
            &mut self.y,
            &mut self.dy,
            &mut self.uy,
            &mut self.gains,
            grad_entries,
        );
        // The P distribution values are restricted to a subset of size n_neighbors for each input
        // sample.
        self.p_values.resize(pairwise_entries, T::zero().into());

        // This vector is used to keep track of the indexes for each nearest neighbors of each
        // sample. There's a one to one correspondence between the elements of p_columns
        // an the elements of p_values: for each row i of length n_neighbors of such matrices it
        // holds that p_columns[i][j] corresponds to the index sample which contributes
        // to p_values[i][j]. This vector is freed inside symmetrize_sparse_matrix.
        let mut p_columns: Vec<CachePadded<usize>> = vec![0.into(); pairwise_entries];

        // Fill the neighbor rows, then fit the per-point Gaussian bandwidth.
        {
            // Distances buffer.
            let mut distances: Vec<CachePadded<T>> = vec![T::zero().into(); pairwise_entries];

            let perplexity = &self.perplexity; // Immutable borrow must be outside.
            self.p_values
                .par_chunks_mut(n_neighbors)
                .zip(distances.par_chunks_mut(n_neighbors))
                .zip(p_columns.par_chunks_mut(n_neighbors))
                .enumerate()
                .for_each(|(index, ((p_values_row, distances_row), p_columns_row))| {
                    // Writes the indices and the distances of the nearest neighbors of the sample.
                    fill_neighbors(index, p_columns_row, distances_row);
                    debug_assert!(!p_columns_row.iter().any(|&i| *i == index));
                    tsne::search_beta(p_values_row, distances_row, perplexity);
                });
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

        // Normalize P, disable the early exaggeration if requested, and seed the embedding.
        self.finalize_p_and_seed(grad_entries);

        // Prepares buffers for Barnes-Hut algorithm.
        let mut positive_forces: Vec<CachePadded<T>> = vec![T::zero().into(); grad_entries];
        let mut negative_forces: Vec<CachePadded<T>> = vec![T::zero().into(); grad_entries];
        let mut forces_buffer: Vec<CachePadded<T>> = vec![T::zero().into(); grad_entries];
        let mut q_sums: Vec<CachePadded<T>> = vec![T::zero().into(); n_samples];

        // Vector used to store the mean values for each embedding dimension. It's used
        // to make the solution zero mean.
        let mut means: Vec<T> = vec![T::zero(); embedding_dim];

        // The callback is moved out of self so that the epoch loop is free to borrow
        // the other fields mutably. It is put back at the end of the fitting.
        let mut epoch_callback = self.epoch_callback.take();
        // Scratch buffer for the embedding snapshots passed to the callback.
        let mut snapshot: Vec<T> = match epoch_callback {
            Some(_) => vec![T::zero(); grad_entries],
            None => Vec::new(),
        };

        // Main Training loop.
        for epoch in 0..self.epochs {
            {
                // Construct space partitioning tree on current embedding.
                let tree = tsne::sptree::SPTree::new(embedding_dim, &self.y, n_samples);
                // Check if the SPTree is correct.
                debug_assert!(tree.is_correct(), "error: SPTree is not correct.");

                // Computes forces using the Barnes-Hut algorithm in parallel.
                // Each chunk of positive_forces and negative_forces is associated to a distinct
                // embedded sample in y. As a consequence of this the computation can be done in
                // parallel.
                positive_forces
                    .par_chunks_mut(embedding_dim)
                    .zip(negative_forces.par_chunks_mut(embedding_dim))
                    .zip(forces_buffer.par_chunks_mut(embedding_dim))
                    .zip(q_sums.par_iter_mut())
                    .zip(self.y.par_chunks(embedding_dim))
                    .enumerate()
                    .for_each(
                        |(
                            index,
                            (
                                (
                                    ((positive_forces_row, negative_forces_row), forces_buffer_row),
                                    q_sum,
                                ),
                                sample,
                            ),
                        )| {
                            tree.compute_edge_forces(
                                index,
                                sample,
                                &self.p_rows,
                                &self.p_columns,
                                &self.p_values,
                                forces_buffer_row,
                                positive_forces_row,
                            );
                            tree.compute_non_edge_forces(
                                index,
                                theta,
                                negative_forces_row,
                                forces_buffer_row,
                                q_sum,
                            );
                        },
                    );
            }

            // Compute final Barnes-Hut t-SNE gradient approximation.
            // Reduces partial sums of Q distribution.
            let q_sum: T = q_sums.par_iter_mut().map(|sum| **sum).sum();
            self.dy
                .par_iter_mut()
                .zip(positive_forces.par_iter_mut())
                .zip(negative_forces.par_iter_mut())
                .for_each(|((grad, pf), nf)| {
                    **grad = **pf - (**nf / q_sum);
                    **pf = T::zero();
                    **nf = T::zero();
                });
            // Zeroes Q-sums.
            q_sums.iter_mut().for_each(|sum| **sum = T::zero());

            // Updates the embedding in parallel with gradient descent.
            tsne::update_solution(
                &mut self.y,
                &self.dy,
                &mut self.uy,
                &mut self.gains,
                &self.learning_rate,
                &self.momentum,
            );

            // Make solution zero-mean.
            tsne::zero_mean(&mut means, &mut self.y, n_samples, embedding_dim);

            // Stop lying about the P-values if the time is right. Epoch 0 is
            // handled before the loop, skip it here to avoid dividing twice.
            if epoch == self.stop_lying_epoch && epoch != 0 {
                tsne::stop_lying(&mut self.p_values);
            }

            // Switches momentum if the time is right.
            if epoch == self.momentum_switch_epoch {
                self.momentum = self.final_momentum;
            }

            // Reports the embedding at the end of the epoch.
            if let Some(callback) = epoch_callback.as_mut() {
                snapshot
                    .iter_mut()
                    .zip(self.y.iter())
                    .for_each(|(dst, src)| *dst = **src);
                callback(epoch, &snapshot);
            }
        }
        // Puts the callback back in place.
        self.epoch_callback = epoch_callback;
        // Clears buffers used for fitting.
        tsne::clear_buffers(&mut self.dy, &mut self.uy, &mut self.gains);
        self.fit = Some(Fit::BarnesHut { theta });

        self
    }

    /// Writes the embedding to a csv file. If the embedding space dimensionality is either equal to
    /// 2 or 3 the resulting csv file will have some simple headers:
    ///
    /// * x, y for 2 dimensions.
    ///
    /// * x, y, z for 3 dimensions.
    ///
    /// # Arguments
    ///
    /// * `file_path` - path of the file to write the embedding to.
    ///
    /// # Errors
    ///
    /// Returns an error is something goes wrong during the I/O operations.
    #[cfg(feature = "csv")]
    pub fn write_csv(&mut self, path: &str) -> Result<&mut Self, Box<dyn Error>>
    where
        T: Float + ToString,
    {
        let mut writer = csv::Writer::from_path(path)?;

        // String-ify the embedding.
        let to_write = self
            .y
            .iter()
            .map(|&el| (*el).to_string())
            .collect::<Vec<String>>();

        // Write headers.
        match self.embedding_dim {
            2 => writer.write_record(["x", "y"])?,
            3 => writer.write_record(["x", "y", "z"])?,
            _ => (), // Write no headers for embedding dimensions greater that 3.
        }
        // Write records.
        for record in to_write.chunks(self.embedding_dim as usize) {
            writer.write_record(record)?
        }
        // Final flush.
        writer.flush()?;

        // Everything went smooth.
        Ok(self)
    }
}

/// Loads data from a csv file.
///
/// # Arguments
///
/// * `file_path` - path of the file to load the data from.
///
/// * `has_headers` - whether the file has headers or not. if set to `true` the function will
///   not parse the first line of the csv file.
///
/// * `skip` - an optional slice that specifies a subset of the file columns that must not be
///   parsed.
///
/// * `f` - function that converts [`String`] into a data sample. It takes as an argument a single
///   record field.
///
/// # Errors
///
/// Returns an error is something goes wrong during the I/O operations.
#[cfg(feature = "csv")]
pub fn load_csv<T, F>(
    path: &str,
    has_headers: bool,
    skip: Option<&[usize]>,
    f: F,
) -> Result<Vec<T>, Box<dyn Error>>
where
    F: Fn(String) -> T,
{
    let mut data: Vec<T> = Vec::new();

    let file = File::open(path)?;

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(has_headers)
        .from_reader(file);

    match skip {
        Some(range) => {
            for result in reader.records() {
                let record = result?;

                (0..record.len())
                    .filter(|column| !range.contains(column))
                    .for_each(|field| data.push(f(record.get(field).unwrap().to_string())));
            }
        }
        None => {
            for result in reader.records() {
                let record = result?;

                (0..record.len())
                    .for_each(|field| data.push(f(record.get(field).unwrap().to_string())));
            }
        }
    }

    Ok(data)
}
