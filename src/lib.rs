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
//! ```
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

pub(crate) use num_traits::{cast::AsPrimitive, Float};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::{ParallelSlice, ParallelSliceMut},
};
#[cfg(feature = "csv")]
use std::{error::Error, fs::File};
use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

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
    p_values: Vec<tsne::Aligned<T>>,
    p_rows: Vec<usize>,
    p_columns: Vec<usize>,
    q_values: Vec<tsne::Aligned<T>>,
    y: Vec<tsne::Aligned<T>>,
    dy: Vec<tsne::Aligned<T>>,
    uy: Vec<tsne::Aligned<T>>,
    gains: Vec<tsne::Aligned<T>>,
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
        }
    }

    /// Sets a new learning rate.
    ///
    /// # Arguments
    ///
    /// `learning_rate` - new value for the learning rate.
    pub fn learning_rate<'a>(&'a mut self, learning_rate: T) -> &'a mut Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets new epochs, i.e the maximum number of fitting iterations.
    ///
    /// # Arguments
    ///
    /// `epochs` - new value for the epochs.
    pub fn epochs<'a>(&'a mut self, epochs: usize) -> &'a mut Self {
        self.epochs = epochs;
        self
    }

    /// Sets a new momentum.
    ///
    /// # Arguments
    ///
    /// `momentum` - new value for the momentum.
    pub fn momentum<'a>(&'a mut self, momentum: T) -> &'a mut Self {
        self.momentum = momentum;
        self
    }

    /// Sets a new final momentum.
    ///
    /// # Arguments
    ///
    /// `final_momentum` - new value for the final momentum.
    pub fn final_momentum<'a>(&'a mut self, final_momentum: T) -> &'a mut Self {
        self.final_momentum = final_momentum;
        self
    }

    /// Sets a new momentum switch epoch, i.e. the epoch after which the algorithm switches to
    /// `final_momentum` for the map update.
    ///
    /// # Arguments
    ///
    /// `momentum_switch_epoch` - new value for the momentum switch epoch.
    pub fn momentum_switch_epoch<'a>(&'a mut self, momentum_switch_epoch: usize) -> &'a mut Self {
        self.momentum_switch_epoch = momentum_switch_epoch;
        self
    }

    /// Sets a new stop lying epoch, i.e. the epoch after which the P distribution values become
    /// true, as defined in the original implementation. For epochs < `stop_lying_epoch` the values
    /// of the P distribution are multiplied by a factor equal to `12.0`.
    ///
    /// # Arguments
    ///
    /// `stop_lying_epoch` - new value for the stop lying epoch.
    pub fn stop_lying_epoch<'a>(&'a mut self, stop_lying_epoch: usize) -> &'a mut Self {
        self.stop_lying_epoch = stop_lying_epoch;
        self
    }

    /// Sets a new value for the embedding dimension.
    ///
    /// # Arguments
    ///
    /// `embedding_dim` - new value for the embedding space dimensionality.
    pub fn embedding_dim<'a>(&'a mut self, embedding_dim: u8) -> &'a mut Self {
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
    pub fn perplexity<'a>(&'a mut self, perplexity: T) -> &'a mut Self {
        self.perplexity = perplexity;
        self
    }

    /// Returns the computed embedding.
    pub fn embedding(&self) -> Vec<T> {
        self.y.iter().map(|x| x.0).collect()
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
    pub fn exact<'a, F: Fn(&U, &U) -> T + Send + Sync>(
        &'a mut self,
        distance_f: F,
    ) -> &'a mut Self {
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
            &grad_entries,
        );
        // Prepare distributions matrices.
        self.p_values.resize(pairwise_entries, T::zero().into()); // P.
        self.q_values.resize(pairwise_entries, T::zero().into()); // Q.

        // Alignment prevents false sharing.
        let mut distances: Vec<tsne::Aligned<T>> = vec![T::zero().into(); pairwise_entries];
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
            &n_samples,
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
                let symmetric = self.p_values[j * n_samples + i].0;
                self.p_values[i * n_samples + j].0 += symmetric;
                self.p_values[j * n_samples + i].0 = self.p_values[i * n_samples + j].0;
            }
        }

        // Normalize P values.
        tsne::normalize_p_values(&mut self.p_values);

        // Initialize solution randomly.
        tsne::random_init(&mut self.y);

        // Vector used to store the mean values for each embedding dimension. It's used
        // to make the solution zero mean.
        let mut means: Vec<T> = vec![T::zero(); embedding_dim];

        // Main fitting loop.
        for epoch in 0..self.epochs {
            // Compute pairwise squared euclidean distances between embeddings in parallel.
            tsne::compute_pairwise_distance_matrix(
                &mut distances,
                |ith: &[tsne::Aligned<T>], jth: &[tsne::Aligned<T>]| {
                    ith.iter()
                        .zip(jth.iter())
                        .map(|(i, j)| (i.0 - j.0).powi(2))
                        .sum()
                },
                |index| &self.y[index * embedding_dim..index * embedding_dim + embedding_dim],
                &n_samples,
            );

            // Computes Q.
            self.q_values
                .par_iter_mut()
                .zip(distances.par_iter())
                .for_each(|(q, d)| q.0 = T::one() / (T::one() + d.0));

            // Computes the exact gradient in parallel.
            let q_values_sum: T = self.q_values.par_iter().map(|q| q.0).sum();

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
                            .for_each(|((p, q), other_sample)| {
                                let m: T = (p.0 - q.0 / q_values_sum) * q.0;
                                dy_sample
                                    .iter_mut()
                                    .zip(y_sample.iter())
                                    .zip(other_sample.iter())
                                    .for_each(|((dy_el, y_el), other_el)| {
                                        dy_el.0 += (y_el.0 - other_el.0) * m
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
            self.dy.iter_mut().for_each(|el| el.0 = T::zero());

            // Make solution zero mean.
            tsne::zero_mean(&mut means, &mut self.y, &n_samples, &embedding_dim);

            // Stop lying about the P-values if the time is right.
            if epoch == self.stop_lying_epoch {
                tsne::stop_lying(&mut self.p_values);
            }

            // Switches momentum if the time is right.
            if epoch == self.momentum_switch_epoch {
                self.momentum = self.final_momentum;
            }
        }
        // Clears buffers used for fitting.
        tsne::clear_buffers(&mut self.dy, &mut self.uy, &mut self.gains);
        self
    }

    /// Performs a parallel Barnes-Hut approximation of the t-SNE algorithm.
    ///
    /// # Arguments
    ///
    /// * `theta` - determines the accuracy of the approximation. Must be **strictly greater than
    /// 0.0**. Large values for θ increase the speed of the algorithm but decrease its accuracy.
    /// For small values of θ it is less probable that a cell in the space partitioning tree will
    /// be treated as a single point. For θ equal to 0.0 the method degenerates in the exact
    /// version.
    ///
    /// * `metric_f` - metric function.
    ///
    ///
    /// **Do note that** `metric_f` **must be a metric distance**, i.e. it must
    /// satisfy the [triangle inequality](https://en.wikipedia.org/wiki/Triangle_inequality).
    pub fn barnes_hut<'a, F: Fn(&U, &U) -> T + Send + Sync>(
        &'a mut self,
        theta: T,
        metric_f: F,
    ) -> &'a mut Self {
        // Checks that theta is valid.
        assert!(
            theta > T::zero(),
            "error: theta value must be greater than 0.0. 
            A value of 0.0 corresponds to using the exact version of the algorithm."
        );

        let data = self.data;
        let n_samples = self.data.len(); // Number of samples in data.

        // Checks that the supplied perplexity is suitable for the number of samples at hand.
        tsne::check_perplexity(&self.perplexity, &n_samples);

        let embedding_dim = self.embedding_dim as usize;
        // Number of  points ot consider when approximating the conditional distribution P.
        let n_neighbors: usize = (T::from(3.0).unwrap() * self.perplexity).as_();
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
            &grad_entries,
        );
        // The P distribution values are restricted to a subset of size n_neighbors for each input
        // sample.
        self.p_values.resize(pairwise_entries, T::zero().into());

        // This vector is used to keep track of the indexes for each nearest neighbors of each
        // sample. There's a one to one correspondence between the elements of p_columns
        // an the elements of p_values: for each row i of length n_neighbors of such matrices it
        // holds that p_columns[i][j] corresponds to the index sample which contributes
        // to p_values[i][j]. This vector is freed inside symmetrize_sparse_matrix.
        let mut p_columns: Vec<tsne::Aligned<usize>> = vec![0.into(); pairwise_entries];

        // Computes sparse input similarities using a vantage point tree.
        {
            // Distances buffer.
            let mut distances: Vec<tsne::Aligned<T>> = vec![T::zero().into(); pairwise_entries];

            // Build ball tree on data set. The tree is freed at the end of the scope.
            let tree = tsne::VPTree::new(data, &metric_f);

            // For each sample in the dataset compute the perplexities using a vantage point tree
            // in parallel.
            {
                let perplexity = &self.perplexity; // Immutable borrow must be outside.
                self.p_values
                    .par_chunks_mut(n_neighbors)
                    .zip(distances.par_chunks_mut(n_neighbors))
                    .zip(p_columns.par_chunks_mut(n_neighbors))
                    .zip(data.par_iter())
                    .enumerate()
                    .for_each(
                        |(index, (((p_values_row, distances_row), p_columns_row), sample))| {
                            // Writes the indices and the distances of the nearest neighbors of sample.
                            tree.search(
                                sample,
                                index,
                                n_neighbors + 1, // The first NN is sample itself.
                                p_columns_row,
                                distances_row,
                                &metric_f,
                            );
                            debug_assert!(!p_columns_row.iter().any(|i| i.0 == index));
                            tsne::search_beta(p_values_row, distances_row, perplexity);
                        },
                    );
            }
        }

        // Symmetrize sparse P matrix.
        tsne::symmetrize_sparse_matrix(
            &mut self.p_rows,
            &mut self.p_columns,
            p_columns,
            &mut self.p_values,
            n_samples,
            &n_neighbors,
        );

        // Normalize P values.
        tsne::normalize_p_values(&mut self.p_values);

        // Initialize solution randomly.
        tsne::random_init(&mut self.y);

        // Prepares buffers for Barnes-Hut algorithm.
        let mut positive_forces: Vec<tsne::Aligned<T>> = vec![T::zero().into(); grad_entries];
        let mut negative_forces: Vec<tsne::Aligned<T>> = vec![T::zero().into(); grad_entries];
        let mut forces_buffer: Vec<tsne::Aligned<T>> = vec![T::zero().into(); grad_entries];
        let mut q_sums: Vec<tsne::Aligned<T>> = vec![T::zero().into(); n_samples];

        // Vector used to store the mean values for each embedding dimension. It's used
        // to make the solution zero mean.
        let mut means: Vec<T> = vec![T::zero(); embedding_dim];

        // Main Training loop.
        for epoch in 0..self.epochs {
            {
                // Construct space partitioning tree on current embedding.
                let tree = tsne::SPTree::new(&embedding_dim, &self.y, &n_samples);
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
                                &theta,
                                negative_forces_row,
                                forces_buffer_row,
                                q_sum,
                            );
                        },
                    );
            }

            // Compute final Barnes-Hut t-SNE gradient approximation.
            // Reduces partial sums of Q distribution.
            let q_sum: T = q_sums.par_iter_mut().map(|sum| sum.0).sum();
            self.dy
                .par_iter_mut()
                .zip(positive_forces.par_iter_mut())
                .zip(negative_forces.par_iter_mut())
                .for_each(|((grad, pf), nf)| {
                    grad.0 = pf.0 - (nf.0 / q_sum);
                    pf.0 = T::zero();
                    nf.0 = T::zero();
                });
            // Zeroes Q-sums.
            q_sums.par_iter_mut().for_each(|sum| sum.0 = T::zero());

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
            tsne::zero_mean(&mut means, &mut self.y, &n_samples, &embedding_dim);

            // Stop lying about the P-values if the time is right.
            if epoch == self.stop_lying_epoch {
                tsne::stop_lying(&mut self.p_values);
            }

            // Switches momentum if the time is right.
            if epoch == self.momentum_switch_epoch {
                self.momentum = self.final_momentum;
            }
        }
        // Clears buffers used for fitting.
        tsne::clear_buffers(&mut self.dy, &mut self.uy, &mut self.gains);
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
    pub fn write_csv<'a>(&'a mut self, path: &str) -> Result<&'a mut Self, Box<dyn Error>>
    where
        T: Float + ToString,
    {
        let mut writer = csv::Writer::from_path(path)?;

        // String-ify the embedding.
        let to_write = self
            .y
            .iter()
            .map(|el| el.0.to_string())
            .collect::<Vec<String>>();

        // Write headers.
        match self.embedding_dim {
            2 => writer.write_record(&["x", "y"])?,
            3 => writer.write_record(&["x", "y", "z"])?,
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
/// not parse the first line of the csv file.
///
/// * `skip` - an optional slice that specifies a subset of the file columns that must not be
/// parsed.
///
/// * `f` - function that converts [`String`] into a data sample. It takes as an argument a single
/// record field.
///
/// # Errors
///
/// Returns an error is something goes wrong during the I/O operations.
#[cfg(feature = "csv")]
pub fn load_csv<T, F: Fn(String) -> T>(
    path: &str,
    has_headers: bool,
    skip: Option<&[usize]>,
    f: F,
) -> Result<Vec<T>, Box<dyn Error>> {
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
