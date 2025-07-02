pub(super) mod sptree;
pub(super) mod vptree;

use std::{
    iter::Sum,
    ops::{Add, AddAssign, DivAssign, MulAssign, SubAssign},
};

use crossbeam::utils::CachePadded;

use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::ParallelSliceMut,
};

use rand_distr::{Distribution, Normal};

use num_traits::{AsPrimitive, Float};

/// Checks whether the perplexity is too large for the number of samples.
///
/// # Arguments
///
/// * `perplexity` - perplexity.
///
/// * `n_samples` - number of data samples.
///
/// # Panics
///
/// If the perplexity is too large.
pub(super) fn check_perplexity<T: Float + AsPrimitive<usize>>(perplexity: &T, n_samples: &usize) {
    if n_samples - 1 < 3 * perplexity.as_() {
        panic!("error: the provided perplexity is too large for the number of data points.\n");
    }
}

/// Prepares the buffers necessary to the computation. Allocates memory freed by `clear_buffers`.
///
/// # Arguments
///
/// * `y` - embedding.
///
/// * `dy` - gradient.
///
/// * `uy` - momentum buffer.
///
/// * `gains` - gains.
pub(super) fn prepare_buffers<T: Float + Send + Sync>(
    y: &mut Vec<CachePadded<T>>,
    dy: &mut Vec<CachePadded<T>>,
    uy: &mut Vec<CachePadded<T>>,
    gains: &mut Vec<CachePadded<T>>,
    grad_entries: usize,
) {
    // Prepares the buffers.
    y.resize(grad_entries, T::zero().into()); // Embeddings.
    dy.resize(grad_entries, T::zero().into()); // Gradient.
    uy.resize(grad_entries, T::zero().into()); // Momentum buffer.
    gains.resize(grad_entries, T::one().into()); // Gains.
}

/// Empties the buffers after the termination of the algorithm. Frees memory allocated by
/// `prepare_buffers`.
///
/// # Arguments
///
/// * `y` - embedding.
///
/// * `dy` - gradient.
///
/// * `uy` - momentum buffer.
///
/// * `gains` - gains.
pub(super) fn clear_buffers<T: Float + Send + Sync>(
    dy: &mut Vec<CachePadded<T>>,
    uy: &mut Vec<CachePadded<T>>,
    gains: &mut Vec<CachePadded<T>>,
) {
    // Empties the buffers.
    *dy = Vec::new(); // Gradient.
    *uy = Vec::new(); // Momentum buffer.
    *gains = Vec::new(); // Gains.
}

/// Random initializes the embedding sampling from a normal distribution with mean 0 and sigma 1e-4.
///
/// # Arguments
///
/// `y` - embedding.
pub(super) fn random_init<T: Float + Send + Sync + Copy>(y: &mut [CachePadded<T>]) {
    let distr = Normal::new(0.0, 1e-4).unwrap();
    let mut rng = rand::rng();
    y.iter_mut()
        .for_each(|el| **el = T::from(distr.sample(&mut rng)).unwrap());
}

/// Computes a squared distance matrix. Computes only the upper triangular entries, excluding the
/// diagonal. The matrix is symmetrized after to get the full distance matrix.
///
/// # Arguments
///
/// * `distances` - distance matrix to fill.
///
/// * `f` - distance function.
///
/// * `g` - a closure that given an index returns the associated sample.
///
/// * `n_samples` - total number of samples.
pub(super) fn compute_pairwise_distance_matrix<'a, T, U, F, G>(
    distances: &mut [CachePadded<T>],
    f: F,
    g: G,
    n_samples: usize,
) where
    T: Float + Send + Sync,
    U: 'a + Send + Sync + ?Sized,
    F: Fn(&U, &U) -> T + Sync + Send,
    G: Fn(&usize) -> &'a U + Sync + Send,
{
    distances
        .par_iter_mut()
        .enumerate()
        .map(|(index, d)| {
            // Picks upper triangular entries excluding the diagonal ones.
            let row_index = index / n_samples;
            let column_index = index % n_samples;

            (row_index, column_index, d)
        })
        .filter(|(row_index, column_index, _)| row_index < column_index)
        .for_each(|(i, j, d)| {
            **d = f(g(&i), g(&j));
        });

    // Symmetrizes the matrix. Effectively filling it.
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            distances[j * n_samples + i] = distances[i * n_samples + j];
        }
    }
}

/// Performs a binary search over the real numbers looking for the optimal bandwidth of the
/// Gaussian kernel relative to the condition distribution `p_values_row`.
///
/// # Arguments
///
/// * `p_values_rows` - conditional distribution relative to the sample.
///
/// * `distances_row` - row of the distance matrix relative to the sample.
///
/// * `perplexity` - given perplexity value.
pub(super) fn search_beta<T>(
    p_values_row: &mut [CachePadded<T>],
    distances_row: &[CachePadded<T>],
    perplexity: &T,
) where
    T: Send + Sync + Copy + Float + Sum + MulAssign + DivAssign,
{
    let mut found = false;
    let mut beta: T = T::one();
    let mut min_beta: T = -T::max_value();
    let mut max_beta: T = T::max_value();
    let tolerance: T = T::from(1e-5).unwrap();
    let mut iteration = 0;
    let mut p_values_row_sum: T = T::zero();

    let two = T::from(2.0).unwrap();
    let zero_point_five = T::from(5.0).unwrap();

    debug_assert_eq!(p_values_row.len(), distances_row.len());

    while !found && iteration < 200 {
        // Here the values of a single row are computed.
        p_values_row
            .iter_mut()
            .zip(distances_row.iter())
            .for_each(|(p, d)| {
                **p = (-beta * d.powi(2)).exp();
            });

        // After that the row is normalized.
        p_values_row_sum = p_values_row.iter().map(|el| **el).sum::<T>() + T::min_positive_value();

        // The conditional distribution's entropy is needed to find the optimal value
        // for beta, i.e. the bandwidth of the Gaussian kernel.
        let mut entropy = p_values_row
            .iter()
            .zip(distances_row.iter())
            .fold(T::zero(), |acc, (p, d)| acc + beta * **p * d.powi(2));
        entropy = entropy / p_values_row_sum + p_values_row_sum.ln();

        // It evaluates whether the entropy is within the tolerance level.
        let entropy_difference = entropy - perplexity.ln();

        if entropy_difference < tolerance && -entropy_difference < tolerance {
            found = true;
        } else {
            if entropy_difference > T::zero() {
                min_beta = beta;

                if max_beta == T::max_value() || max_beta == -T::max_value() {
                    beta *= two;
                } else {
                    beta = (beta + max_beta) / two;
                }
            } else {
                max_beta = beta;

                if min_beta == -T::max_value() || min_beta == T::max_value() {
                    if beta < T::zero() {
                        beta *= two;
                    } else if beta <= T::one() {
                        beta = zero_point_five;
                    } else {
                        beta *= zero_point_five;
                    }
                } else {
                    beta = (beta + min_beta) / two;
                }
            }
            // Checks for overflows.
            if beta.is_infinite() && beta.is_sign_positive() {
                beta = T::max_value()
            }
            if beta.is_infinite() && beta.is_sign_negative() {
                beta = -T::max_value()
            }
        }
        iteration += 1;
    }

    // Row normalization.
    p_values_row
        .iter_mut()
        .for_each(|p| **p /= p_values_row_sum + T::epsilon());
}

/// Normalizes the P values.
///
/// # Arguments
///
/// `p_values` - values of the P distribution.
pub(super) fn normalize_p_values<T: Float + Send + Sync + MulAssign + DivAssign + Sum>(
    p_values: &mut [CachePadded<T>],
) {
    let p_values_sum: T = p_values.par_iter().map(|p| **p).sum();
    let twelve = T::from(12.0).unwrap();
    p_values.par_iter_mut().for_each(|p| {
        **p /= p_values_sum + T::epsilon();
        **p *= twelve;
    });
}

/// Symmetrizes a sparse P matrix.
///
/// # Arguments
///
/// * `p_columns` - for each sample, the indices of its nearest neighbors found with the vantage point tree.
///
/// * `p_values` - P distribution values.
///
/// * `n_samples` - number of samples.
///
/// * `n_neighbors` - number of nearest neighbors to consider.
pub(super) fn symmetrize_sparse_matrix<T>(
    sym_p_rows: &mut Vec<usize>,
    sym_p_columns: &mut Vec<usize>,
    p_columns: Vec<CachePadded<usize>>,
    p_values: &mut Vec<CachePadded<T>>,
    n_samples: usize,
    n_neighbors: &usize,
) where
    T: Float + Add + DivAssign + Send + Sync + MulAssign,
{
    // Each entry of row_counts corresponds to the number of elements for each corresponding row of
    // the symmetric sparse final P matrix.
    let mut row_counts: Vec<usize> = vec![0; n_samples];
    // This sparse and symmetric matrix, due to the nature of the joint probability distribution P,
    // has possibly less entries than the current buffer p_values. In order to construct such a
    // sparse representation, the number of elements contained in each row is needed.
    // Recall that each i-th row corresponds to the joint distribution of the i-th sample.
    let p_rows = |i| (i * n_neighbors);

    for n in 0..n_samples {
        for i in p_rows(n)..p_rows(n + 1) {
            row_counts[n] += 1;
            if !p_columns[p_rows(*p_columns[i])..p_rows(*p_columns[i] + 1)]
                .iter()
                .any(|el| **el == n)
            {
                row_counts[*p_columns[i]] += 1;
            }
        }
    }

    let total: usize = row_counts.iter().sum();

    let mut sym_row_p: Vec<usize> = vec![0; n_samples + 1];
    let mut sym_col_p: Vec<usize> = vec![0; total];
    let mut sym_val_p: Vec<CachePadded<T>> = vec![T::zero().into(); total];

    sym_row_p[0] = 0;
    for _n in 0..n_samples {
        sym_row_p[_n + 1] = sym_row_p[_n] + row_counts[_n];
    }

    let mut offset: Vec<usize> = vec![0; n_samples];

    for _n in 0..n_samples {
        for i in p_rows(_n)..p_rows(_n + 1) {
            // Check whether element (col_P[i], n) is present.
            let mut present: bool = false;
            // Considering element(n, col_P[i]).
            for m in p_rows(*p_columns[i])..p_rows(*p_columns[i] + 1) {
                if *p_columns[m] == _n {
                    present = true;
                    // Make sure we do not add elements twice.
                    if _n <= *p_columns[i] {
                        sym_col_p[sym_row_p[_n] + offset[_n]] = *p_columns[i];
                        sym_col_p[sym_row_p[*p_columns[i]] + offset[*p_columns[i]]] = _n;
                        *sym_val_p[sym_row_p[_n] + offset[_n]] = *p_values[i] + *p_values[m];
                        *sym_val_p[sym_row_p[*p_columns[i]] + offset[*p_columns[i]]] =
                            *p_values[i] + *p_values[m];
                    }
                }
            }
            // If (col_P[i], n) is not present, there is no addition involved.
            if !present {
                sym_col_p[sym_row_p[_n] + offset[_n]] = *p_columns[i];
                sym_col_p[sym_row_p[*p_columns[i]] + offset[*p_columns[i]]] = _n;
                *sym_val_p[sym_row_p[_n] + offset[_n]] = *p_values[i];
                *sym_val_p[sym_row_p[*p_columns[i]] + offset[*p_columns[i]]] = *p_values[i];
            }
            // Update offsets.
            if !present || _n <= *p_columns[i] {
                offset[_n] += 1;
                if *p_columns[i] != _n {
                    offset[*p_columns[i]] += 1;
                }
            }
        }
    }

    // Divide result by two.
    let zero_point_five = T::from(0.5).unwrap();
    sym_val_p.iter_mut().for_each(|p| **p *= zero_point_five);

    *p_values = sym_val_p;
    *sym_p_rows = sym_row_p;
    *sym_p_columns = sym_col_p;
}

/// Updates the embedding.
///
/// # Arguments
///
/// * `y` - embedding.
///
/// * `dy` - tSNE gradient.
///
/// * `uy` - momentum buffer.
///
/// * `gains` - gains.
///
/// * `learning_rate` - learning rate.
///
/// * `momentum` - momentum coefficient.
pub(super) fn update_solution<T>(
    y: &mut [CachePadded<T>],
    dy: &[CachePadded<T>],
    uy: &mut [CachePadded<T>],
    gains: &mut [CachePadded<T>],
    learning_rate: &T,
    momentum: &T,
) where
    T: Float + Send + Sync + AddAssign,
{
    let zero_point_two = T::from(0.2).unwrap();
    let zero_point_eight = T::from(0.8).unwrap();
    let zero_point_zero_one = T::from(0.01).unwrap();

    y.par_iter_mut()
        .zip(dy.par_iter())
        .zip(uy.par_iter_mut())
        .zip(gains.par_iter_mut())
        .for_each(|(((y_el, dy_el), uy_el), gains_el)| {
            **gains_el = if dy_el.signum() != uy_el.signum() {
                **gains_el + zero_point_two
            } else {
                **gains_el * zero_point_eight
            };
            if **gains_el < zero_point_zero_one {
                **gains_el = zero_point_zero_one;
            }
            **uy_el = *momentum * **uy_el - *learning_rate * **gains_el * **dy_el;
            **y_el += **uy_el
        });
}

/// Adjust the P distribution values to the original ones in parallel.
///
/// # Arguments
///
/// `p_values` - P distribution.
pub(super) fn stop_lying<T: Float + Send + Sync + DivAssign>(p_values: &mut [CachePadded<T>]) {
    let twelve = T::from(12.0).unwrap();
    p_values.par_iter_mut().for_each(|p| **p /= twelve);
}

/// Makes the solution zero mean. The embedded samples are taken in chunks of size
/// corresponding to the embedding space dimensionality, so that each chunks
/// corresponds to a sample. Each of the so obtained samples is summed element wise
/// to the means buffer. At last, each mean of the means buffer is divided by the number
/// of samples.
///
/// # Arguments
///
/// * ` means` - where the means of each component of the embedding are stored.
///
/// * `y` - embedding.
///
/// * `n_samples` -  number of samples in the embedding.
///
/// * `embedding_dim`- dimensionality of the embedding space.
pub(super) fn zero_mean<T>(
    means: &mut [T],
    y: &mut [CachePadded<T>],
    n_samples: usize,
    embedding_dim: usize,
) where
    T: Float + Send + Sync + Copy + AddAssign + DivAssign + SubAssign,
{
    // Not parallel as it accumulates into means.
    y.chunks(embedding_dim).for_each(|embedded_sample| {
        means
            .iter_mut()
            .zip(embedded_sample.iter())
            .for_each(|(mean, el)| *mean += **el);
    });

    let n_samples = T::from(n_samples).unwrap();
    means.iter_mut().for_each(|el| *el /= n_samples);

    y.par_chunks_mut(embedding_dim).for_each(|embedded_sample| {
        embedded_sample
            .iter_mut()
            .zip(means.iter())
            .for_each(|(el, mean)| **el -= *mean);
    });

    // Zeroes the mean buffer.
    means.iter_mut().for_each(|el| *el = T::zero());
}

/// Evaluate t-SNE cost function exactly.
///
/// # Arguments
///
/// * `p_values` - values of the P distribution.
///
/// * `y` - current embedding.
///
/// * `n_samples` - number of samples in the embedding;
///
/// * `embedding_dim` - dimensionality of the embedding space.
#[cfg(test)]
pub(crate) fn evaluate_error<T>(
    p_values: &[CachePadded<T>],
    y: &[CachePadded<T>],
    n_samples: usize,
    embedding_dim: usize,
) -> T
where
    T: Float + Send + Sync + AddAssign + Add + DivAssign + Sum,
{
    let mut distances: Vec<CachePadded<T>> = vec![T::zero().into(); n_samples * n_samples];
    compute_pairwise_distance_matrix(
        &mut distances,
        |a: &[CachePadded<T>], b: &[CachePadded<T>]| {
            a.iter()
                .zip(b.iter())
                .map(|(aa, bb)| (**aa - **bb).powi(2))
                .sum::<T>()
        },
        |i| &y[i * embedding_dim..(i + 1) * embedding_dim],
        n_samples,
    );

    let mut q_values: Vec<CachePadded<T>> = vec![T::zero().into(); n_samples * n_samples];
    q_values
        .par_iter_mut()
        .zip(distances.par_iter())
        .for_each(|(q, d)| **q = T::one() / (T::one() + **d));

    let q_sum = q_values.par_iter().map(|q| **q).sum::<T>();
    q_values.par_iter_mut().for_each(|q| **q /= q_sum);

    // Kullback-Leibler divergence.
    p_values
        .par_iter()
        .zip(q_values.par_iter())
        .fold(
            || T::zero(),
            |c, (p, q)| {
                c + **p * ((**p + T::min_positive_value()) / (**q + T::min_positive_value())).ln()
            },
        )
        .sum::<T>()
}

/// Evaluate t-SNE cost function approximately.
///
/// # Arguments
///
/// * `p_rows` - rows of the sparse, symmetric P distribution matrix.
///
/// * `p_columns` - columns of the sparse, symmetric P distribution matrix.
///
/// * `p_values` - sparse symmetric P distribution matrix.
///
/// * `y` - current embedding.
///
/// * `n_samples` - number of samples.
///
/// * `embedding_dim` - dimensionality of the embedding space.
///
/// * `theta` - threshold for the Barnes-Hut algorithm.
#[cfg(test)]
pub(crate) fn evaluate_error_approximately<T>(
    p_rows: &[usize],
    p_columns: &[usize],
    p_values: &[CachePadded<T>],
    y: &[CachePadded<T>],
    n_samples: usize,
    embedding_dim: usize,
    theta: T,
) -> T
where
    T: Float + Send + Sync + Sum + AddAssign + SubAssign + MulAssign + DivAssign,
{
    // Get estimate of normalization term.
    let q_sum = {
        let tree = sptree::SPTree::new(embedding_dim, y, n_samples);
        let mut q_sums: Vec<CachePadded<T>> = vec![T::zero().into(); n_samples];

        let mut buffer: Vec<CachePadded<T>> = vec![T::zero().into(); n_samples * embedding_dim];
        let mut negative_forces: Vec<CachePadded<T>> =
            vec![T::zero().into(); n_samples * embedding_dim];

        q_sums
            .par_iter_mut()
            .zip(negative_forces.par_chunks_mut(embedding_dim))
            .zip(buffer.par_chunks_mut(embedding_dim))
            .enumerate()
            .for_each(|(index, ((sum, negative_forces_row), buffer_row))| {
                tree.compute_non_edge_forces(index, theta, negative_forces_row, buffer_row, sum);
            });

        q_sums.par_iter().map(|sum| **sum).sum::<T>()
    };

    let mut partials: Vec<CachePadded<T>> = vec![T::zero().into(); n_samples];

    partials
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, cost)| {
            let sample_a = &y[index * embedding_dim..(index + 1) * embedding_dim];
            for n in p_rows[index]..p_rows[index + 1] {
                let sample_b = &y[p_columns[n] * embedding_dim..(p_columns[n] + 1) * embedding_dim];

                let mut q = sample_a
                    .iter()
                    .zip(sample_b.iter())
                    .map(|(a, b)| (**a - **b).powi(2))
                    .sum::<T>();
                q = (T::one() / (T::one() + q)) / q_sum;

                // Kullback-Leibler divergence.
                **cost += *p_values[index]
                    * ((*p_values[index] + T::min_positive_value())
                        / (q + T::min_positive_value()))
                    .ln();
            }
        });

    partials.par_iter().map(|partial| **partial).sum::<T>()
}
