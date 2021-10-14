mod sptree;
mod vptree;
use num_traits::{AsPrimitive, Float};
use rand_distr::{Distribution, Normal};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::ParallelSliceMut,
};
pub(crate) use sptree::SPTree;
use std::{
    fmt::{Debug, Display},
    iter::Sum,
    ops::{Add, AddAssign, DivAssign, MulAssign, SubAssign},
};
pub(crate) use vptree::VPTree;

/// Cache aligned floating point number. It's used to prevent false sharing.
#[repr(align(64))]
#[derive(Clone, Copy, Debug)]
pub(super) struct Aligned<T: Send + Sync + Copy>(pub(super) T);

impl<T: Send + Sync + Copy> From<T> for Aligned<T> {
    fn from(scalar: T) -> Self {
        Self(scalar)
    }
}

// Obviously safe.
unsafe impl<T: Send + Sync + Copy> Send for Aligned<T> {}
unsafe impl<T: Send + Sync + Copy> Sync for Aligned<T> {}

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
pub(super) fn check_perplexity<T: Float + Display + AsPrimitive<usize>>(
    perplexity: &T,
    n_samples: &usize,
) {
    if n_samples - 1 < 3 * perplexity.as_() {
        panic!(
            "error: the provided perplexity {} is too large for the number of data points.\n",
            perplexity
        );
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
    y: &mut Vec<Aligned<T>>,
    dy: &mut Vec<Aligned<T>>,
    uy: &mut Vec<Aligned<T>>,
    gains: &mut Vec<Aligned<T>>,
    grad_entries: &usize,
) {
    // Prepares the buffers.
    y.resize(*grad_entries, T::zero().into()); // Embeddings.
    dy.resize(*grad_entries, T::zero().into()); // Gradient.
    uy.resize(*grad_entries, T::zero().into()); // Momentum buffer.
    gains.resize(*grad_entries, T::one().into()); // Gains.
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
    dy: &mut Vec<Aligned<T>>,
    uy: &mut Vec<Aligned<T>>,
    gains: &mut Vec<Aligned<T>>,
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
pub(super) fn random_init<T: Float + Send + Sync + Copy>(y: &mut [Aligned<T>]) {
    let distr = Normal::new(0.0, 1e-4).unwrap();
    let mut rng = rand::thread_rng();
    y.iter_mut()
        .for_each(|el| el.0 = T::from(distr.sample(&mut rng)).unwrap());
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
    distances: &mut [Aligned<T>],
    f: F,
    g: G,
    n_samples: &usize,
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
            d.0 = f(g(&i), g(&j));
        });
    // Symmetrizes the matrix. Effectively filling it.
    for i in 0..*n_samples {
        for j in (i + 1)..*n_samples {
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
    p_values_row: &mut [Aligned<T>],
    distances_row: &[Aligned<T>],
    perplexity: &T,
) where
    T: Send + Sync + Copy + Float + Sum + MulAssign + DivAssign + Debug,
{
    let mut found = false;
    let mut beta: T = T::one();
    let mut min_beta: T = -T::max_value();
    let mut max_beta: T = T::max_value();
    let tolerance: T = T::from(1e-5).unwrap();
    let mut iteration = 0;
    let mut p_values_row_sum: T = T::zero();

    debug_assert_eq!(p_values_row.len(), distances_row.len());

    while !found && iteration < 200 {
        // Here the values of a single row are computed.
        p_values_row
            .iter_mut()
            .zip(distances_row.iter())
            .for_each(|(p, d)| {
                p.0 = (-beta * d.0.powi(2)).exp();
            });

        // After that the row is normalized.
        p_values_row_sum = p_values_row.iter().map(|el| el.0).sum::<T>() + T::min_positive_value();

        // The conditional distribution's entropy is needed to find the optimal value
        // for beta, i.e. the bandwidth of the Gaussian kernel.
        let mut entropy = p_values_row
            .iter()
            .zip(distances_row.iter())
            .fold(T::zero(), |acc, (p, d)| acc + beta * p.0 * d.0.powi(2));
        entropy = entropy / p_values_row_sum + p_values_row_sum.ln();

        // It evaluates whether the entropy is within the tolerance level.
        let entropy_difference = entropy - perplexity.ln();

        if entropy_difference < tolerance && -entropy_difference < tolerance {
            found = true;
        } else {
            if entropy_difference > T::zero() {
                min_beta = beta;

                if max_beta == T::max_value() || max_beta == -T::max_value() {
                    beta *= T::from(2.0).unwrap();
                } else {
                    beta = (beta + max_beta) / T::from(2.0).unwrap();
                }
            } else {
                max_beta = beta;

                if min_beta == -T::max_value() || min_beta == T::max_value() {
                    if beta < T::zero() {
                        beta *= T::from(2.0).unwrap();
                    } else if beta <= T::one() {
                        beta = T::from(0.5).unwrap();
                    } else {
                        beta *= T::from(0.5).unwrap();
                    }
                } else {
                    beta = (beta + min_beta) / T::from(2.0).unwrap();
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
        .for_each(|p| p.0 /= p_values_row_sum);
}

/// Normalizes the P values.
///
/// # Arguments
///
/// `p_values` - values of the P distribution.
pub(super) fn normalize_p_values<T: Float + Send + Sync + MulAssign + DivAssign + Sum>(
    p_values: &mut [Aligned<T>],
) {
    let p_values_sum: T = p_values.par_iter().map(|p| p.0).sum();
    p_values.par_iter_mut().for_each(|p| {
        p.0 /= p_values_sum;
        p.0 *= T::from(12.0).unwrap();
    });
}

/// Symmetrizes a sparse P matrix.
///
/// # Arguments
///
/// * `p_columns` - for each sample, the indices of its nearest neighbors found with the
/// vantage point tree.
///
/// * `p_values` - P distribution values.
///
/// * `n_samples` - number of samples.
///
/// * `n_neighbors` - number of nearest neighbors to consider.
pub(super) fn symmetrize_sparse_matrix<T>(
    p_columns: Vec<Aligned<usize>>,
    p_values: &mut Vec<Aligned<T>>,
    n_samples: usize,
    n_neighbors: &usize,
) -> (Vec<usize>, Vec<usize>)
where
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

    for _n in 0..n_samples {
        for i in p_rows(_n)..p_rows(_n + 1) {
            row_counts[_n] += 1;
            if p_columns[p_rows(p_columns[i].0)..p_rows(p_columns[i].0 + 1)]
                .iter()
                .any(|el| el.0 == _n)
            {
                row_counts[p_columns[i].0] += 1;
            }
        }
    }

    let total: usize = row_counts.iter().sum();

    let mut sym_row_p: Vec<usize> = vec![0; n_samples + 1];
    let mut sym_col_p: Vec<usize> = vec![0; total];
    let mut sym_val_p: Vec<Aligned<T>> = vec![T::zero().into(); total];

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
            for m in p_rows(p_columns[i].0)..p_rows(p_columns[i].0 + 1) {
                if p_columns[m].0 == _n {
                    present = true;
                    // Make sure we do not add elements twice.
                    if _n <= p_columns[i].0 {
                        sym_col_p[sym_row_p[_n] + offset[_n]] = p_columns[i].0;
                        sym_col_p[sym_row_p[p_columns[i].0] + offset[p_columns[i].0]] = _n;
                        sym_val_p[sym_row_p[_n] + offset[_n]].0 = p_values[i].0 + p_values[m].0;
                        sym_val_p[sym_row_p[p_columns[i].0] + offset[p_columns[i].0]].0 =
                            p_values[i].0 + p_values[m].0;
                    }
                }
            }
            // If (col_P[i], n) is not present, there is no addition involved.
            if !present {
                sym_col_p[sym_row_p[_n] + offset[_n]] = p_columns[i].0;
                sym_col_p[sym_row_p[p_columns[i].0] + offset[p_columns[i].0]] = _n;
                sym_val_p[sym_row_p[_n] + offset[_n]].0 = p_values[i].0;
                sym_val_p[sym_row_p[p_columns[i].0] + offset[p_columns[i].0]].0 = p_values[i].0;
            }
            // Update offsets.
            if !present || _n <= p_columns[i].0 {
                offset[_n] += 1;
                if p_columns[i].0 != _n {
                    offset[p_columns[i].0] += 1;
                }
            }
        }
    }

    // Divide result by two.
    sym_val_p
        .iter_mut()
        .for_each(|p| p.0 *= T::from(0.5).unwrap());

    *p_values = sym_val_p;
    (sym_row_p, sym_col_p)
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
    y: &mut [Aligned<T>],
    dy: &[Aligned<T>],
    uy: &mut [Aligned<T>],
    gains: &mut [Aligned<T>],
    learning_rate: &T,
    momentum: &T,
) where
    T: Float + Send + Sync + AddAssign,
{
    y.par_iter_mut()
        .zip(dy.par_iter())
        .zip(uy.par_iter_mut())
        .zip(gains.par_iter_mut())
        .for_each(|(((y_el, dy_el), uy_el), gains_el)| {
            gains_el.0 = if dy_el.0.signum() != uy_el.0.signum() {
                gains_el.0 + T::from(0.2).unwrap()
            } else {
                gains_el.0 * T::from(0.8).unwrap()
            };
            if gains_el.0 < T::from(0.01).unwrap() {
                gains_el.0 = T::from(0.01).unwrap();
            }
            uy_el.0 = *momentum * uy_el.0 - *learning_rate * gains_el.0 * dy_el.0;
            y_el.0 += uy_el.0
        });
}

/// Adjust the P distribution values to the original ones in parallel.
///
/// # Arguments
///
/// `p_values` - P distribution.
pub(super) fn stop_lying<T: Float + Send + Sync + DivAssign>(p_values: &mut [Aligned<T>]) {
    p_values
        .par_iter_mut()
        .for_each(|p| p.0 /= T::from(12.0).unwrap());
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
    y: &mut [Aligned<T>],
    n_samples: &usize,
    embedding_dim: &usize,
) where
    T: Float + Send + Sync + Copy + AddAssign + DivAssign + SubAssign,
{
    // Not parallel as it accumulates into means.
    y.chunks(*embedding_dim).for_each(|embedded_sample| {
        means
            .iter_mut()
            .zip(embedded_sample.iter())
            .for_each(|(mean, el)| *mean += el.0);
    });
    means
        .iter_mut()
        .for_each(|el| *el /= T::from(*n_samples).unwrap());
    y.par_chunks_mut(*embedding_dim)
        .for_each(|embedded_sample| {
            embedded_sample
                .iter_mut()
                .zip(means.iter())
                .for_each(|(el, mean)| el.0 -= *mean);
        });
    // Zeroes the mean buffer.
    means.iter_mut().for_each(|el| *el = T::zero());
}

// TODO: rewrite this.
// /// Evaluate t-SNE cost function exactly.
// #[allow(dead_code)]
// pub fn evaluate_error<T>(p: &mut [T], y: &[T], n: usize, d: usize) -> T
// where
//     T: Float + std::ops::AddAssign + std::ops::Add + std::ops::DivAssign + std::iter::Sum,
// {
//     let mut dd: Vec<T> = vec![T::zero(); n * n];
//     let mut q: Vec<T> = vec![T::zero(); n * n];
//     compute_squared_euclidean_distance(y, n, d, &mut dd);

//     // Compute Q-matrix and normalization sum.
//     let mut nn: usize = 0;
//     let mut sum_q = T::min_positive_value();
//     for _n in 0..n {
//         for m in 0..n {
//             if _n != m {
//                 q[nn + m] = T::one() / (T::one() + dd[nn + m]);
//                 sum_q += q[nn + m];
//             } else {
//                 q[nn + m] = T::min_positive_value();
//             }
//         }
//         nn += n;
//     }

//     for el in &mut q[..] {
//         *el /= sum_q;
//     }

//     let mut c: T = T::zero();
//     for _n in 0..(n * n) {
//         c += p[_n] * ((p[_n] + T::min_positive_value()) / (q[_n] + T::min_positive_value())).ln()
//     }
//     c
// }

// /// Evaluate t-SNE cost function approximately.
// #[allow(dead_code)]
// pub fn evaluate_error_approx<'a, T>(
//     row_p: &mut [usize],
//     col_p: &mut [usize],
//     val_p: &mut [T],
//     y: &'a [T],
//     n: usize,
//     d: usize,
//     theta: T,
// ) -> T
// where
//     T: Float
//         + std::ops::AddAssign
//         + std::ops::SubAssign
//         + std::ops::MulAssign
//         + std::ops::DivAssign
//         + NumCast,
// {
//     // Get estimate of normalization term.
//     let mut tree: SPTree<T> = SPTree::new(d, y, n);
//     let mut buff: Vec<T> = vec![T::zero(); d];
//     let mut sum_q: T = T::zero();
//     for _n in 0..n {
//         tree.compute_non_edge_forces(_n, theta, &mut buff, &mut sum_q);
//     }
//     // Loop over all edges to compute t-SNE error.
//     let mut ind1;
//     let mut ind2;
//     let mut c: T = T::zero();
//     let mut q;

//     for _n in 0..n {
//         ind1 = _n * d;
//         for i in row_p[_n]..row_p[_n + 1] {
//             q = T::zero();
//             ind2 = col_p[i] * d;

//             buff[..d].clone_from_slice(&y[ind1..(d + ind1)]);

//             for (i, el) in buff.iter_mut().enumerate() {
//                 *el -= y[ind2 + i];
//             }

//             for el in &buff[..] {
//                 q += *el * *el;
//             }
//             q = (T::one() / (T::one() + q)) / sum_q;
//             c += val_p[i]
//                 * ((val_p[i] + T::min_positive_value()) / (q + T::min_positive_value())).ln();
//         }
//     }
//     c
// }
