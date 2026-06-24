pub(super) mod arena;
pub(super) mod morton;
pub(super) mod vptree;

use std::{
    iter::Sum,
    ops::{Add, AddAssign, DivAssign, MulAssign, SubAssign},
};

use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::{ParallelSlice, ParallelSliceMut},
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
    y: &mut Vec<T>,
    dy: &mut Vec<T>,
    uy: &mut Vec<T>,
    gains: &mut Vec<T>,
    grad_entries: usize,
) {
    // Prepares the buffers.
    y.resize(grad_entries, T::zero()); // Embeddings.
    dy.resize(grad_entries, T::zero()); // Gradient.
    uy.resize(grad_entries, T::zero()); // Momentum buffer.
    gains.resize(grad_entries, T::one()); // Gains.
}

/// Empties the buffers after the termination of the algorithm. Frees memory allocated by
/// `prepare_buffers`.
///
/// # Arguments
///
/// * `dy` - gradient.
///
/// * `uy` - momentum buffer.
///
/// * `gains` - gains.
pub(super) fn clear_buffers<T: Float + Send + Sync>(
    dy: &mut Vec<T>,
    uy: &mut Vec<T>,
    gains: &mut Vec<T>,
) {
    // Empties the buffers.
    *dy = Vec::new(); // Gradient.
    *uy = Vec::new(); // Momentum buffer.
    *gains = Vec::new(); // Gains.
}

/// Returns the source of randomness used by the crate. On targets where an entropy
/// source is available this is the thread local generator. This includes
/// wasm32-unknown-unknown when the wasm_js feature is enabled, in which case entropy
/// comes from the JavaScript host through getrandom. Otherwise, on
/// wasm32-unknown-unknown no entropy source exists, so a small deterministic generator
/// with a fixed seed is used instead.
#[cfg(any(
    not(all(target_arch = "wasm32", target_os = "unknown")),
    feature = "wasm_js"
))]
pub(super) fn make_rng() -> impl rand::Rng {
    rand::rng()
}

/// Returns the source of randomness used by the crate. On targets where an entropy
/// source is available this is the thread local generator. This includes
/// wasm32-unknown-unknown when the wasm_js feature is enabled, in which case entropy
/// comes from the JavaScript host through getrandom. Otherwise, on
/// wasm32-unknown-unknown no entropy source exists, so a small deterministic generator
/// with a fixed seed is used instead.
#[cfg(all(
    target_arch = "wasm32",
    target_os = "unknown",
    not(feature = "wasm_js")
))]
pub(super) fn make_rng() -> impl rand::Rng {
    use rand::SeedableRng;
    rand::rngs::SmallRng::seed_from_u64(0x6268_7473_6e65)
}

/// Random initializes the embedding sampling from a normal distribution with mean 0 and sigma 1e-4.
///
/// # Arguments
///
/// `y` - embedding.
pub(super) fn random_init<T: Float + Send + Sync + Copy>(y: &mut [T]) {
    let distr = Normal::new(0.0, 1e-4).unwrap();
    let mut rng = make_rng();
    y.iter_mut()
        .for_each(|el| *el = T::from(distr.sample(&mut rng)).unwrap());
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
    distances: &mut [T],
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
            *d = f(g(&i), g(&j));
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
pub(super) fn search_beta<T>(p_values_row: &mut [T], distances_row: &[T], perplexity: &T)
where
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

    debug_assert_eq!(p_values_row.len(), distances_row.len());

    while !found && iteration < 200 {
        // Here the values of a single row are computed.
        p_values_row
            .iter_mut()
            .zip(distances_row.iter())
            .for_each(|(p, d)| {
                *p = (-beta * d.powi(2)).exp();
            });

        // After that the row is normalized.
        p_values_row_sum = p_values_row.iter().copied().sum::<T>() + T::min_positive_value();

        // The conditional distribution's entropy is needed to find the optimal value
        // for beta, i.e. the bandwidth of the Gaussian kernel.
        let mut entropy = p_values_row
            .iter()
            .zip(distances_row.iter())
            .fold(T::zero(), |acc, (p, d)| acc + beta * *p * d.powi(2));
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
                    beta /= two;
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
        .for_each(|p| *p /= p_values_row_sum + T::epsilon());
}

/// Normalizes the P values.
///
/// # Arguments
///
/// `p_values` - values of the P distribution.
pub(super) fn normalize_p_values<T: Float + Send + Sync + MulAssign + Sum>(p_values: &mut [T]) {
    let p_values_sum: T = p_values.par_iter().copied().sum::<T>();
    // Fold the normalization and the early-exaggeration factor into one reciprocal, so the hot pass
    // is a single multiply per value rather than a divide and a multiply.
    let scale = T::from(12.0).unwrap() / (p_values_sum + T::epsilon());
    p_values.par_iter_mut().for_each(|p| *p *= scale);
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
    sym_p_columns: &mut Vec<u32>,
    p_columns: Vec<u32>,
    p_values: &mut Vec<T>,
    n_samples: usize,
    n_neighbors: &usize,
) where
    T: Float + Add + DivAssign + Send + Sync + MulAssign,
{
    // Neighbor indices are stored as u32 to halve the index memory at large n; `col` reads one as a
    // usize wherever it is used to index into a buffer.
    let col = |i: usize| p_columns[i] as usize;
    // Each entry of row_counts corresponds to the number of elements for each corresponding row of
    // the symmetric sparse final P matrix.
    let mut row_counts: Vec<usize> = vec![0; n_samples];
    // This sparse and symmetric matrix, due to the nature of the joint probability distribution P,
    // has possibly less entries than the current buffer p_values. In order to construct such a
    // sparse representation, the number of elements contained in each row is needed.
    // Recall that each i-th row corresponds to the joint distribution of the i-th sample.
    let p_rows = |i| i * n_neighbors;

    for n in 0..n_samples {
        for i in p_rows(n)..p_rows(n + 1) {
            row_counts[n] += 1;
            if !p_columns[p_rows(col(i))..p_rows(col(i) + 1)].contains(&(n as u32)) {
                row_counts[col(i)] += 1;
            }
        }
    }

    let total: usize = row_counts.iter().sum();

    let mut sym_row_p: Vec<usize> = vec![0; n_samples + 1];
    let mut sym_col_p: Vec<u32> = vec![0; total];
    let mut sym_val_p: Vec<T> = vec![T::zero(); total];

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
            for m in p_rows(col(i))..p_rows(col(i) + 1) {
                if col(m) == _n {
                    present = true;
                    // Make sure we do not add elements twice.
                    if _n <= col(i) {
                        sym_col_p[sym_row_p[_n] + offset[_n]] = p_columns[i];
                        sym_col_p[sym_row_p[col(i)] + offset[col(i)]] = _n as u32;
                        sym_val_p[sym_row_p[_n] + offset[_n]] = p_values[i] + p_values[m];
                        sym_val_p[sym_row_p[col(i)] + offset[col(i)]] = p_values[i] + p_values[m];
                    }
                }
            }
            // If (col_P[i], n) is not present, there is no addition involved.
            if !present {
                sym_col_p[sym_row_p[_n] + offset[_n]] = p_columns[i];
                sym_col_p[sym_row_p[col(i)] + offset[col(i)]] = _n as u32;
                sym_val_p[sym_row_p[_n] + offset[_n]] = p_values[i];
                sym_val_p[sym_row_p[col(i)] + offset[col(i)]] = p_values[i];
            }
            // Update offsets.
            if !present || _n <= col(i) {
                offset[_n] += 1;
                if col(i) != _n {
                    offset[col(i)] += 1;
                }
            }
        }
    }

    // Divide result by two.
    let zero_point_five = T::from(0.5).unwrap();
    sym_val_p.iter_mut().for_each(|p| *p *= zero_point_five);

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
    y: &mut [T],
    dy: &[T],
    uy: &mut [T],
    gains: &mut [T],
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
            *gains_el = if dy_el.signum() != uy_el.signum() {
                *gains_el + zero_point_two
            } else {
                *gains_el * zero_point_eight
            };
            if *gains_el < zero_point_zero_one {
                *gains_el = zero_point_zero_one;
            }
            *uy_el = *momentum * *uy_el - *learning_rate * *gains_el * *dy_el;
            *y_el += *uy_el
        });
}

/// Adjust the P distribution values to the original ones in parallel.
///
/// # Arguments
///
/// `p_values` - P distribution.
pub(super) fn stop_lying<T: Float + Send + Sync + MulAssign>(p_values: &mut [T]) {
    let scale = T::one() / T::from(12.0).unwrap();
    p_values.par_iter_mut().for_each(|p| *p *= scale);
}

/// Recenters `y` to zero mean. The per-dimension sums and the subtraction are both parallel passes.
pub(super) fn zero_mean<T, const D: usize>(y: &mut [T], n_samples: usize)
where
    T: Float + Send + Sync + Copy + AddAssign + DivAssign + SubAssign,
{
    let mut means = y
        .par_chunks_exact(D)
        .fold(
            || [T::zero(); D],
            |mut totals, sample| {
                totals
                    .iter_mut()
                    .zip(sample.iter())
                    .for_each(|(total, el)| *total += *el);
                totals
            },
        )
        .reduce(
            || [T::zero(); D],
            |mut left, right| {
                left.iter_mut()
                    .zip(right.iter())
                    .for_each(|(total, partial)| *total += *partial);
                left
            },
        );
    let n_samples = T::from(n_samples).unwrap();
    means.iter_mut().for_each(|mean| *mean /= n_samples);

    y.par_chunks_mut(D).for_each(|sample| {
        sample
            .iter_mut()
            .zip(means.iter())
            .for_each(|(el, mean)| *el -= *mean);
    });
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
pub(crate) fn evaluate_error<T, const D: usize>(p_values: &[T], y: &[T], n_samples: usize) -> T
where
    T: Float + Send + Sync + AddAssign + Add + DivAssign + Sum,
{
    let mut distances: Vec<T> = vec![T::zero(); n_samples * n_samples];
    let (points, _) = y.as_chunks::<D>();
    compute_pairwise_distance_matrix(
        &mut distances,
        |a: &[T; D], b: &[T; D]| {
            a.iter()
                .zip(b.iter())
                .map(|(aa, bb)| (*aa - *bb).powi(2))
                .sum::<T>()
        },
        |i| &points[*i],
        n_samples,
    );

    let mut q_values: Vec<T> = vec![T::zero(); n_samples * n_samples];
    q_values
        .par_iter_mut()
        .zip(distances.par_iter())
        .for_each(|(q, d)| *q = T::one() / (T::one() + *d));

    let q_sum = q_values.par_iter().map(|q| *q).sum::<T>();
    let inverse_q_sum = T::one() / q_sum;
    q_values
        .par_iter_mut()
        .for_each(|q| *q = *q * inverse_q_sum);

    // Kullback-Leibler divergence.
    p_values
        .par_iter()
        .zip(q_values.par_iter())
        .fold(
            || T::zero(),
            |c, (p, q)| {
                c + *p * ((*p + T::min_positive_value()) / (*q + T::min_positive_value())).ln()
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
/// * `theta` - threshold for the Barnes-Hut algorithm.
pub(crate) fn evaluate_error_approximately<T, const D: usize>(
    p_rows: &[usize],
    p_columns: &[u32],
    p_values: &[T],
    y: &[T],
    n_samples: usize,
    theta: T,
) -> T
where
    T: Float + Send + Sync + Sum + AddAssign + SubAssign + MulAssign + DivAssign,
    morton::Dim<D>: morton::Morton<D>,
{
    // Get estimate of normalization term.
    let q_sum = {
        let arena = arena::Arena::<T, D>::new(y, n_samples);
        let theta_sq = theta * theta;
        let mut q_sums: Vec<T> = vec![T::zero(); n_samples];

        q_sums.par_iter_mut().enumerate().for_each(|(index, sum)| {
            // Local scratch: the repulsive forces are not needed here, only their q_sum.
            let mut negative_forces = [T::zero(); D];
            let mut stack = [0u32; arena::STACK_CAP];
            arena.compute_non_edge_forces(
                index,
                theta_sq,
                y,
                &mut negative_forces,
                sum,
                &mut stack,
            );
        });

        q_sums.par_iter().map(|sum| *sum).sum::<T>()
    };
    let inverse_q_sum = T::one() / q_sum;

    let mut partials: Vec<T> = vec![T::zero(); n_samples];

    partials
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, cost)| {
            let sample_a = &y[index * D..(index + 1) * D];
            for &column in &p_columns[p_rows[index]..p_rows[index + 1]] {
                let column = column as usize;
                let sample_b = &y[column * D..(column + 1) * D];

                let mut q = sample_a
                    .iter()
                    .zip(sample_b.iter())
                    .map(|(a, b)| (*a - *b).powi(2))
                    .sum::<T>();
                q = (T::one() / (T::one() + q)) * inverse_q_sum;

                // Kullback-Leibler divergence.
                *cost += p_values[index]
                    * ((p_values[index] + T::min_positive_value()) / (q + T::min_positive_value()))
                        .ln();
            }
        });

    partials.par_iter().map(|partial| *partial).sum::<T>()
}
