mod sptree;
mod vptree;

use super::{Float, NumCast};
use sptree::SPTree;
use vptree::{DataPoint, VPTree};

/// Computes the exact t-SNE gradient.
pub fn compute_gradient<T>(p: &mut [T], y: &[T], n: usize, d: usize, dc: &mut [T])
where
    T: Float + std::ops::Add + std::ops::Sub + std::ops::Mul + std::ops::AddAssign + std::iter::Sum,
{
    // Make sure the current gradient contains zeros.
    for el in &mut dc[..] {
        *el = T::zero();
    }

    let mut dd: Vec<T> = vec![T::zero(); n * n];
    // Compute the squared Euclidean distance matrix.
    compute_squared_euclidean_distance(y, n, d, &mut dd);

    // Compute Q-matrix and normalization sum.
    let mut q: Vec<T> = vec![T::zero(); n * n];
    let mut sum_q: T = T::zero();

    let mut nn: usize = 0;
    for _n in 0..n {
        for m in 0..n {
            if _n != m {
                q[nn + m] = T::one() / (T::one() + dd[nn + m]);
                sum_q += q[nn + m];
            }
        }
        nn += n;
    }

    // Perform the computation of the gradient.
    nn = 0;
    let mut nd: usize = 0;
    for _n in 0..n {
        let mut md: usize = 0;
        for m in 0..n {
            if _n != m {
                let mult: T = (p[nn + m] - (q[nn + m] / sum_q)) * q[nn + m];
                for _d in 0..d {
                    dc[nd + _d] += (y[nd + _d] - y[md + _d]) * mult;
                }
            }
            md += d;
        }
        nn += n;
        nd += d;
    }
}

/// Computes an approximation of the t-SNE gradient.
pub fn compute_gradient_approx<T>(
    inp_row_p: &mut [usize],
    inp_col_p: &mut [usize],
    inp_val_p: &mut [T],
    y: &[T],
    n: usize,
    d: usize,
    dc: &mut [T],
    theta: T,
) where
    T: Float
        + std::ops::Sub
        + std::ops::Div
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + NumCast,
{
    // Construct space partitioning tree on current map.
    let mut tree: SPTree<T> = SPTree::new(d, y, n);
    // Compute all terms required for t-SNE gradient.
    let mut sum_q: T = T::zero();
    let mut pos_f: Vec<T> = vec![T::zero(); n * d];
    let mut neg_f: Vec<T> = vec![T::zero(); n * d];

    tree.compute_edge_forces(inp_row_p, inp_col_p, inp_val_p, n, &mut pos_f);
    for _n in 0..n {
        tree.compute_non_edge_forces(_n, theta, &mut neg_f[_n * d.._n * d + d], &mut sum_q)
    }
    // Compute final t-SNE gradient.
    for i in 0..(n * d) {
        dc[i] = pos_f[i] - (neg_f[i] / sum_q);
    }
}

/// Evaluate t-SNE cost function exactly.
#[allow(dead_code)]
pub fn evaluate_error<T>(p: &mut [T], y: &[T], n: usize, d: usize) -> T
where
    T: Float + std::ops::AddAssign + std::ops::Add + std::ops::DivAssign + std::iter::Sum,
{
    let mut dd: Vec<T> = vec![T::zero(); n * n];
    let mut q: Vec<T> = vec![T::zero(); n * n];
    compute_squared_euclidean_distance(y, n, d, &mut dd);

    // Compute Q-matrix and normalization sum.
    let mut nn: usize = 0;
    let mut sum_q = T::min_positive_value();
    for _n in 0..n {
        for m in 0..n {
            if _n != m {
                q[nn + m] = T::one() / (T::one() + dd[nn + m]);
                sum_q += q[nn + m];
            } else {
                q[nn + m] = T::min_positive_value();
            }
        }
        nn += n;
    }

    for el in &mut q[..] {
        *el /= sum_q;
    }

    let mut c: T = T::zero();
    for _n in 0..(n * n) {
        c += p[_n] * ((p[_n] + T::min_positive_value()) / (q[_n] + T::min_positive_value())).ln()
    }
    c
}

/// Evaluate t-SNE cost function approximately.
#[allow(dead_code)]
pub fn evaluate_error_approx<'a, T>(
    row_p: &mut [usize],
    col_p: &mut [usize],
    val_p: &mut [T],
    y: &'a [T],
    n: usize,
    d: usize,
    theta: T,
) -> T
where
    T: Float
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + NumCast,
{
    // Get estimate of normalization term.
    let mut tree: SPTree<T> = SPTree::new(d, y, n);
    let mut buff: Vec<T> = vec![T::zero(); d];
    let mut sum_q: T = T::zero();
    for _n in 0..n {
        tree.compute_non_edge_forces(_n, theta, &mut buff, &mut sum_q);
    }
    // Loop over all edges to compute t-SNE error.
    let mut ind1;
    let mut ind2;
    let mut c: T = T::zero();
    let mut q;

    for _n in 0..n {
        ind1 = _n * d;
        for i in row_p[_n]..row_p[_n + 1] {
            q = T::zero();
            ind2 = col_p[i] * d;

            buff[..d].clone_from_slice(&y[ind1..(d + ind1)]);

            for (i, el) in buff.iter_mut().enumerate() {
                *el -= y[ind2 + i];
            }

            for el in &buff[..] {
                q += *el * *el;
            }
            q = (T::one() / (T::one() + q)) / sum_q;
            c += val_p[i]
                * ((val_p[i] + T::min_positive_value()) / (q + T::min_positive_value())).ln();
        }
    }
    c
}

/// Compute input similarities with a fixed perplexity.
pub fn compute_fixed_gaussian_perplexity<T>(
    x: &mut [T],
    n: usize,
    d: usize,
    p: &mut [T],
    perplexity: T,
) where
    T: Float
        + NumCast
        + std::ops::Div
        + std::ops::DivAssign
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::iter::Sum,
{
    // Compute the squared Euclidean distance matrix.
    let mut dd: Vec<T> = vec![T::zero(); n * n];
    compute_squared_euclidean_distance(x, n, d, &mut dd);

    // Compute the gaussian kernel row by row.
    let mut nn: usize = 0;

    for _n in 0..n {
        let mut found: bool = false;
        let mut beta: T = T::one();
        let mut min_beta: T = -T::max_value();
        let mut max_beta: T = T::max_value();
        let tol: T = T::from(1e-5).unwrap();

        let mut sum_p: T = T::zero();

        let mut iter: u64 = 0;

        // Iterate until we found a good perplexity.
        while !found && iter < 200 {
            // Compute Gaussian kernel row.
            for m in 0..n {
                p[nn + m] = (-beta * dd[nn + m]).exp();
            }
            p[nn + _n] = T::min_positive_value();

            // Compute entropy of current row.
            sum_p = T::min_positive_value();
            for m in 0..n {
                sum_p += p[nn + m];
            }

            let mut h: T = T::zero();
            for m in 0..n {
                h += beta * (dd[nn + m] * p[nn + m]);
            }
            h = (h / sum_p) + sum_p.ln();

            // Evaluate whether the entropy is within the tolerance level.
            let h_diff: T = h - perplexity.ln();

            if h_diff < tol && -h_diff < tol {
                found = true;
            } else {
                if h_diff > T::zero() {
                    min_beta = beta;

                    if max_beta == T::max_value() || max_beta == -T::max_value() {
                        beta *= T::from(2.0).unwrap();
                    } else {
                        beta = (beta + max_beta) / T::from(2.0).unwrap();
                    }
                } else {
                    max_beta = beta;

                    if min_beta == -T::max_value() || min_beta == T::max_value() {
                        beta /= T::from(2.0).unwrap();
                    } else {
                        beta = (beta + min_beta) / T::from(2.0).unwrap();
                    }
                }
                // Check for overflows.
                if beta.is_infinite() && beta.is_sign_positive() {
                    beta = T::max_value()
                }
                if beta.is_infinite() && beta.is_sign_negative() {
                    beta = -T::max_value()
                }
            }
            // Update iteration counter.
            iter += 1;
        }
        // Row normalize P.
        for m in 0..n {
            p[nn + m] /= sum_p;
        }
        nn += n;
    }
}

/// Compute input similarities with a fixed perplexity using ball trees.
pub fn compute_gaussian_perplexity<T>(
    x: &mut [T],
    n: usize,
    d: usize,
    row_p: &mut Vec<usize>,
    col_p: &mut Vec<usize>,
    val_p: &mut Vec<T>,
    perplexity: T,
    k: usize,
) where
    T: Float
        + NumCast
        + std::iter::Sum
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
{
    let mut cur_p: Vec<T> = vec![T::zero(); n - 1];

    row_p[0] = 0;
    for _n in 0..n {
        row_p[_n + 1] = row_p[_n] + k;
    }

    // Build ball tree on data set.
    let mut items: Vec<DataPoint<&[T]>> = Vec::new();
    for (index, chunk) in x.chunks_exact(d).enumerate() {
        items.push(DataPoint::new(index as u64, chunk, d));
    }
    let mut slice_of_ref: Vec<&DataPoint<&[T]>> = items.iter().collect::<Vec<&DataPoint<&[T]>>>();
    let mut tree: VPTree<T> = VPTree::new(&mut slice_of_ref);

    let mut indices: Vec<u64> = Vec::new();
    let mut distances: Vec<T> = Vec::new();

    // Loop over all points to find nearest neighbors.
    for _n in 0..n {
        // Find nearest neighbors.
        tree.search(&items[_n], k + 1, &mut indices, &mut distances);
        // Initialize some variables for binary search.
        let mut found: bool = false;
        let mut beta: T = T::one();
        let mut max_beta: T = T::max_value();
        let mut min_beta: T = -T::max_value();
        let tol: T = T::from(1e-5).unwrap();

        let mut iter: u64 = 0;
        let mut sum_p: T = T::zero();
        // Iterate until we found a good perplexity.
        while !found && iter < 200 {
            for m in 0..k {
                cur_p[m] = (-beta * distances[m + 1] * distances[m + 1]).exp();
            }
            // Compute entropy of current row.
            sum_p = T::min_positive_value();
            for el in &cur_p[..] {
                sum_p += *el;
            }

            let mut h: T = T::zero();
            for m in 0..k {
                h += beta * (distances[m + 1] * distances[m + 1] * cur_p[m]);
            }
            h = (h / sum_p) + sum_p.ln();

            // Evaluate whether the entropy is within the tolerance level.
            let h_diff: T = h - perplexity.ln();
            if h_diff < tol && -h_diff < tol {
                found = true;
            } else {
                if h_diff > T::zero() {
                    min_beta = beta;
                    if max_beta == T::max_value() || max_beta == -T::max_value() {
                        beta *= T::from(2.0).unwrap()
                    } else {
                        beta = (beta + max_beta) / T::from(2.0).unwrap();
                    }
                } else {
                    max_beta = beta;
                    if min_beta == T::max_value() || min_beta == -T::max_value() {
                        beta /= T::from(2.0).unwrap();
                    } else {
                        beta = (beta + min_beta) / T::from(2.0).unwrap();
                    }
                }
                // Check for overflows.
                if beta.is_infinite() && beta.is_sign_positive() {
                    beta = T::max_value()
                }
                if beta.is_infinite() && beta.is_sign_negative() {
                    beta = -T::max_value()
                }
            }
            // Update iteration counter.
            iter += 1;
        }

        // Row-normalize current row of P and store in matrix.
        for el in &mut cur_p[..] {
            *el /= sum_p;
        }

        for m in 0..k {
            col_p[row_p[_n] + m] = indices[m + 1] as usize;
            val_p[row_p[_n] + m] = cur_p[m];
        }
    }
}

/// Symmetrizes a sparse matrix.
pub fn symmetrize_matrix<T>(
    row_p: &mut Vec<usize>,
    col_p: &mut Vec<usize>,
    val_p: &mut Vec<T>,
    n: usize,
) where
    T: Float + NumCast + std::ops::Add + std::ops::DivAssign,
{
    // Count number of elements and row counts of symmetric matrix.
    let mut row_counts: Vec<usize> = vec![0; n];

    for _n in 0..n {
        for i in row_p[_n]..row_p[_n + 1] {
            // Check whether element (col_P[i], _n) is present.
            let mut present: bool = false;
            for el in col_p.iter().take(row_p[col_p[i] + 1]).skip(row_p[col_p[i]]) {
                if *el == _n {
                    present = true;
                }
            }
            if present {
                row_counts[_n] += 1;
            } else {
                row_counts[_n] += 1;
                row_counts[col_p[i]] += 1;
            }
        }
    }

    let mut no_el: usize = 0;
    for el in &row_counts[..] {
        no_el += *el;
    }

    let mut sym_row_p: Vec<usize> = vec![0; n + 1];
    let mut sym_col_p: Vec<usize> = vec![0; no_el];
    let mut sym_val_p: Vec<T> = vec![T::zero(); no_el];

    sym_row_p[0] = 0;
    for _n in 0..n {
        sym_row_p[_n + 1] = sym_row_p[_n] + row_counts[_n];
    }

    let mut offset: Vec<usize> = vec![0; n];

    for _n in 0..n {
        for i in row_p[_n]..row_p[_n + 1] {
            // Check whether element (col_P[i], n) is present.
            let mut present: bool = false;
            // Considering element(n, col_P[i]).
            for m in row_p[col_p[i]]..row_p[col_p[i] + 1] {
                if col_p[m] == _n {
                    present = true;
                    // Make sure we do not add elements twice.
                    if _n <= col_p[i] {
                        sym_col_p[sym_row_p[_n] + offset[_n]] = col_p[i];
                        sym_col_p[sym_row_p[col_p[i]] + offset[col_p[i]]] = _n;
                        sym_val_p[sym_row_p[_n] + offset[_n]] = val_p[i] + val_p[m];
                        sym_val_p[sym_row_p[col_p[i]] + offset[col_p[i]]] = val_p[i] + val_p[m];
                    }
                }
            }
            // If (col_P[i], n) is not present, there is no addition involved.
            if !present {
                sym_col_p[sym_row_p[_n] + offset[_n]] = col_p[i];
                sym_col_p[sym_row_p[col_p[i]] + offset[col_p[i]]] = _n;
                sym_val_p[sym_row_p[_n] + offset[_n]] = val_p[i];
                sym_val_p[sym_row_p[col_p[i]] + offset[col_p[i]]] = val_p[i];
            }
            // Update offsets.
            if !present || _n <= col_p[i] {
                offset[_n] += 1;
                if col_p[i] != _n {
                    offset[col_p[i]] += 1;
                }
            }
        }
    }

    // Divide the result by two
    for el in &mut sym_val_p[..] {
        *el /= T::from(2.0).unwrap();
    }

    // Return symmetrized matrices.
    *row_p = sym_row_p;
    *col_p = sym_col_p;
    *val_p = sym_val_p;
}

/// Compute squared Euclidean distance matrix.
pub fn compute_squared_euclidean_distance<T>(x: &[T], n: usize, d: usize, dd: &mut [T])
where
    T: Float + std::ops::Sub + std::ops::Mul + std::ops::AddAssign + std::iter::Sum,
{
    let mut ptr: &[T] = x;
    let mut i = 0;

    while ptr.len() > d {
        let mut j = i * (n + 1);
        dd[j] = T::zero();
        j += 1;

        let (v, o): (&[T], &[T]) = ptr.split_at(d);
        for chunk in o.chunks(d) {
            dd[j] = {
                v.iter()
                    .zip(chunk.iter())
                    .map(|(v, c)| (*v - *c) * (*v - *c))
                    .sum()
            };
            dd[(n * j) % (n * n - 1)] = dd[j];
            j += 1;
        }
        ptr = o;
        i += 1;
    }
}

/// Makes data zero-mean.
pub fn zero_mean<T>(x: &mut [T], n: usize, d: usize)
where
    T: Float + NumCast + std::ops::AddAssign + std::ops::SubAssign + std::ops::DivAssign,
{
    // Compute data mean
    let mut mean: Vec<T> = vec![T::zero(); d];
    let mut n_d: usize = 0;
    for _n in 0..n {
        for _d in 0..d {
            mean[_d] += x[n_d + _d];
        }
        n_d += d;
    }
    for el in &mut mean[..] {
        *el /= T::from(n).unwrap();
    }

    // Subtract data mean
    n_d = 0;
    for _n in 0..n {
        for _d in 0..d {
            x[n_d + _d] -= mean[_d];
        }
        n_d += d;
    }
}
