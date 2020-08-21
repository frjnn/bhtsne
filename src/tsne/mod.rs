//! # tsne
//!
//! `tsne` contains the implementations of the
//! vanilla version of t-SNE and the Barnes-Hut optimization
//! and all of the necessary data structures such as VP-trees and
//! SP-trees.
mod sptree;
mod vptree;

use rand_distr::{Distribution, Normal};
use sptree::SPTree;
use vptree::{DataPoint, VPTree};

/// Computes the exact t-SNE gradient.
pub fn compute_gradient(p: &mut [f64], y: &[f64], n: usize, d: usize, dc: &mut [f64]) {
    // Make sure the current gradient contains zeros.
    for i in 0..(n * d) {
        dc[i] = 0.0;
    }

    let mut dd: Vec<f64> = vec![0.0; n * n];
    // Compute the squared Euclidean distance matrix recursively.
    compute_squared_euclidean_distance(y, n, d, &mut dd, 0);

    // Compute Q-matrix and normalization sum.
    let mut q: Vec<f64> = vec![0.0; n * n];
    let mut sum_q: f64 = 0.0;

    let mut nn: usize = 0;
    for _n in 0..n {
        for m in 0..n {
            if _n != m {
                q[nn + m] = 1.0 / (1.0 + dd[nn + m]);
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
                let mult: f64 = (p[nn + m] - q[nn + m] / sum_q) * q[nn + m];

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
pub fn compute_gradient_approx(
    inp_row_p: &mut [usize],
    inp_col_p: &mut [usize],
    inp_val_p: &mut [f64],
    y: &[f64],
    n: usize,
    d: usize,
    dc: &mut [f64],
    theta: f64,
) {
    // Construct space partitioning tree on current map.
    let mut tree: SPTree = SPTree::new(d, y, n);
    // Compute all terms required for t-SNE gradient.
    let mut sum_q: f64 = 0.0;
    let mut pos_f: Vec<f64> = vec![0.0; n * d];
    let mut neg_f: Vec<f64> = vec![0.0; n * d];

    tree.compute_edge_forces(inp_row_p, inp_col_p, inp_val_p, n, &mut pos_f);
    for _n in 0..n {
        tree.compute_non_edge_forces(_n, theta, &mut neg_f[_n * d.._n * d + d], &mut sum_q)
    }
    // Compute final t-SNE gradient.
    for i in 0..(n * d) {
        dc[i] = pos_f[i] - (neg_f[i] / sum_q);
    }
}

/// Evaluate tnse cost function exactly.
pub fn evaluate_error(p: &mut [f64], y: &[f64], n: usize, d: usize) -> f64 {
    let mut dd: Vec<f64> = vec![0.0; n * n];
    let mut q: Vec<f64> = vec![0.0; n * n];
    compute_squared_euclidean_distance(y, n, d, &mut dd, 0);

    // Compute Q-matrix and normalization sum.
    let mut nn: usize = 0;
    let mut sum_q = std::f64::MIN_POSITIVE;
    for _n in 0..n {
        for m in 0..n {
            if _n != m {
                q[nn + m] = 1.0 / (1.0 + dd[nn + m]);
                sum_q += q[nn + m];
            } else {
                q[nn + m] = std::f64::MIN_POSITIVE;
            }
        }
        nn += n;
    }

    for i in 0..(n * n) {
        q[i] /= sum_q;
    }

    let mut c: f64 = 0.0;
    for _n in 0..(n * n) {
        c += p[_n]
            * ((p[_n] + std::f32::MIN_POSITIVE as f64) / (q[_n] + std::f32::MIN_POSITIVE as f64))
                .ln()
    }
    c
}

/// Evaluate t-SNE cost function approximately.
pub fn evaluate_error_approx<'a>(
    row_p: &mut [usize],
    col_p: &mut [usize],
    val_p: &mut [f64],
    y: &'a [f64],
    n: usize,
    d: usize,
    theta: f64,
) -> f64 {
    // Get estimate of normalization term.
    let mut tree: SPTree = SPTree::new(d, y, n);
    let mut buff: Vec<f64> = vec![0.0; d];
    let mut sum_q: f64 = 0.0;
    for _n in 0..n {
        tree.compute_non_edge_forces(_n, theta, &mut buff, &mut sum_q);
    }
    // Loop over all edges to compute t-SNE error.
    let mut ind1;
    let mut ind2;
    let mut c: f64 = 0.0;
    let mut q;

    for _n in 0..n {
        ind1 = _n * d;
        for i in row_p[_n]..row_p[_n + 1] {
            q = 0.0;
            ind2 = col_p[i] * d;

            for _d in 0..d {
                buff[_d] = y[ind1 + _d];
            }
            for _d in 0..d {
                buff[_d] -= y[ind2 + _d];
            }
            for _d in 0..d {
                q += buff[_d] * buff[_d];
            }
            q = (1.0 / (1.0 + q)) / sum_q;
            //println!("q: {}", q);
            c += val_p[i]
                * ((val_p[i] + std::f32::MIN_POSITIVE as f64)
                    / (q + std::f32::MIN_POSITIVE as f64))
                    .ln();
        }
    }
    c
}

/// Compute input similarities with a fixed perplexity.
pub fn compute_fixed_gaussian_perplexity(
    x: &mut [f64],
    n: usize,
    d: usize,
    p: &mut [f64],
    perplexity: f64,
) {
    // Compute the squared Euclidean distance matrix.
    let mut dd: Vec<f64> = vec![0.0; n * n];
    compute_squared_euclidean_distance(x, n, d, &mut dd, 0);

    // Compute the gaussian kernel row by row.
    let mut nn: usize = 0;
    for _n in 0..n {
        let mut found: bool = false;
        let mut beta: f64 = 1.0;
        let mut min_beta: f64 = -std::f64::MAX;
        let mut max_beta: f64 = std::f64::MAX;
        const TOL: f64 = 1e-5;

        let mut sum_p: f64 = 0.0;

        let mut iter: u64 = 0;

        // Iterate until we found a good perplexity.
        while !found && iter < 200 {
            // Compute Gaussian kernel row.
            for m in 0..n {
                p[nn + m] = (-beta * dd[nn + m]).exp();
            }
            p[nn + _n] = std::f64::MIN_POSITIVE;
            // Compute entropy of current row.
            sum_p = std::f64::MIN_POSITIVE;
            for m in 0..n {
                sum_p += p[nn + m];
            }

            let mut h: f64 = 0.0;
            for m in 0..n {
                h += beta * (dd[nn + m] * p[nn + m]);
            }
            h = h / sum_p + sum_p.ln();

            // Evaluate whether the entropy is within the tolerance level.
            let h_diff: f64 = h - perplexity.ln();
            if h_diff < TOL && -h_diff < TOL {
                found = true;
            } else {
                if h_diff > 0.0 {
                    min_beta = beta;

                    if max_beta == std::f64::MAX || max_beta == -std::f64::MAX {
                        beta *= 2.0;
                    } else {
                        beta = (beta + max_beta) / 2.0;
                    }
                } else {
                    max_beta = beta;

                    if min_beta == -std::f64::MAX || min_beta == std::f64::MAX {
                        beta /= 2.0;
                    } else {
                        beta = (beta + min_beta) / 2.0;
                    }
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
pub fn compute_gaussian_perplexity(
    x: &mut [f64],
    n: usize,
    d: usize,
    row_p: &mut Vec<usize>,
    col_p: &mut Vec<usize>,
    val_p: &mut Vec<f64>,
    perplexity: f64,
    k: usize,
) {
    let mut cur_p: Vec<f64> = vec![0.0; n - 1];

    row_p[0] = 0;
    for _n in 0..n {
        row_p[_n + 1] = row_p[_n] + k;
    }

    print!("Building VPTree...\n\n");
    // Build ball tree on data set.
    let mut index: u64 = 0;
    let mut items: Vec<DataPoint<f64>> = Vec::new();
    for chunk in x.chunks_exact(d) {
        items.push(DataPoint::new(index, chunk, d));
        index += 1;
    }
    let mut slice_of_ref: Vec<&DataPoint<f64>> = items.iter().collect::<Vec<&DataPoint<f64>>>();
    let mut tree: VPTree<DataPoint<f64>> = VPTree::new(&mut slice_of_ref);

    let mut indices: Vec<DataPoint<f64>> = Vec::new();
    let mut distances: Vec<f64> = Vec::new();

    // Loop over all points to find nearest neighbors.
    for _n in 0..n {
        // Find nearest neighbors.
        tree.search(&items[_n], k + 1, &mut indices, &mut distances);
        // Initialize some variables for binary search.
        let mut found: bool = false;
        let mut beta: f64 = 1.0;
        let mut max_beta: f64 = std::f64::MAX;
        let mut min_beta: f64 = -std::f64::MAX;
        const TOL: f64 = 1e-5;

        let mut iter: u64 = 0;
        let mut sum_p: f64 = 0.0;
        // Iterate until we found a good perplexity.
        while !found && iter < 200 {
            for m in 0..k {
                cur_p[m] = (-beta * distances[m + 1] * distances[m + 1]).exp();
            }
            // Compute entropy of current row.
            sum_p = std::f64::MIN_POSITIVE;
            for m in 0..k {
                sum_p += cur_p[m];
            }

            let mut h: f64 = 0.0;
            for m in 0..k {
                h += beta * (distances[m + 1] * distances[m + 1] * cur_p[m]);
            }
            h = (h / sum_p) + sum_p.ln();

            // Evaluate whether the entropy is within the tolerance level.
            let h_diff: f64 = h - perplexity.ln();
            if h_diff < TOL && -h_diff < TOL {
                found = true;
            } else {
                if h_diff > 0.0 {
                    min_beta = beta;
                    if max_beta == std::f64::MAX || max_beta == -std::f64::MAX {
                        beta *= 2.0
                    } else {
                        beta = (beta + max_beta) / 2.0;
                    }
                } else {
                    max_beta = beta;
                    if min_beta == std::f64::MAX || min_beta == -std::f64::MAX {
                        beta /= 2.0;
                    } else {
                        beta = (beta + min_beta) / 2.0;
                    }
                }
            }
            // Update iteration counter.
            iter += 1;
        }

        // Row-normalize current row of P and store in matrix.
        for m in 0..k {
            cur_p[m] /= sum_p;
        }

        for m in 0..k {
            col_p[row_p[_n] + m] = indices[m + 1].ind as usize;
            val_p[row_p[_n] + m] = cur_p[m];
        }
    }
}

/// Symmetrizes a sparse matrix.
pub fn symmetrize_matrix(
    row_p: &mut Vec<usize>,
    col_p: &mut Vec<usize>,
    val_p: &mut Vec<f64>,
    n: usize,
) {
    // Count number of elements and row counts of symmetric matrix.
    let mut row_counts: Vec<usize> = vec![0; n];

    for _n in 0..n {
        for i in row_p[_n]..row_p[_n + 1] {
            // Check whether element (col_P[i], _n) is present.
            let mut present: bool = false;
            for m in row_p[col_p[i]]..row_p[col_p[i] + 1] {
                if col_p[m] == _n {
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
    for _n in 0..n {
        no_el += row_counts[_n];
    }

    let mut sym_row_p: Vec<usize> = vec![0; n + 1];
    let mut sym_col_p: Vec<usize> = vec![0; no_el];
    let mut sym_val_p: Vec<f64> = vec![0.0; no_el];

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
            if !present || (present && _n <= col_p[i]) {
                offset[_n] += 1;
                if col_p[i] != _n {
                    offset[col_p[i]] += 1;
                }
            }
        }
    }

    // Divide the result by two
    for i in 0..no_el {
        sym_val_p[i] /= 2.0;
    }

    // Return symmetrized matrices.
    *row_p = sym_row_p;
    *col_p = sym_col_p;
    *val_p = sym_val_p;
}

/// Compute squared Euclidean distance matrix recursively.
pub fn compute_squared_euclidean_distance(x: &[f64], n: usize, d: usize, dd: &mut [f64], i: usize) {
    if x.len() == d {
        return;
    }

    let mut j = (i / d) * (n + 1);
    dd[j] = 0.0;
    j += 1;

    let (v, o): (&[f64], &[f64]) = x.split_at(d);
    for chunk in o.chunks(d) {
        dd[j] = {
            let mut ed: f64 = 0.0;
            for _d in 0..d {
                ed += (v[_d] - chunk[_d]) * (v[_d] - chunk[_d])
            }
            ed
        };
        dd[(j + (n - 1) * j) % (n * n - 1)] = dd[j];
        j += 1;
    }
    compute_squared_euclidean_distance(o, n, d, dd, i + d);
}

/// Makes data zero-mean.
pub fn zero_mean(x: &mut [f64], n: usize, d: usize) {
    // Compute data mean
    let mut mean: Vec<f64> = vec![0.0; d];
    let mut n_d: usize = 0;
    for _n in 0..n {
        for _d in 0..d {
            mean[_d] += x[n_d + _d];
        }
        n_d += d;
    }
    for _d in 0..d {
        mean[_d] /= n as f64;
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

/// Generates a Gaussian random number.
pub fn randn() -> f64 {
    Normal::new(0.0, 1e-4)
        .unwrap()
        .sample(&mut rand::thread_rng())
}
