//! Spectral embedding of the t-SNE affinity graph, used as an initialization for the
//! gradient descent.
//!
//! The entry point is [`spectral_embedding`], which estimates the leading nontrivial
//! eigenvectors of the similarity matrix `T = D^{-1/2} P D^{-1/2}` of the sparse
//! symmetric affinity graph `P` (`D` is the degree diagonal) and rescales them to the
//! seed magnitude expected by the optimizer. The solver is a Chebyshev-filtered
//! subspace iteration with Rayleigh-Ritz projections, chosen because the top
//! eigenvalues of kNN affinity graphs are packed extremely close together and plain
//! power iteration would need thousands of matvecs to separate them. Every reduction
//! runs through a fixed chunking so the result is reproducible run to run regardless
//! of rayon's scheduling.

use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, Range, SubAssign},
};

use num_traits::Float;

use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
    },
    slice::ParallelSliceMut,
};

use super::morton::Dim;

/// Seals [`SpectralBlock`], keeping the trait free to evolve.
mod sealed {
    pub trait Sealed {}
}

/// Fixed-width row storage of the spectral solver, in the spirit of the `Morton`
/// associated types: each dimensionality names its concrete `[T; D + 8]` row type,
/// giving the solver compile-time block widths on stable Rust. The fused kernel
/// unrolls and vectorizes over these rows for about twice the throughput of a
/// dynamic-width loop.
///
/// Sealed, and implemented for [`Dim<D>`] up to `D = 7`, matching the highest
/// Barnes-Hut dimensionality. The `spectral_block` invocation below extends it.
pub trait SpectralBlock: sealed::Sealed {
    /// Width of a block row, the dimensionality plus the oversampling of 8.
    const WIDTH: usize;

    /// A block row, concretely `[T; D + 8]`. Its `Default` value is the zeroed
    /// accumulator row of the fused kernel.
    type Row<T: Float + Default + Send + Sync>: AsRef<[T]> + AsMut<[T]> + Default + Send + Sync;

    /// Views a flat row-major block as typed rows.
    fn as_rows<T: Float + Default + Send + Sync>(flat: &[T]) -> &[Self::Row<T>];

    /// Views a flat row-major mutable block as typed rows.
    fn as_rows_mut<T: Float + Default + Send + Sync>(flat: &mut [T]) -> &mut [Self::Row<T>];
}

macro_rules! spectral_block {
    ($(($dim:literal, $width:literal)),* $(,)?) => {$(
        // The row width must stay in lockstep with the oversampling.
        const _: () = assert!($dim + SPECTRAL_OVERSAMPLE == $width);

        impl sealed::Sealed for Dim<$dim> {}

        impl SpectralBlock for Dim<$dim> {
            const WIDTH: usize = $width;

            type Row<T: Float + Default + Send + Sync> = [T; $width];

            #[inline]
            fn as_rows<T: Float + Default + Send + Sync>(flat: &[T]) -> &[[T; $width]] {
                flat.as_chunks::<$width>().0
            }

            #[inline]
            fn as_rows_mut<T: Float + Default + Send + Sync>(flat: &mut [T]) -> &mut [[T; $width]] {
                flat.as_chunks_mut::<$width>().0
            }
        }
    )*};
}

spectral_block!((1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),);

/// Extra columns carried beyond the embedding dimensionality during the filtered
/// subspace iteration, fixed at the type level through [`SpectralBlock::Row`]. The
/// oversampled block captures the leading eigenspace even when the top eigenvalues
/// are tightly clustered, which is the norm for kNN affinity graphs, and absorbs the
/// unwanted eigenvectors that the filter amplifies alongside the wanted ones. It
/// also caps how many connected components the solver can tell apart: a graph with
/// more than roughly `D + SPECTRAL_OVERSAMPLE` components exhausts the block and the
/// surplus indicators mix arbitrarily.
const SPECTRAL_OVERSAMPLE: usize = 8;

/// Default for [`SpectralParams::rounds`].
const DEFAULT_ROUNDS: usize = 5;

/// Default for [`SpectralParams::degree`].
const DEFAULT_DEGREE: usize = 20;

/// Default for [`SpectralParams::seed_std`].
const DEFAULT_SEED_STD: f64 = 1e-4;

/// Initial upper bound of the unwanted spectral interval `[0, b]` of the shifted
/// operator. Deliberately crude, the Ritz values of the first round replace it.
const CHEBYSHEV_INITIAL_BOUND: f64 = 0.75;

/// Rows per chunk in the deterministic parallel reductions of the spectral solver.
const REDUCTION_ROWS: usize = 4096;

/// Tunable parameters of the spectral embedding initialization, accepted by
/// [`tSNE::spectral_init_with`] and [`tSNE::spectral_embedding_with`]. The defaults
/// resolve the leading eigenvectors of a 70k point MNIST affinity graph to a
/// fraction of a degree, so tuning is only warranted to trade accuracy for speed or
/// to handle unusually structured graphs.
///
/// The solver's work is proportional to `rounds * degree` sparse matvecs over the
/// affinity graph, each spanning the `D + 8` columns of the block.
///
/// ```
/// use bhtsne::SpectralParams;
///
/// let params = SpectralParams::new().rounds(3).degree(12);
/// ```
///
/// [`tSNE::spectral_init_with`]: crate::tSNE::spectral_init_with
/// [`tSNE::spectral_embedding_with`]: crate::tSNE::spectral_embedding_with
#[derive(Clone, Copy, Debug)]
pub struct SpectralParams {
    pub(crate) rounds: usize,
    pub(crate) degree: usize,
    pub(crate) seed_std: f64,
}

impl Default for SpectralParams {
    fn default() -> Self {
        Self {
            rounds: DEFAULT_ROUNDS,
            degree: DEFAULT_DEGREE,
            seed_std: DEFAULT_SEED_STD,
        }
    }
}

impl SpectralParams {
    /// Parameters with the default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Outer rounds of Chebyshev filtering, each followed by orthonormalization and
    /// a Rayleigh-Ritz projection that refines the filter bound. More rounds sharpen
    /// the eigenvectors, fewer trade accuracy for speed (a t-SNE seed tolerates a
    /// fairly rough solve, the early exaggeration phase refines it). Defaults to 5.
    ///
    /// # Panics
    ///
    /// Panics if `rounds` is zero.
    pub fn rounds(mut self, rounds: usize) -> Self {
        assert!(
            rounds >= 1,
            "at least one spectral solver round is required"
        );
        self.rounds = rounds;

        self
    }

    /// Degree of the Chebyshev polynomial applied in each round. Higher degrees
    /// amplify the wanted leading eigenvalues more aggressively per round, which
    /// speeds convergence on tightly clustered spectra. Together with `rounds` this
    /// fixes the matvec budget of the solver. Defaults to 20.
    ///
    /// # Panics
    ///
    /// Panics if `degree` is zero.
    pub fn degree(mut self, degree: usize) -> Self {
        assert!(
            degree >= 1,
            "the Chebyshev filter degree must be at least 1"
        );
        self.degree = degree;

        self
    }

    /// Standard deviation each output column is scaled to. The default matches the
    /// magnitude of the random initialization the seed replaces, which is what the
    /// optimizer's early exaggeration phase expects. Defaults to `1e-4`.
    ///
    /// # Panics
    ///
    /// Panics if `seed_std` is not strictly positive and finite.
    pub fn seed_std(mut self, seed_std: f64) -> Self {
        assert!(
            seed_std.is_finite() && seed_std > 0.0,
            "the seed standard deviation must be strictly positive and finite"
        );
        self.seed_std = seed_std;

        self
    }
}

/// Spectral embedding of the affinity graph in CSR form. Returns a flat row-major
/// `n * D` matrix whose columns are the leading nontrivial eigenvectors of the
/// normalized graph Laplacian eigenmap, centered and scaled to std
/// `params.seed_std`. See [`SPECTRAL_OVERSAMPLE`] for the connected component
/// limit inherent to the fixed block width.
#[allow(clippy::needless_range_loop)]
pub(crate) fn spectral_embedding<T, const D: usize>(
    p_rows: &[usize],
    p_columns: &[u32],
    p_values: &[T],
    params: SpectralParams,
) -> Vec<T>
where
    T: Float + Default + Sum + AddAssign + SubAssign + MulAssign + DivAssign + Send + Sync,
    Dim<D>: SpectralBlock,
{
    let d_out = D;
    let n = p_rows.len().saturating_sub(1);
    assert!(
        n > 0,
        "the spectral embedding requires affinities to be built"
    );
    let one = T::one();

    // 1. Degrees: row sums of P.
    let degrees: Vec<T> = (0..n)
        .map(|i| p_values[p_rows[i]..p_rows[i + 1]].iter().copied().sum())
        .collect();

    // 2. Inv-sqrt degrees. Isolated nodes get a vanishing weight instead, which
    // decouples them from the solve and sends them to the origin of the seed.
    let floor = T::from(1e-12).unwrap();
    let inv_sqrt_d: Vec<T> = degrees
        .iter()
        .map(|&d| {
            let s = d.sqrt();
            if s > floor { one / s } else { floor }
        })
        .collect();

    // 3. Build (column, scaled weight) pairs sorted by column within each row.
    // Sorting improves spatial locality: consecutive edges access nearby rows of the
    // iterate during the matvec.
    let inv_sqrt_d_ref = &inv_sqrt_d;
    let sorted_rows: Vec<Vec<(u32, T)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut pairs: Vec<(u32, T)> = p_columns[p_rows[i]..p_rows[i + 1]]
                .iter()
                .zip(p_values[p_rows[i]..p_rows[i + 1]].iter())
                .map(|(&col, &val)| (col, val * inv_sqrt_d_ref[col as usize]))
                .collect();
            pairs.sort_unstable_by_key(|&(col, _)| col);
            pairs
        })
        .collect();
    let mut edge_cols: Vec<usize> = Vec::with_capacity(p_columns.len());
    let mut edge_weights: Vec<T> = Vec::with_capacity(p_columns.len());
    for row in &sorted_rows {
        for &(col, weight) in row {
            edge_cols.push(col as usize);
            edge_weights.push(weight);
        }
    }

    // 4. Trivial Perron eigenvector v0 = sqrt(d) / ||sqrt(d)||.
    let sqrt_d: Vec<T> = degrees.iter().map(|d| d.sqrt()).collect();
    let norm_v0 = sqrt_d.iter().map(|v| *v * *v).sum::<T>().sqrt();
    let v0: Vec<T> = if norm_v0 > T::zero() {
        sqrt_d.iter().map(|v| *v / norm_v0).collect()
    } else {
        vec![T::zero(); n]
    };

    // 5. Leading nontrivial eigenvectors, flat row-major n x d_out.
    let mut v = chebyshev_rayleigh_ritz::<T, Dim<D>>(
        n,
        &v0,
        &inv_sqrt_d,
        p_rows,
        &edge_cols,
        &edge_weights,
        params,
    );

    // 6. Back to eigenmap coordinates: u = D^{-1/2} v.
    v.par_chunks_mut(d_out).enumerate().for_each(|(i, row)| {
        for d in 0..d_out {
            row[d] *= inv_sqrt_d[i];
        }
    });

    // 7. Center each column at zero mean.
    let n_t = T::from(n as f64).unwrap();
    let sums = chunked_column_reduce(n, d_out, |range, acc: &mut [T]| {
        for i in range {
            let row = &v[i * d_out..(i + 1) * d_out];
            for d in 0..d_out {
                acc[d] += row[d];
            }
        }
    });
    v.par_chunks_mut(d_out).for_each(|row| {
        for d in 0..d_out {
            row[d] -= sums[d] / n_t;
        }
    });

    // 8. Scale each column independently so all have std params.seed_std.
    let target_std = T::from(params.seed_std).unwrap();
    let var_sums = chunked_column_reduce(n, d_out, |range, acc: &mut [T]| {
        for i in range {
            let row = &v[i * d_out..(i + 1) * d_out];
            for d in 0..d_out {
                acc[d] += row[d] * row[d];
            }
        }
    });
    let scales: Vec<T> = var_sums
        .iter()
        .map(|&var_sum| {
            let std_d = (var_sum / n_t).sqrt();
            if std_d > T::from(1e-30).unwrap() {
                target_std / std_d
            } else {
                T::one()
            }
        })
        .collect();
    v.par_chunks_mut(d_out).for_each(|row| {
        for d in 0..d_out {
            row[d] *= scales[d];
        }
    });

    v
}

/// Estimates the leading nontrivial eigenvectors of the similarity matrix
/// `T = D^{-1/2} P D^{-1/2}` by Chebyshev-filtered subspace iteration on the shifted
/// operator `M = (I + T) / 2` with Rayleigh-Ritz projections, returning them as a
/// flat row-major `n` by dimensionality matrix.
///
/// The shift maps the spectrum of `T` from `[-1, 1]` to `[0, 1]` so the iteration
/// cannot lock onto large negative eigenvalues arising from near-bipartite structure.
/// The Perron vector `v0` (eigenvalue one) is projected out during the
/// orthonormalization after every filter round. Each round applies a degree
/// `params.degree` Chebyshev polynomial that is bounded on the unwanted interval
/// `[0, b]` and grows exponentially beyond it, which resolves the tightly clustered
/// leading eigenvalues in a small fixed matvec budget. The bound `b` starts crude and
/// is refined from the Ritz values of the previous round. The recurrence is evaluated
/// in its scaled form (each term divided by its value at the spectrum edge) so
/// intermediates stay bounded in `f32`.
#[allow(clippy::too_many_arguments)]
fn chebyshev_rayleigh_ritz<T, S>(
    n: usize,
    v0: &[T],
    inv_sqrt_d: &[T],
    p_rows: &[usize],
    edge_cols: &[usize],
    edge_weights: &[T],
    params: SpectralParams,
) -> Vec<T>
where
    T: Float + Default + Sum + AddAssign + SubAssign + DivAssign + Send + Sync,
    S: SpectralBlock,
{
    // The rotation after the round loop reads the Ritz decomposition of the last
    // round, so at least one round must have run. The SpectralParams setters uphold
    // these invariants, the assert guards direct construction within the crate.
    assert!(
        params.rounds >= 1
            && params.degree >= 1
            && params.seed_std.is_finite()
            && params.seed_std > 0.0
    );

    // The block width is fixed at the type level. On graphs with fewer than k
    // independent directions the surplus columns collapse during orthonormalization,
    // are refreshed once, and end up zeroed, so no runtime clamp is needed. The
    // output dimensionality follows from the same width, upheld by the const
    // asserts of the spectral_block macro.
    let k = S::WIDTH;
    let d_out = S::WIDTH - SPECTRAL_OVERSAMPLE;

    let zero = T::zero();
    let one = T::one();
    let two = one + one;

    // Deterministic pseudo random start block.
    let mut v: Vec<T> = (0..n * k)
        .map(|index| splitmix_unit::<T>(index as u64))
        .collect();
    let mut y_prev = vec![zero; n * k];
    let mut y_next = vec![zero; n * k];
    let mut ritz_vectors: Vec<T> = Vec::new();
    let mut ritz_order: Vec<usize> = Vec::new();

    let mut bound = T::from(CHEBYSHEV_INITIAL_BOUND).unwrap();
    for round in 0..params.rounds {
        // Degree params.degree filter in scaled form. With t(x) = (2x - b) / b the
        // iterates are y_j = T_j(t(M)) v / T_j(t(1)) and alpha_j tracks the scaling
        // ratio T_{j-1}(t(1)) / T_j(t(1)).
        let t1 = (two - bound) / bound;
        let mut alpha = one / t1;
        // y_1 = alpha * t(M) v = (2 alpha / b) M v - alpha v.
        y_prev.copy_from_slice(&v);
        matvec_combine::<T, S>(
            &y_prev,
            &y_prev,
            &mut v,
            inv_sqrt_d,
            p_rows,
            edge_cols,
            edge_weights,
            two * alpha / bound,
            -alpha,
            zero,
        );
        for _ in 2..=params.degree {
            // y_next = 2 alpha_next t(M) y - alpha_next alpha y_prev.
            let alpha_next = one / (two * t1 - alpha);
            matvec_combine::<T, S>(
                &v,
                &y_prev,
                &mut y_next,
                inv_sqrt_d,
                p_rows,
                edge_cols,
                edge_weights,
                two * two * alpha_next / bound,
                -two * alpha_next,
                -alpha_next * alpha,
            );
            std::mem::swap(&mut y_prev, &mut v);
            std::mem::swap(&mut v, &mut y_next);
            alpha = alpha_next;
        }

        // Distinct refresh seeds per round, offset past the initial block indices.
        // Widen before multiplying so 32-bit targets cannot wrap into the initial
        // index range.
        let refresh_seed = (round as u64 + 1) * n as u64 * k as u64;
        orthonormalize_block(&mut v, n, k, v0, refresh_seed);

        // Rayleigh-Ritz projection: b = V^T M V.
        matvec_combine::<T, S>(
            &v,
            &v,
            &mut y_next,
            inv_sqrt_d,
            p_rows,
            edge_cols,
            edge_weights,
            one,
            zero,
            zero,
        );
        let mut b = chunked_column_reduce(n, k * k, |range, acc: &mut [T]| {
            for i in range {
                let v_row = &v[i * k..(i + 1) * k];
                let w_row = &y_next[i * k..(i + 1) * k];
                for p in 0..k {
                    for q in 0..k {
                        acc[p * k + q] += v_row[p] * w_row[q];
                    }
                }
            }
        });
        // Symmetrize against rounding, M itself is symmetric.
        for p in 0..k {
            for q in (p + 1)..k {
                let mean = (b[p * k + q] + b[q * k + p]) / two;
                b[p * k + q] = mean;
                b[q * k + p] = mean;
            }
        }

        let s = jacobi_eigen(&mut b, k);

        // Order the Ritz pairs by decreasing Ritz value.
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_unstable_by(|&lhs, &rhs| {
            b[rhs * k + rhs]
                .partial_cmp(&b[lhs * k + lhs])
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        if round + 1 < params.rounds {
            // Refine the filter bound: cut just below the smallest wanted eigenvalue,
            // estimated by the first clearly unwanted Ritz value. Ritz values
            // interlace the spectrum from below, so this never dampens a wanted
            // eigenvector.
            let cut = order[(d_out + 1).min(k - 1)];
            let ritz_cut = b[cut * k + cut];
            bound = ritz_cut
                .max(T::from(0.1).unwrap())
                .min(T::from(0.9995).unwrap());
        }
        ritz_vectors = s;
        ritz_order = order;
    }

    // Rotate the basis onto the Ritz vectors selected in the last round: u = V s.
    let top = &ritz_order[..d_out];
    let mut out = vec![zero; n * d_out];
    out.par_chunks_mut(d_out)
        .enumerate()
        .for_each(|(i, out_row)| {
            let v_row = &v[i * k..(i + 1) * k];
            for (d, &col) in top.iter().enumerate() {
                let mut acc = zero;
                for p in 0..k {
                    acc += v_row[p] * ritz_vectors[p * k + col];
                }
                out_row[d] = acc;
            }
        });

    out
}

/// Deterministic parallel column reduction. Splits the `n` rows into fixed chunks,
/// accumulates each chunk independently into a zeroed buffer of length `width` with
/// `accumulate`, then combines the partial buffers serially in chunk order. The result
/// is therefore independent of rayon's scheduling, which keeps the embedding
/// reproducible run to run.
fn chunked_column_reduce<T, F>(n: usize, width: usize, accumulate: F) -> Vec<T>
where
    T: Float + AddAssign + Send + Sync,
    F: Fn(Range<usize>, &mut [T]) + Send + Sync,
{
    let chunks = n.div_ceil(REDUCTION_ROWS).max(1);
    let partials: Vec<Vec<T>> = (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let start = chunk * REDUCTION_ROWS;
            let end = ((chunk + 1) * REDUCTION_ROWS).min(n);
            let mut acc = vec![T::zero(); width];
            accumulate(start..end, &mut acc);
            acc
        })
        .collect();
    let mut total = vec![T::zero(); width];
    for partial in partials {
        for (t, p) in total.iter_mut().zip(partial) {
            *t += p;
        }
    }
    total
}

/// Euclidean norm of column `c` of the flat row-major block, via a deterministic
/// reduction.
fn column_norm<T>(v: &[T], n: usize, k: usize, c: usize) -> T
where
    T: Float + AddAssign + Send + Sync,
{
    chunked_column_reduce(n, 1, |range, acc: &mut [T]| {
        for i in range {
            let val = v[i * k + c];
            acc[0] += val * val;
        }
    })[0]
        .sqrt()
}

/// Deterministic uniform value in `[-0.5, 0.5)` derived from a splitmix64 hash of the
/// entry index. Gives the subspace iteration a reproducible random start without
/// touching the crate RNG.
#[inline]
fn splitmix_unit<T: Float>(index: u64) -> T {
    let mut z = index.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let unit = (z >> 11) as f64 / (1u64 << 53) as f64;
    T::from(unit - 0.5).unwrap()
}

/// Fused kernel of the solver, one parallel pass over the rows computing
/// `out = mul_mv * M v + mul_v * v + mul_prev * v_prev` for all columns, where
/// `M = (I + D^{-1/2} P D^{-1/2}) / 2` is the shifted similarity operator. The edge
/// weights are expected to be pre-scaled by the column's inverse square root degree.
/// With coefficients `(1, 0, 0)` this is the plain operator application, the other
/// combinations implement the scaled Chebyshev recurrence.
///
/// Works over the associated row type of `S`, whose compile-time width lets the
/// inner loops unroll and keeps the accumulator in registers.
#[allow(clippy::too_many_arguments)]
fn matvec_combine<T, S>(
    v: &[T],
    v_prev: &[T],
    out: &mut [T],
    inv_sqrt_d: &[T],
    p_rows: &[usize],
    edge_cols: &[usize],
    edge_weights: &[T],
    mul_mv: T,
    mul_v: T,
    mul_prev: T,
) where
    T: Float + Default + AddAssign + Send + Sync,
    S: SpectralBlock,
{
    let half = T::from(0.5).unwrap();
    let v_rows = S::as_rows(v);
    let prev_rows = S::as_rows(v_prev);
    let out_rows = S::as_rows_mut(out);
    out_rows
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, out_row)| {
            let mut acc = <S::Row<T>>::default();
            let acc = acc.as_mut();
            for e in p_rows[i]..p_rows[i + 1] {
                let j = edge_cols[e];
                let w = edge_weights[e];
                let v_row = v_rows[j].as_ref();
                for d in 0..S::WIDTH {
                    acc[d] += w * v_row[d];
                }
            }
            let scale = inv_sqrt_d[i];
            let v_row = v_rows[i].as_ref();
            let prev_row = prev_rows[i].as_ref();
            let out_row = out_row.as_mut();
            for d in 0..S::WIDTH {
                let mv = half * (v_row[d] + scale * acc[d]);
                out_row[d] = mul_mv * mv + mul_v * v_row[d] + mul_prev * prev_row[d];
            }
        });
}

/// Orthonormalizes the `k` columns of the flat row-major block with classical
/// Gram-Schmidt against the Perron vector `v0` and all previous columns, batching the
/// dot products into a single deterministic reduction per pass. The projection runs
/// twice per column (CGS2), which restores orthogonality where a single classical
/// Gram-Schmidt pass loses it to cancellation in `f32`.
///
/// A column whose residual norm does not exceed `sqrt(eps)` of its starting norm is
/// pure rounding noise, and normalizing that noise (correlated across columns) would
/// hand Rayleigh-Ritz a rank-deficient basis with meaningless values, which an
/// aggressive filter causes whenever the graph has fewer well separated leading
/// eigenvectors than the block is wide. Such a column is replaced by a fresh
/// deterministic pseudo random direction and reorthogonalized, and zeroed if it
/// collapses again.
fn orthonormalize_block<T>(v: &mut [T], n: usize, k: usize, v0: &[T], refresh_seed: u64)
where
    T: Float + AddAssign + SubAssign + DivAssign + Send + Sync,
{
    let collapse = T::epsilon().sqrt();
    for c in 0..k {
        let mut refreshed = false;
        loop {
            // Norm before the projections, the baseline for collapse detection.
            let norm_pre = column_norm(v, n, k, c);

            for _pass in 0..2 {
                // Batched dot products: acc[p] against previous column p, acc[c]
                // against v0.
                let dots = chunked_column_reduce(n, c + 1, |range, acc: &mut [T]| {
                    for i in range {
                        let row = &v[i * k..(i + 1) * k];
                        for p in 0..c {
                            acc[p] += row[p] * row[c];
                        }
                        acc[c] += v0[i] * row[c];
                    }
                });
                v.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
                    let mut delta = dots[c] * v0[i];
                    for p in 0..c {
                        delta += dots[p] * row[p];
                    }
                    row[c] -= delta;
                });
            }

            let norm = column_norm(v, n, k, c);
            if norm > norm_pre * collapse {
                v.par_chunks_mut(k).for_each(|row| row[c] /= norm);
                break;
            }
            if refreshed {
                v.par_chunks_mut(k).for_each(|row| row[c] = T::zero());
                break;
            }
            refreshed = true;
            v.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
                row[c] = splitmix_unit(refresh_seed.wrapping_add((c * n + i) as u64));
            });
        }
    }
}

/// Cyclic Jacobi eigensolver for the small dense symmetric matrix `b` (`k * k`,
/// row-major). On return `b` is nearly diagonal with the eigenvalues on the diagonal
/// and the returned `k * k` matrix holds the corresponding eigenvectors as columns.
fn jacobi_eigen<T>(b: &mut [T], k: usize) -> Vec<T>
where
    T: Float + Sum,
{
    let zero = T::zero();
    let one = T::one();
    let two = one + one;

    let mut s = vec![zero; k * k];
    for p in 0..k {
        s[p * k + p] = one;
    }
    if k < 2 {
        return s;
    }

    // Converged when the squared off-diagonal mass reaches the rounding noise floor
    // of the diagonal, `(eps * k)^2` accounts for error accumulation across sweeps.
    let eps_k = T::epsilon() * T::from(k as f64).unwrap();
    for _sweep in 0..50 {
        let diag_sq: T = (0..k).map(|p| b[p * k + p] * b[p * k + p]).sum();
        let off: T = (0..k)
            .flat_map(|p| ((p + 1)..k).map(move |q| (p, q)))
            .map(|(p, q)| b[p * k + q] * b[p * k + q])
            .sum();
        if off <= eps_k * eps_k * diag_sq {
            break;
        }
        for p in 0..k {
            for q in (p + 1)..k {
                let apq = b[p * k + q];
                if apq == zero {
                    continue;
                }
                let theta = (b[q * k + q] - b[p * k + p]) / (two * apq);
                let t = if theta >= zero {
                    one / (theta + (theta * theta + one).sqrt())
                } else {
                    one / (theta - (theta * theta + one).sqrt())
                };
                let cos = one / (t * t + one).sqrt();
                let sin = t * cos;
                // b = J^T b J for the Givens rotation J in the (p, q) plane.
                for r in 0..k {
                    let brp = b[r * k + p];
                    let brq = b[r * k + q];
                    b[r * k + p] = cos * brp - sin * brq;
                    b[r * k + q] = sin * brp + cos * brq;
                }
                for col in 0..k {
                    let bpc = b[p * k + col];
                    let bqc = b[q * k + col];
                    b[p * k + col] = cos * bpc - sin * bqc;
                    b[q * k + col] = sin * bpc + cos * bqc;
                }
                // Accumulate the eigenvectors: s = s J.
                for r in 0..k {
                    let srp = s[r * k + p];
                    let srq = s[r * k + q];
                    s[r * k + p] = cos * srp - sin * srq;
                    s[r * k + q] = sin * srp + cos * srq;
                }
            }
        }
    }

    s
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest::proptest! {
        #[test]
        fn splitmix_unit_stays_in_range(index in any::<u64>()) {
            let value: f64 = splitmix_unit(index);
            prop_assert!((-0.5..0.5).contains(&value));
        }

        /// The deterministic chunked reduction must agree with a plain serial sum,
        /// including row counts crossing the REDUCTION_ROWS chunk boundary and
        /// widths exercising the per-column combine.
        #[test]
        fn chunked_reduce_matches_serial_sum(
            (width, values) in (1usize..=4).prop_flat_map(|width| {
                (
                    Just(width),
                    proptest::collection::vec(-1.0f64..1.0, width..9000),
                )
            }),
        ) {
            let n = values.len() / width;
            let totals = chunked_column_reduce(n, width, |range, acc: &mut [f64]| {
                for i in range {
                    for w in 0..width {
                        acc[w] += values[i * width + w];
                    }
                }
            });
            for w in 0..width {
                let serial: f64 = (0..n).map(|i| values[i * width + w]).sum();
                prop_assert!(
                    (totals[w] - serial).abs() <= 1e-9 * (1.0 + serial.abs()),
                    "column {w}: chunked {} vs serial {serial}",
                    totals[w]
                );
            }
        }

        /// For any symmetric matrix, the Jacobi solver must return an orthogonal
        /// eigenvector matrix satisfying `a s = s diag(b)`.
        #[test]
        fn jacobi_diagonalizes_symmetric_matrices(
            (k, entries) in (1usize..=10).prop_flat_map(|k| {
                (Just(k), proptest::collection::vec(-1.0f64..1.0, k * k))
            }),
        ) {
            let mut a = vec![0.0f64; k * k];
            for p in 0..k {
                for q in 0..k {
                    a[p * k + q] = (entries[p * k + q] + entries[q * k + p]) / 2.0;
                }
            }
            let mut b = a.clone();
            let s = jacobi_eigen(&mut b, k);

            // Orthogonality of the eigenvector matrix.
            for p in 0..k {
                for q in 0..k {
                    let dot: f64 = (0..k).map(|r| s[r * k + p] * s[r * k + q]).sum();
                    let expected = if p == q { 1.0 } else { 0.0 };
                    prop_assert!(
                        (dot - expected).abs() < 1e-9,
                        "s^T s deviates at ({p}, {q}): {dot}"
                    );
                }
            }
            // Eigenpair equation against the untouched input matrix.
            for p in 0..k {
                for q in 0..k {
                    let lhs: f64 = (0..k).map(|r| a[p * k + r] * s[r * k + q]).sum();
                    let rhs = s[p * k + q] * b[q * k + q];
                    prop_assert!(
                        (lhs - rhs).abs() < 1e-8,
                        "a s != s diag at ({p}, {q}): {lhs} vs {rhs}"
                    );
                }
            }
        }

        /// After orthonormalization every column is unit norm or exactly zero,
        /// orthogonal to all previous columns and to the Perron vector. The
        /// `collapse` flag forces a numerically rank-one block, exercising the
        /// refresh path that guards against the correlated-noise degeneracy.
        #[test]
        fn orthonormalize_yields_orthonormal_or_zero_columns(
            (n, k, entries, v0_raw, collapse, seed) in (1usize..=30, 1usize..=6)
                .prop_flat_map(|(n, k)| {
                    (
                        Just(n),
                        Just(k),
                        proptest::collection::vec(-1.0f64..1.0, n * k),
                        proptest::collection::vec(-1.0f64..1.0, n),
                        any::<bool>(),
                        any::<u64>(),
                    )
                }),
        ) {
            let mut v = entries;
            if collapse {
                for i in 0..n {
                    for c in 1..k {
                        v[i * k + c] = v[i * k] * (c as f64 + 0.5);
                    }
                }
            }
            let norm0: f64 = v0_raw.iter().map(|x| x * x).sum::<f64>().sqrt();
            let v0: Vec<f64> = if norm0 > 0.0 {
                v0_raw.iter().map(|x| x / norm0).collect()
            } else {
                v0_raw
            };

            orthonormalize_block(&mut v, n, k, &v0, seed);

            for c in 0..k {
                let norm: f64 = (0..n).map(|i| v[i * k + c] * v[i * k + c]).sum::<f64>().sqrt();
                prop_assert!(
                    norm == 0.0 || (norm - 1.0).abs() < 1e-9,
                    "column {c} has norm {norm}, expected 1 or exactly 0"
                );
                if norm > 0.5 {
                    let against_v0: f64 = (0..n).map(|i| v[i * k + c] * v0[i]).sum();
                    prop_assert!(against_v0.abs() < 1e-8, "column {c} not deflated: {against_v0}");
                    for p in 0..c {
                        let dot: f64 = (0..n).map(|i| v[i * k + p] * v[i * k + c]).sum();
                        prop_assert!(dot.abs() < 1e-8, "columns {p} and {c} not orthogonal: {dot}");
                    }
                }
            }
        }

    }
}
