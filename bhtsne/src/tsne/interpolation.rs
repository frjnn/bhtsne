//! FFT-accelerated, interpolation-based repulsive forces: the FIt-SNE method of
//! Linderman et al., *Fast interpolation-based t-SNE for improved visualization of
//! single-cell RNA-seq data* (Nature Methods, 2019).
use std::iter::Sum;

use num_traits::{AsPrimitive, Float};

use rayon::prelude::*;

use rustfft::{FftNum, num_complex::Complex};

use super::fft::FftGrid;

/// Sealed marker restricting the FFT interpolation path to the embedding
/// dimensionalities whose grid stays tractable, `D in {1, 2}`.
#[doc(hidden)]
pub trait FftDim {}

impl FftDim for barnes_hut_tree::Dim<1> {}
impl FftDim for barnes_hut_tree::Dim<2> {}

/// Interpolation nodes per box along each axis (`p` in the paper). Three nodes give
/// the quadratic interpolation FIt-SNE and openTSNE use by default.
const INTERPOLATION_POINTS: usize = 3;

/// Lower bound on the number of boxes per axis, matching openTSNE's
/// `min_num_intervals`. Keeps the grid fine enough for tightly packed early
/// embeddings where the spread is still tiny, while staying small so the transform
/// is cheap before the embedding expands.
const MIN_BOXES_PER_AXIS: usize = 10;

/// Embedding extent covered by a single box (openTSNE's `ints_in_interval`): the
/// box count grows one per unit of spread once past [`MIN_BOXES_PER_AXIS`], so the
/// grid resolution tracks the embedding as it expands.
const EXTENT_PER_BOX: f64 = 1.0;

/// Upper bound on the number of boxes per axis, matching openTSNE's cap. A lone
/// outlier can stretch the bounding box arbitrarily; without a ceiling the
/// `fft_len^D` grid would grow without bound, so the resolution is capped here (the
/// embedding then simply spans coarser boxes) rather than risking a runaway
/// allocation.
const MAX_BOXES_PER_AXIS: usize = 1000;

/// Reusable workspace for the interpolation-based repulsive force, sized to the
/// current grid and grown in place as the embedding spreads across epochs.
///
/// Owns every per-epoch buffer so a fit allocates at most a handful of times (only
/// when the grid grows), keeping the hot loop allocation-free in steady state.
pub(crate) struct Interpolant<T: FftNum, const D: usize> {
    /// Interpolation nodes per axis, `boxes_per_axis * INTERPOLATION_POINTS`.
    nodes_per_axis: usize,
    /// FFT length per axis: a small-prime-factored size at least `2 * nodes_per_axis - 1`.
    fft_len: usize,
    /// FFT plan for the current [`Self::fft_len`], rebuilt only when it changes.
    plan: FftGrid<T>,
    /// Real `fft_len^D` grid: charge spread target before the forward transform and
    /// the recovered potentials after the inverse. Reused for the kernel build.
    real_grid: Vec<T>,
    /// Kernel half-spectrum, pre-scaled by the inverse-transform normalization so the
    /// convolution needs no extra scaling pass.
    kernel_spec: Vec<Complex<T>>,
    /// Per-term convolution scratch, one half-spectrum.
    work_spec: Vec<Complex<T>>,
    /// Interpolated node potentials gathered back to points, `n_samples * (D + 2)`.
    potentials: Vec<T>,
    /// Per-point, per-axis box index, `n_samples * D`.
    box_index: Vec<usize>,
    /// Per-point, per-axis Lagrange weights, `n_samples * D * INTERPOLATION_POINTS`.
    weights: Vec<T>,
}

impl<T, const D: usize> Interpolant<T, D>
where
    T: Send + Sync + Float + FftNum + AsPrimitive<usize> + Sum,
{
    /// Number of charge/potential terms, `1` density term plus `D` first moments
    /// plus `1` squared-norm term.
    const TERMS: usize = const { D + 2 };

    /// Creates an empty interpolant.
    pub(crate) fn new() -> Self {
        Self {
            nodes_per_axis: 0,
            fft_len: 1,
            plan: FftGrid::new(1),
            real_grid: Vec::new(),
            kernel_spec: Vec::new(),
            work_spec: Vec::new(),
            potentials: Vec::new(),
            box_index: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Computes the repulsive forces and the `Z` normalizer for the current
    /// embedding `y` (`n_samples * D` row-major coordinates).
    ///
    /// On return `negative_forces[i * D + d]` holds `F_rep,i^d` and `*z` holds the
    /// global normalizer.
    pub(crate) fn repulsive_forces(
        &mut self,
        y: &[T],
        n_samples: usize,
        negative_forces: &mut [T],
        z: &mut T,
    ) {
        if n_samples == 0 {
            *z = T::zero();
            return;
        }

        // Square domain over the joint coordinate range, as in FIt-SNE: a common
        // min/max across axes keeps the grid spacing equal on every axis, so the
        // grid kernel is isotropic.
        let (min, max) = coordinate_bounds(y);
        let extent = (max - min).max(T::min_positive_value());

        let boxes_per_axis = box_count(extent);
        let nodes_per_axis = boxes_per_axis * INTERPOLATION_POINTS;

        // Linear convolution of two `nodes_per_axis`-long sequences needs at least
        // `2 * nodes_per_axis - 1` samples to avoid wrap-around aliasing; round up to
        // a small-prime-factored length so the mixed-radix transform stays fast.
        let fft_len = next_smooth(2 * nodes_per_axis - 1);
        self.resize(nodes_per_axis, fft_len, n_samples);

        let box_width = extent / T::from(boxes_per_axis).unwrap();
        let spacing = box_width / T::from(INTERPOLATION_POINTS).unwrap();

        self.locate_points(y, min, box_width, boxes_per_axis);
        self.build_kernel_spectrum(spacing);
        self.convolve_terms(y);
        self.reconstruct(y, n_samples, negative_forces, z);
    }

    /// Sizes the workspace exactly to the current grid and sample count, rebuilding
    /// the FFT plan only when the transform length changes. Sizing the FFT buffers
    /// exactly keeps every transform's length equal to `fft_len^D`; the [`Vec`]s keep
    /// their capacity when they shrink, so a steady-state embedding stops
    /// reallocating.
    fn resize(&mut self, nodes_per_axis: usize, fft_len: usize, n_samples: usize) {
        self.nodes_per_axis = nodes_per_axis;
        if fft_len != self.fft_len {
            self.fft_len = fft_len;
            self.plan = FftGrid::new(fft_len);
        }
        let grid = fft_len.pow(D as u32);
        let spectrum = self.plan.spectrum_len::<D>();
        let zero = Complex::new(T::zero(), T::zero());
        self.real_grid.resize(grid, T::zero());
        self.kernel_spec.resize(spectrum, zero);
        self.work_spec.resize(spectrum, zero);
        self.potentials.resize(n_samples * Self::TERMS, T::zero());
        self.box_index.resize(n_samples * D, 0);
        self.weights
            .resize(n_samples * D * INTERPOLATION_POINTS, T::zero());
    }

    /// Assigns every point to a box per axis and computes its Lagrange
    /// interpolation weights against that box's nodes. Independent per point, so the
    /// pass fans out across the thread pool.
    fn locate_points(&mut self, y: &[T], min: T, box_width: T, boxes_per_axis: usize) {
        let inv_box_width = box_width.recip();
        let last_box = boxes_per_axis - 1;
        let basis = LagrangeBasis::<INTERPOLATION_POINTS, T>::new();
        let (box_index_chunks, _) = self.box_index.as_chunks_mut::<D>();
        let (y_chunks, _) = y.as_chunks::<D>();
        box_index_chunks
            .par_iter_mut()
            .zip(self.weights.par_chunks_mut(D * INTERPOLATION_POINTS))
            .zip(y_chunks.par_iter())
            .for_each(|((box_row, weight_row), point)| {
                for axis in 0..D {
                    // Locate the box and the point's relative position within it.
                    let offset = (point[axis] - min) * inv_box_width;
                    let box_id = offset.floor();
                    let box_id = if box_id > T::zero() {
                        box_id.to_usize().unwrap_or(last_box).min(last_box)
                    } else {
                        0
                    };
                    box_row[axis] = box_id;
                    let local = (offset - T::from(box_id).unwrap())
                        .max(T::zero())
                        .min(T::one());
                    basis.weights(local, &mut weight_row[axis * INTERPOLATION_POINTS..]);
                }
            });
    }

    /// Fills [`Self::kernel_spec`] with the real FFT of the squared Cauchy kernel
    /// sampled over the grid's signed lags, ready to multiply against each charge
    /// spectrum. It is pre-scaled by `1 / fft_len^D` so the unnormalized inverse
    /// transform yields the true circular convolution with no extra pass.
    ///
    /// Lags wrap circularly: index `c` past the midpoint stands for the negative lag
    /// `c - fft_len`. Entries beyond the valid `±(nodes_per_axis - 1)` lag window are
    /// never read by the zero-padded charge grids, so the whole padded array can be
    /// filled unconditionally.
    fn build_kernel_spectrum(&mut self, spacing: T) {
        let fft_len = self.fft_len;
        let one = T::one();
        self.real_grid
            .par_iter_mut()
            .enumerate()
            .for_each(|(flat, value)| {
                // Squared distance between grid nodes `spacing * lag` apart.
                let mut sq = T::zero();
                let mut rem = flat;
                for _ in 0..D {
                    let coord = rem % fft_len;
                    rem /= fft_len;
                    let lag = signed_lag(coord, fft_len);
                    let displacement = spacing * T::from(lag).unwrap();
                    sq = sq + displacement * displacement;
                }
                let cauchy = (one + sq).recip();
                *value = cauchy * cauchy;
            });
        self.plan
            .forward::<D>(&mut self.real_grid, &mut self.kernel_spec);

        // Fold the inverse-transform normalization into the kernel spectrum.
        let inv_total = T::from(fft_len.pow(D as u32)).unwrap().recip();
        self.kernel_spec
            .par_iter_mut()
            .for_each(|value| *value = *value * inv_total);
    }

    /// Runs the `[spread -> convolve -> gather]` pipeline for every charge term, leaving
    /// the interpolated potentials in [`Self::potentials`] as `n_samples * TERMS`.
    fn convolve_terms(&mut self, y: &[T]) {
        for term in 0..Self::TERMS {
            self.spread_charges(y, term);
            self.plan
                .forward::<D>(&mut self.real_grid, &mut self.work_spec);
            self.multiply_by_kernel();
            self.plan
                .inverse::<D>(&mut self.work_spec, &mut self.real_grid);
            self.gather_potentials(term);
        }
    }

    /// Spreads one charge term onto the real grid (S2N), zeroing it first.
    ///
    /// Sequential because points sharing a box scatter into overlapping nodes; the
    /// pass is `O(n_samples * INTERPOLATION_POINTS^D)` and cheap against the
    /// transforms and the attractive forces.
    fn spread_charges(&mut self, y: &[T], term: usize) {
        let fft_len = self.fft_len;
        let combinations = INTERPOLATION_POINTS.pow(D as u32);

        // Disjoint field borrows: scatter into the grid while reading per-point boxes
        // and weights. The scatter target is a computed node index, so the grid stays
        // indexed; the per-point rows are walked as aligned chunk iterators.
        let real_grid = &mut self.real_grid;
        let (y_chunks, _) = y.as_chunks::<D>();
        let (box_chunks, _) = self.box_index.as_chunks::<D>();
        real_grid.iter_mut().for_each(|v| *v = T::zero());
        for ((point, box_row), weight_row) in y_chunks
            .iter()
            .zip(box_chunks.iter())
            .zip(self.weights.chunks_exact(D * INTERPOLATION_POINTS))
        {
            let charge = charge_value::<T, D>(point, term);
            for combo in 0..combinations {
                let (flat, weight) = node_of_combo::<T, D>(combo, box_row, weight_row, fft_len);
                real_grid[flat] = real_grid[flat] + weight * charge;
            }
        }
    }

    /// Multiplies the charge spectrum in place by the kernel spectrum, the
    /// frequency-domain form of the grid convolution.
    fn multiply_by_kernel(&mut self) {
        self.work_spec
            .par_iter_mut()
            .zip(self.kernel_spec.par_iter())
            .for_each(|(w, &k)| *w = *w * k);
    }

    /// Interpolates the node potentials back to the points (N2S) for one term.
    /// Read-only on the grid, so points are gathered in parallel.
    fn gather_potentials(&mut self, term: usize) {
        let fft_len = self.fft_len;
        let combinations = INTERPOLATION_POINTS.pow(D as u32);
        let real_grid = &self.real_grid;
        let (box_chunks, _) = self.box_index.as_chunks::<D>();
        let weights = &self.weights;

        self.potentials
            .par_chunks_mut(Self::TERMS)
            .zip(box_chunks.par_iter())
            .zip(weights.par_chunks(D * INTERPOLATION_POINTS))
            .for_each(|((point_terms, box_row), weight_row)| {
                let mut phi = T::zero();
                for combo in 0..combinations {
                    let (flat, weight) = node_of_combo::<T, D>(combo, box_row, weight_row, fft_len);
                    phi = phi + weight * real_grid[flat];
                }
                point_terms[term] = phi;
            });
    }

    /// Reconstructs the repulsive forces and the global `Z` from the interpolated
    /// potentials, per the `D + 2` term identity in the module docs.
    fn reconstruct(&self, y: &[T], n_samples: usize, negative_forces: &mut [T], z: &mut T) {
        let two = T::from(2.0).unwrap();
        let (negative_forces_chunks, _) = negative_forces.as_chunks_mut::<D>();
        let (y_chunks, _) = y.as_chunks::<D>();
        let z_sum: T = negative_forces_chunks
            .par_iter_mut()
            .zip(y_chunks.par_iter())
            .zip(self.potentials.par_chunks(Self::TERMS))
            .map(|((force_row, point), phi)| {
                let density = phi[0];
                let squared_norm_potential = phi[D + 1];

                let mut norm_sq = T::zero();
                let mut cross = T::zero();
                for d in 0..D {
                    let first_moment = phi[1 + d];
                    force_row[d] = point[d] * density - first_moment;
                    norm_sq = norm_sq + point[d] * point[d];
                    cross = cross + point[d] * first_moment;
                }

                (T::one() + norm_sq) * density - two * cross + squared_norm_potential
            })
            .sum();

        // Remove the N analytic self-interactions (`B_ii (1 + 0) = 1` each).
        *z = z_sum - T::from(n_samples).unwrap();
    }
}

/// Joint minimum and maximum over every coordinate of `y`, defining the square grid
/// domain. `y` is assumed non-empty.
fn coordinate_bounds<T: Float + Send + Sync>(y: &[T]) -> (T, T) {
    y.par_iter()
        .copied()
        .fold(
            || (T::infinity(), T::neg_infinity()),
            |(lo, hi), v| (lo.min(v), hi.max(v)),
        )
        .reduce(
            || (T::infinity(), T::neg_infinity()),
            |(lo_a, hi_a), (lo_b, hi_b)| (lo_a.min(lo_b), hi_a.max(hi_b)),
        )
}

/// Number of boxes per axis for a given embedding extent: one per
/// [`EXTENT_PER_BOX`] units of spread, clamped to
/// `[MIN_BOXES_PER_AXIS, MAX_BOXES_PER_AXIS]`.
fn box_count<T: Float + AsPrimitive<usize>>(extent: T) -> usize {
    let scaled = (extent / T::from(EXTENT_PER_BOX).unwrap()).ceil();
    let scaled = if scaled > T::zero() { scaled.as_() } else { 0 };

    scaled.clamp(MIN_BOXES_PER_AXIS, MAX_BOXES_PER_AXIS)
}

/// Smallest 7-smooth integer (only prime factors 2, 3, 5, 7) at least `n`, the size
/// class `rustfft`'s mixed-radix kernels transform fastest, mirroring the
/// FFTW-optimal sizing openTSNE rounds to. Smooth numbers are dense, so the linear
/// scan settles within a handful of steps; `n` here is at most a few thousand.
fn next_smooth(n: usize) -> usize {
    let mut candidate = n.max(1);
    loop {
        let mut value = candidate;
        for prime in [2, 3, 5, 7] {
            while value.is_multiple_of(prime) {
                value /= prime;
            }
        }
        if value == 1 {
            return candidate;
        }
        candidate += 1;
    }
}

/// Maps a circular grid index to its signed lag: indices past the midpoint stand
/// for negative lags `coord - fft_len`.
#[inline]
const fn signed_lag(coord: usize, fft_len: usize) -> isize {
    if coord <= fft_len / 2 {
        coord as isize
    } else {
        coord as isize - fft_len as isize
    }
}

/// Node positions and reciprocal denominator products for the `I`
/// Lagrange basis. Both depend only on the equispaced nodes at `(k + 0.5) / p`, not on
/// the sample, so they are built once per [`Interpolant::repulsive_forces`] and
/// reused across every per-point, per-axis weight evaluation.
struct LagrangeBasis<const I: usize, T> {
    nodes: [T; I],
    inv_denom: [T; I],
}

impl<const I: usize, T: Float> LagrangeBasis<I, T> {
    fn new() -> Self {
        let half = T::from(0.5).unwrap();
        let p_recip = T::from(I).unwrap().recip();

        let mut nodes = [T::zero(); I];
        for (k, node_k) in nodes.iter_mut().enumerate() {
            *node_k = (T::from(k).unwrap() + half) * p_recip;
        }

        let mut inv_denom = [T::one(); I];
        for (k, inv_k) in inv_denom.iter_mut().enumerate() {
            let mut denom = T::one();
            for (m, &node_m) in nodes.iter().enumerate() {
                if m != k {
                    denom = denom * (nodes[k] - node_m);
                }
            }
            *inv_k = denom.recip();
        }

        Self { nodes, inv_denom }
    }

    /// Writes the Lagrange basis weights interpolating a value at `local in [0, 1]`
    /// into the first `I` elements of `out`.
    ///
    /// `out` must have at least `I` elements; the first that many are
    /// overwritten.
    #[inline]
    fn weights(&self, local: T, out: &mut [T]) {
        for (k, out_k) in out[..I].iter_mut().enumerate() {
            let mut num = T::one();
            for (m, &node_m) in self.nodes.iter().enumerate() {
                if m != k {
                    num = num * (local - node_m);
                }
            }
            *out_k = num * self.inv_denom[k];
        }
    }
}

/// Charge of sample `point` for a given term: `1` for the density term `0`, the
/// coordinate `y^d` for first-moment term `1 + d`, and `||y||^2` for the last term.
#[inline]
fn charge_value<T: Float + Sum, const D: usize>(point: &[T; D], term: usize) -> T {
    if term == 0 {
        return T::one();
    }

    if (1..=D).contains(&term) {
        return point[term - 1];
    }

    point.iter().map(|&c| c * c).sum()
}

/// Resolves one of the `INTERPOLATION_POINTS^D` node combinations of a point's box
/// into a flat grid index and the tensor-product Lagrange weight. The `combo`
/// counter is read as a base-`INTERPOLATION_POINTS` numeral, one digit per axis.
#[inline]
fn node_of_combo<T: Float, const D: usize>(
    combo: usize,
    box_row: &[usize; D],
    weight_row: &[T],
    fft_len: usize,
) -> (usize, T) {
    let p = INTERPOLATION_POINTS;
    let mut rem = combo;
    let mut flat = 0usize;
    let mut weight = T::one();
    for axis in 0..D {
        let k = rem % p;
        rem /= p;
        let node = box_row[axis] * p + k;
        flat = flat * fft_len + node;
        weight = weight * weight_row[axis * p + k];
    }

    (flat, weight)
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;

    /// Brute-force `O(n^2)` repulsive forces and `Z`, the reference the interpolation
    /// is checked against. `force[i*D+d] = sum_j (1 + ||y_i-y_j||^2)^-2 (y_i^d - y_j^d)`
    /// and `Z = sum_{i!=j} (1 + ||y_i-y_j||^2)^-1`.
    fn brute_force<const D: usize>(y: &[f64], n: usize) -> (Vec<f64>, f64) {
        let mut forces = vec![0.0; n * D];
        let mut z = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let mut dist_sq = 0.0;
                let mut delta = [0.0; D];
                for d in 0..D {
                    delta[d] = y[i * D + d] - y[j * D + d];
                    dist_sq += delta[d] * delta[d];
                }
                let cauchy = 1.0 / (1.0 + dist_sq);
                z += cauchy;
                let sq = cauchy * cauchy;
                for d in 0..D {
                    forces[i * D + d] += sq * delta[d];
                }
            }
        }

        (forces, z)
    }

    /// The relative error of the interpolated forces and `Z` against the brute-force
    /// reference must stay within the interpolation tolerance for a well-spread cloud.
    #[test]
    fn repulsive_forces_match_brute_force_2d() {
        const D: usize = 2;
        let n = 800;
        // Seeded so the accuracy bound below is checked against a fixed cloud.
        let mut rng = StdRng::seed_from_u64(0xBADC0DE);
        // Spread over several units so the grid boxes are populated, the regime the
        // interpolation is built for.
        let y: Vec<f64> = (0..n * D).map(|_| rng.random_range(-5.0..5.0)).collect();

        let (ref_forces, ref_z) = brute_force::<D>(&y, n);

        let mut interpolant = Interpolant::<f64, D>::new();
        let mut forces = vec![0.0; n * D];
        let mut z = 0.0;
        interpolant.repulsive_forces(&y, n, &mut forces, &mut z);

        assert!(
            (z - ref_z).abs() / ref_z < 1e-3,
            "Z relative error too large: got {z}, expected {ref_z}"
        );

        // Compare forces by relative L2 norm of the difference.
        let mut diff_sq = 0.0;
        let mut ref_sq = 0.0;
        for k in 0..n * D {
            diff_sq += (forces[k] - ref_forces[k]).powi(2);
            ref_sq += ref_forces[k].powi(2);
        }
        // The grid uses openTSNE's coarse default (min 10 boxes per axis), so a
        // couple of percent L2 error is expected and harmless for the optimization.
        let rel = (diff_sq / ref_sq).sqrt();
        assert!(rel < 3e-2, "force relative L2 error too large: {rel}");
    }

    /// The same property holds for a one-dimensional embedding.
    #[test]
    fn repulsive_forces_match_brute_force_1d() {
        const D: usize = 1;
        let n = 500;
        // Seeded so the accuracy bound below is checked against a fixed cloud.
        let mut rng = StdRng::seed_from_u64(0xFEED_F00D);
        let y: Vec<f64> = (0..n * D).map(|_| rng.random_range(-4.0..4.0)).collect();

        let (ref_forces, ref_z) = brute_force::<D>(&y, n);

        let mut interpolant = Interpolant::<f64, D>::new();
        let mut forces = vec![0.0; n * D];
        let mut z = 0.0;
        interpolant.repulsive_forces(&y, n, &mut forces, &mut z);

        assert!((z - ref_z).abs() / ref_z < 1e-3, "Z error: {z} vs {ref_z}");
        let mut diff_sq = 0.0;
        let mut ref_sq = 0.0;
        for k in 0..n * D {
            diff_sq += (forces[k] - ref_forces[k]).powi(2);
            ref_sq += ref_forces[k].powi(2);
        }
        // Same coarse-grid tolerance as the 2-D case: a couple of percent L2 error is
        // expected and harmless for the optimization.
        assert!((diff_sq / ref_sq).sqrt() < 3e-2);
    }

    /// The Lagrange basis is a partition of unity (weights sum to one) and is
    /// cardinal at the nodes (weight one at its own node, zero at the others).
    #[test]
    fn lagrange_weights_partition_of_unity() {
        let basis = LagrangeBasis::<INTERPOLATION_POINTS, f64>::new();
        for step in 0..=10 {
            let local = step as f64 / 10.0;
            let mut w = [0.0; INTERPOLATION_POINTS];
            basis.weights(local, &mut w);
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "weights sum to {sum} at {local}");
        }
        // Cardinality: at node s_j = (j + 0.5) / p the basis is one at j and zero
        // at every other node.
        for j in 0..INTERPOLATION_POINTS {
            let node_j = (j as f64 + 0.5) / INTERPOLATION_POINTS as f64;
            let mut w = [0.0; INTERPOLATION_POINTS];
            basis.weights(node_j, &mut w);
            for (k, &w_k) in w.iter().enumerate() {
                let expected = if k == j { 1.0 } else { 0.0 };
                assert!((w_k - expected).abs() < 1e-12, "w[{k}] = {w_k} at node {j}");
            }
        }
    }

    /// An empty embedding yields a zero normalizer and writes no forces.
    #[test]
    fn empty_input_is_a_no_op() {
        let mut interpolant = Interpolant::<f64, 2>::new();
        let mut forces: Vec<f64> = Vec::new();
        let mut z = 1.0;
        interpolant.repulsive_forces(&[], 0, &mut forces, &mut z);
        assert_eq!(z, 0.0);
    }
}
