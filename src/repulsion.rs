//! Per-epoch repulsion strategies for the approximate fitting loop.
//!
//! The attractive force, the fused gradient-descent update, and the zero-mean
//! recentering around it are identical across approximate fits; only the way the
//! repulsive forces and the `Q` normalizer `Z` are approximated differs. That
//! difference is captured by the [`Repulsion`] trait, implemented here by the
//! Barnes-Hut and FIt-SNE strategies.

use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use num_traits::{Float, cast::AsPrimitive};

use rustfft::FftNum;

use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use crate::tsne;

/// Per-epoch repulsion strategy: produces the attractive and repulsive force rows
/// for the current embedding and the reciprocal of the `Q` normalizer `Z`, and
/// evaluates the matching KL divergence. The generic `tSNE::run_loop` owns one of
/// these; the attractive force, the fused gradient-descent update, and the zero-mean
/// recentering around it are identical across strategies, so only this trait differs
/// between the Barnes-Hut and FIt-SNE paths.
pub(crate) trait Repulsion<T, const D: usize> {
    /// Fills the attractive (`positive`) and repulsive (`negative`) force rows for
    /// the current embedding `y` and returns the reciprocal of the `Q` normalizer `Z`.
    fn step(
        &mut self,
        y: &[T],
        p_rows: &[usize],
        p_columns: &[u32],
        p_values: &[T],
        positive: &mut [T],
        negative: &mut [T],
    ) -> T;

    /// Evaluates the KL divergence (the t-SNE loss) of the current embedding under
    /// this strategy's repulsion approximation.
    fn error(
        &self,
        p_rows: &[usize],
        p_columns: &[u32],
        p_values: &[T],
        y: &[T],
        n_samples: usize,
    ) -> T;
}

/// Barnes-Hut repulsion. A Morton arena summarizes the repulsive forces, computed in
/// the same fused parallel pass as the attractive forces and the per-sample `Q`
/// contributions, the latter reduced into `Z` afterwards.
pub(crate) struct BarnesHutRepulsion<T, W, const D: usize>
where
    W: tsne::morton::MortonWord,
{
    /// Morton arena, rebuilt over the embedding each epoch so its buffers persist.
    arena: tsne::arena::Arena<T, W, D>,
    /// Per-sample contribution to the `Q` normalizer, reduced after the force pass.
    /// Sized to the sample count on the first epoch and reused thereafter.
    q_sums: Vec<T>,
    /// Approximation accuracy, retained for the KL-divergence evaluation.
    theta: T,
    /// `theta` squared, the form the cell-acceptance test compares against.
    theta_sq: T,
}

impl<T, W, const D: usize> BarnesHutRepulsion<T, W, D>
where
    T: Float + Send + Sync + AddAssign,
    W: tsne::morton::MortonWord,
{
    pub(crate) fn new(theta: T) -> Self {
        Self {
            arena: tsne::arena::Arena::empty(),
            q_sums: Vec::new(),
            theta,
            theta_sq: theta * theta,
        }
    }
}

impl<T, W, const D: usize> Repulsion<T, D> for BarnesHutRepulsion<T, W, D>
where
    T: Float + Send + Sync + Sum + AddAssign + SubAssign + MulAssign + DivAssign,
    W: tsne::morton::MortonWord,
    tsne::morton::Dim<D>: tsne::morton::Morton<D, Word = W>,
{
    fn step(
        &mut self,
        y: &[T],
        p_rows: &[usize],
        p_columns: &[u32],
        p_values: &[T],
        positive: &mut [T],
        negative: &mut [T],
    ) -> T {
        let n_samples = y.len() / D;
        // Rebuild the Morton arena over the current embedding.
        self.arena.rebuild(y, n_samples);
        self.q_sums.resize(n_samples, T::zero());

        // Barnes-Hut forces in parallel: a positive and negative chunk per sample, plus its
        // `q_sum` term. The attractive pass reads coordinates directly, the repulsive pass walks
        // the arena with a fixed-size stack. The stack lives in the `for_each_init` state, a
        // stack-allocated array so the traversal never touches the heap: rayon calls the init
        // once per work split, so a heap stack here would allocate per split and collapse the
        // scaling at high thread counts.
        let theta_sq = self.theta_sq;
        let arena = &self.arena;
        positive
            .par_chunks_mut(D)
            .zip(negative.par_chunks_mut(D))
            .zip(self.q_sums.par_iter_mut())
            .enumerate()
            .for_each_init(
                || {
                    (
                        [T::zero(); D],
                        [T::zero(); D],
                        <tsne::morton::Dim<D> as tsne::morton::Morton<D>>::empty_stack(),
                    )
                },
                |(edge_row, nonedge_row, stack),
                 (index, ((positive_out, negative_out), q_sum_out))| {
                    // Write each output row once to avoid false sharing.
                    *edge_row = [T::zero(); D];
                    *nonedge_row = [T::zero(); D];
                    let mut q_sum = T::zero();
                    tsne::arena::compute_edge_forces::<T, D>(
                        index, y, p_rows, p_columns, p_values, edge_row,
                    );
                    arena.compute_non_edge_forces(
                        index,
                        theta_sq,
                        y,
                        nonedge_row,
                        &mut q_sum,
                        stack.as_mut(),
                    );
                    positive_out.copy_from_slice(&edge_row[..]);
                    negative_out.copy_from_slice(&nonedge_row[..]);
                    *q_sum_out = q_sum;
                },
            );

        // Sequential q_sum: barrier-free and negligible against the forces. The reciprocal lets
        // the fused update multiply instead of dividing per value.
        let q_sum: T = self.q_sums.iter().copied().sum();

        q_sum.recip()
    }

    fn error(
        &self,
        p_rows: &[usize],
        p_columns: &[u32],
        p_values: &[T],
        y: &[T],
        n_samples: usize,
    ) -> T {
        tsne::evaluate_error_approximately::<T, D>(
            p_rows, p_columns, p_values, y, n_samples, self.theta,
        )
    }
}

/// FIt-SNE repulsion. The attractive forces come from the sparse graph exactly as in
/// the Barnes-Hut path, while the repulsive forces and the `Q` normalizer come from
/// FFT-accelerated interpolation on an equal-spaced grid.
pub(crate) struct InterpolatedRepulsion<T: FftNum, const D: usize> {
    /// Interpolation workspace, grown in place as the embedding spreads.
    interpolant: tsne::interpolation::Interpolant<T, D>,
}

impl<T, const D: usize> InterpolatedRepulsion<T, D>
where
    T: Send + Sync + Float + FftNum + AsPrimitive<usize> + Sum,
{
    pub(crate) fn new() -> Self {
        Self {
            interpolant: tsne::interpolation::Interpolant::new(),
        }
    }
}

impl<T, const D: usize> Repulsion<T, D> for InterpolatedRepulsion<T, D>
where
    T: Float
        + FftNum
        + Send
        + Sync
        + Sum
        + AsPrimitive<usize>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign,
{
    fn step(
        &mut self,
        y: &[T],
        p_rows: &[usize],
        p_columns: &[u32],
        p_values: &[T],
        positive: &mut [T],
        negative: &mut [T],
    ) -> T {
        let n_samples = y.len() / D;
        // Attractive (positive) forces from the sparse graph, parallel per sample.
        // Identical to the Barnes-Hut edge forces, hence the shared routine.
        positive
            .par_chunks_mut(D)
            .enumerate()
            .for_each(|(index, row)| {
                row.fill(T::zero());
                tsne::arena::compute_edge_forces::<T, D>(
                    index, y, p_rows, p_columns, p_values, row,
                );
            });

        // Repulsive (negative) forces and the Q normalizer Z via FFT interpolation.
        let mut z = T::zero();
        self.interpolant
            .repulsive_forces(y, n_samples, negative, &mut z);

        z.recip()
    }

    fn error(
        &self,
        p_rows: &[usize],
        p_columns: &[u32],
        p_values: &[T],
        y: &[T],
        n_samples: usize,
    ) -> T {
        tsne::evaluate_error_interpolated::<T, D>(p_rows, p_columns, p_values, y, n_samples)
    }
}
