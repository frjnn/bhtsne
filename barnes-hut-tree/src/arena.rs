//! Morton (Z-order) linear quadtree in a contiguous arena, the Barnes-Hut tree the optimization
//! loop summarizes repulsive forces over.
//!
//! Every cell lives in one [`Vec<Node>`]. The build quantizes the embedding into per-axis integer
//! coordinates, interleaves them into Morton codes, sorts a `(code, index)` permutation, and walks
//! the sorted codes breadth first to emit nodes whose children occupy a contiguous arena range
//! reached through `first_child`. A cell stores only the summary the traversal reads (center of
//! mass, count, level), never raw coordinates.
//!
//! Morton quantization is total, so every point maps to exactly one cell and point conservation is
//! automatic, which the build asserts (leaf masses sum to `n`).

use std::{array, ops::AddAssign};

use num_traits::Float;

use rayon::prelude::*;

use crate::{
    ForceKernel,
    morton::{Dim, Morton, MortonWord, quantize},
};

/// Minimum sample count before parallel rayon iterations use `with_min_len` to avoid
/// overhead on small workloads.
const ARENA_MIN_CHUNK: usize = 4096;

/// `first_child` value marking a leaf, a node with no children in the arena.
const SENTINEL: u32 = u32::MAX;

/// Slack fraction.
const SLACK_FRACTION: f64 = 0.03;

/// One cell of the arena. The Barnes-Hut traversal reads `center_of_mass`, `count`, and `level`
/// (which sizes the cell through the per-tree half-width table), and follows `first_child` for the
/// `child_count` children stored contiguously. A leaf is marked by `first_child == SENTINEL`.
///
/// Children are only the non-empty orthants of a cell, stored back to back, rather than a fixed
/// `2^D` slots: empty orthants are never emitted, which keeps the arena lean and the traversal off
/// dead cells. `child_count` is at most `2^D`.
#[derive(Debug)]
struct Node<T, const D: usize> {
    /// Mass-weighted center of mass of the points the cell contains.
    center_of_mass: [T; D],
    /// Summed mass of the points the cell contains. `T::from(count)` under
    /// `rebuild_uniform`, the sum of caller-supplied per-point masses under `rebuild`.
    mass: T,
    /// Number of points the cell contains, guarded by the point-conservation invariant.
    count: u32,
    /// Arena index of the first child, or [`SENTINEL`] for a leaf.
    first_child: u32,
    /// Number of contiguous children starting at `first_child`.
    child_count: u8,
    /// Tree level, indexing the per-tree squared half-width table.
    level: u8,
}

/// A Morton linear tree over the embedding, in one arena, for `D` in `{2, 3, 4, 5, 6, 7}`.
/// `W` is the Morton code word type: `u64` for `D <= 4` and `u128` for `D >= 5`.
#[derive(Debug)]
pub struct BarnesHutTree<T, W, const D: usize>
where
    W: MortonWord,
{
    nodes: Vec<Node<T, D>>,
    /// Build scratch: each emitted node's half-open window in `sorted`. Retained with `nodes` so a
    /// rebuild reuses the allocation rather than reallocating it every epoch.
    ranges: Vec<(u32, u32)>,
    /// The `(Morton code, point index)` permutation, retained across rebuilds to reuse its allocation.
    sorted: Vec<(W, u32)>,
    /// Squared maximum half-width of a cell at each level, indexed by `Node::level`. The theta
    /// acceptance test compares this against `theta^2 * dist`, the squared form of the reference
    /// `max_half_width / sqrt(dist) < theta`, avoiding a square root per visit.
    level_half_width_sq: Vec<T>,
}

/// Per-axis bounding box of the embedding, reduced in parallel.
fn bounding_box<T, const D: usize>(y_chunks: &[[T; D]]) -> ([T; D], [T; D])
where
    T: Float + Send + Sync,
{
    y_chunks
        .par_iter()
        .with_min_len(ARENA_MIN_CHUNK)
        .fold(
            || ([T::max_value(); D], [-T::max_value(); D]),
            |(mut min, mut max), point| {
                for axis in 0..D {
                    min[axis] = min[axis].min(point[axis]);
                    max[axis] = max[axis].max(point[axis]);
                }

                (min, max)
            },
        )
        .reduce(
            || ([T::max_value(); D], [-T::max_value(); D]),
            |(mut min_a, mut max_a), (min_b, max_b)| {
                for axis in 0..D {
                    min_a[axis] = min_a[axis].min(min_b[axis]);
                    max_a[axis] = max_a[axis].max(max_b[axis]);
                }

                (min_a, max_a)
            },
        )
}

impl<T, W, const D: usize> BarnesHutTree<T, W, D>
where
    T: Float + Send + Sync + AddAssign,
    W: MortonWord,
{
    /// Creates an empty arena. The epoch loop holds one of these and rebuilds it each epoch so the
    /// buffers persist and are reused.
    pub fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            ranges: Vec::new(),
            sorted: Vec::new(),
            level_half_width_sq: Vec::new(),
        }
    }

    /// Returns the number of points the arena holds (the root node's mass).
    /// Zero when the arena is empty.
    pub fn root_count(&self) -> usize {
        self.nodes.first().map_or(0, |node| node.count as usize)
    }

    /// Builds a fresh arena over `points`, a flat `masses.len() * D` buffer, weighted by
    /// `masses`. Every mass must be strictly positive: a cell of zero total mass has no
    /// well-defined center of mass.
    ///
    /// # Panics
    ///
    /// If `points.len() != masses.len() * D`.
    pub fn new(points: &[T], masses: &[T]) -> Self
    where
        Dim<D>: Morton<D, Word = W>,
    {
        let mut tree = Self::empty();
        tree.rebuild(points, masses);

        tree
    }

    /// Builds a fresh arena over `points` with unit mass per point. Internal cell mass then
    /// equals `T::from(count)`, matching t-SNE's count-weighted arena.
    ///
    /// # Panics
    ///
    /// If `points.len() % D != 0`.
    pub fn new_uniform(points: &[T]) -> Self
    where
        Dim<D>: Morton<D, Word = W>,
    {
        let mut tree = Self::empty();
        tree.rebuild_uniform(points);

        tree
    }

    /// Rebuilds over `points` weighted by `masses`, in place, reusing the retained buffers.
    /// See [`BarnesHutTree::new`] for the mass contract.
    ///
    /// # Panics
    ///
    /// If `points.len() != masses.len() * D`.
    pub fn rebuild(&mut self, points: &[T], masses: &[T])
    where
        Dim<D>: Morton<D, Word = W>,
    {
        assert_eq!(
            points.len(),
            masses.len() * D,
            "points must hold masses.len() points of D components each"
        );
        self.rebuild_impl(points, masses.len(), |index| masses[index]);
    }

    /// Rebuilds over `points` with unit mass per point, in place, reusing the retained
    /// buffers.
    ///
    /// # Panics
    ///
    /// If `points.len() % D != 0`.
    pub fn rebuild_uniform(&mut self, points: &[T])
    where
        Dim<D>: Morton<D, Word = W>,
    {
        assert_eq!(
            points.len() % D,
            0,
            "points must hold a whole number of D-component points"
        );
        self.rebuild_impl(points, points.len() / D, |_| T::one());
    }

    /// Shared rebuild body. `mass_of` returns the mass of point `index`, the only difference
    /// between the mass-weighted and uniform entry points.
    fn rebuild_impl<F>(&mut self, points: &[T], n_samples: usize, mass_of: F)
    where
        Dim<D>: Morton<D, Word = W>,
        F: Fn(usize) -> T + Sync,
    {
        let bits = <Dim<D> as Morton<D>>::BITS;

        self.nodes.clear();
        self.ranges.clear();
        self.sorted.clear();

        if n_samples == 0 {
            self.level_half_width_sq.clear();
            return;
        }

        let (y_chunks, _) = points.as_chunks::<D>();

        // 1. Bounding box and the derived quantization scale.
        let (min, max) = bounding_box::<T, D>(y_chunks);
        let extent: [T; D] = array::from_fn(|axis| max[axis] - min[axis]);
        let scale = T::from(1u64 << bits).unwrap();
        let max_bucket = ((1u64 << bits) - 1) as u32;
        let inv_scale: [T; D] = array::from_fn(|axis| {
            if extent[axis] > T::zero() {
                scale / extent[axis]
            } else {
                T::zero()
            }
        });

        // The squared half-width per level for the theta test: cell full width per axis at level L
        // is extent / 2^L, so the maximum half-width is max(extent) / 2^(L+1).
        let max_extent = extent.iter().copied().fold(T::zero(), T::max);
        self.level_half_width_sq.clear();
        for level in 0..=bits {
            let half_width = max_extent / T::from(1u64 << (level + 1)).unwrap();
            self.level_half_width_sq.push(half_width * half_width);
        }

        // 2-4. Quantize, encode, and sort a (code, index) permutation into Z-order. `sorted` is
        // refilled from the cleared buffer, reusing its capacity across epochs.
        self.sorted.par_extend(
            (0..n_samples)
                .into_par_iter()
                .with_min_len(ARENA_MIN_CHUNK)
                .map(|i| {
                    let point = &y_chunks[i];
                    let code = <Dim<D> as Morton<D>>::encode(quantize::<T, D>(
                        point, &min, &inv_scale, max_bucket,
                    ));

                    (code, i as u32)
                }),
        );
        self.sorted.par_sort_unstable_by_key(|&(code, _)| code);

        let sorted = &self.sorted;
        let nodes = &mut self.nodes;
        let ranges = &mut self.ranges;

        // 5. Breadth-first emission: each node's children (the non-empty orthant groups at the
        // node's tightest enclosing level) occupy a contiguous arena range. `ranges` carries each
        // node's window in `sorted`. Every internal node has at least two children (single-child
        // chains are skipped by the tightest-enclosing-level jump), so an arena over `n` points
        // holds at most `2n - 1` nodes; reserving that up front lets the sequential emission push
        // without reallocating (a no-op once the buffers have grown on the first epoch).
        nodes.reserve(2 * n_samples - 1);
        ranges.reserve(2 * n_samples - 1);
        nodes.push(Node {
            center_of_mass: [T::zero(); D],
            mass: T::zero(),
            count: n_samples as u32,
            first_child: SENTINEL,
            child_count: 0,
            level: bits as u8,
        });
        ranges.push((0, n_samples as u32));

        let mask = W::d_bit_mask(D as u32);
        let mut node = 0usize;
        while node < nodes.len() {
            let (start, end) = ranges[node];
            // A single point, or a cell of points that share a full code (closer than one grid
            // cell, the duplicate case), is a leaf. Sorted codes make the all-equal test O(1).
            if end - start <= 1 || sorted[start as usize].0 == sorted[(end - 1) as usize].0 {
                node += 1;
                continue;
            }

            // Tightest enclosing level: the highest bit at which the range's extreme codes differ
            // sits in the D-bit group that first splits the range, so the node skips straight to
            // that level rather than emitting single-child chains.
            let xor = sorted[start as usize].0 ^ sorted[(end - 1) as usize].0;
            let highest_diff = xor.msb_position();
            let level = (bits - 1) - highest_diff / D as u32;
            let shift = D as u32 * (bits - 1 - level);
            nodes[node].level = level as u8;

            let first_child = nodes.len() as u32;
            let mut child_count: u8 = 0;
            let mut child_start = start;
            while child_start < end {
                let group = (sorted[child_start as usize].0 >> shift) & mask;
                let mut child_end = child_start + 1;
                while child_end < end && (sorted[child_end as usize].0 >> shift) & mask == group {
                    child_end += 1;
                }
                nodes.push(Node {
                    center_of_mass: [T::zero(); D],
                    mass: T::zero(),
                    count: (child_end - child_start),
                    first_child: SENTINEL,
                    child_count: 0,
                    level: bits as u8,
                });
                ranges.push((child_start, child_end));
                child_count += 1;
                child_start = child_end;
            }
            // A D-bit group has at most 2^D distinct values, so a cell never exceeds its orthants.
            debug_assert!(child_count as usize <= <Dim<D> as Morton<D>>::CHILDREN);
            nodes[node].first_child = first_child;
            nodes[node].child_count = child_count;
            node += 1;
        }

        // 6a. Leaf mass and mass-weighted center of mass, computed in parallel per leaf.
        // Internal centers come from the bottom-up reduction below. `count` is already set (the
        // range length) for every node, so only the summary remains.
        let mass_ref = &mass_of;
        nodes
            .par_iter_mut()
            .zip(ranges.par_iter())
            .with_min_len(ARENA_MIN_CHUNK)
            .filter(|(node, _)| node.first_child == SENTINEL)
            .for_each(|(node, &(start, end))| {
                let mut center = [T::zero(); D];
                let mut total = T::zero();
                for slot in start..end {
                    let index = sorted[slot as usize].1 as usize;
                    let weight = mass_ref(index);
                    total += weight;
                    let point = &y_chunks[index];
                    center
                        .iter_mut()
                        .zip(point.iter())
                        .for_each(|(ci, pi)| *ci += weight * *pi);
                }
                let inverse = total.recip();
                center
                    .iter_mut()
                    .for_each(|value| *value = *value * inverse);
                node.mass = total;
                node.center_of_mass = center;
            });

        // 6b. Internal mass and mass-weighted center of mass, bottom up. Children always have a
        // higher arena index than their parent (breadth-first emission), so a reverse pass sees
        // every child finished.
        for node in (0..nodes.len()).rev() {
            if nodes[node].first_child == SENTINEL {
                continue;
            }
            let first_child = nodes[node].first_child as usize;
            let child_count = nodes[node].child_count as usize;
            let mut center = [T::zero(); D];
            let mut total = T::zero();
            for child in &nodes[first_child..first_child + child_count] {
                total += child.mass;
                center
                    .iter_mut()
                    .zip(child.center_of_mass.iter())
                    .for_each(|(value, &component)| *value += component * child.mass);
            }
            let inverse = total.recip();
            center
                .iter_mut()
                .for_each(|value| *value = *value * inverse);
            nodes[node].mass = total;
            nodes[node].center_of_mass = center;
        }

        // Point conservation: every input point lands in exactly one leaf, so the leaf counts
        // must sum to the input count. Morton quantization guarantees this in exact arithmetic,
        // and the check guards the breadth-first range bookkeeping in debug builds.
        debug_assert_eq!(
            nodes
                .iter()
                .filter(|node| node.first_child == SENTINEL)
                .map(|node| node.count as u64)
                .sum::<u64>() as usize,
            n_samples,
            "arena lost or invented points"
        );

        debug_assert!(check_coms_within_cells::<T, W, D>(
            nodes, ranges, sorted, &min, &extent, bits
        ));
    }

    /// Accumulates the non-edge (repulsive) Barnes-Hut forces on point `index` into
    /// `negative_forces_row` and the normalization term `q_sum`. Iterative traversal over the arena
    /// using `stack` as reusable scratch, a mutable slice of [`crate::morton::Morton::Stack`].
    ///
    /// A node is summarized by its center of mass when it is a leaf or passes the theta test, and
    /// otherwise its children are pushed. The leaf holding the query's own point has zero distance
    /// exactly the query coordinate. `theta_sq` is `theta * theta`.
    ///
    /// # Panics
    ///
    /// If `index >= y.len() / D` (out-of-bounds sample index) or if `stack` is shorter than
    /// the [`Morton::Stack`] for dimension `D`.
    pub fn compute_non_edge_forces(
        &self,
        index: usize,
        theta_sq: T,
        y: &[T],
        negative_forces_row: &mut [T; D],
        q_sum: &mut T,
        stack: &mut [u32],
    ) {
        if self.nodes.is_empty() {
            return;
        }
        let (y_chunks, _) = y.as_chunks::<D>();
        let query = &y_chunks[index];
        // Explicit stack with a local top cursor over `stack`, sized per dimensionality by
        // `Morton::Stack`. No heap allocation occurs on this hot path.
        let mut top = 0usize;
        stack[top] = 0;
        top += 1;
        while top > 0 {
            top -= 1;
            let node = &self.nodes[stack[top] as usize];

            let mut displacement = [T::zero(); D];
            let mut distance = T::zero();
            for axis in 0..D {
                let delta = query[axis] - node.center_of_mass[axis];
                displacement[axis] = delta;
                distance += delta * delta;
            }

            if node.first_child == SENTINEL {
                // Skip the query's own leaf (zero displacement), excluding the self-interaction.
                if distance == T::zero() {
                    continue;
                }
            } else if self.level_half_width_sq[node.level as usize] >= theta_sq * distance {
                // The cell subtends too large an angle: descend into its children.
                for child in 0..node.child_count as u32 {
                    stack[top] = node.first_child + child;
                    top += 1;
                }
                continue;
            }

            // Summarize the cell by its center of mass.
            let inverse = (T::one() + distance).recip();
            let mut magnitude = node.mass * inverse;
            *q_sum += magnitude;
            magnitude = magnitude * inverse;
            for axis in 0..D {
                negative_forces_row[axis] += magnitude * displacement[axis];
            }
        }
    }

    /// Accumulates the Barnes-Hut force on `query` (`D` components) into `force`, applying
    /// `kernel` to every summarised cell. Iterative traversal using `stack` as scratch, at
    /// least [`Morton::Stack`] entries wide.
    ///
    /// A cell is summarised when it is a leaf or passes the theta test, otherwise its
    /// children are pushed. Zero-distance cells (the query's own leaf) are skipped.
    /// `theta == 0` descends to leaves and is exact.
    ///
    /// # Panics
    ///
    /// If `query.len() < D`, or if `stack` is shorter than [`Morton::Stack`] for `D`.
    pub fn accumulate_forces<K>(
        &self,
        query: &[T],
        theta: T,
        kernel: &K,
        force: &mut [T; D],
        stack: &mut [u32],
    ) where
        K: ForceKernel<T, D>,
    {
        if self.nodes.is_empty() {
            return;
        }
        assert!(query.len() >= D, "query must hold at least D components");

        let theta_sq = theta * theta;
        let mut top = 0usize;
        stack[top] = 0;
        top += 1;
        while top > 0 {
            top -= 1;
            let node = &self.nodes[stack[top] as usize];

            let mut displacement = [T::zero(); D];
            let mut distance_sq = T::zero();
            for axis in 0..D {
                let delta = query[axis] - node.center_of_mass[axis];
                displacement[axis] = delta;
                distance_sq += delta * delta;
            }

            let is_leaf = node.first_child == SENTINEL;
            if is_leaf {
                // Skip the query's own leaf (zero displacement), excluding the self-interaction
                // and guarding the kernel from a zero-distance singularity.
                if distance_sq <= T::zero() {
                    continue;
                }
            } else if self.level_half_width_sq[node.level as usize] >= theta_sq * distance_sq {
                for child in 0..node.child_count as u32 {
                    stack[top] = node.first_child + child;
                    top += 1;
                }
                continue;
            }

            kernel.accumulate(&displacement, distance_sq, node.mass, is_leaf, force);
        }
    }

    /// Parallel driver over [`BarnesHutTree::accumulate_forces`]: force on every
    /// `D`-component row of `queries` into the matching row of `out`. Each row runs on its
    /// own rayon task with a stack-allocated traversal stack. `out` is overwritten per row.
    ///
    /// # Panics
    ///
    /// If `queries.len() % D != 0`, or `out.len() != queries.len()`.
    pub fn accumulate_all<K>(&self, queries: &[T], theta: T, kernel: &K, out: &mut [T])
    where
        K: ForceKernel<T, D> + Sync,
        Dim<D>: Morton<D, Word = W>,
    {
        assert_eq!(
            queries.len() % D,
            0,
            "queries must hold a whole number of D-component points"
        );
        assert_eq!(queries.len(), out.len(), "out must match queries in length");

        out.par_chunks_mut(D)
            .zip(queries.par_chunks(D))
            .with_min_len(ARENA_MIN_CHUNK)
            .for_each_init(<Dim<D> as Morton<D>>::empty_stack, |stack, (row, query)| {
                let mut force = [T::zero(); D];
                self.accumulate_forces(query, theta, kernel, &mut force, stack.as_mut());
                row.copy_from_slice(&force);
            });
    }
}

/// Whether every node's center of mass lies inside its Morton cell. Used only at build time in
/// debug builds. Derives each cell from a representative member code, the node level, and the
/// bounding box, so it needs the still-live sorted codes and ranges rather than per-node storage.
fn check_coms_within_cells<T, W, const D: usize>(
    nodes: &[Node<T, D>],
    ranges: &[(u32, u32)],
    sorted: &[(W, u32)],
    min: &[T; D],
    extent: &[T; D],
    bits: u32,
) -> bool
where
    T: Float,
    W: MortonWord,
    Dim<D>: Morton<D, Word = W>,
{
    let slack_fraction = T::from(SLACK_FRACTION).unwrap();

    nodes.iter().zip(ranges.iter()).all(|(node, &(start, _))| {
        let level = node.level as u32;
        let coords = <Dim<D> as Morton<D>>::decode(sorted[start as usize].0);
        let cells = T::from(1u64 << level).unwrap();

        (0..D).all(|axis| {
            let width = extent[axis] / cells;
            let cell_index = T::from((coords[axis] as u64) >> (bits - level)).unwrap();
            let low = min[axis] + cell_index * width;
            let high = low + width;
            let magnitude = low.abs().max(high.abs());
            let slack = slack_fraction * (extent[axis] + magnitude) + T::min_positive_value();

            node.center_of_mass[axis] >= low - slack && node.center_of_mass[axis] <= high + slack
        })
    })
}

#[cfg(test)]
impl<T, W, const D: usize> BarnesHutTree<T, W, D>
where
    W: MortonWord,
{
    /// Root node's center of mass as a slice.
    pub(crate) fn root_center_of_mass(&self) -> &[T] {
        self.nodes.first().map_or(&[], |node| &node.center_of_mass)
    }

    /// Number of nodes in the arena.
    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// First child of the root (SENTINEL if leaf).
    pub(crate) fn root_first_child(&self) -> u32 {
        self.nodes.first().map_or(SENTINEL, |node| node.first_child)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic linear congruential generator, so the tests need no RNG dependency.
    fn lcg_cloud(n: usize, dim: usize, mut state: u64) -> Vec<f32> {
        let mut data = Vec::with_capacity(n * dim);
        for _ in 0..n * dim {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            data.push((state >> 40) as f32 / (1u64 << 24) as f32);
        }

        data
    }

    /// The mean of every point, the value the root center of mass must equal.
    fn mean<const D: usize>(y: &[f32], n: usize) -> [f32; D] {
        let mut sum = [0.0f32; D];
        for point in y.chunks_exact(D) {
            for axis in 0..D {
                sum[axis] += point[axis];
            }
        }
        sum.iter_mut().for_each(|value| *value /= n as f32);

        sum
    }

    #[test]
    fn build_conserves_points_and_keeps_coms_in_cells_2d() {
        const N: usize = 2_000;
        let mut data = lcg_cloud(N, 2, 17);
        // Offset far from the origin so a center of mass dragged toward it escapes its cell.
        for value in data.iter_mut() {
            *value += 100.0;
        }
        let arena = BarnesHutTree::<f32, u64, 2>::new_uniform(&data);
        assert_eq!(arena.root_count(), N);
    }

    #[test]
    fn build_conserves_points_3d() {
        const N: usize = 1_500;
        let data = lcg_cloud(N, 3, 23);
        let arena = BarnesHutTree::<f32, u64, 3>::new_uniform(&data);
        assert_eq!(arena.root_count(), N);
    }

    /// The root center of mass is the mean of all points, regardless of tree structure.
    #[test]
    fn root_center_of_mass_equals_the_mean() {
        const N: usize = 1_000;
        let data = lcg_cloud(N, 2, 5);
        let arena = BarnesHutTree::<f32, u64, 2>::new_uniform(&data);
        let expected = mean::<2>(&data, N);
        let root = arena.root_center_of_mass();
        assert!((root[0] - expected[0]).abs() < 1e-3);
        assert!((root[1] - expected[1]).abs() < 1e-3);
    }

    /// Points closer than one grid cell collapse to a single leaf with the summed mass, the
    /// duplicate-handling path. Every point still appears exactly once (conservation).
    #[test]
    fn duplicate_points_collapse_to_one_leaf() {
        const N: usize = 500;
        let data = vec![3.5f32; N * 2];
        let arena = BarnesHutTree::<f32, u64, 2>::new_uniform(&data);
        assert_eq!(arena.root_count(), N);
        // All identical: the root itself is the single collapsed leaf.
        assert_eq!(arena.node_count(), 1);
        assert_eq!(arena.root_first_child(), SENTINEL);
    }

    /// A single point builds a one-node arena whose center of mass is that point.
    #[test]
    fn single_point_builds_a_leaf_root() {
        let data = [2.0f32, -1.0];
        let arena = BarnesHutTree::<f32, u64, 2>::new_uniform(&data);
        assert_eq!(arena.root_count(), 1);
        assert_eq!(arena.node_count(), 1);
        let com = arena.root_center_of_mass();
        assert_eq!(com, &[2.0, -1.0]);
    }

    /// An empty input builds an empty arena and the force pass is a no-op.
    #[test]
    fn empty_input_builds_empty_arena() {
        let arena = BarnesHutTree::<f32, u64, 2>::new_uniform(&[]);
        assert_eq!(arena.root_count(), 0);
        let mut forces = [0.0f32; 2];
        let mut q_sum = 0.0f32;
        let mut stack = <Dim<2> as Morton<2>>::empty_stack();
        arena.compute_non_edge_forces(0, 0.25, &[], &mut forces, &mut q_sum, &mut stack);
        assert_eq!(forces, [0.0, 0.0]);
        assert_eq!(q_sum, 0.0);
    }

    /// Smoke test for the 4D arena path: the build conserves points and produces a tree.
    #[test]
    fn build_conserves_points_4d() {
        const N: usize = 1_000;
        let data = lcg_cloud(N, 4, 31);
        let arena = BarnesHutTree::<f32, u64, 4>::new_uniform(&data);
        assert_eq!(arena.root_count(), N);
    }

    /// Non-trivial force computation: with two well-separated points the repulsive force on
    /// each must point away from the other, and `q_sum` must be positive.
    #[test]
    fn repulsive_forces_point_away_from_each_other() {
        // Two points far apart on the x-axis.
        let data = [-5.0f32, 0.0, 5.0, 0.0];
        let arena = BarnesHutTree::<f32, u64, 2>::new_uniform(&data);
        assert_eq!(arena.root_count(), 2);

        let theta_sq = 0.25;
        let mut stack = <Dim<2> as Morton<2>>::empty_stack();

        // Force on point 0 (at -5, 0) should point left (negative x).
        let mut forces0 = [0.0f32; 2];
        let mut q_sum0 = 0.0f32;
        arena.compute_non_edge_forces(0, theta_sq, &data, &mut forces0, &mut q_sum0, &mut stack);
        assert!(
            forces0[0] < 0.0,
            "force on point 0 should point away from point 1"
        );
        assert!(q_sum0 > 0.0, "q_sum must be positive");

        // Force on point 1 (at 5, 0) should point right (positive x).
        let mut forces1 = [0.0f32; 2];
        let mut q_sum1 = 0.0f32;
        arena.compute_non_edge_forces(1, theta_sq, &data, &mut forces1, &mut q_sum1, &mut stack);
        assert!(
            forces1[0] > 0.0,
            "force on point 1 should point away from point 0"
        );
        assert!(q_sum1 > 0.0, "q_sum must be positive");
    }

    /// Force computation with multiple points spread across quadrants: the net force on a
    /// quadrant center must point roughly toward the quadrant diagonal.
    #[test]
    fn repulsive_forces_with_multiple_points() {
        // Four points at the corners of a square.
        let data = [-1.0f32, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
        let arena = BarnesHutTree::<f32, u64, 2>::new_uniform(&data);
        assert_eq!(arena.root_count(), 4);

        let theta_sq = 0.25;
        let mut stack = <Dim<2> as Morton<2>>::empty_stack();

        // Force on point 0 (at -1, -1) should point roughly down-left.
        let mut forces = [0.0f32; 2];
        let mut q_sum = 0.0f32;
        arena.compute_non_edge_forces(0, theta_sq, &data, &mut forces, &mut q_sum, &mut stack);
        assert!(forces[0] < 0.0, "x-force should be negative");
        assert!(forces[1] < 0.0, "y-force should be negative");
        assert!(q_sum > 0.0, "q_sum must be positive");
    }

    /// Smoke test for the 5D arena path using the u128 Morton word type.
    #[test]
    fn build_conserves_points_5d() {
        const N: usize = 500;
        let data = lcg_cloud(N, 5, 37);
        let arena = BarnesHutTree::<f32, u128, 5>::new_uniform(&data);
        assert_eq!(arena.root_count(), N);
    }

    /// Smoke test for the 6D arena path using the u128 Morton word type.
    #[test]
    fn build_conserves_points_6d() {
        const N: usize = 400;
        let data = lcg_cloud(N, 6, 39);
        let arena = BarnesHutTree::<f32, u128, 6>::new_uniform(&data);
        assert_eq!(arena.root_count(), N);
    }

    /// Smoke test for the 7D arena path using the u128 Morton word type.
    #[test]
    fn build_conserves_points_7d() {
        const N: usize = 300;
        let data = lcg_cloud(N, 7, 41);
        let arena = BarnesHutTree::<f32, u128, 7>::new_uniform(&data);
        assert_eq!(arena.root_count(), N);
    }

    /// Rebuild reuses buffers: the second build over a different cloud must produce
    /// the correct point count and a different center of mass.
    #[test]
    fn rebuild_reuses_buffers() {
        let mut arena = BarnesHutTree::<f32, u64, 2>::empty();
        let data1 = lcg_cloud(500, 2, 10);
        arena.rebuild_uniform(&data1);
        assert_eq!(arena.root_count(), 500);
        let com1: Vec<f32> = arena.root_center_of_mass().to_vec();

        let data2 = lcg_cloud(500, 2, 99);
        arena.rebuild_uniform(&data2);
        assert_eq!(arena.root_count(), 500);
        let com2: Vec<f32> = arena.root_center_of_mass().to_vec();

        // Different seeds produce different clouds, so centers of mass must differ.
        assert!(com1[0] != com2[0] || com1[1] != com2[1]);
    }

    /// Force computation in 3D: with three points forming a triangle the repulsive
    /// force on each must point away from the triangle centroid.
    #[test]
    fn repulsive_forces_in_3d() {
        // Three points forming an equilateral triangle in the xy-plane.
        let data = [0.0f32, 0.0, 0.0, 10.0, 0.0, 0.0, 5.0, 8.66, 0.0];
        let arena = BarnesHutTree::<f32, u64, 3>::new_uniform(&data);
        assert_eq!(arena.root_count(), 3);

        let theta_sq = 0.25;
        let mut stack = <Dim<3> as Morton<3>>::empty_stack();

        // Centroid is at (5, 2.887, 0). Force on point 0 (0,0,0) should point away.
        let mut forces = [0.0f32; 3];
        let mut q_sum = 0.0f32;
        arena.compute_non_edge_forces(0, theta_sq, &data, &mut forces, &mut q_sum, &mut stack);
        // The force should have a component pointing away from the centroid.
        assert!(q_sum > 0.0, "q_sum must be positive");
        // The force magnitude should be nonzero (points are well-separated).
        let mag = (forces[0] * forces[0] + forces[1] * forces[1] + forces[2] * forces[2]).sqrt();
        assert!(mag > 1e-6, "force magnitude should be nonzero");
    }
}
