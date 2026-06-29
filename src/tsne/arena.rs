//! Morton (Z-order) linear quadtree in a contiguous arena, the Barnes-Hut tree the optimization
//! loop summarizes repulsive forces over.
//!
//! Every cell lives in one [`Vec<Node>`]. The build quantizes the embedding into per-axis integer
//! coordinates, interleaves them into `u64` Morton codes, sorts a `(code, index)` permutation, and
//! walks the sorted codes breadth first to emit nodes whose children occupy a contiguous arena
//! range reached through `first_child`. A cell stores only the summary the traversal reads (center
//! of mass, count, level), never raw coordinates.
//!
//! Morton quantization is total, so every point maps to exactly one cell and point conservation is
//! automatic, which the build asserts (leaf masses sum to `n`).

use std::ops::AddAssign;

use num_traits::Float;

use rayon::prelude::*;

use crate::PARALLEL_CODE_THRESHOLD;

use super::morton::{Dim, Morton, quantize};

/// `first_child` value marking a leaf, a node with no children in the arena.
const SENTINEL: u32 = u32::MAX;

/// Capacity of the traversal stack. A root-to-leaf path crosses at most `BITS` internal cells (a
/// child's level is strictly greater than its parent's), and each pushes at most `2^D - 1`
/// unvisited siblings, so the live stack never exceeds `BITS * (2^D - 1)` plus one final fan-out.
/// That is 96 at `D = 2` (32 bits) and 147 at `D = 3` (21 bits), so 256 holds both with margin.
/// Kept fixed and stack-allocated so the repulsive traversal never touches the heap.
pub(crate) const STACK_CAP: usize = 256;

/// One cell of the arena. The Barnes-Hut traversal reads `center_of_mass`, `count`, and `level`
/// (which sizes the cell through the per-tree half-width table), and follows `first_child` for the
/// `child_count` children stored contiguously. A leaf is marked by `first_child == SENTINEL`.
///
/// Children are only the non-empty orthants of a cell, stored back to back, rather than a fixed
/// `2^D` slots: empty orthants are never emitted, which keeps the arena lean and the traversal off
/// dead cells. `child_count` is at most `2^D`.
struct Node<T, const D: usize> {
    center_of_mass: [T; D],
    count: u32,
    first_child: u32,
    child_count: u8,
    level: u8,
}

/// A Morton linear quadtree (`D == 2`) or octree (`D == 3`) over the embedding, in one arena.
pub(crate) struct Arena<T, const D: usize> {
    nodes: Vec<Node<T, D>>,
    /// Build scratch: each emitted node's half-open window in `sorted`. Retained with `nodes` so a
    /// rebuild reuses the allocation rather than reallocating it every epoch.
    ranges: Vec<(u32, u32)>,
    /// The `(Morton code, point index)` permutation, retained across rebuilds to reuse its allocation.
    sorted: Vec<(u64, u32)>,
    /// Squared maximum half-width of a cell at each level, indexed by `Node::level`. The theta
    /// acceptance test compares this against `theta^2 * dist`, the squared form of the reference
    /// `max_half_width / sqrt(dist) < theta`, avoiding a square root per visit.
    level_half_width_sq: Vec<T>,
}

/// Per-axis bounding box of the embedding, reduced in parallel.
fn bounding_box<T, const D: usize>(y: &[T]) -> ([T; D], [T; D])
where
    T: Float + Send + Sync,
{
    y.par_chunks_exact(D)
        .with_min_len(PARALLEL_CODE_THRESHOLD)
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

impl<T, const D: usize> Arena<T, D>
where
    T: Float + Send + Sync + AddAssign,
{
    /// Creates an empty arena. The epoch loop holds one of these and rebuilds it each epoch so the
    /// buffers persist and are reused.
    pub(crate) fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            ranges: Vec::new(),
            sorted: Vec::new(),
            level_half_width_sq: Vec::new(),
        }
    }

    /// Builds a fresh arena over `y`, a flat buffer of `n_samples` points of `D` components each.
    /// Convenience for one-shot callers and tests; the epoch loop instead holds one arena and calls
    /// [`Arena::rebuild`] so the buffers persist and are reused.
    ///
    /// # Panics
    ///
    /// If the leaf masses do not sum to `n_samples` (point conservation), which a correct Morton
    /// build always satisfies.
    pub(crate) fn new(y: &[T], n_samples: usize) -> Self
    where
        Dim<D>: Morton<D>,
    {
        let mut arena = Self::empty();
        arena.rebuild(y, n_samples);

        arena
    }

    /// Rebuilds the arena over `y`, a flat buffer of `n_samples` points of `D` components each, in
    /// place, reusing the retained buffers rather than reallocating them each epoch.
    ///
    /// # Panics
    ///
    /// If the leaf masses do not sum to `n_samples` (point conservation), which a correct Morton
    /// build always satisfies.
    pub(crate) fn rebuild(&mut self, y: &[T], n_samples: usize)
    where
        Dim<D>: Morton<D>,
    {
        let bits = <Dim<D> as Morton<D>>::BITS;

        self.nodes.clear();
        self.ranges.clear();
        self.sorted.clear();

        if n_samples == 0 {
            self.level_half_width_sq.clear();
            return;
        }

        // 1. Bounding box and the derived quantization scale.
        let (min, max) = bounding_box::<T, D>(y);
        let mut extent = [T::zero(); D];
        for axis in 0..D {
            extent[axis] = max[axis] - min[axis];
        }
        let scale = T::from(1u64 << bits).unwrap();
        let max_bucket = ((1u64 << bits) - 1) as u32;
        let mut inv_scale = [T::zero(); D];
        for axis in 0..D {
            inv_scale[axis] = if extent[axis] > T::zero() {
                scale / extent[axis]
            } else {
                T::zero()
            };
        }

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
                .with_min_len(PARALLEL_CODE_THRESHOLD)
                .map(|i| {
                    let point = &y[i * D..i * D + D];
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
            count: n_samples as u32,
            first_child: SENTINEL,
            child_count: 0,
            level: bits as u8,
        });
        ranges.push((0, n_samples as u32));

        let mask = (1u64 << D) - 1;
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
            let highest_diff = 63 - xor.leading_zeros();
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

        // 6a. Leaf centers of mass: the mean of the leaf's member points, computed in parallel
        // per leaf. Internal centers come from the bottom-up reduction below. `count` is already
        // set (the range length) for every node, so only the centers remain.
        nodes
            .par_iter_mut()
            .zip(ranges.par_iter())
            .with_min_len(PARALLEL_CODE_THRESHOLD)
            .for_each(|(node, &(start, end))| {
                if node.first_child == SENTINEL {
                    let mut center = [T::zero(); D];
                    for slot in start..end {
                        let index = sorted[slot as usize].1 as usize;
                        let point = &y[index * D..index * D + D];
                        for axis in 0..D {
                            center[axis] += point[axis];
                        }
                    }
                    let inverse = T::from(node.count).unwrap().recip();
                    center
                        .iter_mut()
                        .for_each(|value| *value = *value * inverse);
                    node.center_of_mass = center;
                }
            });

        // 6b. Internal centers of mass, bottom up. Children always have a higher arena index than
        // their parent (breadth-first emission), so a reverse pass sees every child finished, a
        // count-weighted hierarchical average.
        for node in (0..nodes.len()).rev() {
            if nodes[node].first_child == SENTINEL {
                continue;
            }
            let first_child = nodes[node].first_child as usize;
            let child_count = nodes[node].child_count as usize;
            let total = T::from(nodes[node].count).unwrap();
            let mut center = [T::zero(); D];
            for child in &nodes[first_child..first_child + child_count] {
                let weight = T::from(child.count).unwrap();
                center
                    .iter_mut()
                    .zip(child.center_of_mass.iter())
                    .for_each(|(value, &component)| *value += component * weight);
            }
            let inverse = total.recip();
            center
                .iter_mut()
                .for_each(|value| *value = *value * inverse);
            nodes[node].center_of_mass = center;
        }

        // Point conservation: every input point lands in exactly one leaf, so the leaf masses
        // must sum to the input count. Morton quantization guarantees this, but the check guards
        // the breadth-first range bookkeeping. Kept as a release assert because correctness is
        // not relaxed.
        debug_assert_eq!(
            nodes
                .iter()
                .filter(|node| node.first_child == SENTINEL)
                .map(|node| node.count as u64)
                .sum::<u64>() as usize,
            n_samples,
            "arena lost or invented points"
        );

        debug_assert!(check_coms_within_cells::<T, D>(
            nodes, ranges, sorted, &min, &extent, bits
        ));
    }

    /// Number of points the arena holds, the root mass. Used by the build-invariant tests, where
    /// it crosses into a sibling module that cannot reach the private node array.
    #[cfg(test)]
    pub(crate) fn root_count(&self) -> usize {
        self.nodes.first().map_or(0, |node| node.count as usize)
    }

    /// Accumulates the non-edge (repulsive) Barnes-Hut forces on point `index` into
    /// `negative_forces_row` and the normalization term `q_sum`. Iterative traversal over the arena
    /// using `stack` as reusable scratch, a slice of at least [`STACK_CAP`] entries.
    ///
    /// A node is summarized by its center of mass when it is a leaf or passes the theta test, and
    /// otherwise its children are pushed. The leaf holding the query's own point has zero distance
    /// to its center and is skipped, the self-interaction exclusion: a singleton leaf's center is
    /// exactly the query coordinate. `theta_sq` is `theta * theta`.
    pub(crate) fn compute_non_edge_forces(
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
        let query = &y[index * D..index * D + D];

        // Explicit stack with a local top cursor over `stack`, which holds at most `STACK_CAP`
        // entries (see its bound). No heap allocation occurs on this hot path.
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
            let mut magnitude = T::from(node.count).unwrap() * inverse;
            *q_sum += magnitude;
            magnitude = magnitude * inverse;
            for axis in 0..D {
                negative_forces_row[axis] += magnitude * displacement[axis];
            }
        }
    }
}

/// Accumulates the edge (attractive) forces on point `index` from its sparse P matrix neighbors
/// into `positive_forces_row`. A free function over the embedding and the P arrays: the attractive
/// pass reads point coordinates directly and never touches the tree.
pub(crate) fn compute_edge_forces<T, const D: usize>(
    index: usize,
    y: &[T],
    p_rows: &[usize],
    p_columns: &[u32],
    p_values: &[T],
    positive_forces_row: &mut [T],
) where
    T: Float + AddAssign,
{
    let sample = &y[index * D..index * D + D];
    for entry in p_rows[index]..p_rows[index + 1] {
        let other = p_columns[entry] as usize;
        let other_sample = &y[other * D..other * D + D];

        let mut displacement = [T::zero(); D];
        let mut distance = T::zero();
        for axis in 0..D {
            let delta = sample[axis] - other_sample[axis];
            displacement[axis] = delta;
            distance += delta * delta;
        }
        let factor = p_values[entry] / (distance + T::one());
        for axis in 0..D {
            positive_forces_row[axis] += factor * displacement[axis];
        }
    }
}

/// Whether every node's center of mass lies inside its Morton cell. Used only at build time in
/// debug builds. Derives each cell from a representative member code, the node level, and the
/// bounding box, so it needs the still-live sorted codes and ranges rather than per-node storage.
fn check_coms_within_cells<T, const D: usize>(
    nodes: &[Node<T, D>],
    ranges: &[(u32, u32)],
    sorted: &[(u64, u32)],
    min: &[T; D],
    extent: &[T; D],
    bits: u32,
) -> bool
where
    T: Float,
    Dim<D>: Morton<D>,
{
    let slack_fraction = T::from(1e-3).unwrap();
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
        let arena = Arena::<f32, 2>::new(&data, N);
        assert_eq!(arena.root_count(), N);
    }

    #[test]
    fn build_conserves_points_3d() {
        const N: usize = 1_500;
        let data = lcg_cloud(N, 3, 23);
        let arena = Arena::<f32, 3>::new(&data, N);
        assert_eq!(arena.root_count(), N);
    }

    /// The root center of mass is the mean of all points, regardless of tree structure.
    #[test]
    fn root_center_of_mass_equals_the_mean() {
        const N: usize = 1_000;
        let data = lcg_cloud(N, 2, 5);
        let arena = Arena::<f32, 2>::new(&data, N);
        let expected = mean::<2>(&data, N);
        let root = &arena.nodes[0];
        assert!((root.center_of_mass[0] - expected[0]).abs() < 1e-3);
        assert!((root.center_of_mass[1] - expected[1]).abs() < 1e-3);
    }

    /// Points closer than one grid cell collapse to a single leaf with the summed mass, the
    /// duplicate-handling path. Every point still appears exactly once (conservation).
    #[test]
    fn duplicate_points_collapse_to_one_leaf() {
        const N: usize = 500;
        let data = vec![3.5f32; N * 2];
        let arena = Arena::<f32, 2>::new(&data, N);
        assert_eq!(arena.root_count(), N);
        // All identical: the root itself is the single collapsed leaf.
        assert_eq!(arena.nodes.len(), 1);
        assert_eq!(arena.nodes[0].first_child, SENTINEL);
        assert_eq!(arena.nodes[0].count as usize, N);
    }

    /// A single point builds a one-node arena whose center of mass is that point.
    #[test]
    fn single_point_builds_a_leaf_root() {
        let data = [2.0f32, -1.0];
        let arena = Arena::<f32, 2>::new(&data, 1);
        assert_eq!(arena.root_count(), 1);
        assert_eq!(arena.nodes.len(), 1);
        assert_eq!(arena.nodes[0].center_of_mass, [2.0, -1.0]);
    }

    /// An empty input builds an empty arena and the force pass is a no-op.
    #[test]
    fn empty_input_builds_empty_arena() {
        let arena = Arena::<f32, 2>::new(&[], 0);
        assert_eq!(arena.root_count(), 0);
        let mut forces = [0.0f32; 2];
        let mut q_sum = 0.0f32;
        let mut stack = [0u32; STACK_CAP];
        arena.compute_non_edge_forces(0, 0.25, &[], &mut forces, &mut q_sum, &mut stack);
        assert_eq!(forces, [0.0, 0.0]);
        assert_eq!(q_sum, 0.0);
    }
}
