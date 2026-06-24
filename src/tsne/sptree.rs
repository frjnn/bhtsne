use std::{
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign},
};

use num_traits::{Float, NumCast};

use rayon::prelude::*;

/// Subtrees of at least this many points build their children in parallel. Smaller ones stay
/// sequential, where the work-stealing overhead would dominate.
const PARALLEL_BUILD_THRESHOLD: usize = 128;

/// A cell of an [`SPTree`]: `corner` is its centre, `width` its half extent along each dimension.
struct SPTreeCell<T: Float + Send + Sync, const D: usize> {
    corner: [T; D],
    width: [T; D],
}

impl<T: Float + Send + Sync, const D: usize> SPTreeCell<T, D> {
    fn new(corner: [T; D], width: [T; D]) -> Self {
        Self { corner, width }
    }

    /// Whether `point` lies in the cell.
    fn contains_point(&self, point: &[T; D]) -> bool {
        !point
            .iter()
            .zip(self.corner.iter())
            .zip(self.width.iter())
            .any(|((p, &c), &w)| c - w > *p || c + w < *p)
    }
}

/// A space partitioning tree over the `D` dimensional embedding (a quadtree for `D == 2`, an
/// octree for `D == 3`). Each cell holds the centre of mass and point count of its subtree, which
/// is what the Barnes-Hut approximation summarizes.
pub struct SPTree<'a, T, const D: usize>
where
    T: Float + NumCast + AddAssign + MulAssign + DivAssign + Add + Mul + Div + Send + Sync + Sum,
{
    is_leaf: bool,
    cumulative_size: i64,
    boundary: SPTreeCell<T, D>,
    data: &'a [T],
    center_of_mass: [T; D],
    index: Option<usize>,
    children: Vec<SPTree<'a, T, D>>,
}

impl<'a, T, const D: usize> SPTree<'a, T, D>
where
    T: Float + NumCast + AddAssign + MulAssign + DivAssign + Add + Mul + Div + Send + Sync + Sum,
{
    /// Builds the tree over `data`, a flat buffer of `n_samples` points of `D` components each.
    pub(crate) fn new(data: &'a [T], n_samples: usize) -> Self {
        // Mean, min and max of each dimension size the root cell.
        let mut mean = [T::zero(); D];
        let mut min = [T::max_value(); D];
        let mut max = [-T::max_value(); D];
        data.chunks_exact(D).for_each(|sample| {
            sample
                .iter()
                .zip(mean.iter_mut())
                .zip(min.iter_mut())
                .zip(max.iter_mut())
                .for_each(|(((s, mean_d), min_d), max_d)| {
                    *mean_d += *s;
                    *min_d = min_d.min(*s);
                    *max_d = max_d.max(*s);
                })
        });

        let denominator = T::from(n_samples).unwrap();
        mean.iter_mut().for_each(|el| *el /= denominator);

        let mut width = [T::zero(); D];
        width
            .iter_mut()
            .zip(mean.iter())
            .zip(min.iter())
            .zip(max.iter())
            .for_each(|(((w, mean_d), min_d), max_d)| {
                *w = (*max_d - *mean_d).max(*mean_d - *min_d) + T::min_positive_value();
            });

        let mut tree = SPTree::new_empty(data, mean, width);
        if n_samples > 0 {
            let mut indices: Vec<usize> = (0..n_samples).collect();
            let mut scratch: Vec<usize> = vec![0usize; n_samples];
            let dropped = tree.build(&mut indices, &mut scratch);
            // Every input point ends up in exactly one leaf or in a boundary-crack drop. The leaf
            // mass (`cumulative_size`, summed up the tree) plus the drops must therefore equal the
            // input count. This is independent of how the mass is aggregated, so it catches a
            // miscount such as the empty-child phantom-mass regression.
            debug_assert_eq!(
                tree.cumulative_size as usize + dropped,
                n_samples,
                "SPTree lost or invented points: {} in leaves + {dropped} dropped != {n_samples}",
                tree.cumulative_size
            );
        }
        tree
    }

    /// An empty cell with the given boundary.
    fn new_empty(data: &'a [T], corner: [T; D], width: [T; D]) -> Self {
        SPTree {
            children: Vec::new(),
            center_of_mass: [T::zero(); D],
            boundary: SPTreeCell::new(corner, width),
            cumulative_size: 0,
            is_leaf: true,
            index: None,
            data,
        }
    }

    /// Point `index` as a fixed size array reference.
    fn point(&self, index: usize) -> &[T; D] {
        (&self.data[index * D..index * D + D])
            .try_into()
            .expect("a point spans exactly D components")
    }

    /// Children of a subdivided cell, one per orthant.
    const fn n_children() -> usize {
        1 << D
    }

    /// Recursively builds the subtree from the point `indices`, using `scratch` (same length) to
    /// group them by child cell. The two buffers swap roles each level so the index storage is
    /// never reallocated. Large subtrees build their children in parallel. The structure does not
    /// depend on insertion order.
    fn build(&mut self, indices: &mut [usize], scratch: &mut [usize]) -> usize {
        let len = indices.len();
        self.cumulative_size = len as i64;

        if len == 0 {
            return 0;
        }
        if len == 1 {
            self.make_leaf(indices[0]);
            return 0;
        }

        // Count points per child into the children's own `cumulative_size` (scratch the recursion
        // overwrites), so no `2^D` bucket array is needed. Points in a boundary crack are dropped,
        // as a sequential insertion would drop them.
        self.create_children();
        for &index in indices.iter() {
            if let Some(child) = self.child_containing(index) {
                self.children[child].cumulative_size += 1;
            }
        }
        let occupied = self
            .children
            .iter()
            .filter(|child| child.cumulative_size > 0)
            .count();
        let placed = self
            .children
            .iter()
            .map(|child| child.cumulative_size as usize)
            .sum::<usize>();

        // Points that do not separate (duplicates, or all dropped) become one leaf, which also
        // terminates the recursion. The leaf stands in for all `len` points, so none are dropped.
        if occupied <= 1 && (placed == 0 || self.all_duplicates(indices)) {
            self.children.clear();
            self.make_leaf(indices[0]);
            return 0;
        }
        self.is_leaf = false;
        // Points that fail `child_containing` sit in a floating point boundary crack and are
        // dropped at this level. Deeper drops are summed from the children below.
        let dropped_here = len - placed;

        // Counts become exclusive-prefix offsets, then the scatter groups indices into `scratch`,
        // advancing each child's cursor to its window end.
        let mut running = 0i64;
        for child in self.children.iter_mut() {
            let count = child.cumulative_size;
            child.cumulative_size = running;
            running += count;
        }
        for &index in indices.iter() {
            if let Some(child) = self.child_containing(index) {
                let position = self.children[child].cumulative_size as usize;
                scratch[position] = index;
                self.children[child].cumulative_size += 1;
            }
        }

        // Give each non-empty child its window `[previous end, cumulative_size)`: grouped indices
        // in `scratch`, fresh scratch in `indices` (the two swap roles one level down).
        let dropped_below: usize = if placed >= PARALLEL_BUILD_THRESHOLD {
            let mut child_indices: &mut [usize] = &mut scratch[..placed];
            let mut child_scratch: &mut [usize] = &mut indices[..placed];
            let mut previous_end = 0usize;
            let mut work: Vec<(&mut SPTree<'a, T, D>, &mut [usize], &mut [usize])> =
                Vec::with_capacity(self.children.len());
            for child in self.children.iter_mut() {
                let count = child.cumulative_size as usize - previous_end;
                previous_end += count;
                let (mine, rest_indices) = child_indices.split_at_mut(count);
                let (my_scratch, rest_scratch) = child_scratch.split_at_mut(count);
                child_indices = rest_indices;
                child_scratch = rest_scratch;
                if count != 0 {
                    work.push((child, mine, my_scratch));
                } else {
                    // Empty child: clear the prefix-sum cursor left in `cumulative_size`,
                    // otherwise it registers as phantom mass in `combine_center_of_mass`.
                    child.cumulative_size = 0;
                }
            }
            work.into_par_iter()
                .map(|(child, mine, my_scratch)| child.build(mine, my_scratch))
                .sum()
        } else {
            // Sequential build, avoiding the work list allocation.
            let mut child_indices: &mut [usize] = &mut scratch[..placed];
            let mut child_scratch: &mut [usize] = &mut indices[..placed];
            let mut previous_end = 0usize;
            let mut dropped_below = 0usize;
            for child in self.children.iter_mut() {
                let count = child.cumulative_size as usize - previous_end;
                previous_end += count;
                let (mine, rest_indices) = child_indices.split_at_mut(count);
                let (my_scratch, rest_scratch) = child_scratch.split_at_mut(count);
                child_indices = rest_indices;
                child_scratch = rest_scratch;
                if count != 0 {
                    dropped_below += child.build(mine, my_scratch);
                } else {
                    child.cumulative_size = 0;
                }
            }
            dropped_below
        };

        // The node's mass is the points actually in the leaves below it: the sum of its children's
        // final `cumulative_size`. This matches the centre of mass `combine_center_of_mass` averages
        // over the same children, and stays correct when a deeper crack drops a point.
        self.cumulative_size = self
            .children
            .iter()
            .map(|child| child.cumulative_size)
            .sum();
        self.combine_center_of_mass();

        dropped_here + dropped_below
    }

    /// The child cell containing point `index`, or `None` if a floating point crack leaves it
    /// outside every child. Same order and predicate as a sequential insertion, so it drops the
    /// same points.
    fn child_containing(&self, index: usize) -> Option<usize> {
        let point = self.point(index);
        let half = T::from(0.5).unwrap();
        (0..Self::n_children()).find(|&child| {
            (0..D).all(|d| {
                let half_width = half * self.boundary.width[d];
                let center = if (child >> d) & 1 == 1 {
                    self.boundary.corner[d] - half_width
                } else {
                    self.boundary.corner[d] + half_width
                };
                center - half_width <= point[d] && point[d] <= center + half_width
            })
        })
    }

    /// Whether every point in `indices` equals the first, within the duplicate tolerance.
    fn all_duplicates(&self, indices: &[usize]) -> bool {
        let first = self.point(indices[0]);
        indices[1..].iter().all(|&index| {
            let point = self.point(index);
            !first
                .iter()
                .zip(point.iter())
                .any(|(&a, &b)| (a - b).abs() >= T::min_positive_value())
        })
    }

    /// Makes this node a leaf for `index`, its centre of mass that point. `cumulative_size` is left
    /// as set, so a leaf standing in for duplicates keeps their count.
    fn make_leaf(&mut self, index: usize) {
        self.index = Some(index);
        self.is_leaf = true;
        self.center_of_mass = *self.point(index);
    }

    /// Creates the `2^D` empty children that tile this cell.
    fn create_children(&mut self) {
        let half = T::from(0.5).unwrap();
        let n_children = Self::n_children();
        self.children.reserve_exact(n_children);
        for i in 0..n_children {
            let mut corner = [T::zero(); D];
            let mut width = [T::zero(); D];
            for d in 0..D {
                let half_width = half * self.boundary.width[d];
                width[d] = half_width;
                corner[d] = if (i >> d) & 1 == 1 {
                    self.boundary.corner[d] - half_width
                } else {
                    self.boundary.corner[d] + half_width
                };
            }
            self.children
                .push(SPTree::new_empty(self.data, corner, width));
        }
    }

    /// Aggregates the centre of mass from the children, weighted by cumulative size and summed in
    /// index order, so the result does not depend on the parallel build order.
    fn combine_center_of_mass(&mut self) {
        let center_of_mass = &mut self.center_of_mass;
        *center_of_mass = [T::zero(); D];
        let mut total = T::zero();
        for child in self.children.iter() {
            if child.cumulative_size == 0 {
                continue;
            }
            let weight = T::from(child.cumulative_size).unwrap();
            total += weight;
            center_of_mass
                .iter_mut()
                .zip(child.center_of_mass.iter())
                .for_each(|(center, &child_center)| *center += child_center * weight);
        }
        if total > T::zero() {
            center_of_mass
                .iter_mut()
                .for_each(|center| *center /= total);
        }
    }

    /// Whether every stored point lies inside the cell that holds it.
    pub(crate) fn is_correct(&self) -> bool {
        let is_correct = match self.index {
            Some(index) => self.boundary.contains_point(self.point(index)),
            None => true,
        };
        if !self.is_leaf && is_correct {
            !self.children.iter().any(|child| !child.is_correct())
        } else {
            is_correct
        }
    }

    /// Whether every non-empty cell's centre of mass lies within its own boundary, allowing a
    /// small slack for floating point error. A correct tree satisfies this because a cell's centre
    /// of mass is a convex combination of the points it holds, all of which lie inside the cell. A
    /// corrupted aggregation, such as counting an empty child as mass at the origin, violates it.
    pub(crate) fn centers_of_mass_within_cells(&self) -> bool {
        let within = self.cumulative_size == 0
            || (0..D).all(|d| {
                let slack =
                    T::from(0.01).unwrap() * self.boundary.width[d] + T::min_positive_value();
                let low = self.boundary.corner[d] - self.boundary.width[d] - slack;
                let high = self.boundary.corner[d] + self.boundary.width[d] + slack;
                self.center_of_mass[d] >= low && self.center_of_mass[d] <= high
            });
        within
            && self
                .children
                .iter()
                .all(|child| child.centers_of_mass_within_cells())
    }

    /// Accumulates the non-edge (repulsive) Barnes-Hut forces on point `index` into
    /// `negative_forces_row` and the normalization term `q_sum`. `forces_buffer` is scratch.
    pub(crate) fn compute_non_edge_forces(
        &self,
        index: usize,
        theta: T,
        negative_forces_row: &mut [T; D],
        forces_buffer: &mut [T; D],
        q_sum: &mut T,
    ) {
        // Skip empty nodes and self-interaction.
        if self.cumulative_size == 0
            || (self.is_leaf && self.index.map(|i| i == index).unwrap_or_default())
        {
            return;
        }

        let point = self.point(index);

        // Displacement to the centre of mass, and its squared length.
        forces_buffer
            .iter_mut()
            .zip(point.iter())
            .zip(self.center_of_mass.iter())
            .for_each(|((fb, &p), cm)| *fb = p - *cm);
        let mut distance: T = forces_buffer.iter().map(|b| b.powi(2)).sum();

        let max_width = self
            .boundary
            .width
            .iter()
            .fold(T::zero(), |acc, bw| if *bw >= acc { *bw } else { acc });

        if self.is_leaf || (max_width / distance.sqrt() < theta) {
            // Summarize the cell by its centre of mass.
            distance = T::one() / (T::one() + distance);
            let mut m: T = T::from(self.cumulative_size).unwrap() * distance;
            *q_sum += m;
            m *= distance;
            negative_forces_row
                .iter_mut()
                .zip(forces_buffer.iter())
                .for_each(|(nf, b)| *nf += m * *b);
        } else {
            self.children.iter().for_each(|child| {
                child.compute_non_edge_forces(
                    index,
                    theta,
                    negative_forces_row,
                    forces_buffer,
                    q_sum,
                )
            });
        }
    }

    /// Accumulates the edge (attractive) forces on point `index` from its sparse P matrix neighbors
    /// into `positive_forces_row`. `forces_buffer` is scratch.
    pub(crate) fn compute_edge_forces(
        &self,
        index: usize,
        p_rows: &[usize],
        p_columns: &[usize],
        p_values: &[T],
        forces_buffer: &mut [T; D],
        positive_forces_row: &mut [T; D],
    ) {
        let sample = self.point(index);
        for i in p_rows[index]..p_rows[index + 1] {
            let other_sample = self.point(p_columns[i]);

            forces_buffer
                .iter_mut()
                .zip(sample.iter())
                .zip(other_sample.iter())
                .for_each(|((fb, s), os)| *fb = *s - *os);

            let mut distance = forces_buffer.iter().map(|fb| fb.powi(2)).sum::<T>();
            distance = p_values[i] / (distance + T::one());

            positive_forces_row
                .iter_mut()
                .zip(forces_buffer.iter())
                .for_each(|(pfr, fb)| *pfr += distance * *fb);
        }
    }
}
