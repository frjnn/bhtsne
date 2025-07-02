use std::{
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign},
};

use num_traits::{Float, NumCast};

use crossbeam::utils::CachePadded;

/// A cell for the SPTree.
struct SPTreeCell<T: Float + Send + Sync> {
    corner: Vec<T>,
    width: Vec<T>,
}

impl<T: Float + Send + Sync> SPTreeCell<T> {
    /// Constructs a cell.
    ///
    /// # Arguments
    ///
    /// * `corner` - corner of the cell.
    ///
    /// * `width` - width of the cell.
    fn new(corner: Vec<T>, width: Vec<T>) -> Self {
        Self { corner, width }
    }

    /// Checks whether a point lies in a cell.
    ///
    /// # Arguments
    ///
    /// `point` - a point.
    fn contains_point(&self, point: &[CachePadded<T>]) -> bool {
        debug_assert_eq!(point.len(), self.corner.len());
        debug_assert_eq!(point.len(), self.width.len());

        // All the point components lie inside the cell.
        !point
            .iter()
            .zip(self.corner.iter())
            .zip(self.width.iter())
            .any(|((p, &c), &w)| c - w > **p || c + w < **p)
    }
}

/// An SPTree.
pub struct SPTree<'a, T>
where
    T: Float + NumCast + AddAssign + MulAssign + DivAssign + Add + Mul + Div + Send + Sync + Sum,
{
    dimension: usize,
    is_leaf: bool,
    cumulative_size: i64,
    boundary: SPTreeCell<T>,
    data: &'a [CachePadded<T>],
    center_of_mass: Vec<T>,
    index: Option<usize>,
    children: Vec<SPTree<'a, T>>,
    n_children: usize,
}

impl<'a, T> SPTree<'a, T>
where
    T: Float + NumCast + AddAssign + MulAssign + DivAssign + Add + Mul + Div + Send + Sync + Sum,
{
    /// The constructor for SPTree, builds the tree too.
    ///
    /// # Arguments
    ///
    /// * `dimension` - dimensions of a point.
    ///
    /// * `data` - data to build the tree from.
    ///
    /// * `n_samples` - number of points in `inp_data`.
    pub(crate) fn new(dimension: usize, data: &'a [CachePadded<T>], n_samples: usize) -> Self {
        // Mean for each dimension.
        let mut mean: Vec<T> = vec![T::zero(); dimension];
        // Min for each dimension.
        let mut min: Vec<T> = vec![T::max_value(); dimension];
        // Max for each dimension.
        let mut max: Vec<T> = vec![-T::max_value(); dimension];
        // Compute the boundaries of SPTree.
        data.chunks(dimension).for_each(|sample| {
            sample
                .iter()
                .zip(mean.iter_mut())
                .zip(min.iter_mut())
                .zip(max.iter_mut())
                .for_each(|(((s, mean_d), min_d), max_d)| {
                    *mean_d += **s;
                    *min_d = min_d.min(**s);
                    *max_d = max_d.max(**s);
                })
        });

        let denominator = T::from(n_samples).unwrap();
        mean.iter_mut().for_each(|el| *el /= denominator);

        // Build SPTree.
        let mut width: Vec<T> = vec![T::zero(); dimension];
        width
            .iter_mut()
            .zip(mean.iter())
            .zip(min.iter())
            .zip(max.iter())
            .for_each(|(((w, mean_d), min_d), max_d)| {
                *w = (*max_d - *mean_d).max(*mean_d - *min_d) + T::min_positive_value();
            });

        let mut tree = SPTree::new_empty(dimension, data, mean, width);
        tree.fill(n_samples);

        tree
    }

    /// Constructs an empty tree.
    ///
    /// # Arguments
    ///
    /// * `dimension` - dimensions of a point.
    ///
    /// * `data` - data to build the tree from.
    ///
    /// * `corner` - a corner for a cell.
    ///
    /// * `width` - cell's width.
    fn new_empty(
        dimension: usize,
        data: &'a [CachePadded<T>],
        corner: Vec<T>,
        width: Vec<T>,
    ) -> Self {
        let n_children = 2usize.pow(dimension as u32);
        let boundary = SPTreeCell::new(corner, width);
        let children = Vec::new();
        let center_of_mass = vec![T::zero(); dimension];

        SPTree {
            children,
            center_of_mass,
            boundary,
            cumulative_size: 0,
            is_leaf: true,
            dimension,
            n_children,
            index: None,
            data,
        }
    }

    /// Inserts a point into the SPTree.
    ///
    /// # Arguments
    ///
    /// `index` - index of the point inside `data`.
    fn insert(&mut self, index: usize) -> bool {
        // Point to insert as a slice of data.
        let point = &self.data[index * self.dimension..(index + 1) * self.dimension];

        // Ignore objects which do not belong in this quad tree.
        if !self.boundary.contains_point(point) {
            false
        } else {
            // Online update of cumulative size and center-of-mass.
            self.cumulative_size += 1;
            // Update center of mass.
            let m1: T =
                T::from(self.cumulative_size - 1).unwrap() / T::from(self.cumulative_size).unwrap();
            let m2: T = T::one() / T::from(self.cumulative_size).unwrap();

            debug_assert_eq!(self.center_of_mass.len(), point.len(),);

            self.center_of_mass
                .iter_mut()
                .zip(point.iter())
                .for_each(|(cm, &p)| {
                    *cm *= m1;
                    *cm += m2 * *p;
                });

            // If there is space in this quad tree and it is a leaf, add the object here.
            if self.is_leaf && self.index.is_none() {
                self.index = Some(index);
                true
            } else {
                // If there's another point checks that the two points are different. Don't add
                // duplicates for now.
                let is_duplicate: bool = if let Some(present) = self.index {
                    !point
                        .iter()
                        .zip(
                            self.data[present * self.dimension..(present + 1) * self.dimension]
                                .iter(),
                        )
                        .any(|(&pa, &pb)| (*pa - *pb).abs() >= T::min_positive_value())
                } else {
                    false
                };
                // The point is already in the tree.
                if is_duplicate {
                    true
                } else {
                    // Otherwise, we need to subdivide the current cell.
                    if self.is_leaf {
                        self.subdivide();
                    }
                    // Insert point in some children.
                    self.children.iter_mut().any(|child| child.insert(index))
                }
            }
        }
    }

    /// Create four children which fully divide this cell into four quads of equal area.
    fn subdivide(&mut self) {
        // Creates new children.
        for i in 0..self.n_children {
            let mut den: usize = 1;
            let mut new_corner: Vec<T> = vec![T::zero(); self.dimension];
            let mut new_width: Vec<T> = vec![T::zero(); self.dimension];

            // Computes new corner and new width.
            let zero_point_five = T::from(0.5).unwrap();
            new_width
                .iter_mut()
                .zip(new_corner.iter_mut())
                .zip(self.boundary.width.iter())
                .zip(self.boundary.corner.iter())
                .for_each(|(((nw, nc), bw), bc)| {
                    *nw = zero_point_five * *bw;
                    if (i / den) % 2 == 1 {
                        *nc = *bc - zero_point_five * *bw;
                    } else {
                        *nc = *bc + zero_point_five * *bw;
                    }
                    den *= 2;
                });
            // Creates a new child.
            self.children.push(SPTree::new_empty(
                self.dimension,
                self.data,
                new_corner,
                new_width,
            ));
        }

        // Move existing points, if any, to correct children.
        if let Some(index) = self.index {
            self.children.iter_mut().any(|child| child.insert(index));
            // Empty parent node.
            self.index = None;
        }
        self.is_leaf = false;
    }

    /// Build SPTree on dataset.
    ///
    /// # Arguments
    ///
    /// `n_samples` - number of points.
    fn fill(&mut self, n_samples: usize) {
        (0..n_samples).for_each(|index| {
            self.insert(index);
        })
    }

    /// Checks whether the tree is correct
    pub(crate) fn is_correct(&self) -> bool {
        // Empty nodes are correct.
        let is_correct = if let Some(index) = self.index {
            self.boundary
                .contains_point(&self.data[index * self.dimension..(index + 1) * self.dimension])
        } else {
            true
        };

        if !self.is_leaf && is_correct {
            // There are no children who are not correct.
            !self.children.iter().any(|child| !child.is_correct())
        } else {
            is_correct
        }
    }

    /// Compute non-edge forces using Barnes-Hut algorithm.
    ///
    /// # Arguments
    ///
    /// * `index` - index of a point.
    ///
    /// * `theta` - parameter specific to Barnes-Hut algorithm.
    ///
    /// * `negative_forces_row` - negative forces.
    ///
    /// * `forces_buffer` - buffer for the forces.
    ///
    /// * `q_sum` - cumulative sum for the multiplicative factors.
    pub(crate) fn compute_non_edge_forces(
        &self,
        index: usize,
        theta: T,
        negative_forces_row: &mut [CachePadded<T>],
        forces_buffer: &mut [CachePadded<T>],
        q_sum: &mut CachePadded<T>,
    ) {
        // Make sure that  no time is spent on empty nodes or self-interactions.
        if self.cumulative_size == 0
            || (self.is_leaf && self.index.map(|i| i == index).unwrap_or_default())
        {
            return;
        }

        debug_assert_eq!(negative_forces_row.len(), forces_buffer.len());

        // Retrieves point slice.
        let point: &[CachePadded<T>] =
            &self.data[index * self.dimension..(index + 1) * self.dimension];

        // Compute distance between point and center-of-mass.
        forces_buffer
            .iter_mut()
            .zip(point.iter())
            .zip(self.center_of_mass.iter())
            .for_each(|((fb, &p), cm)| **fb = *p - *cm);

        let mut distance: T = forces_buffer.iter().map(|b| b.powi(2)).sum();

        // Check whether we can use this node as a summary.
        let max_width = self
            .boundary
            .width
            .iter()
            .fold(T::zero(), |acc, bw| if *bw >= acc { *bw } else { acc });

        if self.is_leaf || (max_width / distance.sqrt() < theta) {
            // Compute and add tSNE forces between point and current node.
            distance = T::one() / (T::one() + distance);

            let mut m: T = T::from(self.cumulative_size).unwrap() * distance;

            **q_sum += m;
            m *= distance;

            negative_forces_row
                .iter_mut()
                .zip(forces_buffer.iter())
                .for_each(|(nf, b)| **nf += m * **b);
        } else {
            // Recursively apply Barnes-Hut to children.
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

    /// Computes edge forces.
    ///
    /// # Arguments
    ///
    /// * `index` = index of a sample.
    ///
    /// * `sample` - sample.
    ///
    /// * `p_rows` - row indices.
    ///
    /// * `p_columns` - column indices.
    ///
    /// * `p_values` - P matrix entries.
    ///
    /// * `n_samples` - number of points.
    ///
    /// * `positive_forces_row` - positive forces.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute_edge_forces(
        &self,
        index: usize,
        sample: &[CachePadded<T>],
        p_rows: &[usize],
        p_columns: &[usize],
        p_values: &[CachePadded<T>],
        forces_buffer: &mut [CachePadded<T>],
        positive_forces_row: &mut [CachePadded<T>],
    ) {
        // Indexes neighbors of sample.
        // index is the index of the sample.
        for i in p_rows[index]..p_rows[index + 1] {
            // Compute pairwise distance and Q-value.
            let other_sample =
                &self.data[p_columns[i] * self.dimension..(p_columns[i] + 1) * self.dimension];

            debug_assert_eq!(sample.len(), other_sample.len(),);
            debug_assert_eq!(forces_buffer.len(), positive_forces_row.len());

            forces_buffer
                .iter_mut()
                .zip(sample.iter())
                .zip(other_sample.iter())
                .for_each(|((fb, s), os)| **fb = **s - **os);

            let mut distance = forces_buffer.iter().map(|fb| fb.powi(2)).sum::<T>();
            distance = *p_values[i] / (distance + T::one());

            // Sum positive force.
            positive_forces_row
                .iter_mut()
                .zip(forces_buffer.iter())
                .for_each(|(pfr, fb)| **pfr += distance * **fb);
        }
    }
}
