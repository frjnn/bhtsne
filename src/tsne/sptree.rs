use super::super::{Float, NumCast};

/// A cell for the SPTree.
struct Cell<T>
where
    T: Float,
{
    dim: usize,
    corner: Vec<T>,
    width: Vec<T>,
}

impl<T> Cell<T>
where
    T: Float,
{
    /// Constructs a cell.
    ///
    /// # Arguments
    ///
    /// * `dim` - dimensions of a point.
    /// * `corner` - corner of the cell.
    /// * `width` - width of the cell.
    fn new(dim: usize, corner: Vec<T>, width: Vec<T>) -> Self {
        Cell { dim, corner, width }
    }

    /// Checks whether a point lies in a cell.
    ///
    /// # Arguments
    ///
    /// `point` - a point.
    fn contains_point(&self, point: &[T]) -> bool {
        let mut res: bool = false;
        for (i, el) in point.iter().enumerate() {
            if self.corner[i] - self.width[i] > *el {
                break;
            }
            if self.corner[i] + self.width[i] < *el {
                break;
            }
            if i == self.dim - 1 {
                res = true;
            }
        }
        res
    }
}

/// An SPTree.
pub struct SPTree<'a, T>
where
    T: Float,
{
    // A buffer we use when doing force computations.
    buff: Vec<T>,
    // Properties of this node in the tree
    dimension: usize,
    is_leaf: bool,
    size: usize,
    cum_size: i64,
    // Axis-aligned bounding box stored as a
    // center with half-dimensions to represent
    // the boundaries of this quad tree.
    boundary: Cell<T>,
    // Indices in this space-partitioning tree node,
    // corresponding center-of-mass, and list of all children.
    data: &'a [T],
    center_of_mass: Vec<T>,
    index: [usize; 1],
    // Children.
    children: Vec<SPTree<'a, T>>,
    no_children: usize,
}

impl<'a, T> SPTree<'a, T>
where
    T: Float
        + NumCast
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::Add
        + std::ops::Mul
        + std::ops::Div,
{
    /// The constructor for SPTree, builds the tree too.
    ///
    /// # Arguments
    ///
    /// * `dim` - dimensions of a point.
    /// * `inp_data` - data to build the tree from.
    /// * `num_points` - number of points in `inp_data`.
    pub fn new(dim: usize, inp_data: &'a [T], num_points: usize) -> Self {
        // Compute mean, width, and height of current
        // map i.e. boundaries of SPTree.
        let mut n_d: usize = 0;

        let mut mean_y: Vec<T> = vec![T::zero(); dim];
        let mut min_y: Vec<T> = vec![T::max_value(); dim];
        let mut max_y: Vec<T> = vec![-T::max_value(); dim];

        for n in 0..num_points {
            for d in 0..dim {
                mean_y[d] += inp_data[n * dim + d];

                if inp_data[n_d + d] < min_y[d] {
                    min_y[d] = inp_data[n_d + d];
                }

                if inp_data[n_d + d] > max_y[d] {
                    max_y[d] = inp_data[n_d + d];
                }
            }
            n_d += dim;
        }
        for el in &mut mean_y[..] {
            *el /= T::from(num_points).unwrap();
        }

        // Construct SPTree.
        let mut inp_width: Vec<T> = vec![T::zero(); dim];

        for d in 0..dim {
            let max_mean = max_y[d] - mean_y[d];
            let mean_min = mean_y[d] - min_y[d];

            inp_width[d] = if max_mean >= mean_min {
                max_mean + T::from(1e-5).unwrap()
            } else {
                mean_min + T::from(1e-5).unwrap()
            };
        }

        let mut empty_tree: SPTree<T> = SPTree::_new(dim, inp_data, mean_y, inp_width);
        empty_tree.fill(num_points);
        empty_tree
    }

    /// Auxiliary function: construct SPTree with particular size, build the tree too!
    ///
    /// # Arguments
    ///
    /// * `inp_dim` - dimensions of a point.
    /// * `inp_data` - data to build the tree from.
    /// * `inp_corner` - a corner for a cell.
    /// * `inp_width` - cell's width.
    fn _new(inp_dim: usize, inp_data: &'a [T], inp_corner: Vec<T>, inp_width: Vec<T>) -> Self {
        let mut _no_children: usize = 2;

        for _ in 1..inp_dim {
            _no_children *= 2;
        }

        let _boundary: Cell<T> = Cell::new(inp_dim, inp_corner, inp_width);
        let _children: Vec<SPTree<T>> = Vec::new();
        let _center_of_mass: Vec<T> = vec![T::zero(); inp_dim];
        let _buff: Vec<T> = vec![T::zero(); inp_dim];
        SPTree {
            buff: _buff,
            children: _children,
            center_of_mass: _center_of_mass,
            boundary: _boundary,
            size: 0,
            cum_size: 0,
            is_leaf: true,
            dimension: inp_dim,
            no_children: _no_children,
            index: [0],
            data: inp_data,
        }
    }

    /// Constructs an empty tree.
    ///
    /// # Arguments
    ///
    /// * `inp_dim` - dimensions of a point.
    /// * `inp_data` - data to build the tree from.
    /// * `inp_corner` - a corner for a cell.
    /// * `inp_width` - cell's width.
    fn new_empty(inp_dim: usize, inp_data: &'a [T], inp_corner: Vec<T>, inp_width: Vec<T>) -> Self {
        SPTree::_new(inp_dim, inp_data, inp_corner, inp_width)
    }

    /// Inserts a point into the SPTree.
    ///
    /// # Arguments
    ///
    /// `new_index` - index of a point.
    fn insert(&mut self, new_index: usize) -> bool {
        let point_bound_l: usize = new_index * self.dimension;
        let point_bound_r: usize = point_bound_l + self.dimension;

        // Ignore objects which do not belong in this quad tree.
        if !self
            .boundary
            .contains_point(&self.data[point_bound_l..point_bound_r])
        {
            false
        } else {
            // Online update of cumulative size and center-of-mass.
            self.cum_size += 1;

            let mult1: T = T::from(self.cum_size - 1).unwrap() / T::from(self.cum_size).unwrap();
            let mult2: T = T::one() / T::from(self.cum_size).unwrap();

            for d in 0..self.dimension {
                self.center_of_mass[d] *= mult1;
            }

            for d in 0..self.dimension {
                self.center_of_mass[d] += mult2 * self.data[point_bound_l..point_bound_r][d];
            }

            // If there is space in this quad tree and it is a leaf, add the object here.
            if self.is_leaf && self.size < 1 {
                self.index[0] = new_index;
                self.size += 1;
                true
            } else {
                // Don't add duplicates for now (this is not very nice).
                let mut any_duplicate: bool = false;

                for n in 0..self.size {
                    let mut duplicate: bool = true;
                    for d in 0..self.dimension {
                        if self.data[point_bound_l..point_bound_r][d]
                            != self.data[self.index[n] * self.dimension
                                ..self.index[n] * self.dimension + self.dimension][d]
                        {
                            duplicate = false;
                            break;
                        }
                    }
                    any_duplicate |= duplicate;
                }

                if any_duplicate {
                    true
                } else {
                    // Otherwise, we need to subdivide the current cell.
                    if self.is_leaf {
                        self.subdivide();
                    }

                    let mut inserted: bool = false;

                    // Find out where the point can be inserted.
                    for i in 0..self.no_children {
                        if self.children[i].insert(new_index) {
                            inserted = true;
                            if inserted {
                                break;
                            }
                        }
                    }
                    inserted
                }
            }
        }
    }

    /// Create four children which fully divide this cell into four quads of equal area.
    fn subdivide(&mut self) {
        // Create new children.
        for i in 0..self.no_children {
            let mut div: usize = 1;
            let mut new_corner: Vec<T> = vec![T::zero(); self.dimension];
            let mut new_width: Vec<T> = vec![T::zero(); self.dimension];

            for d in 0..self.dimension {
                new_width[d] = T::from(0.5).unwrap() * self.boundary.width[d];
                if (i / div) % 2 == 1 {
                    new_corner[d] =
                        self.boundary.corner[d] - T::from(0.5).unwrap() * self.boundary.width[d];
                } else {
                    new_corner[d] =
                        self.boundary.corner[d] + T::from(0.5).unwrap() * self.boundary.width[d];
                }
                div *= 2;
            }
            self.children.push(SPTree::new_empty(
                self.dimension,
                self.data,
                new_corner,
                new_width,
            ));
        }

        // Move existing points to correct children.
        for i in 0..self.size {
            let mut success: bool = false;
            for j in 0..self.no_children {
                if !success {
                    success = self.children[j].insert(self.index[i]);
                }
            }
            self.index[i] = 0;
        }
        // Empty parent node.
        self.size = 0;
        self.is_leaf = false;
    }

    /// Build SPTree on dataset.
    ///
    /// # Arguments
    ///
    /// `data_n` - number of points.
    fn fill(&mut self, data_n: usize) {
        (0..data_n).for_each(|n| {
            self.insert(n);
        })
    }

    /// Checks whether the tree is correct
    #[allow(dead_code)]
    #[cfg(not(tarpaulin_include))]
    pub fn is_correct(&self) -> bool {
        let mut ok: bool = true;

        for i in 0..self.size {
            let point: &[T] = &self.data[self.index[i] * self.dimension..];
            if !self.boundary.contains_point(point) {
                ok = false;
            }
        }

        if ok {
            if !self.is_leaf {
                let mut correct: bool = true;
                for i in 0..self.no_children {
                    correct = correct && self.children[i].is_correct();
                }
                correct
            } else {
                ok
            }
        } else {
            ok
        }
    }

    /// Compute non-edge forces using Barnes-Hut algorithm.
    ///
    /// # Arguments
    ///
    /// * `point_index` - index of a point.
    /// * `theta` - parameter specific to Barnes-Hut algorithm.
    /// * `neg_f` - negative forces.
    /// * `sum_q` - cumulative sum for the multiplicative factors.
    pub fn compute_non_edge_forces(
        &mut self,
        point_index: usize,
        theta: T,
        neg_f: &mut [T],
        sum_q: &mut T,
    ) {
        // Make sure that we spend no time on empty nodes or self-interactions.
        if self.cum_size == 0 || (self.is_leaf && self.size == 1 && self.index[0] == point_index) {
            return;
        }

        // Compute distance between point and center-of-mass.
        let mut distance: T = T::zero();
        let ind: usize = point_index * self.dimension;

        for d in 0..self.dimension {
            self.buff[d] = self.data[ind + d] - self.center_of_mass[d];
        }
        for d in 0..self.dimension {
            distance += self.buff[d] * self.buff[d];
        }

        // Check whether we can use this node as a summary.
        let mut max_width: T = T::zero();
        let mut cur_width: T;
        for d in 0..self.dimension {
            cur_width = self.boundary.width[d];
            max_width = if max_width > cur_width {
                max_width
            } else {
                cur_width
            }
        }

        if self.is_leaf || max_width / distance.sqrt() < theta {
            // Compute and add tsne force between point and current node.
            distance = T::one() / (T::one() + distance);
            let mut mult: T = T::from(self.cum_size).unwrap() * distance;
            *sum_q += mult;
            mult *= distance;

            for (i, el) in neg_f.iter_mut().enumerate() {
                *el += mult * self.buff[i];
            }
        } else {
            // Recursively apply Barnes-Hut to children.
            for i in 0..self.no_children {
                self.children[i].compute_non_edge_forces(point_index, theta, neg_f, sum_q)
            }
        }
    }

    /// Computes edge forces.
    ///
    /// # Arguments
    ///
    /// * `row_p` - row indices.
    /// * `col_p` - column indices.
    /// * `val_p` - P matrix entries.
    /// * `num` - number of points.
    /// * `pos_f` - positive forces.
    pub fn compute_edge_forces(
        &mut self,
        row_p: &mut [usize],
        col_p: &mut [usize],
        val_p: &mut [T],
        num: usize,
        pos_f: &mut [T],
    ) {
        // Loop over all edges in the graph.
        let mut ind1: usize = 0;
        let mut ind2: usize;
        let mut distance: T;

        for n in 0..num {
            for i in row_p[n]..row_p[n + 1] {
                // Compute pairwise distance and Q-value.
                distance = T::one();
                ind2 = col_p[i] * self.dimension;

                for d in 0..self.dimension {
                    self.buff[d] = self.data[ind1 + d] - self.data[ind2 + d];
                }
                for d in 0..self.dimension {
                    distance += self.buff[d] * self.buff[d];
                }
                distance = val_p[i] / distance;
                // Sum positive force.
                for d in 0..self.dimension {
                    pos_f[ind1 + d] += distance * self.buff[d];
                }
            }
            ind1 += self.dimension;
        }
    }
}
