use num_traits::Float;
use rand::Rng;
use std::{
    cmp::Ordering,
    collections::BinaryHeap,
    fmt::{Debug, Display},
};

/// A node of the vantage point tree.
pub(crate) struct Node<T: Float + Debug + Display> {
    index: usize,
    threshold: T,
    left: Option<Box<Node<T>>>,
    right: Option<Box<Node<T>>>,
}

impl<T: Float + Debug + Display> Node<T> {
    /// Creates an empty node without children and  with index equal to 0.
    fn new() -> Self {
        Node {
            index: 0,
            threshold: T::zero(),
            left: None,
            right: None,
        }
    }
}

/// An item on the intermediate result heap. It is used to store results from the nearest neighbors
/// search performed of the vantage point tree.
struct HeapItem<T: Float + Debug + Display> {
    // Index of a sample.
    index: usize,
    // Distance of the sample from the target.
    distance: T,
}

impl<T: Float + Debug + Display> PartialOrd for HeapItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl<T: Float + Debug + Display> PartialEq for HeapItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<T: Float + Debug + Display> Eq for HeapItem<T> {}

impl<T: Float + Debug + Display> Ord for HeapItem<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap()
    }
}

/// Auxiliary struct used to build the vantage point tree.
struct VPTreeBuilder<'a, T: Float + Debug + Display> {
    root: &'a mut Option<Box<Node<T>>>,
    lower: usize,
    upper: usize,
}

impl<'a, T: Float + Debug + Display> VPTreeBuilder<'a, T> {
    /// VPTreeBuilder constructor.
    ///
    /// # Arguments
    ///
    /// * `node` - current node.
    ///
    /// * `lower` - lower bound.
    ///
    /// * `upper` - upper bound.
    fn new(root: &'a mut Option<Box<Node<T>>>, lower: usize, upper: usize) -> Self {
        Self { root, lower, upper }
    }
}

/// Vantage Point tree.
pub(crate) struct VPTree<'a, T: Float + Debug + Display + Send + Sync, U> {
    items: Vec<(usize, &'a U)>,
    pub(crate) root: Option<Box<Node<T>>>,
}

impl<'a, T: Float + Debug + Display + Send + Sync, U> VPTree<'a, T, U> {
    /// Constructor for the `VPTree` struct.
    ///
    /// # Arguments
    ///
    /// * `items` - **original** items to build the tree on.
    ///
    /// * `metric_f` - metric function.
    pub fn new<F>(items: &'a [U], metric_f: F) -> Self
    where
        F: Fn(&U, &U) -> T,
    {
        let mut tree = VPTree {
            // Need to swap some references around, don't want to move original data.
            // Also need to keep track of the original position, as it will differ.
            items: items.iter().enumerate().collect(),
            root: None,
        };
        let n_samples = tree.items.len(); // Immutable borrow must be kept outside.
        tree.build_from_points(0, n_samples, &metric_f);
        tree
    }

    /// Function that iteratively fills the tree.
    ///
    /// # Arguments
    ///
    /// * `lower' - lower bound.
    ///
    /// * `upper` - upper bound.
    ///
    /// * `metric_f` - metric function.
    fn build_from_points<F: Fn(&U, &U) -> T>(&mut self, lower: usize, upper: usize, metric_f: F) {
        let mut stack = vec![VPTreeBuilder::new(&mut self.root, lower, upper)];
        let mut thread_rng = rand::thread_rng();

        while let Some(builder) = stack.pop() {
            let VPTreeBuilder { root, lower, upper } = builder;

            // Lower index is center of current node.
            if upper != lower {
                *root = Some(Box::new(Node::new()));
                let mut node = root.as_deref_mut().unwrap();
                node.index = lower;

                if upper - lower > 1 {
                    // If we did not arrive at leaf yet choose an arbitrary point
                    // and move it to the start.
                    let i = thread_rng.gen_range(lower..upper);
                    self.items.swap(lower, i);

                    let (_, to_cmp) = self.items[lower];
                    // Partition around the median distances.
                    let median: usize = (upper + lower) / 2;
                    self.items[lower + 1..upper].select_nth_unstable_by(
                        median,
                        &mut |a: &(usize, &U), b: &(usize, &U)| {
                            if metric_f(to_cmp, a.1) < metric_f(to_cmp, b.1) {
                                Ordering::Less
                            } else if metric_f(to_cmp, a.1) == metric_f(to_cmp, b.1) {
                                Ordering::Equal
                            } else {
                                Ordering::Greater
                            }
                        },
                    );

                    // Threshold of the new node will be the distances to the median.
                    node.threshold = metric_f(self.items[lower].1, self.items[median].1);

                    stack.push(VPTreeBuilder::new(&mut node.left, lower + 1, median));
                    stack.push(VPTreeBuilder::new(&mut node.right, median, upper));
                }
            }
        }
    }

    /// Auxiliary function that searches for the k nearest neighbors of an item.
    fn look_up<F: Fn(&U, &U) -> T>(
        &self,
        tau: &mut T, //Tracks the distances to the farthest point in the results.
        target: &U,
        k: usize,
        heap: &mut BinaryHeap<HeapItem<T>>,
        metric_f: F,
    ) {
        let mut stack: Vec<&Option<Box<Node<T>>>> = vec![&self.root];

        while let Some(next_in_stack) = stack.pop() {
            if let Some(ref node) = next_in_stack {
                let (original_position, point) = &self.items[node.index];
                // Compute distances between target and current node.
                let distance: T = metric_f(point, target);
                if distance < *tau {
                    // If current node is within the radius tau
                    // remove furthest node from result list (if it already contains k results),
                    // add current node to result list and
                    // update value of tau (farthest point in result list).
                    if heap.len() == k {
                        heap.pop();
                    }
                    heap.push(HeapItem {
                        index: *original_position,
                        distance,
                    });
                    if heap.len() == k {
                        *tau = heap.peek().unwrap().distance;
                    }
                }

                match (node.left.as_ref(), node.right.as_ref()) {
                    // Return if we arrived at a leaf.
                    (None, None) => continue,
                    (_, _) => {
                        // If the target lies within the radius of ball.
                        if distance < node.threshold {
                            // if there can still be neighbors inside the ball, recursively search left child first.
                            if distance - *tau <= node.threshold {
                                stack.push(&node.left)
                            }
                            // if there can still be neighbors outside the ball, recursively search right child.
                            if distance + *tau >= node.threshold {
                                stack.push(&node.right)
                            }
                        } else {
                            // If the target lies outsize the radius of the ball.
                            // If there can still be neighbors outside the ball, recursively search right child.
                            if distance + *tau >= node.threshold {
                                stack.push(&node.right)
                            }
                            // if there can still be neighbors inside the ball, recursively search left child.
                            if distance - *tau <= node.threshold {
                                stack.push(&node.left)
                            }
                        }
                    }
                }
            }
        }
    }

    /// Function that searches the tree and finds the k nearest neighbors of `target`.
    ///
    /// # Arguments
    ///
    /// * `target` -  target data point.
    ///
    /// * `index` - index of the target.
    ///
    /// * `k` - number of nearest neighbors.
    ///
    /// * `results` - vector in which the index of the nearest neighbors will be saved.
    ///
    /// * `distances` - vector storing relative distances of the nearest neighbors.
    ///
    /// * `metric_f` - metric function.
    pub fn search<F>(
        &self,
        target: &U,
        target_index: usize,
        k: usize,
        neighbors_indices: &mut [super::Aligned<usize>],
        distances: &mut [super::Aligned<T>],
        metric_f: F,
    ) where
        F: Fn(&U, &U) -> T,
    {
        debug_assert_eq!(neighbors_indices.len(), distances.len());

        // Use a priority queue to store intermediate results on.
        let mut heap: BinaryHeap<HeapItem<T>> = BinaryHeap::with_capacity(k);
        // Perform the search.
        self.look_up(&mut T::max_value(), target, k, &mut heap, metric_f);
        // Gather final results.
        let results = heap.into_sorted_vec();

        // Avoid the target itself.
        neighbors_indices
            .iter_mut()
            .zip(distances.iter_mut())
            .zip(results.iter().filter(|result| {
                let HeapItem { index, distance: _ } = result;
                target_index != *index
            }))
            .for_each(|((idx, d), result)| {
                let HeapItem { index, distance } = result;
                idx.0 = *index;
                d.0 = *distance;
            });
    }
}
