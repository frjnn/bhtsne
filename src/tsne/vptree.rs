use super::super::Float;
use rand::Rng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Trait that must be satisfied in order to build the tree. We must have a metric
/// between its elements.
pub trait Measurable<T>
where
    T: Float,
{
    fn metric(a: &Self, b: &Self) -> T;
}

/// Datapoint struct.
pub struct DataPoint<T> {
    pub ind: u64,
    content: T,
}

impl<'a, T> Measurable<T> for DataPoint<&[T]>
where
    T: Float + std::iter::Sum + std::ops::Add,
{
    // Euclidean distance between slices of `T`.
    fn metric(point_a: &Self, point_b: &DataPoint<&[T]>) -> T {
        let vec_v: &[T] = &point_a.content;
        let vec_w: &[T] = &point_b.content;
        let sum: T = vec_v
            .iter()
            .zip(vec_w.iter())
            .map(|(v, w)| (*v - *w) * (*v - *w))
            .sum();
        sum.sqrt()
    }
}

impl<'a, T> DataPoint<&'a [T]>
where
    T: Float,
    Self: Measurable<T>,
{
    /// A simple constructor.
    pub fn new(ind: u64, content: &'a [T]) -> Self {
        DataPoint { ind, content }
    }

    /// Implements a comparison between `self` and two other `DataPoints`.
    fn compare(&self, a: &Self, b: &Self) -> Ordering {
        if Self::metric(self, a) < Self::metric(self, b) {
            Ordering::Less
        } else if Self::metric(self, a) == Self::metric(self, b) {
            Ordering::Equal
        } else {
            Ordering::Greater
        }
    }
}

/// A node of the tree. It has a point and radius,
/// left children are closer to point than the radius.
pub struct Node<T> {
    index: usize,
    threshold: T,
    left: Option<Box<Node<T>>>,
    right: Option<Box<Node<T>>>,
}

impl<T> Node<T>
where
    T: Float,
{
    /// A simple constructor.
    fn new() -> Self {
        Node {
            index: 0,             // Index of point in node.
            threshold: T::zero(), // Radius.
            left: None,           // Points closer by than threshold.
            right: None,          // Points farther away than threshold.
        }
    }
}

/// An item on the intermediate result queue.
struct HeapItem<T>
where
    T: Float,
{
    index: usize,
    dist: T,
}

impl<T> PartialOrd for HeapItem<T>
where
    T: Float,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}

impl<T> PartialEq for HeapItem<T>
where
    T: Float,
{
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl<T> Eq for HeapItem<T> where T: Float {}

impl<T> Ord for HeapItem<T>
where
    T: Float,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap()
    }
}

/// Auxiliary struct used to build the vantage point tree.
struct VPTreeBuilder<'a, T> {
    root: &'a mut Option<Box<Node<T>>>,
    lower: usize,
    upper: usize,
}

impl<'a, T> VPTreeBuilder<'a, T> {
    /// VPTreeBuilder constructor.
    ///
    /// # Arguments
    ///
    /// * `node` - current node.
    /// * `lower` - lower bound.
    /// * `upper` - upper bound.
    fn new(root: &'a mut Option<Box<Node<T>>>, lower: usize, upper: usize) -> Self {
        Self { root, lower, upper }
    }
}

/// Vantage Point tree.
pub struct VPTree<'a, T> {
    items: &'a mut [&'a DataPoint<&'a [T]>],
    tau: T,
    pub root: Option<Box<Node<T>>>,
}

impl<'a, T> VPTree<'a, T>
where
    T: Float,
    DataPoint<&'a [T]>: Measurable<T>,
{
    /// Function that iteratively fills the tree.
    ///
    /// # Arguments
    ///
    /// * `root` - root of the vantage point tree.
    /// * `items` - items to build the tree with.
    /// * `lower' - lower bound.
    /// * `upper` - upper bound.
    fn build_from_points(
        root: &mut Option<Box<Node<T>>>,
        items: &mut [&DataPoint<&'a [T]>],
        lower: usize,
        upper: usize,
    ) {
        let mut stack: Vec<VPTreeBuilder<T>> = vec![VPTreeBuilder::new(root, lower, upper)];
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
                    let i: usize = thread_rng.gen_range(lower..upper) as usize;
                    items.swap(lower, i);

                    let to_cmp = items[lower];
                    let mut c = |&a: &&DataPoint<&'a [T]>, &b: &&DataPoint<&'a [T]>| -> Ordering {
                        to_cmp.compare(a, b)
                    };
                    // Partition around the median distance.
                    let median: usize = (upper + lower) / 2;
                    pdqselect::select_by(&mut items[lower + 1..upper], median, &mut c);

                    // Threshold of the new node will be the distance to the median.
                    node.threshold = DataPoint::<&[T]>::metric(&items[lower], &items[median]);

                    stack.push(VPTreeBuilder::new(&mut node.left, lower + 1, median));
                    stack.push(VPTreeBuilder::new(&mut node.right, median, upper));
                }
            }
        }
    }

    /// Auxiliary function that searches for the k nearest neighbors of an item.
    fn _search(
        items: &[&DataPoint<&'a [T]>],
        tau: &mut T,
        node: &Option<Box<Node<T>>>,
        target: &DataPoint<&'a [T]>,
        k: usize,
        heap: &mut BinaryHeap<HeapItem<T>>,
    ) {
        let mut stack: Vec<&Option<Box<Node<T>>>> = vec![node];

        while let Some(_node) = stack.pop() {
            if let Some(ref node) = _node {
                // Compute distance between target and current node.
                let dist: T = DataPoint::<&'a [T]>::metric(items[node.index], target);
                if dist < *tau {
                    // If current node is within the radius tau
                    // remove furthest node from result list (if we already have k results),
                    // add current node to result list and
                    // update value of tau (farthest point in result list).
                    if heap.len() == k {
                        heap.pop();
                    }
                    heap.push(HeapItem {
                        index: node.index,
                        dist,
                    });
                    if heap.len() == k {
                        *tau = heap.peek().unwrap().dist;
                    }
                }

                match (node.left.as_ref(), node.right.as_ref()) {
                    // Return if we arrived at a leaf.
                    (None, None) => continue,
                    (_, _) => {
                        // If the target lies within the radius of ball.
                        if dist < node.threshold {
                            // if there can still be neighbors inside the ball, recursively search left child first.
                            if dist - *tau <= node.threshold {
                                stack.push(&node.left)
                            }
                            // if there can still be neighbors outside the ball, recursively search right child.
                            if dist + *tau >= node.threshold {
                                stack.push(&node.right)
                            }
                        } else {
                            // If the target lies outsize the radius of the ball.
                            // If there can still be neighbors outside the ball, recursively search right child.
                            if dist + *tau >= node.threshold {
                                stack.push(&node.right)
                            }
                            // if there can still be neighbors inside the ball, recursively search left child.
                            if dist - *tau <= node.threshold {
                                stack.push(&node.left)
                            }
                        }
                    }
                }
            }
        }
    }

    /// Constructor for the `VPTree` struct.
    ///
    /// # Arguments
    ///
    /// `items` - items to build the tree with.
    pub fn new(items: &'a mut [&'a DataPoint<&'a [T]>]) -> Self
    where
        T: Float,
        DataPoint<&'a [T]>: Measurable<T>,
    {
        let mut tree: VPTree<T> = VPTree {
            items,
            tau: T::zero(),
            root: None,
        };
        let len = tree.items.len();
        VPTree::build_from_points(&mut tree.root, tree.items, 0, len);
        tree
    }

    /// Function that searches the tree and finds the k nearest neighbors of `target`.
    ///
    /// # Arguments
    ///
    /// * `target` -  target data point.
    /// * `k` - number of nearest neighbors.
    /// * `results` - vector in which the index of the nearest neighbors will be saved.
    /// * `distances` - vector storing relative distances of the nearest neighbors.
    pub fn search(
        &mut self,
        target: &DataPoint<&'a [T]>,
        k: usize,
        results: &mut Vec<u64>,
        distances: &mut Vec<T>,
    ) {
        // Use a priority queue to store intermediate results on.
        let mut heap: BinaryHeap<HeapItem<T>> = BinaryHeap::new();
        // Variable that tracks the distance to the farthest point in our results.
        self.tau = T::max_value();

        // Perform the search.
        VPTree::_search(&self.items, &mut self.tau, &self.root, target, k, &mut heap);

        // Empties the important vectors.
        results.clear();
        distances.clear();
        // Gather final results.
        while let Some(el) = heap.pop() {
            results.push(self.items[el.index].ind);
            distances.push(el.dist);
        }
        // Results are in reverse order.
        results.reverse();
        distances.reverse();
    }
}
