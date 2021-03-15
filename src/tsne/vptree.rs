use pdqselect::select_by;
use rand::Rng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Trait that must be satisfied in order to build the tree. We must have a metric
/// between its elements.
pub trait Measurable {
    fn metric(a: &Self, b: &Self) -> f32;
}

#[derive(Clone)]
/// The datapoint struct.
pub struct DataPoint<T> {
    pub ind: u64,
    content: T,
    dims: usize,
}

impl<'a> Measurable for DataPoint<&[f32]> {
    // Euclidean distance between slices of `f32`.
    fn metric(point_a: &Self, point_b: &DataPoint<&[f32]>) -> f32 {
        let vec_v: &[f32] = &point_a.content;
        let vec_w: &[f32] = &point_b.content;
        let sum: f32 = vec_v
            .iter()
            .zip(vec_w.iter())
            .map(|(v, w)| (*v - *w) * (*v - *w))
            .sum();
        sum.sqrt().into()
    }
}

impl<T> DataPoint<T>
where
    Self: Measurable,
{
    /// A simple constructor.
    pub fn new(ind: u64, content: T, dims: usize) -> DataPoint<T> {
        DataPoint {
            ind: ind,
            content: content,
            dims: dims,
        }
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
pub struct Node {
    index: usize,
    threshold: f32,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

impl Node {
    /// A simple constructor.
    fn new() -> Self {
        Node {
            index: 0,       // Index of point in node.
            threshold: 0.0, // Radius.
            left: None,     // Points closer by than threshold.
            right: None,    // Points farther away than threshold.
        }
    }
}

/// An item on the intermediate result queue.
struct HeapItem {
    index: usize,
    dist: f32,
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl Eq for HeapItem {}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap()
    }
}

/// Vantage Point tree.
pub struct VPTree<'a, T> {
    items: &'a mut [&'a DataPoint<T>],
    tau: f32,
    pub root: Option<Box<Node>>,
}

impl<'a, T> VPTree<'a, T>
where
    DataPoint<T>: Measurable,
{
    /// Function that recursively fills the tree.
    fn build_from_points(
        root: &mut Option<Box<Node>>,
        items: &mut [&DataPoint<T>],
        lower: usize,
        upper: usize,
    ) {
        if upper == lower {
            // Indicates that we're done here!
            *root = None;
            return;
        }
        // Lower index is center of current node.
        let mut node: Node = Node::new();
        node.index = lower;

        if upper - lower > 1 {
            // If we did not arrive at leaf yet choose an arbitrary point
            // and move it to the start.
            let i: usize = rand::thread_rng().gen_range(lower..upper) as usize;
            items.swap(lower, i);

            // Partition around the median distance.
            let median: usize = (upper + lower) / 2;

            let to_cmp = items[lower];
            let mut c =
                |&a: &&DataPoint<T>, &b: &&DataPoint<T>| -> Ordering { to_cmp.compare(a, b) };
            select_by(&mut items[lower + 1..upper], median, &mut c);

            // Threshold of the new node will be the distance to the median.
            node.threshold = DataPoint::<T>::metric(&items[lower], &items[median]);
            // Recursively build tree.
            node.index = lower;
            VPTree::build_from_points(&mut node.left, items, lower + 1, median);
            VPTree::build_from_points(&mut node.right, items, median, upper);
        }
        *root = Some(Box::new(node));
    }

    /// Auxiliary function that searches for the k nearest neighbors of an item.
    fn _search(
        items: &[&DataPoint<T>],
        tau: &mut f32,
        node: &Option<Box<Node>>,
        target: &DataPoint<T>,
        k: usize,
        heap: &mut BinaryHeap<HeapItem>,
    ) {
        // Indicates that we're done here!
        let node: &Node = match node {
            Some(_node) => _node.as_ref(),
            None => return,
        };

        // Compute distance between target and current node.
        let dist: f32 = DataPoint::<T>::metric(items[node.index], target);
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
                dist: dist,
            });
            if heap.len() == k {
                *tau = heap.peek().unwrap().dist;
            }
        }

        match (node.left.as_ref(), node.right.as_ref()) {
            // Return if we arrived at a leaf.
            (None, None) => return,
            (_, _) => {
                // If the target lies within the radius of ball.
                if dist < node.threshold {
                    // if there can still be neighbors inside the ball, recursively search left child first.
                    if dist - *tau <= node.threshold {
                        VPTree::_search(items, tau, &node.left, target, k, heap)
                    }
                    // if there can still be neighbors outside the ball, recursively search right child.
                    if dist + *tau >= node.threshold {
                        VPTree::_search(items, tau, &node.right, target, k, heap)
                    }
                } else {
                    // If the target lies outsize the radius of the ball.
                    // If there can still be neighbors outside the ball, recursively search right child.
                    if dist + *tau >= node.threshold {
                        VPTree::_search(items, tau, &node.right, target, k, heap)
                    }
                    // if there can still be neighbors inside the ball, recursively search left child.
                    if dist - *tau <= node.threshold {
                        VPTree::_search(items, tau, &node.left, target, k, heap)
                    }
                }
            }
        }
    }

    /// Default constructor for the `VPTree` struct.
    pub fn new(items: &'a mut [&'a DataPoint<T>]) -> Self
    where
        DataPoint<T>: Measurable,
    {
        let mut tree: VPTree<T> = VPTree {
            items: items,
            tau: 0.0,
            root: None,
        };
        let len = tree.items.len();
        VPTree::build_from_points(&mut tree.root, tree.items.as_mut(), 0, len);
        tree
    }

    /// Function that uses the tree to find the k nearest neighbors of `target`.
    pub fn search(
        &mut self,
        target: &DataPoint<T>,
        k: usize,
        results: &mut Vec<u64>,
        distances: &mut Vec<f32>,
    ) {
        // Use a priority queue to store intermediate results on.
        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
        // Variable that tracks the distance to the farthest point in our results.
        self.tau = std::f32::MAX;

        // Perform the search.
        VPTree::_search(
            &mut self.items,
            &mut self.tau,
            &self.root,
            target,
            k,
            &mut heap,
        );

        // Empties the important vectors.
        results.clear();
        distances.clear();
        // Gather final results.
        while !heap.is_empty() {
            let el: HeapItem = heap.pop().unwrap();
            results.push(self.items[el.index].ind);
            distances.push(el.dist);
        }
        // Results are in reverse order.
        results.reverse();
        distances.reverse();
    }
}
