use std::{cmp::Ordering, collections::BinaryHeap};

use rand::Rng;

use num_traits::Float;

/// `left`/`right` value marking the absence of a child.
const SENTINEL: u32 = u32::MAX;

/// One node of the flat vantage point tree. Node `0` is the root.
#[derive(Clone, Debug)]
struct Node<T: Float> {
    item: u32,
    threshold: T,
    left: u32,
    right: u32,
}

/// Which side of its parent a frame's node hangs off, so the build can wire the link once the
/// node's arena index is known.
enum Side {
    Root,
    Left,
    Right,
}

impl<T: Float> Default for Node<T> {
    fn default() -> Self {
        Node {
            item: 0,
            threshold: T::zero(),
            left: SENTINEL,
            right: SENTINEL,
        }
    }
}

/// An item on the intermediate result heap. It is used to store results from the nearest neighbors
/// search performed of the vantage point tree.
struct HeapItem<T: Float> {
    // Index of a sample.
    index: usize,
    // Distance of the sample from the target.
    distance: T,
}

/// Reusable per-search scratch.
pub(crate) struct SearchScratch<T: Float> {
    /// Bounded max-heap of the current k best candidates, ordered by distance.
    heap: BinaryHeap<HeapItem<T>>,
    /// Explicit traversal stack standing in for recursion in [`VPTree::look_up`].
    stack: Vec<u32>,
    /// The heap drained into ascending order, read back into the caller's output rows.
    results: Vec<HeapItem<T>>,
}

impl<T: Float> Default for SearchScratch<T> {
    fn default() -> Self {
        Self {
            heap: BinaryHeap::new(),
            stack: Vec::new(),
            results: Vec::new(),
        }
    }
}

impl<T: Float> PartialOrd for HeapItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Float> PartialEq for HeapItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<T: Float> Eq for HeapItem<T> {}

impl<T: Float> Ord for HeapItem<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap()
    }
}

/// Vantage Point tree.
pub(crate) struct VPTree<'a, T: Float + Send + Sync, U> {
    items: Vec<(usize, &'a U)>,
    nodes: Vec<Node<T>>,
}

impl<'a, T: Float + Send + Sync, U> VPTree<'a, T, U> {
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
            nodes: Vec::new(),
        };
        tree.build(&metric_f);

        tree
    }

    /// Builds the flat tree over `items`. Iterative, with an explicit frame stack standing in for
    /// recursion: each frame is a half-open `items` range and the parent slot its node fills.
    ///
    /// # Arguments
    ///
    /// * `metric_f` - metric function.
    fn build<F>(&mut self, metric_f: F)
    where
        F: Fn(&U, &U) -> T,
    {
        let n = self.items.len();
        if n == 0 {
            return;
        }

        let mut rng = super::make_rng();
        // Cached (distance, item) pairs for the current partition, reused across frames.
        let mut partition: Vec<(T, (usize, &'a U))> = Vec::new();
        let mut stack = vec![(0usize, n, SENTINEL, Side::Root)];

        while let Some((lower, upper, parent, side)) = stack.pop() {
            // An empty range contributes no node; the parent's link stays SENTINEL.
            if lower == upper {
                continue;
            }

            // Allocate this node and wire it to its parent.
            let node_index = self.nodes.len() as u32;
            self.nodes.push(Node {
                item: lower as u32,
                ..Node::default()
            });
            match side {
                Side::Root => {}
                Side::Left => self.nodes[parent as usize].left = node_index,
                Side::Right => self.nodes[parent as usize].right = node_index,
            }

            if upper - lower > 1 {
                // Choose an arbitrary vantage point and move it to the start.
                let i = rng.random_range(lower..upper);
                self.items.swap(lower, i);

                let vantage = self.items[lower].1;
                let median = (upper + lower) / 2;

                // Cache the vantage-to-candidate distances once, then select the median on the
                // cached scalars.
                partition.clear();
                partition.extend(
                    self.items[lower + 1..upper]
                        .iter()
                        .map(|pair @ (_, sample)| (metric_f(vantage, sample), *pair)),
                );
                let k = median - lower - 1;
                partition.select_nth_unstable_by(k, |(a, ..), (b, ..)| {
                    a.partial_cmp(b).unwrap_or(Ordering::Equal)
                });
                // Write the partitioned order back into `items`.
                for (slot, &(_, pair)) in self.items[lower + 1..upper]
                    .iter_mut()
                    .zip(partition.iter())
                {
                    *slot = pair;
                }

                // Threshold of the new node is the distance to the median.
                self.nodes[node_index as usize].threshold = metric_f(vantage, self.items[median].1);

                stack.push((lower + 1, median, node_index, Side::Left));
                stack.push((median, upper, node_index, Side::Right));
            }
        }
    }

    /// Auxiliary function that searches for the k nearest neighbors of an item, accumulating them on
    /// `heap`.
    fn look_up<F>(
        &self,
        tau: &mut T, // Tracks the distances to the farthest point in the results.
        target: &U,
        k: usize,
        heap: &mut BinaryHeap<HeapItem<T>>,
        stack: &mut Vec<u32>,
        metric_f: F,
    ) where
        F: Fn(&U, &U) -> T,
    {
        if self.nodes.is_empty() {
            return;
        }
        // Reused across searches: start each traversal from the root.
        stack.clear();
        stack.push(0);

        while let Some(node_index) = stack.pop() {
            let node = &self.nodes[node_index as usize];
            let (original_position, point) = self.items[node.item as usize];
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
                    index: original_position,
                    distance,
                });
                if heap.len() == k {
                    *tau = heap.peek().unwrap().distance;
                }
            }

            // Return if we arrived at a leaf.
            if node.left == SENTINEL && node.right == SENTINEL {
                continue;
            }

            if distance < node.threshold {
                // If the target lies within the radius of the ball, search the left (near) child
                // first; the right (far) child only if neighbors could still lie outside the ball.
                if distance + *tau >= node.threshold && node.right != SENTINEL {
                    stack.push(node.right);
                }
                if distance - *tau <= node.threshold && node.left != SENTINEL {
                    stack.push(node.left);
                }
            } else {
                // If the target lies outside the radius of the ball, search the right (far) child
                // first; the left (near) child only if neighbors could still lie inside the ball.
                if distance - *tau <= node.threshold && node.left != SENTINEL {
                    stack.push(node.left);
                }
                if distance + *tau >= node.threshold && node.right != SENTINEL {
                    stack.push(node.right);
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
    /// * `target_index` - index of the target.
    ///
    /// * `k` - number of nearest neighbors.
    ///
    /// * `out` - the paired output rows the result is written into: the nearest neighbor indices and
    ///   their distances, in ascending distance order.
    ///
    /// * `metric_f` - metric function.
    pub fn search<F>(
        &self,
        target: &U,
        target_index: usize,
        k: usize,
        out: (&mut [u32], &mut [T]),
        scratch: &mut SearchScratch<T>,
        metric_f: F,
    ) where
        F: Fn(&U, &U) -> T,
    {
        let (neighbors_indices, distances) = out;
        debug_assert_eq!(neighbors_indices.len(), distances.len());

        // Reuse the per-worker priority queue to store intermediate results on.
        scratch.heap.clear();
        // Perform the search.
        self.look_up(
            &mut T::max_value(),
            target,
            k,
            &mut scratch.heap,
            &mut scratch.stack,
            metric_f,
        );
        scratch.results.clear();
        while let Some(item) = scratch.heap.pop() {
            scratch.results.push(item);
        }

        // Avoid the target itself.
        neighbors_indices
            .iter_mut()
            .zip(distances.iter_mut())
            .zip(
                scratch
                    .results
                    .iter()
                    .rev()
                    .filter(|result| target_index != result.index),
            )
            .for_each(|((idx, d), result)| {
                *idx = result.index as u32;
                *d = result.distance;
            });
    }
}
