# Changelog

## 0.1.1

`no_std` + `alloc` support. `rayon` is now the optional, default-on `parallel` feature, with a sequential fallback when it is off (identical results up to floating-point summation order). The float bound is relaxed from `num_traits::Float` to `num_traits::float::FloatCore`, which the tree's arithmetic fits entirely, so dropping the default `std` feature builds the crate on bare metal with no math backend and `--no-default-features` alone compiles.

## 0.1.0

Initial extraction from the `bhtsne` crate into a reusable Barnes-Hut primitive.

`BarnesHutTree<T, W, D>` is a Morton (Z-order) linear tree in a contiguous arena. `D` runs 2 to 7 (`u64` codes for `D <= 4`, `u128` for `D >= 5`), breadth-first build with tightest-enclosing-level jumps, iterative theta-criterion traversal, heap-free reusable scratch stack per dimensionality.

Every cell carries a summed `mass: T` alongside its point count. `new(points, masses)` and `rebuild(points, masses)` populate it from caller-supplied masses. `new_uniform(points)` and `rebuild_uniform(points)` use `T::from(count)`. `compute_non_edge_forces` is the t-SNE specialisation, accumulating the `q_sum` normaliser next to the force.

A `ForceKernel<T, const D: usize>` trait, blanket-implemented for `Fn(&[T; D], T, T, bool, &mut [T; D])`, drives `accumulate_forces` and `accumulate_all` on top of the same traversal.
