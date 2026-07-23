<div align="center"> <h1 align="center"> barnes-hut-tree </h1> </div>

<div align="center">

[![CI](https://github.com/frjnn/bhtsne/actions/workflows/ci.yml/badge.svg)](https://github.com/frjnn/bhtsne/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/barnes-hut-tree.svg)](https://crates.io/crates/barnes-hut-tree)
[![docs.rs](https://docs.rs/barnes-hut-tree/badge.svg)](https://docs.rs/barnes-hut-tree)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>


Morton (Z-order) linear tree in a contiguous arena for spatial force approximation, the structure the Barnes-Hut algorithm walks to summarize repulsive forces from distant points. Companion crate to [`bhtsne`](https://crates.io/crates/bhtsne), published standalone so any other force-approximation problem can reuse it without the tSNE optimizer.

Embedding dimensionality `D` runs 2 to 7. The `Morton<D>` trait picks the code word type per `D`: `Dim<2>`, `Dim<3>`, and `Dim<4>` interleave into a `u64` (32, 21, and 16 bits per axis), while `Dim<5>`, `Dim<6>`, and `Dim<7>` use a `u128` (25, 21, and 18 bits per axis).

`BarnesHutTree<T, W, D>` owns the flat node buffer, the sorted `(code, index)` permutation, and the per-level squared half-widths the theta test compares against. Rebuilding it in place reuses those buffers, so a long fit allocates only on the first rebuild. Each rebuild quantizes the points into per-axis integer coordinates, interleaves them into Morton codes, sorts the `(code, index)` permutation in parallel, and walks the sorted codes breadth first to emit cells whose children sit contiguously in the arena. The traversal reads `center_of_mass`, `mass`, and `level` per cell, and follows `first_child` when the theta test rejects a summary.

## Installation

Add this line to your `Cargo.toml`:
```toml
[dependencies]
barnes-hut-tree = "0.1"
```

## Example

```rust
use barnes_hut_tree::{BarnesHutTree, Dim, Morton};

// Build a 3D arena over four points.
let embedding: Vec<f32> = vec![
    0.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
];
let tree: BarnesHutTree<f32, <Dim<3> as Morton<3>>::Word, 3> =
    BarnesHutTree::new_uniform(&embedding);

// Approximate the repulsive force on point 0 and its contribution to the Q normalizer.
let theta_sq = 0.5f32 * 0.5;
let mut stack = <Dim<3> as Morton<3>>::empty_stack();
let mut forces = [0.0f32; 3];
let mut q_sum = 0.0f32;
tree.compute_non_edge_forces(
    0,
    theta_sq,
    &embedding,
    &mut forces,
    &mut q_sum,
    stack.as_mut(),
);
```

## Parallelism and `no_std`

The default `parallel` feature runs the arena build and force reductions on [rayon](https://github.com/rayon-rs/rayon). Turning it off swaps in a sequential fallback with the same results up to floating-point summation order. Dropping the default `std` feature makes the crate `no_std` + `alloc` for bare metal. The tree only needs `num_traits::float::FloatCore`, so no math backend is required:

```toml
[dependencies]
barnes-hut-tree = { version = "0.1", default-features = false }
```

## License

Licensed under the [MIT License](https://opensource.org/licenses/MIT).
