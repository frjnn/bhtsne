# Changelog

## 0.7.11

Add 5D support for the Barnes-Hut path: `barnes_hut` and `barnes_hut_with_neighbors` now accept an embedding dimensionality `D` of 5 (in addition to 2, 3, and 4). The Morton codec uses 25 bits per axis at `D = 5`, with a `u128` Z-order code (125 bits used), so points closer than `1 / 33554432` of the bounding-box width on any axis collapse into the same leaf cell.

## 0.7.10
Add 4D support for the Barnes-Hut path: `barnes_hut` and `barnes_hut_with_neighbors` now accept an embedding dimensionality `D` of 4 (in addition to 2 and 3). The Morton codec uses 16 bits per axis at `D = 4`, so points closer than `1 / 65536` of the bounding-box width on any axis collapse into the same leaf cell.

## 0.7.4

Add `tSNE::fit_sne` and `tSNE::fit_sne_with_neighbors`, an FFT-accelerated, interpolation-based fitting path (the FIt-SNE method of Linderman et al., 2019, the same algorithm openTSNE defaults to). It builds the sparse affinity graph exactly as `barnes_hut` does, sharing the vantage point tree, the per-point Gaussian bandwidth search, the symmetrization, and the affinity cache, but approximates the repulsive forces in `O(n)` per epoch on a coarse equispaced grid rather than over a space-partitioning tree, and carries no `theta` accuracy knob. The repulsive force and the `Q` normalizer are both recovered from one convolution with the squared Cauchy kernel via the `D + 2` term identity (charges `[1, y_1, ..., y_D, ||y||^2]`), matching openTSNE's formulation. Both produce the `negative_forces` and `Z` the Barnes-Hut loop expects, so the two repulsion strategies feed the very same fused gradient-descent update, which is now factored into a shared `gradient_descent_step`; the affinity construction is shared through a `build_affinities` helper.

## 0.6.0

Make the embedding space dimensionality a const generic, `tSNE<T, U, const D: usize>` (default 2), replacing the `embedding_dim` builder: choose it as the type parameter, for example `tSNE::<f32, &[f32], 3>::new(..)`. Cells now store their coordinates as inline `[T; D]` arrays. The Barnes-Hut optimization loop is parallelized and several times faster per epoch for both `f32` and `f64`, about 4x at 2000 points and more as the point count grows, by fusing the gradient into the gradient-descent update and reducing the Q term with a barrier-free sequential sum.

The Barnes-Hut tree is then a Morton (Z-order) linear quadtree built in one contiguous arena, which replaces the recursive pointer tree and makes the optimization loop both faster and leaner at large `n`. Per epoch the repulsive pass is about 2x faster at 10k and 100k points and about 3.6x faster at 1M on 64 threads, and peak resident memory drops by roughly a third, while the 10-NN same-label quality and `kl_divergence` match the recursive tree within stochastic noise. Each epoch quantizes the embedding into per-axis integer coordinates, interleaves them into `u64` codes, sorts a `(code, index)` permutation in parallel, and walks the sorted codes to emit cells whose children sit contiguously in the arena, reached through a `first_child` index rather than pointers. The repulsive traversal is iterative over a fixed-size stack array so it never touches the heap. Point conservation is automatic because Morton quantization is total, so the build asserts that the leaf masses sum to `n`. The tree-accelerated `barnes_hut` and `barnes_hut_with_neighbors` are restricted to embedding dimensionality `D` of 2 or 3 through a trait bound, the two dimensionalities a `u64` code covers with ample precision (32 bits per axis at `D` of 2, lossless for `f32`, and 21 bits per axis at `D` of 3), and `kl_divergence` after a Barnes-Hut fit carries the same bound. The exact `exact` path stays general for any `D`. Neighbor indices (`p_columns`) are stored as `u32` instead of `usize`, the bulk of the memory saving at 1M. Cross-thread determinism is relaxed (unstable sort, plain parallel reductions), but correctness is not: any thread schedule yields a valid embedding.

## 0.5.11

Store the working buffers as dense `Vec<T>` instead of wrapping every scalar in `CachePadded`, which had put each value on its own 128 byte cache line and blocked vectorization. The exact gradient loop is several times faster and the Barnes-Hut path is faster as well, for both `f32` and `f64`, with bitwise identical embeddings. The Barnes-Hut force loop now accumulates into thread local scratch and writes once to avoid false sharing. Drops the `crossbeam` dependency and adds a criterion benchmark of the optimization loop.

## 0.5.9

Add `tSNE::barnes_hut_with_neighbors`, an index-accelerated entry point that takes caller-supplied nearest neighbors instead of building a vantage point tree, plus the public `Neighbor` struct.

## 0.5.3

Bump dependencies [(#13)](https://github.com/frjnn/bhtsne/pull/13)


## 0.5.2

Fix index out of bounds in `symmetrize_sparse_matrix` [(#12)](https://github.com/frjnn/bhtsne/pull/12)