# Changelog

## 0.6.0

Make the embedding space dimensionality a const generic, `tSNE<T, U, const D: usize>` (default 2), replacing the `embedding_dim` builder: choose it as the type parameter, for example `tSNE::<f32, &[f32], 3>::new(..)`. Cells now store their coordinates as inline `[T; D]` arrays. The Barnes-Hut optimization loop is parallelized and several times faster per epoch for both `f32` and `f64`, about 4x at 2000 points and more as the point count grows: the space-partitioning tree is built top-down on the thread pool (its structure is insertion-order independent), and the per-epoch fork-joins drop from five to three by fusing the gradient into the gradient-descent update and reducing the Q term with a barrier-free sequential sum. The embedding is now reproducible bit for bit across runs on the default multi-threaded pool.

## 0.5.11

Store the working buffers as dense `Vec<T>` instead of wrapping every scalar in `CachePadded`, which had put each value on its own 128 byte cache line and blocked vectorization. The exact gradient loop is several times faster and the Barnes-Hut path is faster as well, for both `f32` and `f64`, with bitwise identical embeddings. The Barnes-Hut force loop now accumulates into thread local scratch and writes once to avoid false sharing. Drops the `crossbeam` dependency and adds a criterion benchmark of the optimization loop.

## 0.5.9

Add `tSNE::barnes_hut_with_neighbors`, an index-accelerated entry point that takes caller-supplied nearest neighbors instead of building a vantage point tree, plus the public `Neighbor` struct.

## 0.5.3

Bump dependencies [(#13)](https://github.com/frjnn/bhtsne/pull/13)


## 0.5.2

Fix index out of bounds in `symmetrize_spare_matrix` [(#12)](https://github.com/frjnn/bhtsne/pull/12)