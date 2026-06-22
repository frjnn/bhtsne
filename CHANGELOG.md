# Changelog

## 0.5.11

Store the working buffers as dense `Vec<T>` instead of wrapping every scalar in `CachePadded`, which had put each value on its own 128 byte cache line and blocked vectorization. The exact gradient loop is several times faster and the Barnes-Hut path is faster as well, for both `f32` and `f64`, with bitwise identical embeddings. The Barnes-Hut force loop now accumulates into thread local scratch and writes once to avoid false sharing. Drops the `crossbeam` dependency and adds a criterion benchmark of the optimization loop.

## 0.5.9

Add `tSNE::barnes_hut_with_neighbors`, an index-accelerated entry point that takes caller-supplied nearest neighbors instead of building a vantage point tree, plus the public `Neighbor` struct.

## 0.5.3

Bump dependencies [(#13)](https://github.com/frjnn/bhtsne/pull/13)


## 0.5.2

Fix index out of bounds in `symmetrize_spare_matrix` [(#12)](https://github.com/frjnn/bhtsne/pull/12)