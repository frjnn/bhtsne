<div align="center"> <h1 align="center"> bhtsne </h1> </div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/frjnn/bhtsne/branch/master/graph/badge.svg)](https://codecov.io/gh/frjnn/bhtsne)

</div>


Parallel Barnes-Hut and exact implementations of the t-SNE algorithm written in Rust. The tree-accelerated version of the algorithm is described with fine detail in [this paper](http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf) by [Laurens van der Maaten](https://github.com/lvdmaaten). The exact, original, version of the algorithm is described in [this other paper](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) by [G. Hinton](https://www.cs.toronto.edu/~hinton/) and Laurens van der Maaten.
Additional implementations of the algorithm, including this one, are listed at [this page](http://lvdmaaten.github.io/tsne/).

## Installation 

Add this line to your `Cargo.toml`:
```toml
[dependencies]
bhtsne = "0.7.3"
```
### Documentation

The API documentation is available [here](https://docs.rs/bhtsne).

### Example

The implementation supports custom data types and custom defined metrics. For instance, general vector data can be handled in the following way.

```rust
use bhtsne;

const N: usize = 150;         // Number of vectors to embed.
const D: usize = 4;           // The dimensionality of the
                              // original space.
const THETA: f32 = 0.5;       // Parameter used by the Barnes-Hut algorithm.
                              // Small values improve accuracy but increase complexity.

const PERPLEXITY: f32 = 10.0; // Perplexity of the conditional distribution.
const EPOCHS: usize = 2000;   // Number of fitting iterations.

// Loads the data from a csv file skipping the first row,
// treating it as headers and skipping the 5th column,
// treating it as a class label.
// Do note that you can also switch to f64s for higher precision.
let data: Vec<f32> = bhtsne::load_csv("iris.csv", true, Some(&[4]), |float| {
    float.parse().unwrap()
})?;
let samples: Vec<&[f32]> = data.chunks(D).collect();

// Executes the Barnes-Hut approximation of the algorithm and writes the embedding to the
// specified csv file.
bhtsne::tSNE::<f32, &[f32], 2>::new(&samples)
    .perplexity(PERPLEXITY)
    .epochs(EPOCHS)
    .barnes_hut(THETA, |sample_a, sample_b| {
        sample_a
            .iter()
            .zip(sample_b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    })
    .write_csv("iris_embedding.csv")?;
```

In the example euclidean distance is used, but any other distance metric on data types of choice, such as strings, can be defined and plugged in.

The tree-accelerated `barnes_hut` and `barnes_hut_with_neighbors` support an embedding dimensionality `D` of 2 or 3, the dimensionalities a `u64` Z-order code covers with ample precision. The exact `exact` path stays general for any `D`.

## FIt-SNE

For larger datasets the crate also provides `fit_sne` and `fit_sne_with_neighbors`, an FFT-accelerated, interpolation-based fitting path (the [FIt-SNE](https://www.nature.com/articles/s41592-018-0308-4) method of Linderman et al., the same algorithm [openTSNE](https://github.com/pavlin-policar/openTSNE) defaults to). They build the affinity graph exactly like `barnes_hut` (same vantage point tree, bandwidth search, symmetrization, and affinity cache) but approximate the repulsive forces in `O(n)` per epoch on an equispaced grid via a real FFT convolution, with no `theta` knob. They drop in place of `barnes_hut`:

```rust
bhtsne::tSNE::<f32, &[f32], 2>::new(&samples)
    .perplexity(PERPLEXITY)
    .epochs(EPOCHS)
    .fit_sne(|sample_a, sample_b| {
        sample_a
            .iter()
            .zip(sample_b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    })
    .write_csv("embedding.csv")?;
```

`fit_sne` is restricted to an embedding dimensionality `D` of 1 or 2 (where the interpolation grid stays tractable). Its per-epoch cost is nearly flat in the point count, so it matches the parallel Barnes-Hut path around 50k points and is faster beyond; roughly 1.3x at 70k points and widening.

## Parallelism 
Being built on [rayon](https://github.com/rayon-rs/rayon), the algorithm uses the same number of threads as the number of CPUs available. Do note that on systems with hyperthreading enabled this equals the number of logical cores and not the physical ones. See [rayon's FAQs](https://github.com/rayon-rs/rayon/blob/master/FAQ.md) for additional informations.

## MNIST embedding
The following embedding has been obtained by preprocessing the [MNIST](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/) train set using PCA to reduce its 
dimensionality to 50. It took approximately **3 minutes and 6 seconds** on a 2.0GHz quad-core 10th-generation i5 MacBook Pro. 
![mnist](imgs/mnist_embedding.png) 
