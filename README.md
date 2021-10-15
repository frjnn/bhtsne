<div align="center"> <h1 align="center"> bhtsne </h1> </div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gethseman](https://circleci.com/gh/frjnn/bhtsne.svg?style=shield)](https://app.circleci.com/pipelines/github/frjnn/bhtsne)
[![codecov](https://codecov.io/gh/frjnn/bhtsne/branch/master/graph/badge.svg)](https://codecov.io/gh/frjnn/bhtsne)

</div>


Parallel Barnes-Hut and exact implementations of the t-SNE algorithm written in Rust. The tree-accelerated version of the algorithm is described with fine detail in [this paper](http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf) by [Laurens van der Maaten](https://github.com/lvdmaaten). The vanilla version of the algorithm is described in [this other paper](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) by G. Hinton and Laurens van der Maaten.
Additional implementations of the algorithm, including this one, are listed at [this page](http://lvdmaaten.github.io/tsne/).

## Usage 

Add this line to your `Cargo.toml`:
```toml
[dependencies]
bhtsne = "0.5.0"
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
 const MAX_ITER: usize = 2000; // Number of fitting iterations.
 const NO_DIMS: u8 = 2;        // Dimensionality of the embedded space.
 const EPOCHS: usize = 2000;
 
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
 bhtsne::tSNE::new(&samples)
     .embedding_dim(NO_DIMS)
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

## MNIST embedding
The following embedding has been obtained by preprocessing the [MNIST](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/) dataset using PCA to reduce its 
dimensionality to 50.
![mnist](imgs/mnist_embedding.png) 
