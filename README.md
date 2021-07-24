<div align="center"> <h1 align="center"> bhtsne </h1> </div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gethseman](https://circleci.com/gh/frjnn/bhtsne.svg?style=shield)](https://app.circleci.com/pipelines/github/frjnn/bhtsne)
[![codecov](https://codecov.io/gh/frjnn/bhtsne/branch/master/graph/badge.svg)](https://codecov.io/gh/frjnn/bhtsne)

</div>


Barnes-Hut and vanilla implementations of the t-SNE algorithm written in Rust. The tree-accelerated version of the algorithm is described with fine detail in [this paper](http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf) by [Laurens van der Maaten](https://github.com/lvdmaaten). The vanilla version of the algorithm is described in [this other paper](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) by G. Hinton and Laurens van der Maaten.
Additional implementations of the algorithm, including this one, are listed at [this page](http://lvdmaaten.github.io/tsne/).

## Usage 

Add this line to your `Cargo.toml`:
```toml
[dependencies]
bhtsne = "0.4.0"
```
### Documentation

The docs are available on [crates.io](https://crates.io/crates/bhtsne).

### Example

```rust
 use bhtsne;

 const N: usize = 150;         // Number of vectors to embed.
 const D: usize = 4;           // The dimensionality of the
                               // original space.
 const THETA: f32 = 0.5;       // Parameter used by the Barnes-Hut algorithm.
                               // When set to 0.0 the exact t-SNE version is
                               // used instead.
    
 const PERPLEXITY: f32 = 10.0; // Perplexity of the conditional distribution.
 const MAX_ITER: u64 = 2000;   // Number of fitting iterations.
 const NO_DIMS: usize = 2;     // Dimensionality of the embedded space.

 // Loads the data from a csv file skipping the first row,
 // treating it as headers and skipping the 5th column,
 // treating it as a class label.
 // Do note that you can also use f64s.
 let mut data: Vec<f32> = bhtsne::load_csv("iris.csv", true, Some(4));

 // The vector where the bi-dimensional embedding
 // will be stored.
 let mut y: Vec<f32> = vec![0.0; N * NO_DIMS];

 // Runs t-SNE.
 bhtsne::run(
     &mut data, N, D, &mut y, NO_DIMS, PERPLEXITY, THETA, false, MAX_ITER, 250, 250,
 );

 // Writes the embedding to a csv file.
 bhtsne::write_csv("iris_embedding.csv", y, NO_DIMS);
```


## MNIST embedding
The following embedding has been obtained by preprocessing the MNIST dataset using PCA to reduce its 
dimensionality to 50. It took approximately 17 mins on a 2.0GHz quad-core 10th-generation i5 MacBook Pro.
![mnist](imgs/mnist_embedding.png) 
