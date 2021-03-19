# bhtsne

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gethseman](https://circleci.com/gh/frjnn/bhtsne.svg?style=shield)](https://app.circleci.com/pipelines/github/frjnn/bhtsne)
[![codecov](https://codecov.io/gh/frjnn/bhtsne/branch/master/graph/badge.svg)](https://codecov.io/gh/frjnn/bhtsne)

Barnes-Hut implementation of t-SNE written in Rust. The algorithm is described with fine detail in [this paper](http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf) by Laurens van der Maaten.

## Installation 

Add this line to your `Cargo.toml`:
```toml
[dependencies]
bhtsne = "0.3.1"
```

## Example

```rust
use bhtsne;

// Parameters template.
let n: usize = 8;     // Number of vectors to embed.
let d: usize = 4;     // The dimensionality of the original space.
let theta: f32 = 0.5; // The parameter used by the Barnes-Hut algorithm. When set to 0.0
                      // the exact t-SNE version is used instead.
   
let perplexity = 1.0; // The perplexity of the conditional distribution.
let max_iter = 2000;  // The number of fitting iterations.
let no_dims = 2;      // The dimensionality of the embedded space.

// Loads data the from a csv file skipping the first row,
// treating it as headers and skipping the 5th column,
// treating it as a class label.
let mut data: Vec<f32> = bhtsne::load_csv("data.csv", true, Some(4));
// This is the vector for the resulting embedding.
let mut y: Vec<f32> = vec![0.0; n * no_dims];
// Runs t-SNE.
bhtsne::run(
    &mut data, n, d, &mut y, no_dims, perplexity, theta, false, max_iter, 250, 250,
);
// Writes the embedding to a csv file.
bhtsne::write_csv("embedding.csv", y, no_dims);
```
Also check the docs available on [crates.io](https://crates.io/crates/bhtsne).

## MNIST embedding
The following embedding has been obtained by preprocessing the MNIST dataset using PCA to reduce its 
dimensionality to 50. It took approximately 17 mins on a 2.0GHz quad-core 10th-generation i5 MacBook Pro.
![mnist](imgs/mnist_embedding.png) 
