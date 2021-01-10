# bhtsne

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gethseman](https://circleci.com/gh/frjnn/bhtsne.svg?style=shield)](https://app.circleci.com/pipelines/github/frjnn/bhtsne)
[![codecov](https://codecov.io/gh/frjnn/bhtsne/branch/master/graph/badge.svg)](https://codecov.io/gh/frjnn/bhtsne)

Barnes-Hut implementation of t-SNE written in Rust. The algorithm is described with fine detail in [this paper](http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf) by Laurens van der Maaten.

## Installation 

Be sure that your `Cargo.toml` looks somewhat like this:
```toml
[dependencies]
bhtsne = "0.2.1"
```

## Examples

A small but exhaustive example is provided in the docs available on [crates.io](https://crates.io/crates/bhtsne).
