//! # bhtsne
//!
//! `bhtsne` contains the implementations of the
//! vanilla version of t-SNE and the Barnes-Hut optimization.
//! 
//! At the moment the only supported metric is the **euclidean distance**.
//!
//!
//! # Example
//!
//! ```
//! use bhtsne;
//!
//! // Parameters template.
//! let n: usize = 150;     // Number of vectors to embed.
//! let d: usize = 4;       // The dimensionality of the original space.
//! let theta: f32 = 0.5;   // The parameter used by the Barnes-Hut algorithm. When set to 0.0
//!                         // the exact t-SNE version is used instead.
//!    
//! let perplexity = 10.0;     // The perplexity of the conditional distribution.
//! let max_iter: u64 = 2000;  // The number of fitting iterations.
//! let no_dims: usize = 2;    // The dimensionality of the embedded space.
//!
//! // Loads data the from a csv file skipping the first row,
//! // treating it as headers and skipping the 5th column,
//! // treating it as a class label.
//! let mut data: Vec<f32> = bhtsne::load_csv("iris.csv", true, Some(4));
//!
//! // This is the vector for the resulting embedding.
//! let mut y: Vec<f32> = vec![0.0; n * no_dims];
//!
//! // Runs t-SNE.
//! bhtsne::run(
//!     &mut data, n, d, &mut y, no_dims, perplexity, theta, false, max_iter, 250, 250,
//! );
//!
//! // Writes the embedding to a csv file.
//! bhtsne::write_csv("iris_embedding.csv", y, no_dims);
//! ```
mod tsne;
pub(crate) use num_traits::{cast::AsPrimitive, cast::NumCast, Float};
use rand_distr::{Distribution, Normal};
use tsne::*;

#[cfg(feature = "csv")]
use std::fs::File;

/// Performs t-SNE.
///
/// # Arguments
/// * `x` - `&mut [T]` containing the data to embed. `T` must implement `num_traits::Float`.
///
/// * `n`- `usize` that represents the number of `d`-dimensional points in `x`.
///
/// * `d` - `usize` that represents the dimension of each point.
///
/// * `y` - `&mut [T]` where to store the resultant embedding.
///
/// * `no_dims` - `usize` that represents the dimension of the embedded points.
///
/// * `perplexity` - `T` representing perplexity value.
///
/// * `theta` - `T` representing the theta value. If `theta` is set to `0.0` the __exact__ version of t-SNE will be used,
/// if set to greater values the __Barnes-Hut__ version will be used instead.
///
/// * `skip_random_init` - `bool` stating whether to skip the random initialization of `y`.
/// In most cases it should be set to `false`.
///
/// * `max_iter` -` u64` indicating the maximum number of fitting iterations.
///
/// * `stop_lying_iter` - `u64` indicating the number of the iteration after which the P distribution
/// values become _“true“_.
///
/// * `mom_switch_iter` - `u64` indicating the number of the iteration after which the momentum's value is
/// set to its final value.
pub fn run<T>(
    x: &mut [T],
    n: usize,
    d: usize,
    y: &mut [T],
    no_dims: usize,
    perplexity: T,
    theta: T,
    skip_random_init: bool,
    max_iter: u64,
    stop_lying_iter: u64,
    mom_switch_iter: u64,
) where
    T: Float
        + NumCast
        + AsPrimitive<usize>
        + std::fmt::Display
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::iter::Sum,
{
    // Determine whether we are using the exact algorithm.
    if n - 1 < 3 * perplexity.as_() {
        panic!(
            "error: perplexity= {} too large for the number of data points.\n",
            perplexity
        );
    }
    let exact: bool = theta == T::zero();

    // Set learning parameters.
    let mut momentum: T = T::from(0.5).unwrap();
    let final_momentum: T = T::from(0.8).unwrap();
    let eta: T = T::from(200.0).unwrap();

    let mut dy: Vec<T> = vec![T::zero(); n * no_dims];
    let mut uy: Vec<T> = vec![T::zero(); n * no_dims];
    let mut gains: Vec<T> = vec![T::one(); n * no_dims];

    // Normalizing input data to prevent numerical problems.
    zero_mean(x, n, d);

    let mut max_x: T = T::zero();
    for el in &x[..] {
        let el_abs = el.abs();
        if el_abs > max_x {
            max_x = el_abs;
        }
    }
    for el in &mut x[..] {
        *el /= max_x;
    }

    let mut p: Vec<T> = Vec::new();
    let mut row_p: Vec<usize> = Vec::new();
    let mut col_p: Vec<usize> = Vec::new();
    let mut val_p: Vec<T> = Vec::new();

    if exact {
        p = vec![T::zero(); n * n];
        // Compute similarities.
        compute_fixed_gaussian_perplexity(x, n, d, &mut p, perplexity);

        // Symmetrize input similarities.
        let mut nn: usize = 0;
        for _n in 0..n {
            let mut mn: usize = (_n + 1) * n;
            for m in (_n + 1)..n {
                let _p = p[mn + _n];
                p[nn + m] += _p;
                p[mn + _n] = p[nn + m];
                mn += n;
            }
            nn += n;
        }

        let mut sum_p: T = T::zero();
        for el in &p[..] {
            sum_p += *el;
        }
        for el in &mut p[..] {
            *el /= sum_p;
        }
    } else {
        let k: usize = (T::from(3.0).unwrap() * perplexity).as_();
        row_p = vec![0; n + 1];
        col_p = vec![0; n * k];
        val_p = vec![T::zero(); n * k];
        // Compute input similarities for approximate algorithm.
        compute_gaussian_perplexity(x, n, d, &mut row_p, &mut col_p, &mut val_p, perplexity, k);

        symmetrize_matrix(&mut row_p, &mut col_p, &mut val_p, n);

        let mut sum_p: T = T::zero();

        for el in val_p.iter().take(row_p[n]) {
            sum_p += *el;
        }
        for el in val_p.iter_mut().take(row_p[n]) {
            *el /= sum_p;
        }
    }

    // Lie about the P values.
    if exact {
        for el in &mut p[..] {
            *el *= T::from(12.0).unwrap();
        }
    } else {
        for el in val_p.iter_mut().take(row_p[n]) {
            *el *= T::from(12.0).unwrap();
        }
    }

    // Initialize solution randomly.
    if !skip_random_init {
        // Generates a Gaussian random number.
        let n_distr = Normal::new(0.0, 1e-4).unwrap();

        for el in &mut y[..] {
            *el = T::from(n_distr.sample(&mut rand::thread_rng()) * 0.0001).unwrap();
        }
    }

    // Main training loop.
    for iter in 0..max_iter {
        // Compute (approximate) gradient.
        if exact {
            compute_gradient(&mut p, y, n, no_dims, &mut dy);
        } else {
            compute_gradient_approx(
                &mut row_p, &mut col_p, &mut val_p, y, n, no_dims, &mut dy, theta,
            );
        }

        // Update gains.
        for (i, el) in gains.iter_mut().enumerate() {
            *el = if dy[i].signum() != uy[i].signum() {
                *el + T::from(0.2).unwrap()
            } else {
                *el * T::from(0.8).unwrap()
            }
        }
        for el in &mut gains[..] {
            if *el < T::from(0.01).unwrap() {
                *el = T::from(0.01).unwrap();
            }
        }

        // Perform gradient update with momentum and gains.
        for i in 0..(n * no_dims) {
            uy[i] = momentum * uy[i] - eta * gains[i] * dy[i];
        }
        for i in 0..(n * no_dims) {
            y[i] += uy[i];
        }

        // Make solution zero mean.
        zero_mean(y, n, no_dims);

        // Stop lying about the P-values after a while, and switch momentum.
        if iter == stop_lying_iter {
            if exact {
                for el in &mut p[..] {
                    *el /= T::from(12.0).unwrap();
                }
            } else {
                for el in val_p.iter_mut().take(row_p[n]) {
                    *el /= T::from(12.0).unwrap();
                }
            }
        }
        if iter == mom_switch_iter {
            momentum = final_momentum;
        }
    }
}

/// Loads data from a csv file.
///
/// # Arguments
///
/// * `file_path` -`&str` that specifies the path of the file to load the data from.
///
/// * `has_headers` - `bool` value that specifies whether the file has headers or not. if `has_headers`
/// is set to `true` the function will not parse the first line of the .csv file.
///
/// * `skip_col` - `Option<usize>` that may specify a column of the file that must not be parsed.
#[cfg(feature = "csv")]
pub fn load_csv<T>(file_path: &str, has_headers: bool, skip_col: Option<usize>) -> Vec<T>
where
    T: Float + std::str::FromStr + std::fmt::Debug,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    // Declaring the vectors where we'll put the parsed data.
    let mut data: Vec<T> = Vec::new();
    // Opening the file.
    let file = match File::open(file_path) {
        Ok(file) => file,
        Err(e) => panic!("error: tsne couldn't open the .csv file, {}", e),
    };
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(has_headers)
        .from_reader(file);

    match skip_col {
        Some(tc) => {
            for result in rdr.records() {
                let record = match result {
                    Ok(res) => res,
                    Err(e) => panic!("error: while parsing records, {}", e),
                };
                for i in 0..record.len() {
                    if i != tc {
                        data.push(record.get(i).unwrap().parse().unwrap())
                    }
                }
            }
        }
        None => {
            for result in rdr.records() {
                let record = match result {
                    Ok(res) => res,
                    Err(e) => panic!("error: while parsing records, {}", e),
                };
                for i in 0..record.len() {
                    data.push(record.get(i).unwrap().parse().unwrap())
                }
            }
        }
    }
    data
}

/// Writes the embedding to a csv file.
///
/// # Arguments
///
/// * `file_path` - `&str` that specifies the path of the file to write the embedding to.
///
/// * `embedding`- `&mut [T]` where the embedding is stored. `T` must implement `num_traits::Float`.
///
/// * `dims` - `usize` representing the dimension of the embedding's space. If the emdedding's space has more than three or less than two dimensions
/// the resultant file won't have headers.
#[cfg(feature = "csv")]
pub fn write_csv<T>(file_path: &str, embedding: Vec<T>, dims: usize)
where
    T: Float + std::fmt::Display,
{
    let mut wtr: csv::Writer<File> = match csv::Writer::from_path(file_path) {
        Ok(writer) => writer,
        Err(e) => panic!("error: during the opening of the file : {}", e),
    };
    // String-ify the embedding.
    let to_write: Vec<String> = embedding
        .iter()
        .map(|el| el.to_string())
        .collect::<Vec<String>>();
    // Write headers if we can.
    match dims {
        2 => match wtr.write_record(&["x", "y"]) {
            Ok(_) => (),
            Err(e) => panic!("error: during write, {}", e),
        },
        3 => match wtr.write_record(&["x", "y", "z"]) {
            Ok(_) => (),
            Err(e) => panic!("error: during write, {}", e),
        },
        _ => (),
    }
    // Write records.
    for chunk in to_write.chunks(dims) {
        match wtr.write_record(chunk) {
            Ok(_) => (),
            Err(e) => panic!("error: during write, {}", e),
        }
    }
    // Final flush.
    match wtr.flush() {
        Ok(_) => (),
        Err(e) => panic!("error: during flush, {}", e),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(not(tarpaulin_include))]
    fn test_run() {
        let n = 150;
        let d = 4;
        let theta = 0.5;
        let perplexity = 1.0;
        let max_iter = 2000;
        let no_dims = 2;

        let mut data: Vec<f32> = super::load_csv("iris.csv", true, Some(4));
        let mut y: Vec<f32> = vec![0.0; n * no_dims];

        super::run(
            &mut data, n, d, &mut y, no_dims, perplexity, theta, false, max_iter, 250, 250,
        );

        super::run(
            &mut data, n, d, &mut y, no_dims, perplexity, 0.0, false, max_iter, 250, 250,
        );

        let mut data: Vec<f64> = super::load_csv("iris.csv", true, Some(4));
        let mut y: Vec<f64> = vec![0.0; n * no_dims];

        super::run(
            &mut data,
            n,
            d,
            &mut y,
            no_dims,
            perplexity as f64,
            theta as f64,
            false,
            max_iter,
            250,
            250,
        );

        super::run(
            &mut data,
            n,
            d,
            &mut y,
            no_dims,
            perplexity as f64,
            0.0,
            false,
            max_iter,
            250,
            250,
        );

        super::write_csv("iris_embedding.csv", y, no_dims);
    }
}
