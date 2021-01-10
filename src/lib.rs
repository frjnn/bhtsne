//! # bhtsne
//!
//! `bhtsne` contains the implementations of the
//! vanilla version of t-SNE and the Barnes-Hut optimization.
//!
//!
//! # Example
//!
//! ```
//! use bhtsne;
//!
//! // Dummy variables for dummy data.
//! let n = 8;
//! let d = 4;
//! let theta = 0.5;
//! let perplexity = 1.0;
//! let max_iter = 2000;
//! let no_dims = 2;
//! // Load data and labels from a csv file. Labes are useful for plotting.
//! let (mut data, labels) = match bhtsne::load_csv("data.csv", true, true, "target", 0) {
//!     (data, None) => panic!("This is not supposed to happen!"),
//!     (data, Some(labels)) => (data, labels),
//! };
//! let mut y: Vec<f64> = vec![0.0; n * no_dims];
//! // Run t-SNE.
//! bhtsne::run(
//!     &mut data, n, d, &mut y, no_dims, perplexity, theta, false, max_iter, 250, 250,
//! );
//! // Writing the embedding to a csv file. Useful for wrappers.
//! bhtsne::write_csv("embedding.csv", y, 2);
//! ```
mod tsne;

use std::fs::File;
use std::time::Instant;
use tsne::*;

/// Performs t-SNE.
///
/// # Arguments
/// * `x` - A `&mut [f64]` containing the data to embed.
///
/// * `n`- A `usize` that represents the number of `d`-dimensional points in `x`.
///
/// * `d` - A `usize` that represents the dimension of each point.
///
/// * `y` - A `&mut [f64]` where to store the resultant embedding.
///
/// * `no_dims` -  A `usize` that represents the dimension of the embedded points.
///
/// * `perplexity` - An `f64` representing perplexity value.
///
/// * `theta` - An `f64` representing the theta value. If `theta` is set to `0.0` the __exact__ version of t-SNE will be used,
/// if set to greater values the __Barnes-Hut__ version will be used instead.
///
/// * `skip_random_init` - A `bool` stating wether to skip the random initialization of `y`.
/// In most cases it should be set to `false`.
///
/// * `max_iter` - A `u64` indicating the maximum number of iterations to perform the fitting.
///
/// * `stop_lying_iter` - A `u64` indicating the number of the iteration after which the P distribution
/// values become _“true“_.
///
/// * `mom_switch_iter` - A `u64` indicating the number of the iteration after which the momentum's value is
/// set to its final value.
pub fn run(
    x: &mut [f64],
    n: usize,
    d: usize,
    y: &mut [f64],
    no_dims: usize,
    perplexity: f64,
    theta: f64,
    skip_random_init: bool,
    max_iter: u64,
    stop_lying_iter: u64,
    mom_switch_iter: u64,
) {
    // Determine whether we are using the exact algorithm.
    if n - 1 < 3 * perplexity as usize {
        panic!(
            "Perplexity: {} too large for the number of data points!\n",
            perplexity
        );
    }
    print!("Using perplexity: {} \n", perplexity,);

    let exact: bool = if theta == 0.0 { true } else { false };

    // Set learning parameters.
    let mut momentum: f64 = 0.5;
    const FINAL_MOMENTUM: f64 = 0.8;
    const ETA: f64 = 200.0;

    let mut start: Instant;
    let mut end: Instant;

    let mut dy: Vec<f64> = vec![0.0; n * no_dims];
    let mut uy: Vec<f64> = vec![0.0; n * no_dims];
    let mut gains: Vec<f64> = vec![1.0; n * no_dims];

    print!("Computing input similarities...\n");
    start = Instant::now();

    // Normalizing input data to prevent numerical problems.
    zero_mean(x, n, d);

    let mut max_x: f64 = 0.0;
    for i in 0..(n * d) {
        if x[i].abs() > max_x {
            max_x = x[i].abs();
        }
    }
    for i in 0..(n * d) {
        x[i] /= max_x;
    }

    let mut p: Vec<f64> = Vec::new();
    let mut row_p: Vec<usize> = Vec::new();
    let mut col_p: Vec<usize> = Vec::new();
    let mut val_p: Vec<f64> = Vec::new();

    if exact {
        print!("Executing exact version.\n");

        p = vec![0.0; n * n];
        // Compute similarities.
        compute_fixed_gaussian_perplexity(x, n, d, &mut p, perplexity);

        print!("Symmetrizing matrix...\n");
        // Symmetrize input similarities.
        let mut nn: usize = 0;
        for _n in 0..n {
            let mut mn: usize = (_n + 1) * n;
            for m in (_n + 1)..n {
                p[nn + m] += p[mn + _n];
                p[mn + _n] += p[nn + m];
                mn += n;
            }
            nn += n;
        }

        let mut sum_p: f64 = 0.0;
        for i in 0..(n * n) {
            sum_p += p[i];
        }
        for i in 0..(n * n) {
            p[i] /= sum_p;
        }
    } else {
        print!("Executing Barnes-Hut version.\n");

        let k: usize = (3.0 * perplexity) as usize;
        row_p = vec![0; n + 1];
        col_p = vec![0; n * k];
        val_p = vec![0.0; n * k];
        // Compute input similarities for approximate algorithm.
        compute_gaussian_perplexity(x, n, d, &mut row_p, &mut col_p, &mut val_p, perplexity, k);

        print!("Symmetrizing matrix...\n");
        symmetrize_matrix(&mut row_p, &mut col_p, &mut val_p, n);

        let mut sum_p: f64 = 0.0;

        for i in 0..row_p[n] {
            sum_p += val_p[i];
        }
        for i in 0..row_p[n] {
            val_p[i] /= sum_p;
        }
    }
    end = Instant::now();

    // Lie about the P values.
    if exact {
        for i in 0..(n * n) {
            p[i] *= 12.0;
        }
    } else {
        for i in 0..row_p[n] {
            val_p[i] *= 12.0;
        }
    }

    // Initialize solution randomly.
    if !skip_random_init {
        print!("Sampling random solution...\n");
        for i in 0..(n * no_dims) {
            y[i] = randn() * 0.0001;
        }
    }

    if exact {
        print!(
            "Input similarities computed in {} seconds.\nLearning embedding...\n",
            end.duration_since(start).as_secs_f32()
        );
    } else {
        print!(
            "Input similarities computed in {} seconds (sparsity = {}).\nLearning embedding...\n",
            end.duration_since(start).as_secs_f32(),
            row_p[n] as f64 / (n * n) as f64
        );
    }

    let start_fitting: Instant = Instant::now();

    // Main training loop.
    for iter in 0..max_iter {
        start = Instant::now();

        // Compute (approximate) gradient.
        if exact {
            compute_gradient(&mut p, y, n, no_dims, &mut dy);
        } else {
            compute_gradient_approx(
                &mut row_p, &mut col_p, &mut val_p, y, n, no_dims, &mut dy, theta,
            );
        }

        // Update gains.
        for i in 0..(n * no_dims) {
            gains[i] = if dy[i].signum() != uy[i].signum() {
                gains[i] + 0.2
            } else {
                gains[i] * 0.8
            }
        }
        for i in 0..(n * no_dims) {
            if gains[i] < 0.01 {
                gains[i] = 0.01;
            }
        }

        // Perform gradient update with momentum and gains.
        for i in 0..(n * no_dims) {
            uy[i] = momentum * uy[i] - ETA * gains[i] * dy[i];
        }
        for i in 0..(n * no_dims) {
            y[i] += uy[i];
        }

        // Make solution zero mean.
        zero_mean(y, n, no_dims);

        // Stop lying about the P-values after a while, and switch momentum.
        if iter == stop_lying_iter {
            if exact {
                for i in 0..(n * n) {
                    p[i] /= 12.0;
                }
            } else {
                for i in 0..row_p[n] {
                    val_p[i] /= 12.0;
                }
            }
        }
        if iter == mom_switch_iter {
            momentum = FINAL_MOMENTUM;
        }

        if iter == 0 {
            let c: f64;
            if exact {
                c = evaluate_error(&mut p, y, n, no_dims);
            } else {
                // Doing approximate computation here!
                c = evaluate_error_approx(&mut row_p, &mut col_p, &mut val_p, y, n, no_dims, theta)
            }
            print!("Iteration 0, error is {}\n", c);
        }

        if iter > 0 && (iter % 50 == 0 || iter == max_iter - 1) {
            end = Instant::now();
            let c: f64;

            if exact {
                c = evaluate_error(&mut p, y, n, no_dims);
            } else {
                c = evaluate_error_approx(&mut row_p, &mut col_p, &mut val_p, y, n, no_dims, theta)
            }

            print!(
                "Iteration {}: error is  {} (50 iterations in {} seconds)\n",
                iter,
                c,
                end.duration_since(start).as_secs_f32()
            );
        }
    }
    print!(
        "Fitting performed in {} seconds.\n",
        end.duration_since(start_fitting).as_secs_f32()
    )
}

/// Loads data from a csv file.
///
/// # Arguments
///
/// * `file_path` - An `&str` that specifies the path of the file to load the data from.
///
/// * `has_headers` - A `bool` value that specifies if the file has headers or not. if `has_headers`
/// and `has_target` are set to `true` the function will locate the column
/// specified by `target_hd` and will parse all of that column's records
/// as data labels thus putting them in the second vector of the result's
/// pair; in this case `target_col` will be ignored. If `has_headers` is set
/// to `false` and `has_target` is specified, `target_col` will be used for
/// the same purpose. When the csv has no target and no headers
/// `has_headers`must be set to `false` or else the first record of the file
/// won't be parsed.
///
/// * `has_target` - A `bool` value that specifies if the file has a target column or not.
///
/// * `target_hd` - An `&str` that specifies the target header of the file.
///
/// * `target_col` - A `usize` that specifies the target column of the file.
pub fn load_csv(
    file_path: &str,
    has_headers: bool,
    has_target: bool,
    target_hd: &str,
    target_col: usize,
) -> (Vec<f64>, Option<Vec<String>>) {
    // Declaring the vectors where we'll put the parsed data.
    let mut data: Vec<f64> = Vec::new();
    let mut labels: Vec<String>;
    // Opening the file.
    let file = match File::open(file_path) {
        Ok(file) => file,
        Err(e) => panic!("Couldn't open the .csv file: {}", e),
    };
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(!has_headers)
        .from_reader(file);

    // This the variable where we'll store the target column.
    let mut tc: usize = 0;

    // If a target is present.
    if has_target {
        labels = Vec::new();
        // If the csv has headers we use the specified target header
        // to find the target column.
        if has_headers {
            let headers: csv::StringRecord = match rdr.records().next() {
                Some(hds) => match hds {
                    Ok(hds) => hds,
                    Err(e) => panic!("An error occurred while parsing headers: {}", e),
                },
                None => panic!("Error: headers not found."),
            };
            for i in 0..headers.len() {
                let header: String = headers.get(i).unwrap().parse().unwrap();
                println!("{}", header);
                if header == target_hd {
                    tc = i;
                }
            }
        } else {
            // If there are no header but there is a
            // target we use the specified target column.
            tc = target_col;
        }
        for result in rdr.records() {
            let record = match result {
                Ok(res) => res,
                Err(e) => panic!("Error while parsing records, {}", e),
            };
            for i in 0..record.len() {
                if i != tc {
                    data.push(record.get(i).unwrap().parse().unwrap())
                } else {
                    labels.push(record.get(i).unwrap().to_string());
                }
            }
        }
        (data, Some(labels))
    // If there are no headers nor a target just put everything in the data vector.
    } else {
        for result in rdr.records() {
            let record = match result {
                Ok(res) => res,
                Err(e) => panic!("Error while parsing records, {}", e),
            };
            for i in 0..record.len() {
                data.push(record.get(i).unwrap().parse().unwrap())
            }
        }
        (data, None)
    }
}

/// Writes the embedding to a csv file.
///
/// # Arguments
///
/// * `file_path` - An `&str` that specifies the path of the file to load the data from.
///
/// * `embedding`- An `&mut [f64]` where the embedding is stored.
///
/// * `dims` - A `usize` representing the dimension of the embedding's space. If the emdedding's space has more than three or less than two dimensions
/// the resultant file won't have headers.
pub fn write_csv(file_path: &str, embedding: Vec<f64>, dims: usize) {
    let mut wtr: csv::Writer<File> = match csv::Writer::from_path(file_path) {
        Ok(writer) => writer,
        Err(e) => panic!(
            "An error has occurred during the opening of the file : {}",
            e
        ),
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
            Err(e) => panic!("Error during write: {}", e),
        },
        3 => match wtr.write_record(&["x", "y", "z"]) {
            Ok(_) => (),
            Err(e) => panic!("Error during write: {}", e),
        },
        _ => println!(
            "Found more than three or less than two dimensions, {}.csv won't have an header.",
            file_path
        ),
    }
    // Write records.
    for chunk in to_write.chunks(dims) {
        match wtr.write_record(chunk) {
            Ok(_) => (),
            Err(e) => panic!("Error during write: {}", e),
        }
    }
    // Final flush.
    match wtr.flush() {
        Ok(_) => (),
        Err(e) => panic!("Couldn't write file: {}", e),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(not(tarpaulin_include))]
    fn it_works() {
        // Dummy variables for dummy data.
        let n = 8;
        let d = 4;
        let theta = 0.5;
        let perplexity = 1.0;
        let max_iter = 2000;
        let no_dims = 2;
        // Load data and labels from a csv file. Labes are useful for plotting.
        let (mut data, labels) = match super::load_csv("data.csv", true, true, "target", 0) {
            (data, None) => panic!("This is not supposed to happen!"),
            (data, Some(labels)) => (data, labels),
        };
        let mut y: Vec<f64> = vec![0.0; n * no_dims];
        // Run Barnes-Hut t-SNE.
        super::run(
            &mut data, n, d, &mut y, no_dims, perplexity, theta, false, max_iter, 250, 250,
        );
        // Run vanilla t-SNE.
        super::run(
            &mut data, n, d, &mut y, no_dims, perplexity, 0.0, false, max_iter, 250, 250,
        );
        // Writing the embedding to a csv file. Useful for wrappers.
        super::write_csv("embedding.csv", y, 2);
    }
}
