use std::{
    error::Error,
    fs::File,
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use num_traits::Float;

use crate::tSNE;

impl<'data, T, U, const D: usize> tSNE<'data, T, U, D>
where
    T: Send + Sync + Float + Sum + DivAssign + MulAssign + AddAssign + SubAssign,
    U: Send + Sync,
{
    /// Writes the embedding to a csv file. If the embedding space dimensionality is either equal to
    /// 2 or 3 the resulting csv file will have some simple headers:
    ///
    /// * x, y for 2 dimensions.
    ///
    /// * x, y, z for 3 dimensions.
    ///
    /// # Arguments
    ///
    /// * `file_path` - path of the file to write the embedding to.
    ///
    /// # Errors
    ///
    /// Returns an error is something goes wrong during the I/O operations.
    pub fn write_csv(&mut self, path: &str) -> Result<&mut Self, Box<dyn Error>>
    where
        T: Float + ToString,
    {
        let mut writer = csv::Writer::from_path(path)?;

        // String-ify the embedding.
        let to_write = self
            .y
            .iter()
            .map(|&el| el.to_string())
            .collect::<Vec<String>>();

        // Write headers.
        match D {
            2 => writer.write_record(["x", "y"])?,
            3 => writer.write_record(["x", "y", "z"])?,
            _ => (), // Write no headers for embedding dimensions greater that 3.
        }
        // Write records.
        for record in to_write.chunks(D) {
            writer.write_record(record)?
        }
        // Final flush.
        writer.flush()?;

        // Everything went smooth.
        Ok(self)
    }
}

/// Loads data from a csv file.
///
/// # Arguments
///
/// * `file_path` - path of the file to load the data from.
///
/// * `has_headers` - whether the file has headers or not. if set to `true` the function will
///   not parse the first line of the csv file.
///
/// * `skip` - an optional slice that specifies a subset of the file columns that must not be
///   parsed.
///
/// * `f` - function that converts [`String`] into a data sample. It takes as an argument a single
///   record field.
///
/// # Errors
///
/// Returns an error is something goes wrong during the I/O operations.
pub fn load_csv<T, F>(
    path: &str,
    has_headers: bool,
    skip: Option<&[usize]>,
    f: F,
) -> Result<Vec<T>, Box<dyn Error>>
where
    F: Fn(String) -> T,
{
    let mut data: Vec<T> = Vec::new();

    let file = File::open(path)?;

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(has_headers)
        .from_reader(file);

    match skip {
        Some(range) => {
            for result in reader.records() {
                let record = result?;

                (0..record.len())
                    .filter(|column| !range.contains(column))
                    .for_each(|field| data.push(f(record.get(field).unwrap().to_string())));
            }
        }
        None => {
            for result in reader.records() {
                let record = result?;

                (0..record.len())
                    .for_each(|field| data.push(f(record.get(field).unwrap().to_string())));
            }
        }
    }

    Ok(data)
}
