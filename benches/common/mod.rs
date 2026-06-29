use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use num_traits::{AsPrimitive, Float};

use bhtsne::{FftNum, Neighbor};

/// The scalar bound bundle the t-SNE solver requires, satisfied by both floats.
pub trait Scalar:
    Float
    + FftNum
    + Send
    + Sync
    + AsPrimitive<usize>
    + Sum
    + DivAssign
    + AddAssign
    + MulAssign
    + SubAssign
{
}

impl<T> Scalar for T where
    T: Float
        + FftNum
        + Send
        + Sync
        + AsPrimitive<usize>
        + Sum
        + DivAssign
        + AddAssign
        + MulAssign
        + SubAssign
{
}

/// Casts an `f64` constant into the scalar in use.
pub fn cast<T: Float>(x: f64) -> T {
    T::from(x).expect("should convert to float")
}

/// Deterministic pseudo random samples, reproducible and RNG-free.
pub fn lcg<T: Scalar>(n: usize, dim: usize, mut state: u64) -> Vec<T> {
    let mut data = Vec::with_capacity(n * dim);
    for _ in 0..n * dim {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        data.push(cast((state >> 33) as f64 / u32::MAX as f64 - 0.5));
    }

    data
}

/// Squared euclidean distance between two samples.
pub fn sq_euclidean<T: Scalar>(x: &[T], y: &[T]) -> T {
    x.iter()
        .zip(y)
        .map(|(xi, yi)| (*xi - *yi).powi(2))
        .sum::<T>()
}

/// Euclidean distance between two samples.
pub fn euclidean<T: Scalar>(a: &[T], b: &[T]) -> T {
    sq_euclidean(a, b).sqrt()
}

/// Exact k nearest neighbors per sample, ascending distance, excluding self.
/// Built once, outside the timed section.
pub fn brute_force_neighbors<T: Scalar>(samples: &[&[T]], k: usize) -> Vec<Vec<Neighbor<T>>> {
    (0..samples.len())
        .map(|i| {
            let mut distances: Vec<(usize, T)> = (0..samples.len())
                .filter(|&j| j != i)
                .map(|j| (j, euclidean(samples[i], samples[j])))
                .collect();
            distances
                .sort_by(|(_, a), (_, b)| a.partial_cmp(b).expect("distance should not be NaN"));
            distances
                .into_iter()
                .take(k)
                .map(|(index, distance)| Neighbor { index, distance })
                .collect()
        })
        .collect()
}
