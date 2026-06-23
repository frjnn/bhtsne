//! Benchmarks the t-SNE optimization loop (gradient descent and force
//! computation), excluding the nearest neighbor index: the Barnes-Hut path runs
//! from neighbors precomputed outside the timed section. Both `f32` and `f64`
//! are measured, each at zero epochs (setup only) and at `EPOCHS`, so the
//! per-epoch loop cost is the difference.

use std::hint::black_box;
use std::iter::Sum;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

use bhtsne::{Neighbor, tSNE};
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main};
use num_traits::{AsPrimitive, Float};

const INPUT_DIM: usize = 50;
const EMBED_DIM: u8 = 2;
const PERPLEXITY: f64 = 30.0;
const THETA: f64 = 0.5;
const EPOCHS: usize = 100;

/// The scalar bound bundle the t-SNE solver requires, satisfied by both floats.
trait Scalar:
    Float + Send + Sync + AsPrimitive<usize> + Sum + DivAssign + AddAssign + MulAssign + SubAssign
{
}
impl Scalar for f32 {}
impl Scalar for f64 {}

fn cast<T: Float>(x: f64) -> T {
    T::from(x).unwrap()
}

/// Deterministic pseudo random samples, reproducible and RNG-free.
fn lcg<T: Scalar>(n: usize, dim: usize, mut state: u64) -> Vec<T> {
    let mut data = Vec::with_capacity(n * dim);
    for _ in 0..n * dim {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        data.push(cast((state >> 33) as f64 / u32::MAX as f64 - 0.5));
    }
    data
}

fn sq_euclidean<T: Scalar>(a: &[T], b: &[T]) -> T {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum()
}

fn euclidean<T: Scalar>(a: &[T], b: &[T]) -> T {
    sq_euclidean(a, b).sqrt()
}

/// Exact k nearest neighbors per sample, ascending distance, excluding self.
/// Built once, outside the timed section.
fn brute_force_neighbors<T: Scalar>(samples: &[&[T]], k: usize) -> Vec<Vec<Neighbor<T>>> {
    (0..samples.len())
        .map(|i| {
            let mut distances: Vec<(usize, T)> = (0..samples.len())
                .filter(|&j| j != i)
                .map(|j| (j, euclidean(samples[i], samples[j])))
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.truncate(k);
            distances
                .into_iter()
                .map(|(index, distance)| Neighbor { index, distance })
                .collect()
        })
        .collect()
}

fn bench_exact<T: Scalar>(group: &mut BenchmarkGroup<'_, WallTime>, dtype: &str) {
    for &n in &[250usize, 500] {
        let data: Vec<T> = lcg(n, INPUT_DIM, 0x00C0_FFEE);
        let samples: Vec<&[T]> = data.chunks(INPUT_DIM).collect();

        for &epochs in &[0usize, EPOCHS] {
            let id = BenchmarkId::new(format!("{dtype}/n{n}"), epochs);
            group.bench_with_input(id, &epochs, |b, &e| {
                b.iter(|| {
                    let mut tsne = tSNE::new(&samples);
                    tsne.embedding_dim(EMBED_DIM)
                        .perplexity(cast(PERPLEXITY))
                        .epochs(e)
                        .exact(|a, b| sq_euclidean(a, b));
                    black_box(tsne.embedding());
                });
            });
        }
    }
}

fn bench_barnes_hut<T: Scalar>(group: &mut BenchmarkGroup<'_, WallTime>, dtype: &str) {
    let k = (3.0 * PERPLEXITY) as usize;
    for &n in &[1000usize, 2000] {
        let data: Vec<T> = lcg(n, INPUT_DIM, 0x00C0_FFEE);
        let samples: Vec<&[T]> = data.chunks(INPUT_DIM).collect();
        let neighbors = brute_force_neighbors(&samples, k);

        for &epochs in &[0usize, EPOCHS] {
            let id = BenchmarkId::new(format!("{dtype}/n{n}"), epochs);
            group.bench_with_input(id, &epochs, |b, &e| {
                b.iter(|| {
                    let mut tsne = tSNE::new(&samples);
                    tsne.embedding_dim(EMBED_DIM)
                        .perplexity(cast(PERPLEXITY))
                        .epochs(e)
                        .barnes_hut_with_neighbors(cast(THETA), black_box(&neighbors));
                    black_box(tsne.embedding());
                });
            });
        }
    }
}

fn exact(c: &mut Criterion) {
    let mut group = c.benchmark_group("tsne_exact");
    group.sample_size(10);
    bench_exact::<f32>(&mut group, "f32");
    bench_exact::<f64>(&mut group, "f64");
    group.finish();
}

fn barnes_hut(c: &mut Criterion) {
    let mut group = c.benchmark_group("tsne_barnes_hut");
    group.sample_size(10);
    bench_barnes_hut::<f32>(&mut group, "f32");
    bench_barnes_hut::<f64>(&mut group, "f64");
    group.finish();
}

criterion_group!(benches, exact, barnes_hut);
criterion_main!(benches);
