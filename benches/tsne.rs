//! Benchmarks the t-SNE optimization loop (gradient descent and force
//! computation), excluding the nearest neighbor index: the Barnes-Hut path runs
//! from neighbors precomputed outside the timed section. Both `f32` and `f64`
//! are measured, each at zero epochs (setup only) and at `EPOCHS`, so the
//! per-epoch loop cost is the difference.
mod common;

use std::hint::black_box;

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main, measurement::WallTime,
};

use bhtsne::tSNE;

use common::{Scalar, brute_force_neighbors, cast, lcg, sq_euclidean};

const INPUT_DIM: usize = 50;
const PERPLEXITY: f64 = 30.0;
const THETA: f64 = 0.5;
const EPOCHS: usize = 100;

fn bench_exact<T: Scalar>(group: &mut BenchmarkGroup<'_, WallTime>, dtype: &str) {
    for &n in &[250usize, 500] {
        let data: Vec<T> = lcg(n, INPUT_DIM, 0x00C0_FFEE);
        let samples: Vec<&[T]> = data.chunks(INPUT_DIM).collect();

        for &epochs in &[0usize, EPOCHS] {
            let id = BenchmarkId::new(format!("{dtype}/n{n}"), epochs);
            group.bench_with_input(id, &epochs, |b, &e| {
                b.iter(|| {
                    let mut tsne: tSNE<T, &[T]> = tSNE::new(&samples);
                    tsne.perplexity(cast(PERPLEXITY))
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
    for &n in &[1000, 2000] {
        let data: Vec<T> = lcg(n, INPUT_DIM, 0x00C0_FFEE);
        let samples: Vec<&[T]> = data.chunks(INPUT_DIM).collect();
        let neighbors = brute_force_neighbors(&samples, k);

        for &epochs in &[0usize, EPOCHS] {
            let id = BenchmarkId::new(format!("{dtype}/n{n}"), epochs);
            group.bench_with_input(id, &epochs, |b, &e| {
                b.iter(|| {
                    let mut tsne: tSNE<T, &[T]> = tSNE::new(&samples);
                    tsne.perplexity(cast(PERPLEXITY))
                        .epochs(e)
                        .barnes_hut_with_neighbors(cast(THETA), black_box(&neighbors));
                    black_box(tsne.embedding());
                });
            });
        }
    }
}

fn bench_fit_sne<T: Scalar>(group: &mut BenchmarkGroup<'_, WallTime>, dtype: &str) {
    let k = (3.0 * PERPLEXITY) as usize;
    for &n in &[1000, 2000] {
        let data: Vec<T> = lcg(n, INPUT_DIM, 0x00C0_FFEE);
        let samples: Vec<&[T]> = data.chunks(INPUT_DIM).collect();
        let neighbors = brute_force_neighbors(&samples, k);

        for &epochs in &[0usize, EPOCHS] {
            let id = BenchmarkId::new(format!("{dtype}/n{n}"), epochs);
            group.bench_with_input(id, &epochs, |b, &e| {
                b.iter(|| {
                    let mut tsne: tSNE<T, &[T]> = tSNE::new(&samples);
                    tsne.perplexity(cast(PERPLEXITY))
                        .epochs(e)
                        .fit_sne_with_neighbors(black_box(&neighbors));
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

fn fit_sne(c: &mut Criterion) {
    let mut group = c.benchmark_group("tsne_fit_sne");
    group.sample_size(10);
    bench_fit_sne::<f32>(&mut group, "f32");
    bench_fit_sne::<f64>(&mut group, "f64");
    group.finish();
}

criterion_group!(benches, exact, barnes_hut, fit_sne);
criterion_main!(benches);
