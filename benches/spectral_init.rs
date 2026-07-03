//! Spectral embedding initialization: `tSNE::spectral_init` across
//! graph sizes and target dimensions. The affinity graph is built once per size
//! (via a short `barnes_hut` run with `epochs(0)`), then `spectral_init` is timed.
mod common;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use bhtsne::tSNE;

use common::{euclidean, lcg};

const DIM: usize = 128;
const PERPLEXITY: f32 = 30.0;
const THETA: f32 = 0.5;
const SIZES: [usize; 6] = [500, 1000, 2000, 4000, 10000, 20000];

// Bench dimensions 2, 3, 4, 7 via separate typed benchmarks.
fn bench_d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_init_d2");
    for &n in &SIZES {
        let data = lcg(n, DIM, 0xDEAD_BEEF);
        let samples: Vec<&[f32]> = data.chunks(DIM).collect();

        let mut tsne: tSNE<f32, &[f32], 2> = tSNE::new(&samples);
        tsne.perplexity(PERPLEXITY)
            .epochs(0)
            .barnes_hut(THETA, |a, b| euclidean(a, b));

        group.throughput(Throughput::Elements((n * 2) as u64));
        group.bench_with_input(BenchmarkId::new("d2", n), &tsne, |b, t| {
            b.iter(|| black_box(t.spectral_embedding()));
        });
    }
    group.finish();
}

fn bench_d3(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_init_d3");
    for &n in &SIZES {
        let data = lcg(n, DIM, 0xDEAD_BEEF);
        let samples: Vec<&[f32]> = data.chunks(DIM).collect();

        let mut tsne: tSNE<f32, &[f32], 3> = tSNE::new(&samples);
        tsne.perplexity(PERPLEXITY)
            .epochs(0)
            .barnes_hut(THETA, |a, b| euclidean(a, b));

        group.throughput(Throughput::Elements((n * 3) as u64));
        group.bench_with_input(BenchmarkId::new("d3", n), &tsne, |b, t| {
            b.iter(|| black_box(t.spectral_embedding()));
        });
    }
    group.finish();
}

fn bench_d4(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_init_d4");
    for &n in &SIZES {
        let data = lcg(n, DIM, 0xDEAD_BEEF);
        let samples: Vec<&[f32]> = data.chunks(DIM).collect();

        let mut tsne: tSNE<f32, &[f32], 4> = tSNE::new(&samples);
        tsne.perplexity(PERPLEXITY)
            .epochs(0)
            .barnes_hut(THETA, |a, b| euclidean(a, b));

        group.throughput(Throughput::Elements((n * 4) as u64));
        group.bench_with_input(BenchmarkId::new("d4", n), &tsne, |b, t| {
            b.iter(|| black_box(t.spectral_embedding()));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_d2, bench_d3, bench_d4);
criterion_main!(benches);
