//! Affinity construction: `barnes_hut` (vantage point tree) vs
//! `barnes_hut_with_neighbors` (precomputed). `epochs(0)` isolates P construction
//! from the shared gradient descent. Neighbors are built once, outside timing.
mod common;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use bhtsne::tSNE;

use common::{brute_force_neighbors, euclidean, lcg};

const DIM: usize = 128;
const PERPLEXITY: f32 = 30.0;
const THETA: f32 = 0.5;
const SIZES: [usize; 4] = [500, 1000, 2000, 4000];

fn bench_affinities(c: &mut Criterion) {
    let k = (3.0 * PERPLEXITY) as usize;
    let mut group = c.benchmark_group("input_affinities");

    for &n in &SIZES {
        let data = lcg(n, DIM, 0x00C0_FFEE);
        let samples: Vec<&[f32]> = data.chunks(DIM).collect();
        let neighbors = brute_force_neighbors(&samples, k);

        group.throughput(Throughput::Elements(n as u64));

        // Vantage point tree path: builds the tree and queries it per sample.
        group.bench_with_input(BenchmarkId::new("vptree", n), &n, |b, _| {
            b.iter(|| {
                let mut tsne = tSNE::new(&samples);
                tsne.embedding_dim(2)
                    .perplexity(PERPLEXITY)
                    .epochs(0)
                    .barnes_hut(THETA, |a, b| euclidean(a, b));
                black_box(tsne.embedding());
            });
        });

        // Index-accelerated path: fills the rows from precomputed neighbors.
        group.bench_with_input(BenchmarkId::new("precomputed_neighbors", n), &n, |b, _| {
            b.iter(|| {
                let mut tsne = tSNE::new(&samples);
                tsne.embedding_dim(2)
                    .perplexity(PERPLEXITY)
                    .epochs(0)
                    .barnes_hut_with_neighbors(THETA, black_box(&neighbors));
                black_box(tsne.embedding());
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_affinities);
criterion_main!(benches);
