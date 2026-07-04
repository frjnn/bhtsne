use std::time::Instant;

use rand::{SeedableRng, rngs::StdRng};

use rand_distr::{Distribution, Normal};

use bhtsne::tSNE;

const DIM: usize = 50;
const K: usize = 10;
const SEED: u64 = 123456;
const THETA: f32 = 0.5;
const SIZES: &[usize] = &[100_000, 200_000, 500_000, 1_000_000, 2_000_000];

/// Euclidean distance.
fn euclid(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y)
        .map(|(xi, yi)| (xi - yi).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// `n` points drawn from `K` isotropic Gaussian blobs in `DIM` dimensions, from a
/// seeded RNG so the dataset is reproducible.
fn blobs(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let center = Normal::new(0.0f32, 6.0).unwrap();
    let noise = Normal::new(0.0f32, 1.0).unwrap();

    let centers: Vec<f32> = (0..K * DIM).map(|_| center.sample(&mut rng)).collect();
    let mut data = Vec::with_capacity(n * DIM);
    for i in 0..n {
        let k = i % K;
        for d in 0..DIM {
            data.push(centers[k * DIM + d] + noise.sample(&mut rng));
        }
    }

    data
}

fn main() {
    println!("Optimization only, f32, 1000 epochs:");
    println!(
        "{:>8}  {:>11}  {:>11}  {:>7}",
        "N", "fit_sne s", "bh s", "bh/fit"
    );

    for &n in SIZES {
        let data = blobs(n, SEED);
        let rows: Vec<&[f32]> = data.chunks_exact(DIM).collect();

        // Build the affinity graph once.
        let aff = {
            let mut warm: tSNE<f32, &[f32]> = tSNE::new(&rows);
            warm.perplexity(30.0).epochs(0).fit_sne(|a, b| euclid(a, b));

            warm.affinities()
                .expect("should have pre-computed affinities")
        };

        // Measures the fitting loop.
        let start_fit_sne = Instant::now();
        tSNE::<f32, _>::new(&rows)
            .with_affinities(aff.clone())
            .epochs(1000)
            .fit_sne(|x, y| euclid(x, y));
        let end_fit_sne = start_fit_sne.elapsed().as_secs_f64();

        let start_bh_tnse = Instant::now();
        tSNE::<f32, _>::new(&rows)
            .with_affinities(aff.clone())
            .epochs(1000)
            .barnes_hut(THETA, |x, y| euclid(x, y));
        let end_bh_tsne = start_bh_tnse.elapsed().as_secs_f64();

        println!(
            "{n:>8}  {end_fit_sne:>11.1}  {end_bh_tsne:>11.1}  {:>6.2}x",
            end_bh_tsne / end_fit_sne
        );
    }
}
