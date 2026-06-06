use super::{tSNE, tsne};
use crossbeam::utils::CachePadded;

const D: usize = 4;
const THETA: f32 = 0.5;
const PERPLEXITY: f32 = 10.;
const EPOCHS: usize = 2_000;
const NO_DIMS: u8 = 2;

#[test]
fn set_learning_rate() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    tsne.learning_rate(15.);
    assert_eq!(tsne.learning_rate, 15.);
}

#[test]
fn set_epochs() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    tsne.epochs(15);
    assert_eq!(tsne.epochs, 15);
}

#[test]
fn set_momentum() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    tsne.momentum(15.);
    assert_eq!(tsne.momentum, 15.);
}

#[test]
fn set_final_momentum() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    tsne.final_momentum(15.);
    assert_eq!(tsne.final_momentum, 15.);
}

#[test]
fn set_momentum_switch_epoch() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    tsne.momentum_switch_epoch(15);
    assert_eq!(tsne.momentum_switch_epoch, 15);
}

#[test]
fn set_stop_lying_epoch() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    tsne.stop_lying_epoch(15);
    assert_eq!(tsne.stop_lying_epoch, 15);
}

#[test]
fn set_embedding_dim() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    tsne.embedding_dim(3);
    assert_eq!(tsne.embedding_dim, 3);
}

#[test]
fn set_perplexity() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    tsne.perplexity(15.);
    assert_eq!(tsne.perplexity, 15.);
}

#[test]
#[ignore = "requires iris dataset"]
fn exact_tsne() {
    let data: Vec<f32> =
        crate::load_csv("iris.csv", true, Some(&[4]), |float| float.parse().unwrap()).unwrap();
    let samples: Vec<&[f32]> = data.chunks(D).collect::<Vec<&[f32]>>();

    let mut tsne = tSNE::new(&samples);
    tsne.embedding_dim(NO_DIMS)
        .perplexity(PERPLEXITY)
        .epochs(EPOCHS)
        .exact(|sample_a, sample_b| {
            sample_a
                .iter()
                .zip(sample_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum()
        });
    tsne.write_csv("iris_embedding_vanilla.csv").unwrap();

    let embedding = tsne.embedding();
    let points: Vec<_> = embedding.chunks(NO_DIMS as usize).collect();

    assert_eq!(points.len(), samples.len());

    assert!(
        tsne::evaluate_error(
            &tsne.p_values,
            &tsne.y,
            samples.len(),
            tsne.embedding_dim as usize
        ) < 0.5
    );
}

#[test]
#[ignore = "requires iris dataset"]
fn barnes_hut_tsne() {
    let data: Vec<f32> =
        crate::load_csv("iris.csv", true, Some(&[4]), |float| float.parse().unwrap()).unwrap();
    let samples: Vec<&[f32]> = data.chunks(D).collect::<Vec<&[f32]>>();

    let mut tsne = tSNE::new(&samples);
    tsne.embedding_dim(NO_DIMS)
        .perplexity(PERPLEXITY)
        .epochs(EPOCHS)
        .barnes_hut(THETA, |sample_a, sample_b| {
            sample_a
                .iter()
                .zip(sample_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        })
        .write_csv("iris_embedding_barnes_hut.csv")
        .unwrap();

    let embedding = tsne.embedding();
    let points: Vec<_> = embedding.chunks(NO_DIMS as usize).collect();

    assert_eq!(points.len(), samples.len());

    assert!(
        tsne::evaluate_error_approximately(
            &tsne.p_rows,
            &tsne.p_columns,
            &tsne.p_values,
            &tsne.y,
            samples.len(),
            tsne.embedding_dim as usize,
            THETA
        ) < 5.0
    );
}

/// Regression test for the Gaussian bandwidth binary search.
///
/// When the neighbour distances are heterogeneous and noticeably larger than
/// 1, matching the target perplexity requires a bandwidth beta well below 1.
/// The descent path of the search (taken while no lower bracket is known yet)
/// must therefore be able to shrink beta indefinitely. Releases 0.5.0-0.5.2
/// clamped it at 0.5 and releases 0.5.3-0.5.4 moved beta upwards instead
/// (a constant named `zero_point_five` was set to 5.0), making the search
/// diverge and the conditional distribution degenerate.
#[test]
fn search_beta_converges_when_optimal_beta_below_one() {
    // 90 neighbours (3 * perplexity) with squared distances spread over
    // [20, 120]: the optimal beta for perplexity 30 is roughly 0.08.
    let distances_row: Vec<CachePadded<f64>> = (0..90)
        .map(|i| CachePadded::new((20.0 + 100.0 * (i as f64 + 1.0) / 90.0_f64).sqrt()))
        .collect();
    let mut p_values_row: Vec<CachePadded<f64>> = vec![CachePadded::new(0.0); 90];
    let perplexity = 30.0;

    tsne::search_beta(&mut p_values_row, &distances_row, &perplexity);

    // The effective number of neighbours encoded by the row, exp(H(P)),
    // must match the requested perplexity.
    let entropy: f64 = p_values_row
        .iter()
        .map(|p| **p)
        .filter(|&p| p > 0.0)
        .map(|p| -p * p.ln())
        .sum();
    let effective_perplexity = entropy.exp();

    assert!(
        (effective_perplexity - perplexity).abs() < 0.1,
        "expected effective perplexity of ~{perplexity}, got {effective_perplexity}"
    );
}

/// End-to-end regression test: two trivially separable clusters whose
/// coordinates are large enough that the bandwidth search must go below
/// beta = 1. A correct t-SNE is invariant to uniform input rescaling, so the
/// embedding must separate the clusters just as it does for small inputs.
#[test]
fn barnes_hut_separates_clusters_at_large_input_scale() {
    const N_PER_CLUSTER: usize = 150;
    const DIM: usize = 10;

    // Deterministic LCG so the test needs no RNG dependency.
    let mut state = 42_u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / u32::MAX as f32) - 0.5
    };

    let mut data = Vec::with_capacity(2 * N_PER_CLUSTER * DIM);
    for cluster in 0..2 {
        let centre = if cluster == 0 { 0.0 } else { 30.0 };
        for _ in 0..N_PER_CLUSTER {
            for _ in 0..DIM {
                data.push(centre + 6.0 * next());
            }
        }
    }
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let mut tsne = tSNE::new(&samples);
    tsne.embedding_dim(2)
        .perplexity(30.0)
        .epochs(500)
        .barnes_hut(THETA, |sample_a, sample_b| {
            sample_a
                .iter()
                .zip(sample_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        });
    let embedding = tsne.embedding();

    // For every point, the nearest embedded neighbour must belong to the
    // same cluster for at least 95% of the points.
    let n = 2 * N_PER_CLUSTER;
    let mut same_cluster = 0;
    for i in 0..n {
        let mut best = f32::MAX;
        let mut best_j = usize::MAX;
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = embedding[2 * i] - embedding[2 * j];
            let dy = embedding[2 * i + 1] - embedding[2 * j + 1];
            let d = dx * dx + dy * dy;
            if d < best {
                best = d;
                best_j = j;
            }
        }
        if (i < N_PER_CLUSTER) == (best_j < N_PER_CLUSTER) {
            same_cluster += 1;
        }
    }
    assert!(
        same_cluster as f64 / n as f64 > 0.95,
        "clusters not separated: only {same_cluster}/{n} points have a same-cluster nearest neighbour"
    );
}
