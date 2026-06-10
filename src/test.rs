use crossbeam::utils::CachePadded;

use super::{Neighbor, tSNE, tsne};

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
fn set_epoch_callback() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    tsne.epoch_callback(|_epoch, _embedding| {});
    assert!(tsne.epoch_callback.is_some());
}

#[test]
fn set_initial_embedding() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    tsne.initial_embedding([1., 2.]);
    assert_eq!(tsne.initial_embedding, Some(vec![1., 2.]));
}

#[test]
fn kl_divergence_is_none_before_fitting() {
    let data = [0.0_f32, 1.0, 2.0, 3.0];
    let samples: Vec<&[f32]> = data.chunks(1).collect();
    let tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
    assert!(tsne.kl_divergence().is_none());
}

#[test]
fn kl_divergence_after_barnes_hut_is_finite_and_nonnegative() {
    const N: usize = 60;
    const DIM: usize = 4;
    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let mut tsne = tSNE::new(&samples);
    tsne.embedding_dim(NO_DIMS)
        .perplexity(PERPLEXITY)
        .epochs(100)
        .barnes_hut(THETA, |a, b| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        });

    let kl = tsne.kl_divergence().expect("fitted");
    assert!(kl.is_finite() && kl >= 0.0, "{kl}");
}

#[test]
fn kl_divergence_after_exact_is_finite_and_nonnegative() {
    const N: usize = 60;
    const DIM: usize = 4;
    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let mut tsne = tSNE::new(&samples);
    tsne.embedding_dim(NO_DIMS)
        .perplexity(PERPLEXITY)
        .epochs(100)
        .exact(|a, b| a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum());

    let kl = tsne.kl_divergence().expect("fitted");
    assert!(kl.is_finite() && kl >= 0.0, "{kl}");
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

    assert!(tsne.kl_divergence().unwrap() < 0.5);
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

    assert!(tsne.kl_divergence().unwrap() < 5.0);
}

/// The epoch callback must be invoked once per epoch, in order, with a snapshot
/// of the embedding whose final value matches the result of `embedding`, and it
/// must survive the fitting so that subsequent runs can reuse it.
#[test]
fn epoch_callback_reports_each_barnes_hut_epoch() {
    const N: usize = 60;
    const DIM: usize = 4;
    const RUN_EPOCHS: usize = 100;

    // Deterministic LCG so the test needs no RNG dependency.
    let mut state = 7_u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / u32::MAX as f32) - 0.5
    };
    let data: Vec<f32> = (0..N * DIM).map(|_| next()).collect();
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let mut epochs_seen: Vec<usize> = Vec::new();
    let mut last_snapshot: Vec<f32> = Vec::new();

    let embedding = {
        let mut tsne = tSNE::new(&samples);
        tsne.embedding_dim(NO_DIMS)
            .perplexity(PERPLEXITY)
            .epochs(RUN_EPOCHS)
            .epoch_callback(|epoch, snapshot| {
                assert_eq!(snapshot.len(), N * NO_DIMS as usize);
                epochs_seen.push(epoch);
                last_snapshot.clear();
                last_snapshot.extend_from_slice(snapshot);
            })
            .barnes_hut(THETA, |sample_a, sample_b| {
                sample_a
                    .iter()
                    .zip(sample_b.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt()
            });
        // The callback must be put back in place once the fitting is over.
        assert!(tsne.epoch_callback.is_some());
        tsne.embedding()
    };

    assert_eq!(epochs_seen, (0..RUN_EPOCHS).collect::<Vec<usize>>());
    assert_eq!(last_snapshot, embedding);
}

/// Same as `epoch_callback_reports_each_barnes_hut_epoch` for the exact version
/// of the algorithm.
#[test]
fn epoch_callback_reports_each_exact_epoch() {
    const N: usize = 60;
    const DIM: usize = 4;
    const RUN_EPOCHS: usize = 50;

    // Deterministic LCG so the test needs no RNG dependency.
    let mut state = 7_u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / u32::MAX as f32) - 0.5
    };
    let data: Vec<f32> = (0..N * DIM).map(|_| next()).collect();
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let mut epochs_seen: Vec<usize> = Vec::new();
    let mut last_snapshot: Vec<f32> = Vec::new();

    let embedding = {
        let mut tsne = tSNE::new(&samples);
        tsne.embedding_dim(NO_DIMS)
            .perplexity(PERPLEXITY)
            .epochs(RUN_EPOCHS)
            .epoch_callback(|epoch, snapshot| {
                assert_eq!(snapshot.len(), N * NO_DIMS as usize);
                epochs_seen.push(epoch);
                last_snapshot.clear();
                last_snapshot.extend_from_slice(snapshot);
            })
            .exact(|sample_a, sample_b| {
                sample_a
                    .iter()
                    .zip(sample_b.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum()
            });
        // The callback must be put back in place once the fitting is over.
        assert!(tsne.epoch_callback.is_some());
        tsne.embedding()
    };

    assert_eq!(epochs_seen, (0..RUN_EPOCHS).collect::<Vec<usize>>());
    assert_eq!(last_snapshot, embedding);
}

/// A warm started fit must begin from the supplied embedding: the first epoch
/// stays close to the seed, far closer than a random init near the origin would.
#[test]
fn warm_start_begins_from_initial_embedding_barnes_hut() {
    const N: usize = 60;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    // A plausible layout to continue from.
    let seed = {
        let mut tsne = tSNE::new(&samples);
        tsne.embedding_dim(NO_DIMS)
            .perplexity(PERPLEXITY)
            .epochs(300)
            .barnes_hut(THETA, |sample_a, sample_b| {
                sample_a
                    .iter()
                    .zip(sample_b.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt()
            });
        tsne.embedding()
    };

    let mut first_snapshot: Vec<f32> = Vec::new();
    {
        let mut tsne = tSNE::new(&samples);
        tsne.embedding_dim(NO_DIMS)
            .perplexity(PERPLEXITY)
            .epochs(5)
            .stop_lying_epoch(0)
            .momentum_switch_epoch(0)
            .initial_embedding(&seed[..])
            .epoch_callback(|epoch, snapshot| {
                if epoch == 0 {
                    first_snapshot.extend_from_slice(snapshot);
                }
            })
            .barnes_hut(THETA, |sample_a, sample_b| {
                sample_a
                    .iter()
                    .zip(sample_b.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt()
            });
    }

    let dim = NO_DIMS as usize;
    let displacement = mean_point_distance(&first_snapshot, &seed, dim);
    let diagonal = bounding_box_diagonal(&seed, dim);
    assert!(
        displacement < 0.05 * diagonal,
        "first epoch strayed {displacement} from the seed, its bounding box diagonal is {diagonal}"
    );

    // A random initialization concentrates every point around the origin, so
    // its mean displacement from the seed is the mean seed point norm.
    let origin = vec![0.0_f32; seed.len()];
    let random_displacement = mean_point_distance(&origin, &seed, dim);
    assert!(
        random_displacement > 10.0 * displacement,
        "warm start indistinguishable from a random initialization: {displacement} against {random_displacement}"
    );
}

/// Same as `warm_start_begins_from_initial_embedding_barnes_hut` for the exact
/// version of the algorithm.
#[test]
fn warm_start_begins_from_initial_embedding_exact() {
    const N: usize = 60;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    // A plausible layout to continue from.
    let seed = {
        let mut tsne = tSNE::new(&samples);
        tsne.embedding_dim(NO_DIMS)
            .perplexity(PERPLEXITY)
            .epochs(300)
            .exact(|sample_a, sample_b| {
                sample_a
                    .iter()
                    .zip(sample_b.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum()
            });
        tsne.embedding()
    };

    let mut first_snapshot: Vec<f32> = Vec::new();
    {
        let mut tsne = tSNE::new(&samples);
        tsne.embedding_dim(NO_DIMS)
            .perplexity(PERPLEXITY)
            .epochs(5)
            .stop_lying_epoch(0)
            .momentum_switch_epoch(0)
            .initial_embedding(&seed[..])
            .epoch_callback(|epoch, snapshot| {
                if epoch == 0 {
                    first_snapshot.extend_from_slice(snapshot);
                }
            })
            .exact(|sample_a, sample_b| {
                sample_a
                    .iter()
                    .zip(sample_b.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum()
            });
    }

    let dim = NO_DIMS as usize;
    let displacement = mean_point_distance(&first_snapshot, &seed, dim);
    let diagonal = bounding_box_diagonal(&seed, dim);
    assert!(
        displacement < 0.05 * diagonal,
        "first epoch strayed {displacement} from the seed, its bounding box diagonal is {diagonal}"
    );

    // A random initialization concentrates every point around the origin, so
    // its mean displacement from the seed is the mean seed point norm.
    let origin = vec![0.0_f32; seed.len()];
    let random_displacement = mean_point_distance(&origin, &seed, dim);
    assert!(
        random_displacement > 10.0 * displacement,
        "warm start indistinguishable from a random initialization: {displacement} against {random_displacement}"
    );
}

/// The Barnes-Hut fit must reject a seed whose length does not match
/// `n_samples * embedding_dim`.
#[test]
#[should_panic(expected = "initial embedding has")]
fn warm_start_rejects_wrong_length_barnes_hut() {
    const N: usize = 60;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let mut tsne = tSNE::new(&samples);
    tsne.embedding_dim(NO_DIMS)
        .perplexity(PERPLEXITY)
        .epochs(1)
        .initial_embedding([0.0; 7])
        .barnes_hut(THETA, |sample_a, sample_b| {
            sample_a
                .iter()
                .zip(sample_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        });
}

/// The exact fit carries its own length check, exercise it independently of the
/// Barnes-Hut one.
#[test]
#[should_panic(expected = "initial embedding has")]
fn warm_start_rejects_wrong_length_exact() {
    const N: usize = 60;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let mut tsne = tSNE::new(&samples);
    tsne.embedding_dim(NO_DIMS)
        .perplexity(PERPLEXITY)
        .epochs(1)
        .initial_embedding([0.0; 7])
        .exact(|sample_a, sample_b| {
            sample_a
                .iter()
                .zip(sample_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum()
        });
}

/// The seed is consumed by the fit, so a second fit with no new seed falls back
/// to a random init near the origin rather than reusing the old seed.
#[test]
fn warm_start_seed_is_consumed_by_the_fit() {
    const N: usize = 60;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let seed = {
        let mut tsne = tSNE::new(&samples);
        tsne.embedding_dim(NO_DIMS)
            .perplexity(PERPLEXITY)
            .epochs(300)
            .barnes_hut(THETA, |sample_a, sample_b| {
                sample_a
                    .iter()
                    .zip(sample_b.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt()
            });
        tsne.embedding()
    };

    let mut second_run_first_snapshot: Vec<f32> = Vec::new();
    let mut tsne = tSNE::new(&samples);
    tsne.embedding_dim(NO_DIMS)
        .perplexity(PERPLEXITY)
        .epochs(1)
        .initial_embedding(&seed[..]);

    // First fit consumes the seed.
    tsne.barnes_hut(THETA, |sample_a, sample_b| {
        sample_a
            .iter()
            .zip(sample_b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    });
    // The builder slot must be empty again.
    assert!(tsne.initial_embedding.is_none());

    // Second fit, no new seed: it must random init, not continue from the seed.
    tsne.epochs(1)
        .epoch_callback(|epoch, snapshot| {
            if epoch == 0 {
                second_run_first_snapshot.extend_from_slice(snapshot);
            }
        })
        .barnes_hut(THETA, |sample_a, sample_b| {
            sample_a
                .iter()
                .zip(sample_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        });
    // The callback keeps a mutable borrow of the snapshot for as long as tsne
    // lives, drop it so the snapshot can be read.
    drop(tsne);

    let dim = NO_DIMS as usize;
    let from_seed = mean_point_distance(&second_run_first_snapshot, &seed, dim);
    let from_origin = mean_point_distance(&second_run_first_snapshot, &vec![0.0; seed.len()], dim);
    assert!(
        from_origin < from_seed,
        "second run continued from the consumed seed instead of random init: \
         {from_origin} from origin against {from_seed} from the seed"
    );
}

/// A stop lying epoch of zero must mean no early exaggeration at all. Two warm
/// started single epoch runs, one with the exaggeration off and one with it on,
/// must take differently sized first steps, since the momentum buffer is zero at
/// epoch 0 the two differ by the exaggeration factor alone.
#[test]
fn stop_lying_epoch_zero_skips_exaggeration_barnes_hut() {
    const N: usize = 60;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    // A plausible layout to continue from.
    let seed = {
        let mut tsne = tSNE::new(&samples);
        tsne.embedding_dim(NO_DIMS)
            .perplexity(PERPLEXITY)
            .epochs(300)
            .barnes_hut(THETA, |sample_a, sample_b| {
                sample_a
                    .iter()
                    .zip(sample_b.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt()
            });
        tsne.embedding()
    };

    let first_step = |stop_lying_epoch: usize| -> f32 {
        let mut first_snapshot: Vec<f32> = Vec::new();
        {
            let mut tsne = tSNE::new(&samples);
            tsne.embedding_dim(NO_DIMS)
                .perplexity(PERPLEXITY)
                .epochs(1)
                .stop_lying_epoch(stop_lying_epoch)
                .initial_embedding(&seed[..])
                .epoch_callback(|_epoch, snapshot| {
                    first_snapshot.extend_from_slice(snapshot);
                })
                .barnes_hut(THETA, |sample_a, sample_b| {
                    sample_a
                        .iter()
                        .zip(sample_b.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt()
                });
        }
        mean_point_distance(&first_snapshot, &seed, NO_DIMS as usize)
    };

    let exaggerated = first_step(1000);
    let truthful = first_step(0);
    assert!(
        truthful < exaggerated / 3.0,
        "first epoch still exaggerated: moved {truthful} against {exaggerated} with 12x P values"
    );
}

/// Same as `stop_lying_epoch_zero_skips_exaggeration_barnes_hut` for the exact
/// version of the algorithm.
#[test]
fn stop_lying_epoch_zero_skips_exaggeration_exact() {
    const N: usize = 60;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    // A plausible layout to continue from.
    let seed = {
        let mut tsne = tSNE::new(&samples);
        tsne.embedding_dim(NO_DIMS)
            .perplexity(PERPLEXITY)
            .epochs(300)
            .exact(|sample_a, sample_b| {
                sample_a
                    .iter()
                    .zip(sample_b.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum()
            });
        tsne.embedding()
    };

    let first_step = |stop_lying_epoch: usize| -> f32 {
        let mut first_snapshot: Vec<f32> = Vec::new();
        {
            let mut tsne = tSNE::new(&samples);
            tsne.embedding_dim(NO_DIMS)
                .perplexity(PERPLEXITY)
                .epochs(1)
                .stop_lying_epoch(stop_lying_epoch)
                .initial_embedding(&seed[..])
                .epoch_callback(|_epoch, snapshot| {
                    first_snapshot.extend_from_slice(snapshot);
                })
                .exact(|sample_a, sample_b| {
                    sample_a
                        .iter()
                        .zip(sample_b.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum()
                });
        }
        mean_point_distance(&first_snapshot, &seed, NO_DIMS as usize)
    };

    let exaggerated = first_step(1000);
    let truthful = first_step(0);
    assert!(
        truthful < exaggerated / 3.0,
        "first epoch still exaggerated: moved {truthful} against {exaggerated} with 12x P values"
    );
}

/// Euclidean distance between two samples, the metric the Barnes-Hut tests use.
fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Exact k nearest neighbors per sample, sorted by ascending distance, excluding
/// self: the same set the vantage point tree finds.
fn brute_force_neighbors(samples: &[&[f32]], n_neighbors: usize) -> Vec<Vec<Neighbor<f32>>> {
    (0..samples.len())
        .map(|i| {
            let mut distances: Vec<(usize, f32)> = (0..samples.len())
                .filter(|&j| j != i)
                .map(|j| (j, euclidean(samples[i], samples[j])))
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.truncate(n_neighbors);
            distances
                .into_iter()
                .map(|(index, distance)| Neighbor { index, distance })
                .collect()
        })
        .collect()
}

/// Fed the neighbors the tree would find, `barnes_hut_with_neighbors` must
/// reproduce the `barnes_hut` embedding bit for bit (the pipeline is deterministic).
#[test]
fn barnes_hut_with_neighbors_matches_vptree_path() {
    const N: usize = 80;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 11);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    // A seed so both fits start from the very same embedding.
    let seed = lcg_samples(N, NO_DIMS as usize, 99);

    // Pin to one rayon thread: the parallel reductions in the fit accumulate in a
    // thread-count dependent order, which the chaotic gradient descent would
    // amplify, so single threaded keeps both paths bitwise reproducible.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();

    let reference = pool.install(|| {
        let mut tsne = tSNE::new(&samples);
        tsne.embedding_dim(NO_DIMS)
            .perplexity(PERPLEXITY)
            .epochs(100)
            .initial_embedding(&seed[..])
            .barnes_hut(THETA, |a, b| euclidean(a, b));
        tsne.embedding()
    });

    let n_neighbors = (3.0 * PERPLEXITY) as usize;
    let neighbors = brute_force_neighbors(&samples, n_neighbors);

    let candidate = pool.install(|| {
        let mut tsne = tSNE::new(&samples);
        tsne.embedding_dim(NO_DIMS)
            .perplexity(PERPLEXITY)
            .epochs(100)
            .initial_embedding(&seed[..])
            .barnes_hut_with_neighbors(THETA, &neighbors);
        tsne.embedding()
    });

    assert_eq!(candidate, reference);
}

/// Ragged neighbor rows must be rejected.
#[test]
#[should_panic(expected = "same length")]
fn barnes_hut_with_neighbors_rejects_ragged_rows() {
    const N: usize = 80;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 11);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let n_neighbors = (3.0 * PERPLEXITY) as usize;
    let mut neighbors = brute_force_neighbors(&samples, n_neighbors);
    // Make one row shorter than the others.
    neighbors[0].pop();

    let mut tsne = tSNE::new(&samples);
    tsne.embedding_dim(NO_DIMS)
        .perplexity(PERPLEXITY)
        .epochs(1)
        .barnes_hut_with_neighbors(THETA, &neighbors);
}

/// An out-of-range neighbor index must be rejected up front.
#[test]
#[should_panic(expected = "out of range")]
fn barnes_hut_with_neighbors_rejects_out_of_range_index() {
    const N: usize = 80;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 11);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let n_neighbors = (3.0 * PERPLEXITY) as usize;
    let mut neighbors = brute_force_neighbors(&samples, n_neighbors);
    // Point one neighbor at a sample that does not exist.
    neighbors[0][0].index = N;

    let mut tsne = tSNE::new(&samples);
    tsne.embedding_dim(NO_DIMS)
        .perplexity(PERPLEXITY)
        .epochs(1)
        .barnes_hut_with_neighbors(THETA, &neighbors);
}

/// Deterministic LCG data so the tests need no RNG dependency.
fn lcg_samples(n: usize, dim: usize, mut state: u64) -> Vec<f32> {
    let mut data = Vec::with_capacity(n * dim);
    for _ in 0..n * dim {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        data.push(((state >> 33) as f32 / u32::MAX as f32) - 0.5);
    }
    data
}

/// Mean euclidean distance between corresponding points of two embeddings.
fn mean_point_distance(a: &[f32], b: &[f32], dim: usize) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len() / dim;
    a.chunks_exact(dim)
        .zip(b.chunks_exact(dim))
        .map(|(p, q)| {
            p.iter()
                .zip(q.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        })
        .sum::<f32>()
        / n as f32
}

/// Diagonal of the bounding box of an embedding.
fn bounding_box_diagonal(points: &[f32], dim: usize) -> f32 {
    (0..dim)
        .map(|d| {
            let component = points.iter().skip(d).step_by(dim);
            let min = component.clone().fold(f32::MAX, |a, &b| a.min(b));
            let max = component.fold(f32::MIN, |a, &b| a.max(b));
            (max - min).powi(2)
        })
        .sum::<f32>()
        .sqrt()
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
