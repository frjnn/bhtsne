use super::{Neighbor, tSNE, tsne};
use rand::Rng;

const D: usize = 4;
const THETA: f32 = 0.5;
const PERPLEXITY: f32 = 10.;
const EPOCHS: usize = 2_000;
const NO_DIMS: u8 = 2;

#[test]
fn set_learning_rate() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    tsne.learning_rate(15.);
    assert_eq!(tsne.learning_rate, Some(15.));
}

#[test]
fn learning_rate_defaults_to_unset() {
    let tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    assert_eq!(tsne.learning_rate, None);
}

#[test]
fn auto_learning_rate_hits_the_floor_for_small_n() {
    // 100 / 12 / 4 is about 2.08, well below the floor, so the rate clamps to 50.
    let tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    assert_eq!(tsne.resolve_learning_rate(100), 50.0);
}

#[test]
fn auto_learning_rate_scales_with_n() {
    // Above the floor the rate is exactly n / early_exaggeration / 4.
    let tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    assert_eq!(tsne.resolve_learning_rate(120_000), 120_000.0 / 12.0 / 4.0);
}

#[test]
fn explicit_learning_rate_overrides_the_auto_default() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    tsne.learning_rate(123.0);
    assert_eq!(tsne.resolve_learning_rate(100), 123.0);
    assert_eq!(tsne.resolve_learning_rate(1_000_000), 123.0);
}

#[test]
fn auto_learning_rate_is_coupled_to_early_exaggeration() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    let with_default = tsne.resolve_learning_rate(120_000);
    tsne.early_exaggeration(6.0);
    let with_half = tsne.resolve_learning_rate(120_000);
    // Halving the exaggeration doubles the auto rate (both are above the floor).
    assert_eq!(with_half, 2.0 * with_default);
}

/// Calibration guard: at n = 10000 with the default factor the auto rate lands
/// near the historical fixed 200, confirming the divisor convention. A wrong
/// divisor would move this far out of band.
#[test]
fn auto_learning_rate_matches_historical_default_at_ten_thousand() {
    let tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    let rate = tsne.resolve_learning_rate(10_000);
    assert!(
        (205.0..=212.0).contains(&rate),
        "auto rate at n=10000 is {rate}, expected close to 208 (= 10000 / 12 / 4)"
    );
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
fn set_early_exaggeration() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    tsne.early_exaggeration(4.);
    assert_eq!(tsne.early_exaggeration, 4.);
}

#[test]
fn early_exaggeration_default_is_twelve() {
    let tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    assert_eq!(tsne.early_exaggeration, 12.);
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

    let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne.perplexity(PERPLEXITY)
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

/// Smoke test for the arena build and the force and reduction passes: the embedding stays finite
/// and correctly sized after a short Barnes-Hut fit.
#[test]
fn parallel_barnes_hut_build_smoke() {
    const N: usize = 160;
    const DIM: usize = 4;
    let data = lcg_samples(N, DIM, 5);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();
    let n_neighbors = (3.0 * PERPLEXITY) as usize;
    let neighbors = brute_force_neighbors(&samples, n_neighbors);

    let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne.perplexity(PERPLEXITY)
        .epochs(3)
        .barnes_hut_with_neighbors(THETA, &neighbors);

    let embedding = tsne.embedding();
    assert_eq!(embedding.len(), N * NO_DIMS as usize);
    assert!(embedding.iter().all(|v| v.is_finite()));
}

#[test]
fn kl_divergence_after_exact_is_finite_and_nonnegative() {
    const N: usize = 60;
    const DIM: usize = 4;
    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne.perplexity(PERPLEXITY)
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

    let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne.perplexity(PERPLEXITY)
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

    let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne.perplexity(PERPLEXITY)
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

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let mut epochs_seen: Vec<usize> = Vec::new();
    let mut last_snapshot: Vec<f32> = Vec::new();

    let embedding = {
        let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
        tsne.perplexity(PERPLEXITY)
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

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let mut epochs_seen: Vec<usize> = Vec::new();
    let mut last_snapshot: Vec<f32> = Vec::new();

    let embedding = {
        let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
        tsne.perplexity(PERPLEXITY)
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

/// The epoch callback is invoked only on the fitting thread, so it need not be
/// `Send` or `Sync`. A closure capturing an `Rc<RefCell<_>>` is neither, which the
/// previous bound rejected; this is exactly the shape a single threaded wasm
/// worker needs to forward progress. If the bound ever tightened again, this test
/// would fail to compile.
#[test]
fn epoch_callback_accepts_non_send_closure() {
    use std::cell::RefCell;
    use std::rc::Rc;

    const N: usize = 40;
    const DIM: usize = 4;
    const RUN_EPOCHS: usize = 10;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    // `Rc<RefCell<_>>` is neither `Send` nor `Sync`, so this closure is `!Send`.
    let epochs_seen = Rc::new(RefCell::new(Vec::<usize>::new()));
    let sink = Rc::clone(&epochs_seen);

    let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne.perplexity(PERPLEXITY)
        .epochs(RUN_EPOCHS)
        .epoch_callback(move |epoch, _snapshot| {
            sink.borrow_mut().push(epoch);
        })
        .barnes_hut(THETA, |sample_a, sample_b| {
            sample_a
                .iter()
                .zip(sample_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        });

    assert_eq!(
        *epochs_seen.borrow(),
        (0..RUN_EPOCHS).collect::<Vec<usize>>()
    );
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
        let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
        tsne.perplexity(PERPLEXITY)
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
        let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
        tsne.perplexity(PERPLEXITY)
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
        let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
        tsne.perplexity(PERPLEXITY)
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
        let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
        tsne.perplexity(PERPLEXITY)
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

/// Exact squared-euclidean distance used by the early-exaggeration fits below.
fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Runs a short exact fit from a fixed seed and returns the embedding snapshot at
/// the end of epoch `capture`, so two configurations can be compared at the same
/// point of the optimization. `configure` sets the knob under test.
fn exact_snapshot_at<F>(capture: usize, configure: F) -> Vec<f32>
where
    F: FnOnce(&mut tSNE<'_, f32, &[f32]>),
{
    const N: usize = 60;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();
    let seed = lcg_samples(N, NO_DIMS as usize, 99);

    let mut snapshot: Vec<f32> = Vec::new();
    {
        let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
        tsne.perplexity(PERPLEXITY)
            .epochs(capture + 1)
            .initial_embedding(&seed[..]);
        configure(&mut tsne);
        tsne.epoch_callback(|epoch, current| {
            if epoch == capture {
                snapshot.extend_from_slice(current);
            }
        })
        .exact(|a, b| squared_euclidean(a, b));
    }
    snapshot
}

/// Setting the factor to its default `12.0` explicitly must match leaving it
/// unset, so existing callers see no behavior change. The two runs execute the
/// same arithmetic, so they may differ only by parallel-reduction noise, far
/// below the embedding scale.
#[test]
fn early_exaggeration_explicit_twelve_matches_default() {
    // Compare after a single step: the optimizer is chaotic, so even identical
    // arithmetic diverges macroscopically over many epochs once parallel-reduction
    // noise is amplified. One step isolates the behavior from that amplification.
    let default_run = exact_snapshot_at(0, |_tsne| {});
    let explicit_run = exact_snapshot_at(0, |tsne| {
        tsne.early_exaggeration(12.0);
    });

    let dim = NO_DIMS as usize;
    let drift = mean_point_distance(&default_run, &explicit_run, dim);
    let diagonal = bounding_box_diagonal(&default_run, dim);
    assert!(
        drift <= 1e-4 * diagonal + 1e-6,
        "explicit 12.0 strayed {drift} from the default, diagonal {diagonal}"
    );
}

/// The factor must reach the optimizer: two fits differing only in
/// `early_exaggeration` pull the embedding apart by different amounts in the
/// early epochs, so their first-epoch snapshots are measurably different.
#[test]
fn early_exaggeration_changes_early_embedding() {
    let strong = exact_snapshot_at(0, |tsne| {
        tsne.early_exaggeration(12.0);
    });
    let weak = exact_snapshot_at(0, |tsne| {
        tsne.early_exaggeration(4.0);
    });

    let dim = NO_DIMS as usize;
    let difference = mean_point_distance(&strong, &weak, dim);
    let diagonal = bounding_box_diagonal(&strong, dim);
    assert!(
        difference > 0.05 * diagonal,
        "exaggeration 12.0 against 4.0 barely moved the first epoch: {difference} against diagonal {diagonal}"
    );
}

/// `early_exaggeration(1.0)` is a second way to express "no exaggeration": it must
/// produce the same first epoch as `stop_lying_epoch(0)`, which normalizes then
/// undoes the lying immediately. Both leave the `P` distribution unexaggerated.
#[test]
fn early_exaggeration_one_matches_stop_lying_zero() {
    let no_exaggeration = exact_snapshot_at(0, |tsne| {
        tsne.early_exaggeration(1.0);
    });
    let lying_disabled = exact_snapshot_at(0, |tsne| {
        tsne.stop_lying_epoch(0);
    });

    let dim = NO_DIMS as usize;
    let drift = mean_point_distance(&no_exaggeration, &lying_disabled, dim);
    let diagonal = bounding_box_diagonal(&no_exaggeration, dim);
    assert!(
        drift <= 1e-4 * diagonal + 1e-6,
        "the two no-exaggeration paths diverged: {drift} against diagonal {diagonal}"
    );
}

/// The Barnes-Hut fit must reject a seed whose length does not match
/// `n_samples * D`.
#[test]
#[should_panic(expected = "initial embedding has")]
fn warm_start_rejects_wrong_length_barnes_hut() {
    const N: usize = 60;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne.perplexity(PERPLEXITY)
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

    let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne.perplexity(PERPLEXITY)
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
        let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
        tsne.perplexity(PERPLEXITY)
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
    let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne.perplexity(PERPLEXITY)
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
        let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
        tsne.perplexity(PERPLEXITY)
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
            let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
            tsne.perplexity(PERPLEXITY)
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
        let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
        tsne.perplexity(PERPLEXITY)
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
            let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
            tsne.perplexity(PERPLEXITY)
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

/// Fed the neighbors the tree would find, `barnes_hut_with_neighbors` reproduces the `barnes_hut`
/// embedding. The parallel reductions are not bit-reproducible across thread schedules (rayon's
/// float reduction order depends on work-stealing), so the two paths are compared on a single-thread
/// pool, which still verifies that the supplied-neighbors entry point matches the vantage-point-tree
/// path exactly.
#[test]
fn barnes_hut_with_neighbors_matches_vptree_path() {
    const N: usize = 80;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 11);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    // A seed so both fits start from the very same embedding.
    let seed = lcg_samples(N, NO_DIMS as usize, 99);

    let n_neighbors = (3.0 * PERPLEXITY) as usize;
    let neighbors = brute_force_neighbors(&samples, n_neighbors);

    // A single-thread pool makes the reductions deterministic, so the two paths are bit-comparable.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let (reference, candidate) = pool.install(|| {
        let reference = {
            let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
            tsne.perplexity(PERPLEXITY)
                .epochs(100)
                .initial_embedding(&seed[..])
                .barnes_hut(THETA, |a, b| euclidean(a, b));
            tsne.embedding()
        };
        let candidate = {
            let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
            tsne.perplexity(PERPLEXITY)
                .epochs(100)
                .initial_embedding(&seed[..])
                .barnes_hut_with_neighbors(THETA, &neighbors);
            tsne.embedding()
        };
        (reference, candidate)
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

    let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne.perplexity(PERPLEXITY)
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

    let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne.perplexity(PERPLEXITY)
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
    let distances_row: Vec<f64> = (0..90)
        .map(|i| (20.0 + 100.0 * (i as f64 + 1.0) / 90.0_f64).sqrt())
        .collect();
    let mut p_values_row: Vec<f64> = vec![0.0; 90];
    let perplexity = 30.0;

    tsne::search_beta(&mut p_values_row, &distances_row, &perplexity);

    // The effective number of neighbours encoded by the row, exp(H(P)),
    // must match the requested perplexity.
    let entropy: f64 = p_values_row
        .iter()
        .copied()
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

    let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne.perplexity(30.0)
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

/// With neighbours supplied (so the vantage point tree's randomness is out of the picture) and a
/// fixed seed, two Barnes-Hut runs must land in the same place. Determinism is relaxed for the
/// arena (unstable sort, plain parallel reductions), so this is a tolerance check rather than a
/// bit-for-bit one: the two embeddings must agree to within a small fraction of the embedding
/// scale, which a correct and stable optimization satisfies. N is above the parallel code
/// threshold, so the build runs in parallel.
#[test]
fn barnes_hut_is_stable_run_to_run() {
    const N: usize = 600;
    const DIM: usize = 4;
    let data = lcg_samples(N, DIM, 11);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();
    let n_neighbors = (3.0 * PERPLEXITY) as usize;
    let neighbors = brute_force_neighbors(&samples, n_neighbors);
    let seed = lcg_samples(N, NO_DIMS as usize, 99);

    let run = || {
        let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
        tsne.perplexity(PERPLEXITY)
            .epochs(150)
            .initial_embedding(&seed[..])
            .barnes_hut_with_neighbors(THETA, &neighbors);
        tsne.embedding().to_vec()
    };

    let first = run();
    let second = run();
    let drift = mean_point_distance(&first, &second, NO_DIMS as usize);
    let diagonal = bounding_box_diagonal(&first, NO_DIMS as usize);
    assert!(
        drift <= 0.05 * diagonal + 1e-4,
        "two runs diverged: mean drift {drift} exceeds tolerance for diagonal {diagonal}"
    );
}

/// Regression test for the parallel build, white box, the phantom-mass class. A corrupted
/// aggregation that counts mass a cell does not hold (an empty orthant, a stale cursor) drags the
/// cell centre of mass off, which this catches: every cell centre of mass must lie within its own
/// Morton cell. Morton quantization makes point conservation automatic, which `Arena::new` asserts
/// (the leaf masses sum to `n`), so the root mass equalling `n` confirms no point was lost or
/// invented. The cloud is offset far from the origin so any centre of mass dragged toward it lands
/// outside its cell. N is above the parallel code threshold.
#[test]
fn arena_build_maintains_invariants() {
    const N: usize = 2_000;
    let mut data = lcg_samples(N, 2, 17);
    for value in data.iter_mut() {
        *value += 100.0;
    }

    let arena = tsne::arena::Arena::<f32, 2>::new(&data, N);

    assert_eq!(arena.root_count(), N, "arena lost or invented points");
    assert!(
        arena.centers_of_mass_within_cells(),
        "a cell centre of mass escaped its cell, the build aggregated phantom mass"
    );
}

/// End-to-end regression test for the same bug, reproducing the symptom directly: corrupted
/// repulsive forces let attraction collapse the whole embedding onto a handful of coordinates. A
/// healthy run spreads the points out, so most embedded positions are distinct.
#[test]
fn barnes_hut_does_not_collapse_embedding() {
    use std::collections::HashSet;

    const N: usize = 500;
    const DIM: usize = 8;
    let data = lcg_samples(N, DIM, 23);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne.perplexity(30.0)
        .epochs(1000)
        .barnes_hut(THETA, |a, b| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        });
    let embedding = tsne.embedding();

    // Count distinct positions, rounded to a hundredth. The collapse piled every point onto three
    // coordinates, a healthy embedding keeps them apart.
    let distinct: HashSet<(i64, i64)> = embedding
        .chunks_exact(2)
        .map(|point| {
            (
                (point[0] * 100.0).round() as i64,
                (point[1] * 100.0).round() as i64,
            )
        })
        .collect();
    assert!(
        distinct.len() > N / 2,
        "embedding collapsed: only {} distinct positions for {N} points",
        distinct.len()
    );
}

/// Round trip: run barnes_hut, extract affinities, inject them with initial_embedding
/// into a second tSNE, and call barnes_hut again. The continuation stays closer
/// to the seed than a random-init run, and cluster structure is preserved.
#[test]
fn affinities_round_trip_barnes_hut() {
    const N: usize = 200;
    let data = lcg_samples(N, D, 42);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    // First run.
    let mut tsne1: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne1
        .perplexity(PERPLEXITY)
        .epochs(EPOCHS)
        .barnes_hut(THETA, |a, b| euclidean(a, b));
    let embedding1 = tsne1.embedding();
    let affinities = tsne1.affinities().expect("should have affinities");

    // Second run: warm start with affinities.
    let mut tsne2: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne2
        .perplexity(PERPLEXITY)
        .epochs(500)
        .initial_embedding(embedding1.clone())
        .with_affinities(affinities);
    tsne2.barnes_hut(THETA, |a, b| euclidean(a, b));
    let embedding2 = tsne2.embedding();

    // The continuation must start near the seed, not restart from random.
    // Compare against a fresh random-init run: the warm-start embedding
    // should be closer to the seed than a random run would be.
    let mut tsne_rand: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne_rand
        .perplexity(PERPLEXITY)
        .epochs(500)
        .barnes_hut(THETA, |a, b| euclidean(a, b));
    let embedding_rand = tsne_rand.embedding();

    let dist_warm = mean_point_distance(&embedding1, &embedding2, D);
    let dist_rand = mean_point_distance(&embedding1, &embedding_rand, D);
    assert!(
        dist_warm < dist_rand,
        "warm start ({}) should be closer to seed than random ({})",
        dist_warm,
        dist_rand,
    );

    // Cluster structure should be preserved: relative ordering of nearby points
    // should be similar.
    let data10 = &data[..D * 10];
    let dist1 = mean_point_distance(data10, &embedding1[..D * 10], D);
    let dist2 = mean_point_distance(data10, &embedding2[..D * 10], D);
    assert!(
        (dist1 - dist2).abs() < dist1 * 0.5,
        "cluster structure changed too much: {} vs {}",
        dist1,
        dist2,
    );
}

/// Equivalence: a second barnes_hut call reusing cached affinities produces
/// the same embedding as a plain barnes_hut continuation within tolerance.
#[test]
fn affinities_equivalence_first_step() {
    const N: usize = 100;
    const CAPTURE: usize = 1; // Compare after 1 epoch.

    let data = lcg_samples(N, D, 99);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    // Build affinities from a reference run.
    let mut tsne_ref: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne_ref
        .perplexity(PERPLEXITY)
        .epochs(EPOCHS)
        .barnes_hut(THETA, |a, b| euclidean(a, b));
    let affinities = tsne_ref.affinities().unwrap();
    let seed = tsne_ref.embedding();

    // Path A: plain barnes_hut continuation from the seed.
    let mut tsne_a: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne_a
        .perplexity(PERPLEXITY)
        .epochs(CAPTURE + 1)
        .initial_embedding(seed.clone());
    tsne_a.barnes_hut(THETA, |a, b| euclidean(a, b));
    let result_a = tsne_a.embedding();

    // Path B: barnes_hut with cached affinities from the same seed.
    let mut tsne_b: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne_b
        .perplexity(PERPLEXITY)
        .epochs(CAPTURE + 1)
        .initial_embedding(seed)
        .with_affinities(affinities);
    tsne_b.barnes_hut(THETA, |a, b| euclidean(a, b));
    let result_b = tsne_b.embedding();

    // Two runs on the same thread pool may differ by parallel-reduction noise,
    // so use a tolerance.
    let max_diff: f32 = result_a
        .iter()
        .zip(result_b.iter())
        .map(|(a, b)| (a - b).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let scale = result_a
        .iter()
        .map(|v| v.abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    assert!(
        max_diff / scale < 1e-3,
        "precomputed path diverged from reference: max_diff={max_diff}, scale={scale}",
    );
}

/// affinities() returns values summing to about 1 regardless of run length.
#[test]
fn affinities_pristine_independent_of_run_length() {
    const N: usize = 100;

    let data = lcg_samples(N, D, 7);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    // Short run: fewer epochs than stop_lying_epoch (250).
    let mut tsne_short: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne_short
        .perplexity(PERPLEXITY)
        .epochs(50)
        .barnes_hut(THETA, |a, b| euclidean(a, b));
    let affinities_short = tsne_short.affinities().unwrap();
    let sum_short: f32 = affinities_short.values.iter().sum();

    // Long run: more epochs than stop_lying_epoch.
    let mut tsne_long: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne_long
        .perplexity(PERPLEXITY)
        .epochs(1000)
        .barnes_hut(THETA, |a, b| euclidean(a, b));
    let affinities_long = tsne_long.affinities().unwrap();
    let sum_long: f32 = affinities_long.values.iter().sum();

    // Both should sum to approximately 1 (pristine P).
    assert!(
        (sum_short - 1.0).abs() < 0.05,
        "short run affinities sum to {sum_short}, expected ~1",
    );
    assert!(
        (sum_long - 1.0).abs() < 0.05,
        "long run affinities sum to {sum_long}, expected ~1",
    );
    // And they should be identical since the data and perplexity are the same.
    assert_eq!(
        affinities_short.rows, affinities_long.rows,
        "row structure differs between short and long runs",
    );
    assert_eq!(
        affinities_short.columns, affinities_long.columns,
        "column structure differs between short and long runs",
    );
    for (a, b) in affinities_short
        .values
        .iter()
        .zip(affinities_long.values.iter())
    {
        assert!((a - b).abs() < 1e-6, "value differs: {} vs {}", a, b,);
    }
}

/// Cached affinities are reused in barnes_hut_with_neighbors when custom
/// neighbors match the cached neighbor indices. When they differ, the
/// custom neighbors are used and affinities are regenerated.
#[test]
fn affinities_work_with_custom_neighbors() {
    const N: usize = 100;

    let data = lcg_samples(N, D, 42);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    // Build affinities from a reference run.
    let mut tsne_ref: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne_ref
        .perplexity(PERPLEXITY)
        .epochs(EPOCHS)
        .barnes_hut(THETA, |a, b| euclidean(a, b));
    let affinities = tsne_ref.affinities().unwrap();
    let seed = tsne_ref.embedding();

    // Build different custom neighbors.
    let mut rng = rand::rng();
    let different_neighbors: Vec<Vec<Neighbor<f32>>> = samples
        .iter()
        .enumerate()
        .map(|(sample_idx, _)| {
            let mut row: Vec<Neighbor<f32>> = (0..N)
                .filter_map(|i| {
                    if i == sample_idx {
                        None
                    } else {
                        Some(Neighbor {
                            index: i,
                            distance: rng.random_range(0.0..100.0),
                        })
                    }
                })
                .collect();
            row.truncate(15);
            row
        })
        .collect();

    // Path B: cached affinities + different custom neighbors.
    let mut tsne_b: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne_b
        .perplexity(PERPLEXITY)
        .epochs(50)
        .initial_embedding(seed.clone())
        .with_affinities(affinities.clone());
    tsne_b.barnes_hut_with_neighbors(THETA, &different_neighbors);
    let result_b = tsne_b.embedding();

    // Path C: cached affinities + barnes_hut (no custom neighbors).
    let mut tsne_c: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne_c
        .perplexity(PERPLEXITY)
        .epochs(50)
        .initial_embedding(seed)
        .with_affinities(affinities);
    tsne_c.barnes_hut(THETA, |a, b| euclidean(a, b));
    let result_c = tsne_c.embedding();

    // When neighbors differ, Path B uses custom neighbors and diverges
    // from the cached path (Path C).
    let max_diff: f32 = result_b
        .iter()
        .zip(result_c.iter())
        .map(|(a, b)| (a - b).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let scale = result_b
        .iter()
        .map(|v| v.abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    assert!(
        max_diff / scale > 0.05,
        "different neighbors should invalidate cached affinities: max_diff={}, scale={}",
        max_diff,
        scale,
    );
}

/// Cached affinities path works with random seed when no initial_embedding is set.
#[test]
fn cached_affinities_random_seed() {
    const N: usize = 50;

    let data = lcg_samples(N, D, 7);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    // Build affinities from a reference run.
    let mut tsne_ref: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne_ref
        .perplexity(PERPLEXITY)
        .epochs(EPOCHS)
        .barnes_hut(THETA, |a, b| euclidean(a, b));
    let affinities = tsne_ref.affinities().unwrap();

    // Cached path with random seed (no initial_embedding).
    let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne.perplexity(PERPLEXITY)
        .epochs(50)
        .with_affinities(affinities);
    tsne.barnes_hut(THETA, |a, b| euclidean(a, b));
    let embedding = tsne.embedding();

    // Basic sanity: embedding has correct shape and finite values.
    assert_eq!(embedding.len(), N * 2);
    assert!(
        embedding.iter().all(|v| v.is_finite()),
        "embedding contains non-finite values",
    );
    // Values should not all be zero (random init should produce variation).
    assert!(
        embedding.iter().any(|v| *v != 0.0),
        "embedding is all zeros, random init may have failed",
    );
}

/// Mismatched dataset size causes cached affinities to be discarded and
/// rebuilt via the VPTree path.
#[test]
fn cached_affinities_discarded_on_dataset_mismatch() {
    const N_SMALL: usize = 50;
    const N_LARGE: usize = 100;

    let data_small = lcg_samples(N_SMALL, D, 7);
    let samples_small: Vec<&[f32]> = data_small.chunks(D).collect();

    // Build affinities from a small dataset.
    let mut tsne_small: tSNE<f32, &[f32]> = tSNE::new(&samples_small);
    tsne_small
        .perplexity(PERPLEXITY)
        .epochs(EPOCHS)
        .barnes_hut(THETA, |a, b| euclidean(a, b));
    let affinities = tsne_small.affinities().unwrap();

    // Inject affinities from a different-sized dataset.
    let data_large = lcg_samples(N_LARGE, D, 7);
    let samples_large: Vec<&[f32]> = data_large.chunks(D).collect();
    let mut tsne_large: tSNE<f32, &[f32]> = tSNE::new(&samples_large);
    tsne_large
        .perplexity(PERPLEXITY)
        .with_affinities(affinities);

    // Should not panic, affinities are silently discarded and rebuilt.
    tsne_large.barnes_hut(THETA, |a, b| euclidean(a, b));
    let embedding = tsne_large.embedding();

    assert_eq!(embedding.len(), N_LARGE * 2);
    assert!(
        embedding.iter().all(|v| v.is_finite()),
        "embedding contains non-finite values",
    );
}
/// Changing perplexity after a fit must invalidate the cached affinities:
/// the second run recomputes P with the new perplexity rather than reusing
/// stale values. Both paths run in a single-thread pool so the Barnes-Hut
/// reductions are deterministic and the embeddings are bit-comparable.
#[test]
fn cached_affinities_invalidated_on_perplexity_change() {
    const N: usize = 80;

    let data = lcg_samples(N, D, 42);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    let new_perplexity = 2.0_f32;
    let n_neighbors = (3.0 * new_perplexity) as usize;
    let neighbors = brute_force_neighbors(&samples, n_neighbors);

    // Seed from a preliminary run.
    let mut tsne_seed: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne_seed.perplexity(PERPLEXITY).epochs(20);
    tsne_seed.barnes_hut(THETA, |a, b| euclidean(a, b));
    let seed = tsne_seed.embedding();

    // Build reference affinities with new_perplexity.
    let mut tsne_ref: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne_ref.perplexity(new_perplexity);
    tsne_ref.barnes_hut_with_neighbors(THETA, &neighbors);
    let affinities = tsne_ref.affinities().unwrap();

    // Single-thread pool for deterministic Barnes-Hut reductions.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let (result_a, result_b) = pool.install(|| {
        // Path A: inject affinities, change perplexity, run.
        // The cache should be invalidated because perplexity changed.
        let mut tsne_a: tSNE<f32, &[f32]> = tSNE::new(&samples);
        tsne_a
            .perplexity(new_perplexity)
            .with_affinities(affinities.clone())
            .epochs(50)
            .initial_embedding(seed.clone());
        // Change perplexity AFTER caching.
        tsne_a.perplexity(PERPLEXITY);
        tsne_a.barnes_hut_with_neighbors(THETA, &neighbors);
        let result_a = tsne_a.embedding();

        // Path B: inject affinities, same perplexity, run.
        // Cache should be reused.
        let mut tsne_b: tSNE<f32, &[f32]> = tSNE::new(&samples);
        tsne_b
            .perplexity(new_perplexity)
            .with_affinities(affinities)
            .epochs(50)
            .initial_embedding(seed);
        tsne_b.barnes_hut_with_neighbors(THETA, &neighbors);
        let result_b = tsne_b.embedding();

        (result_a, result_b)
    });

    // Path A recomputes P (cache invalidated by perplexity change),
    // Path B reuses cached P. Different P distributions produce
    // different embeddings, so they must NOT match.
    let any_diff = result_a
        .iter()
        .zip(result_b.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        any_diff,
        "embeddings are identical: perplexity change did not invalidate cache",
    );
}
/// Running barnes_hut twice on the same instance (second run hits the
/// cached-affinities path) must produce the same result as running on a fresh
/// instance with injected affinities. If `stop_lying_fired` is not reset before
/// the second run, `stop_lying` never fires and the exaggeration is never
/// removed, producing a different embedding.
#[test]
fn cached_affinities_reset_stop_lying_flag() {
    const N: usize = 80;
    const SL_EPOCH: usize = 5;
    const RUN_EPOCHS: usize = 20;

    let data = lcg_samples(N, D, 42);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    let n_neighbors = (3.0 * PERPLEXITY) as usize;
    let neighbors = brute_force_neighbors(&samples, n_neighbors);

    // Reference run to build affinities.
    let mut tsne_ref: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne_ref.perplexity(PERPLEXITY);
    tsne_ref.barnes_hut_with_neighbors(THETA, &neighbors);
    let affinities = tsne_ref.affinities().unwrap();

    // Seed from a preliminary run.
    let mut tsne_seed: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne_seed.perplexity(PERPLEXITY).epochs(20);
    tsne_seed.barnes_hut(THETA, |a, b| euclidean(a, b));
    let seed = tsne_seed.embedding();

    // Single-thread pool for deterministic reductions.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let (result_a, result_b) = pool.install(|| {
        // Path A: fresh instance with injected affinities.
        // stop_lying_fired starts false; stop_lying fires at SL_EPOCH.
        let mut tsne_a: tSNE<f32, &[f32]> = tSNE::new(&samples);
        tsne_a
            .perplexity(PERPLEXITY)
            .with_affinities(affinities.clone())
            .stop_lying_epoch(SL_EPOCH)
            .epochs(RUN_EPOCHS)
            .initial_embedding(seed.clone());
        tsne_a.barnes_hut_with_neighbors(THETA, &neighbors);
        let result_a = tsne_a.embedding();

        // Path B: run twice on the same instance.
        // First run sets stop_lying_fired = true.
        // Second run should reset it; if not, stop_lying never fires.
        let mut tsne_b: tSNE<f32, &[f32]> = tSNE::new(&samples);
        tsne_b
            .perplexity(PERPLEXITY)
            .stop_lying_epoch(SL_EPOCH)
            .epochs(RUN_EPOCHS);
        tsne_b.barnes_hut_with_neighbors(THETA, &neighbors);
        tsne_b.epochs(RUN_EPOCHS).initial_embedding(seed);
        tsne_b.barnes_hut_with_neighbors(THETA, &neighbors);
        let result_b = tsne_b.embedding();

        (result_a, result_b)
    });

    assert_eq!(
        result_a, result_b,
        "second cached-affinities run produced different embedding: stop_lying_fired was not reset",
    );
}

#[test]
#[should_panic(expected = "cached affinities were built with a different perplexity")]
fn with_affinities_panics_on_perplexity_mismatch() {
    let data: Vec<f32> = (0..800).map(|i| i as f32).collect();
    let samples: Vec<&[f32]> = data.chunks(4).collect();

    // Build affinities at perplexity 30.
    let mut tsne: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne.perplexity(30.0).epochs(20);
    tsne.barnes_hut(THETA, |a, b| euclidean(a, b));
    let affinities = tsne.affinities().unwrap();

    // Try to inject into an instance with a different perplexity.
    let mut tsne2: tSNE<f32, &[f32]> = tSNE::new(&samples);
    tsne2.perplexity(5.0);
    tsne2.with_affinities(affinities);
}
