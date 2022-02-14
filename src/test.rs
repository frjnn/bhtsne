use super::{tSNE, tsne};

const D: usize = 4;
const THETA: f32 = 0.5;
const PERPLEXITY: f32 = 10.;
const EPOCHS: usize = 2_000;
const NO_DIMS: u8 = 2;

#[test]
#[cfg(not(tarpaulin_include))]
fn set_learning_rate() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    tsne.learning_rate(15.);
    assert_eq!(tsne.learning_rate, 15.);
}

#[test]
#[cfg(not(tarpaulin_include))]
fn set_epochs() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    tsne.epochs(15);
    assert_eq!(tsne.epochs, 15);
}

#[test]
#[cfg(not(tarpaulin_include))]
fn set_momentum() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    tsne.momentum(15.);
    assert_eq!(tsne.momentum, 15.);
}

#[test]
#[cfg(not(tarpaulin_include))]
fn set_final_momentum() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    tsne.final_momentum(15.);
    assert_eq!(tsne.final_momentum, 15.);
}

#[test]
#[cfg(not(tarpaulin_include))]
fn set_momentum_switch_epoch() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    tsne.momentum_switch_epoch(15);
    assert_eq!(tsne.momentum_switch_epoch, 15);
}

#[test]
#[cfg(not(tarpaulin_include))]
fn set_stop_lying_epoch() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    tsne.stop_lying_epoch(15);
    assert_eq!(tsne.stop_lying_epoch, 15);
}

#[test]
#[cfg(not(tarpaulin_include))]
fn set_embedding_dim() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    tsne.embedding_dim(3);
    assert_eq!(tsne.embedding_dim, 3);
}

#[test]
#[cfg(not(tarpaulin_include))]
fn set_perplexity() {
    let mut tsne: tSNE<f32, f32> = tSNE::new(&[0.]);
    tsne.perplexity(15.);
    assert_eq!(tsne.perplexity, 15.);
}

#[test]
#[cfg(not(tarpaulin_include))]
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
            &samples.len(),
            &(tsne.embedding_dim as usize)
        ) < 0.5
    );
}

#[test]
#[cfg(not(tarpaulin_include))]
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
            &samples.len(),
            &(tsne.embedding_dim as usize),
            &THETA
        ) < 5.0
    );
}
