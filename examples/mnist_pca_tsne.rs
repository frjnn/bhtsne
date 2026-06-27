//! Run Barnes-Hut t-SNE on MNIST pre-reduced to 20 dimensions by PCA.
//!
//! The PCA-compressed dataset is shipped in `data/mnist_pca20.npz`
//! (numpy compressed archive with `data` and `labels` arrays).
//!
//! Produces a CSV embedding and a PNG scatter plot colored by digit.

use std::error::Error;
use std::time::Instant;

use bhtsne::tSNE;
use chrono::Utc;
use npyz::npz::NpzArchive;
use plotters::prelude::*;

const PCA_DIMS: usize = 20;
const PERPLEXITY: f64 = 30.0;
const THETA: f64 = 0.5;
const EPOCHS: usize = 1000;

/// Distinct colors for the ten MNIST digits.
const DIGIT_COLORS: [RGBAColor; 10] = [
    RGBAColor(214, 39, 40, 0.7),
    RGBAColor(37, 150, 190, 0.7),
    RGBAColor(26, 152, 69, 0.7),
    RGBAColor(230, 126, 34, 0.7),
    RGBAColor(155, 89, 182, 0.7),
    RGBAColor(22, 160, 133, 0.7),
    RGBAColor(192, 57, 43, 0.7),
    RGBAColor(44, 62, 80, 0.7),
    RGBAColor(211, 84, 0, 0.7),
    RGBAColor(149, 165, 166, 0.7),
];

fn main() -> Result<(), Box<dyn Error>> {
    let t0 = Instant::now();
    let mut npz = NpzArchive::open("data/mnist_pca20.npz")?;
    let data: Vec<f32> = npz.by_name("data")?.expect("missing 'data'").into_vec()?;
    let labels: Vec<u8> = npz
        .by_name("labels")?
        .expect("missing 'labels'")
        .into_vec()?;
    let n_samples = data.len() / PCA_DIMS;
    let rows: Vec<&[f32]> = data.chunks_exact(PCA_DIMS).collect();
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!("Loaded {n_samples} samples x {PCA_DIMS} PCA components");

    let t1 = Instant::now();
    let mut tsne: tSNE<f64, &[f32]> = tSNE::new(&rows);
    tsne.perplexity(PERPLEXITY)
        .epochs(EPOCHS)
        .epoch_callback(|epoch, _embedding| {
            if epoch % 100 == 0 || epoch == EPOCHS - 1 {
                println!("  epoch {epoch}/{EPOCHS}");
            }
        })
        .barnes_hut(THETA, |a, b| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| {
                    let d = (*x as f64) - (*y as f64);
                    d * d
                })
                .sum::<f64>()
                .sqrt()
        });
    let tsne_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let embedding = tsne.embedding();
    let kl = tsne.kl_divergence().unwrap_or(f64::NAN);

    println!("t-SNE done: KL divergence {kl:.4}");

    let csv_out = "mnist_pca20_tsne.csv";
    tsne.write_csv(csv_out)?;
    println!("Embedding written to {csv_out} ({n_samples} rows, 2 columns)");

    let png_out = "mnist_pca20_tsne.png";
    println!("Writing scatter plot to {png_out}...");
    plot_embedding(&embedding, &labels, png_out, load_ms, tsne_ms)?;
    println!("Done");
    Ok(())
}

fn plot_embedding(
    embedding: &[f64],
    labels: &[u8],
    path: &str,
    load_ms: f64,
    tsne_ms: f64,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (1200, 1280)).into_drawing_area();
    root.fill(&WHITE)?;

    let (x_min, x_max, y_min, y_max) = embedding.chunks_exact(2).fold(
        (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ),
        |(xlo, xhi, ylo, yhi), c| (xlo.min(c[0]), xhi.max(c[0]), ylo.min(c[1]), yhi.max(c[1])),
    );

    let x_span = x_max - x_min;
    let y_span = y_max - y_min;
    let x_range = (x_min - x_span * 0.02)..(x_max + x_span * 0.02);
    let y_range = (y_max + y_span * 0.02)..(y_min - y_span * 0.02);

    let mut chart = ChartBuilder::on(&root)
        .margin(60)
        .margin_bottom(80)
        .build_cartesian_2d(x_range, y_range)?;

    for digit in 0..10u8 {
        let points: Vec<(f64, f64)> = embedding
            .chunks_exact(2)
            .zip(labels.iter())
            .filter_map(|(c, &l)| if l == digit { Some((c[0], c[1])) } else { None })
            .collect();

        let color = DIGIT_COLORS[digit as usize];

        chart
            .draw_series(PointSeries::of_element(
                points,
                1,
                &color,
                &|pt: (f64, f64), size, style| Circle::new(pt, size, style),
            ))?
            .label(format!("{digit}"))
            .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], color.filled()));
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .border_style(BLACK.mix(0.3))
        .background_style(WHITE.mix(0.8))
        .draw()?;

    let commit = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    let date = Utc::now().format("%Y-%m-%d");
    let footer = format!(
        "commit {}  •  {}  •  load {:.0} ms  •  t-SNE {:.0} ms",
        commit, date, load_ms, tsne_ms,
    );
    root.draw(&Text::new(
        footer,
        (600, 1260),
        ("sans-serif", 16).into_font().color(&BLACK),
    ))?;

    root.present()?;
    Ok(())
}
