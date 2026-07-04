//! Real-input FFT engine behind the interpolation-based repulsion of
//! [`super::interpolation`] (the FIt-SNE path), built on [`rustfft`] and
//! [`realfft`].

use std::sync::Arc;

use num_traits::Float;

use rayon::prelude::*;

use rustfft::{Fft, FftNum, FftPlanner, num_complex::Complex};

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// Transpose block size.
const TR_BLOCK: usize = 32;

/// Cached real and complex plans for one per-axis transform length, with the separable
/// 1-D/2-D drivers and a reused transpose scratch.
#[derive(Clone)]
pub(crate) struct FftGrid<T: FftNum> {
    len: usize,
    spectrum_cols: usize,
    r2c: Arc<dyn RealToComplex<T>>,
    c2r: Arc<dyn ComplexToReal<T>>,
    col_forward: Arc<dyn Fft<T>>,
    col_inverse: Arc<dyn Fft<T>>,
    transpose: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
    scratch_len: usize,
}

impl<T: FftNum + Float> FftGrid<T> {
    /// Plans the real and complex transforms for the given per-axis `len`.
    ///
    /// # Arguments
    ///
    /// `len` - axis length.
    pub(crate) fn new(len: usize) -> Self {
        let mut real_planner = RealFftPlanner::<T>::new();
        let mut planner = FftPlanner::<T>::new();
        let spectrum_cols = len / 2 + 1;

        let r2c = real_planner.plan_fft_forward(len);
        let c2r = real_planner.plan_fft_inverse(len);
        let col_forward = planner.plan_fft_forward(len);
        let col_inverse = planner.plan_fft_inverse(len);

        // One scratch slot per worker, each sized to the most demanding of the four
        // plans. The worker count is a starting estimate; `resize_scratch_to_pool`
        // refits it to the pool actually running each transform.
        let scratch_len = r2c
            .get_scratch_len()
            .max(c2r.get_scratch_len())
            .max(col_forward.get_inplace_scratch_len())
            .max(col_inverse.get_inplace_scratch_len())
            .max(1);
        let workers = rayon::current_num_threads().max(1);

        Self {
            len,
            spectrum_cols,
            r2c,
            c2r,
            col_forward,
            col_inverse,
            transpose: vec![Complex::new(T::zero(), T::zero()); len * spectrum_cols],
            scratch: vec![Complex::new(T::zero(), T::zero()); workers * scratch_len],
            scratch_len,
        }
    }

    /// Rows per parallel job so the work splits into at most one batch per scratch
    /// slot; each job then walks its rows through a single preallocated slot.
    #[inline]
    fn batch_rows(&self, n_rows: usize) -> usize {
        let workers = self.scratch.len() / self.scratch_len;

        n_rows.div_ceil(workers.max(1)).max(1)
    }

    /// Re-sizes the scratch pool to one slot per thread of the pool the transform is
    /// *currently* running in, rather than the one active when the grid was built.
    fn resize_scratch_to_pool(&mut self) {
        let needed = rayon::current_num_threads().max(1) * self.scratch_len;
        if self.scratch.len() != needed {
            self.scratch
                .resize(needed, Complex::new(T::zero(), T::zero()));
        }
    }

    /// Number of complex samples in the half-spectrum of a `DIMS`-dimensional grid:
    /// `len^(DIMS - 1) * (len / 2 + 1)`.
    pub(crate) const fn spectrum_len<const DIMS: usize>(&self) -> usize {
        self.len.pow((DIMS - 1) as u32) * self.spectrum_cols
    }

    /// Forward transform of a real `len^DIMS` grid into its half `spectrum`.
    ///
    /// `DIMS` is the caller's embedding dimensionality and is either `1` or `2`.
    pub(crate) fn forward<const DIMS: usize>(
        &mut self,
        grid: &mut [T],
        spectrum: &mut [Complex<T>],
    ) {
        self.resize_scratch_to_pool();
        self.rows_r2c(grid, spectrum);
        if DIMS == 2 {
            self.column_pass(true, spectrum);
        }
    }

    /// Inverse transform of a half `spectrum` back to a real `len^DIMS` grid.
    ///
    /// Unnormalized, like the underlying transforms; the caller folds the
    /// `1 / len^DIMS` factor into the kernel spectrum.
    ///
    /// The `spectrum` is consumed as scratch.
    pub(crate) fn inverse<const DIMS: usize>(
        &mut self,
        spectrum: &mut [Complex<T>],
        grid: &mut [T],
    ) {
        self.resize_scratch_to_pool();
        if DIMS == 2 {
            self.column_pass(false, spectrum);
        }
        clear_real_axis_imag(spectrum, self.len, self.spectrum_cols);
        self.rows_c2r(spectrum, grid);
    }

    /// Transforms the second axis of a 2-D half-spectrum by transposing it so
    /// each column becomes a contiguous row, running the length-`len` complex
    /// transform over those rows, and transposing back.
    fn column_pass(&mut self, forward: bool, spectrum: &mut [Complex<T>]) {
        let (m, h) = (self.len, self.spectrum_cols);

        transpose(spectrum, &mut self.transpose, m, h);
        self.column_rows(m, forward);
        transpose(&self.transpose, spectrum, h, m);
    }

    /// Real-to-complex transform of every contiguous length-`len` row of `grid` into
    /// the matching length-`spectrum_cols` row of `spectrum`.
    fn rows_r2c(&mut self, grid: &mut [T], spectrum: &mut [Complex<T>]) {
        let (m, h, slot_len) = (self.len, self.spectrum_cols, self.scratch_len);
        let batch = self.batch_rows(grid.len() / m);
        let r2c = &self.r2c;
        let scratch = &mut self.scratch;

        grid.par_chunks_mut(batch * m)
            .zip(spectrum.par_chunks_mut(batch * h))
            .zip(scratch.par_chunks_mut(slot_len))
            .for_each(|((grid_batch, spectrum_batch), slot)| {
                for (grid_row, spectrum_row) in
                    grid_batch.chunks_mut(m).zip(spectrum_batch.chunks_mut(h))
                {
                    r2c.process_with_scratch(grid_row, spectrum_row, slot)
                        .expect("real-to-complex transform");
                }
            });
    }

    /// Complex-to-real transform of every length-`spectrum_cols` row of `spectrum`
    /// back to a length-`len` row of `grid`. The spectrum rows are consumed as
    /// scratch.
    fn rows_c2r(&mut self, spectrum: &mut [Complex<T>], grid: &mut [T]) {
        let (m, h, slot_len) = (self.len, self.spectrum_cols, self.scratch_len);
        let batch = self.batch_rows(grid.len() / m);
        let c2r = &self.c2r;
        let scratch = &mut self.scratch;

        grid.par_chunks_mut(batch * m)
            .zip(spectrum.par_chunks_mut(batch * h))
            .zip(scratch.par_chunks_mut(slot_len))
            .for_each(|((grid_batch, spectrum_batch), slot)| {
                for (grid_row, spectrum_row) in
                    grid_batch.chunks_mut(m).zip(spectrum_batch.chunks_mut(h))
                {
                    c2r.process_with_scratch(spectrum_row, grid_row, slot)
                        .expect("complex-to-real transform");
                }
            });
    }

    /// In-place complex transform of every contiguous length-`m` row of the transpose
    /// buffer (the columns of the half-spectrum), batched per worker over the pool.
    fn column_rows(&mut self, m: usize, forward: bool) {
        let slot_len = self.scratch_len;
        let batch = self.batch_rows(self.transpose.len() / m);
        let fft = if forward {
            &self.col_forward
        } else {
            &self.col_inverse
        };
        let buffer = &mut self.transpose;
        let scratch = &mut self.scratch;

        buffer
            .par_chunks_mut(batch * m)
            .zip(scratch.par_chunks_mut(slot_len))
            .for_each(|(batch_rows, slot)| {
                for row in batch_rows.chunks_mut(m) {
                    fft.process_with_scratch(row, slot);
                }
            });
    }
}

/// Zeros the imaginary part of the bins; they are real in exact arithmetic,
/// so this only discards accumulated float noise.
#[inline]
fn clear_real_axis_imag<T: FftNum + Float>(spectrum: &mut [Complex<T>], m: usize, h: usize) {
    let even = m.is_multiple_of(2);

    spectrum.chunks_mut(h).for_each(|row| {
        row[0].im = T::zero();
        if even {
            row[h - 1].im = T::zero();
        }
    });
}

/// Out-of-place transpose of a `rows * cols` row-major grid into a `cols * rows` one:
/// `dst[c, r] = src[r, c]`.
#[inline]
fn transpose<T: FftNum>(src: &[Complex<T>], dst: &mut [Complex<T>], rows: usize, cols: usize) {
    dst.par_chunks_mut(TR_BLOCK * rows)
        .enumerate()
        .for_each(|(block, dst_block)| {
            let col0 = block * TR_BLOCK;
            let block_cols = dst_block.len() / rows;
            let mut row0 = 0;
            while row0 < rows {
                let row1 = (row0 + TR_BLOCK).min(rows);
                for r in row0..row1 {
                    let src_tile = &src[r * cols + col0..r * cols + col0 + block_cols];
                    dst_block[r..]
                        .iter_mut()
                        .step_by(rows)
                        .zip(src_tile)
                        .for_each(|(slot, &value)| *slot = value);
                }
                row0 = row1;
            }
        });
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;

    #[test]
    fn round_trip_2d() {
        let m = 12;
        let total = m * m;
        let mut rng = StdRng::seed_from_u64(0xABCD);
        let original: Vec<f64> = (0..total).map(|_| rng.random_range(-0.5..0.5)).collect();

        let mut grid = FftGrid::<f64>::new(m);
        let mut real = original.clone();
        let mut spectrum = vec![Complex::new(0.0, 0.0); grid.spectrum_len::<2>()];
        grid.forward::<2>(&mut real, &mut spectrum);
        grid.inverse::<2>(&mut spectrum, &mut real);
        let inv_total = 1.0 / total as f64;
        for (got, want) in real.iter().zip(original.iter()) {
            assert!(
                (got * inv_total - want).abs() < 1e-12,
                "got {got}, want {want}"
            );
        }
    }

    #[test]
    fn round_trip_1d() {
        let m = 16;
        let mut rng = StdRng::seed_from_u64(0xBEEF);
        let original: Vec<f64> = (0..m).map(|_| rng.random_range(-0.5..0.5)).collect();

        let mut grid = FftGrid::<f64>::new(m);
        let mut real = original.clone();
        let mut spectrum = vec![Complex::new(0.0, 0.0); grid.spectrum_len::<1>()];
        grid.forward::<1>(&mut real, &mut spectrum);
        grid.inverse::<1>(&mut spectrum, &mut real);
        let inv_total = 1.0 / m as f64;
        for (got, want) in real.iter().zip(original.iter()) {
            assert!((got * inv_total - want).abs() < 1e-12);
        }
    }

    #[test]
    fn fft_circular_convolution_2d_matches_direct() {
        let m = 8;
        let total = m * m;
        let mut rng = StdRng::seed_from_u64(0x5151);
        let a: Vec<f64> = (0..total).map(|_| rng.random_range(-0.5..0.5)).collect();
        let b: Vec<f64> = (0..total).map(|_| rng.random_range(-0.5..0.5)).collect();

        // Direct 2-D circular convolution.
        let mut direct = vec![0.0; total];
        for ki in 0..m {
            for kj in 0..m {
                let mut acc = 0.0;
                for ji in 0..m {
                    for jj in 0..m {
                        let ai = (ki + m - ji) % m;
                        let aj = (kj + m - jj) % m;
                        acc += a[ai * m + aj] * b[ji * m + jj];
                    }
                }
                direct[ki * m + kj] = acc;
            }
        }

        let mut grid = FftGrid::<f64>::new(m);
        let len = grid.spectrum_len::<2>();
        let mut fa = a.clone();
        let mut sa = vec![Complex::new(0.0, 0.0); len];
        grid.forward::<2>(&mut fa, &mut sa);
        let mut fb = b.clone();
        let mut sb = vec![Complex::new(0.0, 0.0); len];
        grid.forward::<2>(&mut fb, &mut sb);

        let mut prod: Vec<Complex<f64>> = sa.iter().zip(sb.iter()).map(|(x, y)| x * y).collect();
        let mut out = vec![0.0; total];
        grid.inverse::<2>(&mut prod, &mut out);
        let inv_total = 1.0 / total as f64;
        for k in 0..total {
            assert!((out[k] * inv_total - direct[k]).abs() < 1e-10, "k={k}");
        }
    }

    #[test]
    fn round_trip_across_thread_pools() {
        let m = 20;
        let total = m * m;
        let mut rng = StdRng::seed_from_u64(0xD00D);
        let original: Vec<f64> = (0..total).map(|_| rng.random_range(-0.5..0.5)).collect();

        // Built against the ambient pool.
        let mut grid = FftGrid::<f64>::new(m);

        for threads in [1usize, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut real = original.clone();
                let mut spectrum = vec![Complex::new(0.0, 0.0); grid.spectrum_len::<2>()];
                grid.forward::<2>(&mut real, &mut spectrum);
                grid.inverse::<2>(&mut spectrum, &mut real);
                let inv_total = 1.0 / total as f64;
                for (got, want) in real.iter().zip(original.iter()) {
                    assert!(
                        (got * inv_total - want).abs() < 1e-12,
                        "threads={threads}: got {got}, want {want}"
                    );
                }
            });
        }
    }
}
