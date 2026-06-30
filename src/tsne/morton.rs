//! Z-order (Morton) bit interleaving for the linear quadtree arena.
//!
//! The arena encodes each embedding point as a single `u64` Morton code: the per-axis quantized
//! coordinates are bit-interleaved so that sorting the codes lays the points out in Z-order, where
//! points sharing a code prefix share a tree cell. Only `D in {2, 3}` are supported, the two
//! dimensionalities a `u64` code covers with ample precision (32 bits per axis at `D = 2`, lossless
//! for `f32`, and 21 bits per axis at `D = 3`). The [`Morton`] trait is implemented for those two
//! dimensions alone, so the bound `Dim<D>: Morton<D>` is what restricts the Barnes-Hut tree path to
//! `D in {2, 3}` at compile time.

use num_traits::Float;

/// Per-dimension Z-order codec, implemented only for [`Dim<2>`] and [`Dim<3>`].
pub trait Morton<const D: usize> {
    /// Number of children a fully occupied internal cell can have, `2^D`.
    const CHILDREN: usize;

    /// Bits of precision per axis in the `u64` code.
    const BITS: u32;

    /// Interleaves the `D` quantized per-axis coordinates into one Z-order code.
    fn encode(coords: [u32; D]) -> u64;

    /// Inverse of [`encode`], recovering the per-axis quantized coordinates.
    ///
    /// [`encode`]: Morton::encode
    fn decode(code: u64) -> [u32; D];
}

/// Carrier type the per-`D` [`Morton`] implementations attach to, since a trait needs a type to
/// dispatch on while `D` stays a const generic on the arena.
pub struct Dim<const D: usize>;

/// Spreads the low 32 bits of `x` into the even bit positions (one zero gap between each), the
/// 2D Morton building block.
#[inline]
const fn part_1by1(mut x: u64) -> u64 {
    x &= 0x0000_0000_ffff_ffff;
    x = (x | (x << 16)) & 0x0000_ffff_0000_ffff;
    x = (x | (x << 8)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;

    (x | (x << 1)) & 0x5555_5555_5555_5555
}

/// Inverse of [`part_1by1`]: gathers the even bit positions back into the low 32 bits.
#[inline]
const fn compact_1by1(mut x: u64) -> u64 {
    x &= 0x5555_5555_5555_5555;
    x = (x | (x >> 1)) & 0x3333_3333_3333_3333;
    x = (x | (x >> 2)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x >> 4)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x >> 8)) & 0x0000_ffff_0000_ffff;

    (x | (x >> 16)) & 0x0000_0000_ffff_ffff
}

/// Spreads the low 21 bits of `x` so each lands three positions apart, the 3D Morton building block.
#[inline]
const fn part_1by2(mut x: u64) -> u64 {
    x &= 0x1f_ffff;
    x = (x | (x << 32)) & 0x001f_0000_0000_ffff;
    x = (x | (x << 16)) & 0x001f_0000_ff00_00ff;
    x = (x | (x << 8)) & 0x100f_00f0_0f00_f00f;
    x = (x | (x << 4)) & 0x10c3_0c30_c30c_30c3;

    (x | (x << 2)) & 0x1249_2492_4924_9249
}

/// Inverse of [`part_1by2`]: gathers every third bit back into the low 21 bits.
#[inline]
const fn compact_1by2(mut x: u64) -> u64 {
    x &= 0x1249_2492_4924_9249;
    x = (x | (x >> 2)) & 0x10c3_0c30_c30c_30c3;
    x = (x | (x >> 4)) & 0x100f_00f0_0f00_f00f;
    x = (x | (x >> 8)) & 0x001f_0000_ff00_00ff;
    x = (x | (x >> 16)) & 0x001f_0000_0000_ffff;

    (x | (x >> 32)) & 0x001f_ffff
}

impl Morton<2> for Dim<2> {
    const CHILDREN: usize = 4;
    const BITS: u32 = 32;

    fn encode([c0, c1]: [u32; 2]) -> u64 {
        part_1by1(c0 as u64) | (part_1by1(c1 as u64) << 1)
    }

    fn decode(code: u64) -> [u32; 2] {
        [compact_1by1(code) as u32, compact_1by1(code >> 1) as u32]
    }
}

impl Morton<3> for Dim<3> {
    const CHILDREN: usize = 8;
    const BITS: u32 = 21;

    fn encode([c0, c1, c2]: [u32; 3]) -> u64 {
        part_1by2(c0 as u64) | (part_1by2(c1 as u64) << 1) | (part_1by2(c2 as u64) << 2)
    }

    fn decode(code: u64) -> [u32; 3] {
        [
            compact_1by2(code) as u32,
            compact_1by2(code >> 1) as u32,
            compact_1by2(code >> 2) as u32,
        ]
    }
}

/// Quantizes one embedding point to `B`-bit per-axis integer coordinates over the bounding box.
///
/// `min` is the per-axis lower corner and `inv_scale[axis] = 2^B / extent[axis]` (or `0` for a
/// degenerate zero-width axis, which collapses to bucket `0`). The result is clamped to
/// `[0, 2^B - 1]`, so the maximum coordinate maps to the last bucket rather than overflowing.
pub(crate) fn quantize<T: Float, const D: usize>(
    point: &[T; D],
    min: &[T; D],
    inv_scale: &[T; D],
    max_bucket: u32,
) -> [u32; D] {
    std::array::from_fn(|axis| {
        let scaled = ((point[axis] - min[axis]) * inv_scale[axis]).floor();

        if scaled > T::zero() {
            scaled
                .to_u64()
                .map_or(max_bucket, |value| value.min(max_bucket as u64) as u32)
        } else {
            0
        }
    })
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;

    #[test]
    fn encode_decode_roundtrips_2d() {
        let mut rng = StdRng::seed_from_u64(0x1234_5678);
        for _ in 0..10_000 {
            let x = rng.random::<u32>();
            let y = rng.random::<u32>();
            let code = Dim::<2>::encode([x, y]);
            assert_eq!(Dim::<2>::decode(code), [x, y]);
        }
    }

    #[test]
    fn encode_decode_roundtrips_3d() {
        let mut rng = StdRng::seed_from_u64(0x9abc_def0);
        let mask = (1u32 << 21) - 1;
        for _ in 0..10_000 {
            let x = rng.random::<u32>() & mask;
            let y = rng.random::<u32>() & mask;
            let z = rng.random::<u32>() & mask;
            let code = Dim::<3>::encode([x, y, z]);
            assert_eq!(Dim::<3>::decode(code), [x, y, z]);
        }
    }

    /// On a 2x2 grid the codes must be `0, 1, 2, 3` for `(0,0), (1,0), (0,1), (1,1)`, the canonical
    /// Z-order with the first axis varying fastest.
    #[test]
    fn encode_matches_known_z_order_2d() {
        assert_eq!(Dim::<2>::encode([0, 0]), 0);
        assert_eq!(Dim::<2>::encode([1, 0]), 1);
        assert_eq!(Dim::<2>::encode([0, 1]), 2);
        assert_eq!(Dim::<2>::encode([1, 1]), 3);
    }

    #[test]
    fn encode_matches_known_z_order_3d() {
        assert_eq!(Dim::<3>::encode([0, 0, 0]), 0);
        assert_eq!(Dim::<3>::encode([1, 0, 0]), 1);
        assert_eq!(Dim::<3>::encode([0, 1, 0]), 2);
        assert_eq!(Dim::<3>::encode([0, 0, 1]), 4);
        assert_eq!(Dim::<3>::encode([1, 1, 1]), 7);
    }

    /// Sorting points by code must group them by shared high-bit prefix: two points in the same
    /// top-level quadrant must sort adjacently relative to one in a different quadrant.
    #[test]
    fn sorting_by_code_yields_z_order() {
        // Quantized so the top bit per axis selects the quadrant (bit 31 for D = 2).
        let top = 1u32 << 31;
        let lower_left = Dim::<2>::encode([1, 1]);
        let lower_left_2 = Dim::<2>::encode([5, 7]);
        let upper_right = Dim::<2>::encode([top, top]);
        let mut codes = [upper_right, lower_left_2, lower_left];
        codes.sort_unstable();
        // The two lower-left points (small codes) precede the upper-right point.
        assert!(codes[0] < codes[2]);
        assert_eq!(codes[2], upper_right);
        assert!(codes[0] == lower_left && codes[1] == lower_left_2);
    }

    /// Quantization maps the box corners to the first and last buckets and is monotone.
    #[test]
    fn quantize_spans_the_bucket_range() {
        let min = [0.0f32, -2.0];
        let extent = [4.0f32, 8.0];
        let max_bucket = (1u64 << 32) - 1;
        let inv_scale = [
            (1u64 << 32) as f32 / extent[0],
            (1u64 << 32) as f32 / extent[1],
        ];
        let low = quantize::<f32, 2>(&[0.0, -2.0], &min, &inv_scale, max_bucket as u32);
        assert_eq!(low, [0, 0]);
        let high = quantize::<f32, 2>(&[4.0, 6.0], &min, &inv_scale, max_bucket as u32);
        assert_eq!(high, [max_bucket as u32, max_bucket as u32]);
        let mid = quantize::<f32, 2>(&[2.0, 2.0], &min, &inv_scale, max_bucket as u32);
        assert!(mid[0] > 0 && mid[0] < max_bucket as u32);
        assert!(mid[1] > 0 && mid[1] < max_bucket as u32);
    }

    /// A degenerate zero-width axis collapses every point to bucket zero on that axis.
    #[test]
    fn quantize_handles_zero_width_axis() {
        let min = [1.0f32, 5.0];
        let inv_scale = [0.0f32, (1u64 << 32) as f32 / 4.0];
        let max_bucket = ((1u64 << 32) - 1) as u32;
        let q = quantize::<f32, 2>(&[1.0, 7.0], &min, &inv_scale, max_bucket);
        assert_eq!(q[0], 0);
        assert!(q[1] > 0);
    }
}
