//! 2D Morton codec: 32 bits per axis (lossless for `f32`).

use super::{Dim, Morton};

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

impl Morton<2> for Dim<2> {
    const CHILDREN: usize = 4;
    const BITS: u32 = 32;
    type Stack = [u32; 128];

    fn empty_stack() -> Self::Stack {
        [0u32; 128]
    }

    fn encode([c0, c1]: [u32; 2]) -> u64 {
        part_1by1(c0 as u64) | (part_1by1(c1 as u64) << 1)
    }

    fn decode(code: u64) -> [u32; 2] {
        [compact_1by1(code) as u32, compact_1by1(code >> 1) as u32]
    }
}

#[cfg(test)]
mod tests {
    use proptest::prop_assert_eq;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::super::{Dim, Morton, quantize};

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

    /// On a 2x2 grid the codes must be `0, 1, 2, 3` for `(0,0), (1,0), (0,1), (1,1)`, the canonical
    /// Z-order with the first axis varying fastest.
    #[test]
    fn encode_matches_known_z_order_2d() {
        assert_eq!(Dim::<2>::encode([0, 0]), 0);
        assert_eq!(Dim::<2>::encode([1, 0]), 1);
        assert_eq!(Dim::<2>::encode([0, 1]), 2);
        assert_eq!(Dim::<2>::encode([1, 1]), 3);
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

    proptest::proptest! {
        #[test]
        fn proptest_roundtrip_2d(x in 0u32.., y in 0u32..) {
            let code = Dim::<2>::encode([x, y]);
            prop_assert_eq!(Dim::<2>::decode(code), [x, y]);
        }
    }
}
