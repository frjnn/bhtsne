//! 3D Morton codec: 21 bits per axis.

use super::{Dim, Morton};

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

impl Morton<3> for Dim<3> {
    const CHILDREN: usize = 8;
    const BITS: u32 = 21;
    type Stack = [u32; 192];

    fn empty_stack() -> Self::Stack {
        [0u32; 192]
    }

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

#[cfg(test)]
mod tests {
    use proptest::prop_assert_eq;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::super::{Dim, Morton};

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

    #[test]
    fn encode_matches_known_z_order_3d() {
        assert_eq!(Dim::<3>::encode([0, 0, 0]), 0);
        assert_eq!(Dim::<3>::encode([1, 0, 0]), 1);
        assert_eq!(Dim::<3>::encode([0, 1, 0]), 2);
        assert_eq!(Dim::<3>::encode([0, 0, 1]), 4);
        assert_eq!(Dim::<3>::encode([1, 1, 1]), 7);
    }

    proptest::proptest! {
        #[test]
        fn proptest_roundtrip_3d(x in 0u32..1 << 21, y in 0u32..1 << 21, z in 0u32..1 << 21) {
            let code = Dim::<3>::encode([x, y, z]);
            prop_assert_eq!(Dim::<3>::decode(code), [x, y, z]);
        }
    }
}
