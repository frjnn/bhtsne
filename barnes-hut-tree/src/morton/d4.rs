//! 4D Morton codec: 16 bits per axis.

use super::{Dim, Morton};

/// Spreads the low 16 bits of `x` so each lands four positions apart, the 4D Morton building block.
#[inline]
const fn part_1by3(mut x: u64) -> u64 {
    x &= 0x0000_0000_0000_ffff;
    x = (x | (x << 24)) & 0x0000_00ff_0000_00ff;
    x = (x | (x << 12)) & 0x000f_000f_000f_000f;
    x = (x | (x << 6)) & 0x0303_0303_0303_0303;

    (x | (x << 3)) & 0x1111_1111_1111_1111
}

/// Inverse of [`part_1by3`]: gathers every fourth bit back into the low 16 bits.
#[inline]
const fn compact_1by3(mut x: u64) -> u64 {
    x &= 0x1111_1111_1111_1111;
    x = (x | (x >> 3)) & 0x0303_0303_0303_0303;
    x = (x | (x >> 6)) & 0x000f_000f_000f_000f;
    x = (x | (x >> 12)) & 0x0000_00ff_0000_00ff;

    (x | (x >> 24)) & 0x0000_0000_0000_ffff
}

impl Morton<4> for Dim<4> {
    const CHILDREN: usize = 16;
    const BITS: u32 = 16;
    type Stack = [u32; 256];
    type Word = u64;

    fn empty_stack() -> Self::Stack {
        [0u32; 256]
    }

    fn encode([c0, c1, c2, c3]: [u32; 4]) -> Self::Word {
        part_1by3(c0 as u64)
            | (part_1by3(c1 as u64) << 1)
            | (part_1by3(c2 as u64) << 2)
            | (part_1by3(c3 as u64) << 3)
    }

    fn decode(code: Self::Word) -> [u32; 4] {
        [
            compact_1by3(code) as u32,
            compact_1by3(code >> 1) as u32,
            compact_1by3(code >> 2) as u32,
            compact_1by3(code >> 3) as u32,
        ]
    }
}

#[cfg(test)]
mod tests {
    use proptest::prop_assert_eq;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::super::{Dim, Morton};

    #[test]
    fn encode_decode_roundtrips_4d() {
        let mut rng = StdRng::seed_from_u64(0xabcd_1234);
        let mask = (1u32 << 16) - 1;
        for _ in 0..10_000 {
            let a = rng.random::<u32>() & mask;
            let b = rng.random::<u32>() & mask;
            let c = rng.random::<u32>() & mask;
            let d = rng.random::<u32>() & mask;
            let code = Dim::<4>::encode([a, b, c, d]);
            assert_eq!(Dim::<4>::decode(code), [a, b, c, d]);
        }
    }

    #[test]
    fn encode_matches_known_z_order_4d() {
        assert_eq!(Dim::<4>::encode([0, 0, 0, 0]), 0);
        assert_eq!(Dim::<4>::encode([1, 0, 0, 0]), 1);
        assert_eq!(Dim::<4>::encode([0, 1, 0, 0]), 2);
        assert_eq!(Dim::<4>::encode([0, 0, 1, 0]), 4);
        assert_eq!(Dim::<4>::encode([0, 0, 0, 1]), 8);
        assert_eq!(Dim::<4>::encode([1, 1, 1, 1]), 15);
    }

    proptest::proptest! {
        #[test]
        fn proptest_roundtrip_4d(a in 0u16.., b in 0u16.., c in 0u16.., d in 0u16..) {
            let code = Dim::<4>::encode([a as u32, b as u32, c as u32, d as u32]);
            prop_assert_eq!(Dim::<4>::decode(code), [a as u32, b as u32, c as u32, d as u32]);
        }
    }
}
