//! 7D Morton codec: 18 bits per axis.
//!
//! Uses a lookup table for the bit spreader. The 18 bits split into three 5-bit chunks (15 bits)
//! plus three remaining bits. Each chunk spreads via a 32-entry lookup table and the results
//! concatenate at 35-bit offsets. The final interleaved code uses 126 bits (18 * 7), fitting in
//! `u128`.

use super::{Dim, Morton};

/// Spreads the low 18 bits of `x` so each lands seven positions apart (six gaps), the 7D Morton
/// building block.
///
/// The low 15 bits are split into three 5-bit chunks. Each chunk spreads via a lookup table
/// and the three results concatenate at 35-bit offsets. The remaining three bits are placed
/// directly at positions 105, 112, and 119.
#[inline]
const fn part_1by6(x: u128) -> u128 {
    // Spread table for a single 5-bit chunk: bit i at position 7*i.
    const SPREAD: [u128; 32] = [
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x0000_0000_0000_0080,
        0x0000_0000_0000_0081,
        0x0000_0000_0000_4000,
        0x0000_0000_0000_4001,
        0x0000_0000_0000_4080,
        0x0000_0000_0000_4081,
        0x0000_0000_0020_0000,
        0x0000_0000_0020_0001,
        0x0000_0000_0020_0080,
        0x0000_0000_0020_0081,
        0x0000_0000_0020_4000,
        0x0000_0000_0020_4001,
        0x0000_0000_0020_4080,
        0x0000_0000_0020_4081,
        0x0000_0000_1000_0000,
        0x0000_0000_1000_0001,
        0x0000_0000_1000_0080,
        0x0000_0000_1000_0081,
        0x0000_0000_1000_4000,
        0x0000_0000_1000_4001,
        0x0000_0000_1000_4080,
        0x0000_0000_1000_4081,
        0x0000_0000_1020_0000,
        0x0000_0000_1020_0001,
        0x0000_0000_1020_0080,
        0x0000_0000_1020_0081,
        0x0000_0000_1020_4000,
        0x0000_0000_1020_4001,
        0x0000_0000_1020_4080,
        0x0000_0000_1020_4081,
    ];

    let x = x & 0x0003_ffff;
    let c0 = x & 0x1f;
    let c1 = (x >> 5) & 0x1f;
    let c2 = (x >> 10) & 0x1f;
    let r = (x >> 15) & 0x7;

    SPREAD[c0 as usize]
        | (SPREAD[c1 as usize] << 35)
        | (SPREAD[c2 as usize] << 70)
        | ((r & 1) << 105)
        | (((r >> 1) & 1) << 112)
        | (((r >> 2) & 1) << 119)
}

/// Inverse of [`part_1by6`]: gathers every seventh bit back into the low 18 bits.
#[inline]
const fn compact_1by6(code: u128) -> u128 {
    (code & 1)
        | (((code >> 7) & 1) << 1)
        | (((code >> 14) & 1) << 2)
        | (((code >> 21) & 1) << 3)
        | (((code >> 28) & 1) << 4)
        | (((code >> 35) & 1) << 5)
        | (((code >> 42) & 1) << 6)
        | (((code >> 49) & 1) << 7)
        | (((code >> 56) & 1) << 8)
        | (((code >> 63) & 1) << 9)
        | (((code >> 70) & 1) << 10)
        | (((code >> 77) & 1) << 11)
        | (((code >> 84) & 1) << 12)
        | (((code >> 91) & 1) << 13)
        | (((code >> 98) & 1) << 14)
        | (((code >> 105) & 1) << 15)
        | (((code >> 112) & 1) << 16)
        | (((code >> 119) & 1) << 17)
}

impl Morton<7> for Dim<7> {
    const CHILDREN: usize = 128;
    const BITS: u32 = 18;
    type Stack = [u32; 2304];
    type Word = u128;

    fn empty_stack() -> Self::Stack {
        [0u32; 2304]
    }

    fn encode([c0, c1, c2, c3, c4, c5, c6]: [u32; 7]) -> Self::Word {
        part_1by6(c0 as u128)
            | (part_1by6(c1 as u128) << 1)
            | (part_1by6(c2 as u128) << 2)
            | (part_1by6(c3 as u128) << 3)
            | (part_1by6(c4 as u128) << 4)
            | (part_1by6(c5 as u128) << 5)
            | (part_1by6(c6 as u128) << 6)
    }

    fn decode(code: Self::Word) -> [u32; 7] {
        [
            compact_1by6(code) as u32,
            compact_1by6(code >> 1) as u32,
            compact_1by6(code >> 2) as u32,
            compact_1by6(code >> 3) as u32,
            compact_1by6(code >> 4) as u32,
            compact_1by6(code >> 5) as u32,
            compact_1by6(code >> 6) as u32,
        ]
    }
}

#[cfg(test)]
mod tests {
    use proptest::prop_assert_eq;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::super::{Dim, Morton};

    #[test]
    fn encode_decode_roundtrips_7d() {
        let mut rng = StdRng::seed_from_u64(0xabcd_ef03);
        let mask = (1u32 << 18) - 1;
        for _ in 0..10_000 {
            let a = rng.random::<u32>() & mask;
            let b = rng.random::<u32>() & mask;
            let c = rng.random::<u32>() & mask;
            let d = rng.random::<u32>() & mask;
            let e = rng.random::<u32>() & mask;
            let f = rng.random::<u32>() & mask;
            let g = rng.random::<u32>() & mask;
            let code = Dim::<7>::encode([a, b, c, d, e, f, g]);
            assert_eq!(Dim::<7>::decode(code), [a, b, c, d, e, f, g]);
        }
    }

    #[test]
    fn encode_matches_known_z_order_7d() {
        assert_eq!(Dim::<7>::encode([0, 0, 0, 0, 0, 0, 0]), 0);
        assert_eq!(Dim::<7>::encode([1, 0, 0, 0, 0, 0, 0]), 1);
        assert_eq!(Dim::<7>::encode([0, 1, 0, 0, 0, 0, 0]), 2);
        assert_eq!(Dim::<7>::encode([0, 0, 1, 0, 0, 0, 0]), 4);
        assert_eq!(Dim::<7>::encode([0, 0, 0, 1, 0, 0, 0]), 8);
        assert_eq!(Dim::<7>::encode([0, 0, 0, 0, 1, 0, 0]), 16);
        assert_eq!(Dim::<7>::encode([0, 0, 0, 0, 0, 1, 0]), 32);
        assert_eq!(Dim::<7>::encode([0, 0, 0, 0, 0, 0, 1]), 64);
        assert_eq!(Dim::<7>::encode([1, 1, 1, 1, 1, 1, 1]), 127);
    }

    proptest::proptest! {
        #[test]
        fn proptest_roundtrip_7d(a in 0u32..1 << 18, b in 0u32..1 << 18, c in 0u32..1 << 18, d in 0u32..1 << 18, e in 0u32..1 << 18, f in 0u32..1 << 18, g in 0u32..1 << 18) {
            let code = Dim::<7>::encode([a, b, c, d, e, f, g]);
            prop_assert_eq!(Dim::<7>::decode(code), [a, b, c, d, e, f, g]);
        }
    }
}
