//! 6D Morton codec: 21 bits per axis.
//!
//! Uses a lookup table for the bit spreader since the final gap of 5 (six positions apart) does
//! not decompose as powers of two, which breaks the standard binary-doubling shift-and-mask
//! algorithm used for D in {2, 3, 4}.
//!
//! The 21 bits are split into four 5-bit chunks (20 bits) plus one remaining bit. Each chunk
//! spreads via a small lookup table and the results concatenate at 30-bit offsets. The final
//! interleaved code uses 126 bits (21 * 6), fitting in `u128`.

use super::{Dim, Morton};

/// Spreads the low 21 bits of `x` so each lands six positions apart (five gaps), the 6D Morton
/// building block.
///
/// The low 20 bits are split into four 5-bit chunks. Each chunk spreads via a lookup table
/// and the four results concatenate at 30-bit offsets. Bit 20 is placed directly at position
/// 120.
#[inline]
const fn part_1by5(x: u128) -> u128 {
    // Spread table for a single 5-bit chunk: bit i at position 6*i.
    const SPREAD: [u128; 32] = [
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x0000_0000_0000_0040,
        0x0000_0000_0000_0041,
        0x0000_0000_0000_1000,
        0x0000_0000_0000_1001,
        0x0000_0000_0000_1040,
        0x0000_0000_0000_1041,
        0x0000_0000_0004_0000,
        0x0000_0000_0004_0001,
        0x0000_0000_0004_0040,
        0x0000_0000_0004_0041,
        0x0000_0000_0004_1000,
        0x0000_0000_0004_1001,
        0x0000_0000_0004_1040,
        0x0000_0000_0004_1041,
        0x0000_0000_0100_0000,
        0x0000_0000_0100_0001,
        0x0000_0000_0100_0040,
        0x0000_0000_0100_0041,
        0x0000_0000_0100_1000,
        0x0000_0000_0100_1001,
        0x0000_0000_0100_1040,
        0x0000_0000_0100_1041,
        0x0000_0000_0104_0000,
        0x0000_0000_0104_0001,
        0x0000_0000_0104_0040,
        0x0000_0000_0104_0041,
        0x0000_0000_0104_1000,
        0x0000_0000_0104_1001,
        0x0000_0000_0104_1040,
        0x0000_0000_0104_1041,
    ];

    let x = x & 0x00_1f_ff_ff;
    let c0 = x & 0x1f;
    let c1 = (x >> 5) & 0x1f;
    let c2 = (x >> 10) & 0x1f;
    let c3 = (x >> 15) & 0x1f;

    SPREAD[c0 as usize]
        | (SPREAD[c1 as usize] << 30)
        | (SPREAD[c2 as usize] << 60)
        | (SPREAD[c3 as usize] << 90)
        | ((x >> 20) & 1) << 120
}

/// Inverse of [`part_1by5`]: gathers every sixth bit back into the low 21 bits.
#[inline]
const fn compact_1by5(code: u128) -> u128 {
    (code & 1)
        | (((code >> 6) & 1) << 1)
        | (((code >> 12) & 1) << 2)
        | (((code >> 18) & 1) << 3)
        | (((code >> 24) & 1) << 4)
        | (((code >> 30) & 1) << 5)
        | (((code >> 36) & 1) << 6)
        | (((code >> 42) & 1) << 7)
        | (((code >> 48) & 1) << 8)
        | (((code >> 54) & 1) << 9)
        | (((code >> 60) & 1) << 10)
        | (((code >> 66) & 1) << 11)
        | (((code >> 72) & 1) << 12)
        | (((code >> 78) & 1) << 13)
        | (((code >> 84) & 1) << 14)
        | (((code >> 90) & 1) << 15)
        | (((code >> 96) & 1) << 16)
        | (((code >> 102) & 1) << 17)
        | (((code >> 108) & 1) << 18)
        | (((code >> 114) & 1) << 19)
        | (((code >> 120) & 1) << 20)
}

impl Morton<6> for Dim<6> {
    const CHILDREN: usize = 64;
    const BITS: u32 = 21;
    type Stack = [u32; 1344];
    type Word = u128;

    fn empty_stack() -> Self::Stack {
        [0u32; 1344]
    }

    fn encode([c0, c1, c2, c3, c4, c5]: [u32; 6]) -> Self::Word {
        part_1by5(c0 as u128)
            | (part_1by5(c1 as u128) << 1)
            | (part_1by5(c2 as u128) << 2)
            | (part_1by5(c3 as u128) << 3)
            | (part_1by5(c4 as u128) << 4)
            | (part_1by5(c5 as u128) << 5)
    }

    fn decode(code: Self::Word) -> [u32; 6] {
        [
            compact_1by5(code) as u32,
            compact_1by5(code >> 1) as u32,
            compact_1by5(code >> 2) as u32,
            compact_1by5(code >> 3) as u32,
            compact_1by5(code >> 4) as u32,
            compact_1by5(code >> 5) as u32,
        ]
    }
}

#[cfg(test)]
mod tests {
    use proptest::prop_assert_eq;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::super::{Dim, Morton};

    #[test]
    fn encode_decode_roundtrips_6d() {
        let mut rng = StdRng::seed_from_u64(0xabcd_ef02);
        let mask = (1u32 << 21) - 1;
        for _ in 0..10_000 {
            let a = rng.random::<u32>() & mask;
            let b = rng.random::<u32>() & mask;
            let c = rng.random::<u32>() & mask;
            let d = rng.random::<u32>() & mask;
            let e = rng.random::<u32>() & mask;
            let f = rng.random::<u32>() & mask;
            let code = Dim::<6>::encode([a, b, c, d, e, f]);
            assert_eq!(Dim::<6>::decode(code), [a, b, c, d, e, f]);
        }
    }

    #[test]
    fn encode_matches_known_z_order_6d() {
        assert_eq!(Dim::<6>::encode([0, 0, 0, 0, 0, 0]), 0);
        assert_eq!(Dim::<6>::encode([1, 0, 0, 0, 0, 0]), 1);
        assert_eq!(Dim::<6>::encode([0, 1, 0, 0, 0, 0]), 2);
        assert_eq!(Dim::<6>::encode([0, 0, 1, 0, 0, 0]), 4);
        assert_eq!(Dim::<6>::encode([0, 0, 0, 1, 0, 0]), 8);
        assert_eq!(Dim::<6>::encode([0, 0, 0, 0, 1, 0]), 16);
        assert_eq!(Dim::<6>::encode([0, 0, 0, 0, 0, 1]), 32);
        assert_eq!(Dim::<6>::encode([1, 1, 1, 1, 1, 1]), 63);
    }

    proptest::proptest! {
        #[test]
        fn proptest_roundtrip_6d(a in 0u32..1 << 21, b in 0u32..1 << 21, c in 0u32..1 << 21, d in 0u32..1 << 21, e in 0u32..1 << 21, f in 0u32..1 << 21) {
            let code = Dim::<6>::encode([a, b, c, d, e, f]);
            prop_assert_eq!(Dim::<6>::decode(code), [a, b, c, d, e, f]);
        }
    }
}
