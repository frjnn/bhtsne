//! 5D Morton codec: 25 bits per axis.
//!
//! Uses a lookup table for the bit spreader since the final gap of 4 (five positions apart) does
//! not decompose as powers of two, which breaks the standard binary-doubling shift-and-mask
//! algorithm used for D in {2, 3, 4}.
//!
//! The 25 bits are split into six 4-bit nibbles (24 bits) plus one remaining bit. Each nibble is
//! spread via a small lookup table and the results are concatenated at 20-bit offsets. The final
//! interleaved code uses 125 bits (25 * 5), fitting in `u128`.

use super::{Dim, Morton};

/// Spreads the low 25 bits of `x` so each lands five positions apart (four gaps), the 5D Morton
/// building block.
///
/// The low 24 bits are split into six 4-bit nibbles. Each nibble is spread via a lookup table
/// and the six results are concatenated at 20-bit offsets. Bit 24 is placed directly at position
/// 120.
#[inline]
const fn part_1by4(x: u128) -> u128 {
    // Spread table for a single 4-bit nibble: bit i at position 5*i.
    const SPREAD: [u128; 16] = [
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x0000_0000_0000_0020,
        0x0000_0000_0000_0021,
        0x0000_0000_0000_0400,
        0x0000_0000_0000_0401,
        0x0000_0000_0000_0420,
        0x0000_0000_0000_0421,
        0x0000_0000_0000_8000,
        0x0000_0000_0000_8001,
        0x0000_0000_0000_8020,
        0x0000_0000_0000_8021,
        0x0000_0000_0000_8400,
        0x0000_0000_0000_8401,
        0x0000_0000_0000_8420,
        0x0000_0000_0000_8421,
    ];

    let x = x & 0x01_ff_ff_ff;
    let n0 = x & 0xf;
    let n1 = (x >> 4) & 0xf;
    let n2 = (x >> 8) & 0xf;
    let n3 = (x >> 12) & 0xf;
    let n4 = (x >> 16) & 0xf;
    let n5 = (x >> 20) & 0xf;

    SPREAD[n0 as usize]
        | (SPREAD[n1 as usize] << 20)
        | (SPREAD[n2 as usize] << 40)
        | (SPREAD[n3 as usize] << 60)
        | (SPREAD[n4 as usize] << 80)
        | (SPREAD[n5 as usize] << 100)
        | ((x >> 24) & 1) << 120
}

/// Inverse of [`part_1by4`]: gathers every fifth bit back into the low 25 bits.
#[inline]
const fn compact_1by4(code: u128) -> u128 {
    (code & 1)
        | (((code >> 5) & 1) << 1)
        | (((code >> 10) & 1) << 2)
        | (((code >> 15) & 1) << 3)
        | (((code >> 20) & 1) << 4)
        | (((code >> 25) & 1) << 5)
        | (((code >> 30) & 1) << 6)
        | (((code >> 35) & 1) << 7)
        | (((code >> 40) & 1) << 8)
        | (((code >> 45) & 1) << 9)
        | (((code >> 50) & 1) << 10)
        | (((code >> 55) & 1) << 11)
        | (((code >> 60) & 1) << 12)
        | (((code >> 65) & 1) << 13)
        | (((code >> 70) & 1) << 14)
        | (((code >> 75) & 1) << 15)
        | (((code >> 80) & 1) << 16)
        | (((code >> 85) & 1) << 17)
        | (((code >> 90) & 1) << 18)
        | (((code >> 95) & 1) << 19)
        | (((code >> 100) & 1) << 20)
        | (((code >> 105) & 1) << 21)
        | (((code >> 110) & 1) << 22)
        | (((code >> 115) & 1) << 23)
        | (((code >> 120) & 1) << 24)
}

impl Morton<5> for Dim<5> {
    const CHILDREN: usize = 32;
    const BITS: u32 = 25;
    type Stack = [u32; 800];
    type Word = u128;

    fn empty_stack() -> Self::Stack {
        [0u32; 800]
    }

    fn encode([c0, c1, c2, c3, c4]: [u32; 5]) -> Self::Word {
        part_1by4(c0 as u128)
            | (part_1by4(c1 as u128) << 1)
            | (part_1by4(c2 as u128) << 2)
            | (part_1by4(c3 as u128) << 3)
            | (part_1by4(c4 as u128) << 4)
    }

    fn decode(code: Self::Word) -> [u32; 5] {
        [
            compact_1by4(code) as u32,
            compact_1by4(code >> 1) as u32,
            compact_1by4(code >> 2) as u32,
            compact_1by4(code >> 3) as u32,
            compact_1by4(code >> 4) as u32,
        ]
    }
}

#[cfg(test)]
mod tests {
    use proptest::prop_assert_eq;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::super::{Dim, Morton};

    #[test]
    fn encode_decode_roundtrips_5d() {
        let mut rng = StdRng::seed_from_u64(0xabcd_ef01);
        let mask = (1u32 << 25) - 1;
        for _ in 0..10_000 {
            let a = rng.random::<u32>() & mask;
            let b = rng.random::<u32>() & mask;
            let c = rng.random::<u32>() & mask;
            let d = rng.random::<u32>() & mask;
            let e = rng.random::<u32>() & mask;
            let code = Dim::<5>::encode([a, b, c, d, e]);
            assert_eq!(Dim::<5>::decode(code), [a, b, c, d, e]);
        }
    }

    #[test]
    fn encode_matches_known_z_order_5d() {
        assert_eq!(Dim::<5>::encode([0, 0, 0, 0, 0]), 0);
        assert_eq!(Dim::<5>::encode([1, 0, 0, 0, 0]), 1);
        assert_eq!(Dim::<5>::encode([0, 1, 0, 0, 0]), 2);
        assert_eq!(Dim::<5>::encode([0, 0, 1, 0, 0]), 4);
        assert_eq!(Dim::<5>::encode([0, 0, 0, 1, 0]), 8);
        assert_eq!(Dim::<5>::encode([0, 0, 0, 0, 1]), 16);
        assert_eq!(Dim::<5>::encode([1, 1, 1, 1, 1]), 31);
    }

    proptest::proptest! {
        #[test]
        fn proptest_roundtrip_5d(a in 0u32..1 << 25, b in 0u32..1 << 25, c in 0u32..1 << 25, d in 0u32..1 << 25, e in 0u32..1 << 25) {
            let code = Dim::<5>::encode([a, b, c, d, e]);
            prop_assert_eq!(Dim::<5>::decode(code), [a, b, c, d, e]);
        }
    }
}
