//! Z-order (Morton) bit interleaving for the linear quadtree arena.
//!
//! The arena encodes each embedding point as a single Morton code: the per-axis quantized
//! coordinates are bit-interleaved so that sorting the codes lays the points out in Z-order, where
//! points sharing a code prefix share a tree cell. `D in {2, 3, 4, 5, 6, 7}` are supported, with 32
//! bits per axis at `D = 2` (lossless for `f32`), 21 bits at `D = 3`, 16 bits at `D = 4`, 25
//! bits at `D = 5`, 21 bits at `D = 6`, and 18 bits at `D = 7`. The [`Morton`] trait is implemented for those six
//! dimensions, so the bound `Dim<D>: Morton<D>` is what restricts the Barnes-Hut tree path at
//! compile time.

mod d2;
mod d3;
mod d4;
mod d5;
mod d6;
mod d7;

use num_traits::Float;

/// Integer type of a Morton code. Covers the operations the arena performs on codes: XOR, shifts,
/// masks, comparison, and finding the most-significant differing bit.
pub trait MortonWord:
    Send
    + Sync
    + Copy
    + Default
    + Ord
    + std::ops::BitXor<Output = Self>
    + std::ops::BitAnd<Output = Self>
    + std::ops::Shr<u32, Output = Self>
{
    /// Total number of bits in the type.
    const BITS: u32;

    /// Index of the most-significant set bit (0-based from the LSB).
    ///
    /// Returns `Self::BITS - 1` when the value is zero (no bits set).
    fn msb_position(self) -> u32;

    /// Returns a value with the low `d` bits set (a D-bit group mask).
    fn d_bit_mask(d: u32) -> Self;
}

impl MortonWord for u64 {
    const BITS: u32 = 64;
    #[inline]
    fn d_bit_mask(d: u32) -> Self {
        (1u64 << d) - 1
    }
    #[inline]
    fn msb_position(self) -> u32 {
        Self::BITS - 1 - self.leading_zeros()
    }
}

impl MortonWord for u128 {
    const BITS: u32 = 128;

    #[inline]
    fn d_bit_mask(d: u32) -> Self {
        (1u128 << d) - 1
    }
    fn msb_position(self) -> u32 {
        Self::BITS - 1 - self.leading_zeros()
    }
}

/// Per-dimension Z-order codec, implemented for [`Dim<2>`], [`Dim<3>`], [`Dim<4>`], [`Dim<5>`], [`Dim<6>`], and [`Dim<7>`].
pub trait Morton<const D: usize> {
    /// Number of children a fully occupied internal cell can have, `2^D`.
    const CHILDREN: usize;

    /// Bits of precision per axis in the Morton code.
    const BITS: u32;

    /// Fixed-size traversal stack for the Barnes-Hut repulsive force pass.
    type Stack: AsMut<[u32]>;

    /// The integer type of the interleaved Morton code.
    type Word: MortonWord;

    /// Allocate an empty stack on the caller's stack frame.
    fn empty_stack() -> Self::Stack;

    /// Interleaves the `D` quantized per-axis coordinates into one Z-order code.
    fn encode(coords: [u32; D]) -> Self::Word;

    /// Inverse of [`encode`], recovering the per-axis quantized coordinates.
    ///
    /// [`encode`]: Morton::encode
    fn decode(code: Self::Word) -> [u32; D];
}

/// Carrier type the per-`D` [`Morton`] implementations attach to, since a trait needs a type to
/// dispatch on while `D` stays a const generic on the arena.
pub struct Dim<const D: usize>;

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
