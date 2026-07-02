//! Z-order (Morton) bit interleaving for the linear quadtree arena.
//!
//! The arena encodes each embedding point as a single `u64` Morton code: the per-axis quantized
//! coordinates are bit-interleaved so that sorting the codes lays the points out in Z-order, where
//! points sharing a code prefix share a tree cell. `D in {2, 3, 4}` are supported, covering a
//! `u64` code with 32 bits per axis at `D = 2` (lossless for `f32`), 21 bits at `D = 3`, and
//! 16 bits at `D = 4`. The [`Morton`] trait is implemented for those three dimensions, so the
//! bound `Dim<D>: Morton<D>` is what restricts the Barnes-Hut tree path at compile time.

mod d2;
mod d3;
mod d4;

use num_traits::Float;

/// Per-dimension Z-order codec, implemented for [`Dim<2>`], [`Dim<3>`], and [`Dim<4>`].
pub trait Morton<const D: usize> {
    /// Number of children a fully occupied internal cell can have, `2^D`.
    const CHILDREN: usize;

    /// Bits of precision per axis in the `u64` code.
    const BITS: u32;

    /// Fixed-size traversal stack for the Barnes-Hut repulsive force pass.
    type Stack: AsMut<[u32]>;

    /// Allocate an empty stack on the caller's stack frame.
    fn empty_stack() -> Self::Stack;

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
