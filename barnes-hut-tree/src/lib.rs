#![cfg_attr(not(test), no_std)]
#![doc = include_str!("../README.md")]

extern crate alloc;

mod arena;
mod morton;

pub use arena::BarnesHutTree;
pub use morton::{Dim, Morton, MortonWord};

/// Per-cell force contribution, the pluggable half of the Barnes-Hut traversal.
///
/// [`BarnesHutTree::accumulate_forces`] and [`BarnesHutTree::accumulate_all`] call
/// [`ForceKernel::accumulate`] once per summarised cell (a leaf, or an internal cell
/// passing the theta test), handing it the cell's summary. The kernel turns that summary
/// into a force, so the force law lives entirely in the kernel and never in the tree.
///
/// The blanket implementation covers any closure of the matching shape, `#[inline]`d so
/// a monomorphised call compiles down to inline arithmetic.
///
/// # Example
///
/// ```
/// use barnes_hut_tree::BarnesHutTree;
///
/// // Inverse-cube repulsion: push the query away from a summarised cell with magnitude
/// // `mass / distance^3` on the displacement, one axis at a time.
/// let kernel = |displacement: &[f32; 2],
///               distance_squared: f32,
///               mass: f32,
///               _is_leaf: bool,
///               force: &mut [f32; 2]| {
///     let scale = mass / (distance_squared * distance_squared.sqrt());
///     force[0] += scale * displacement[0];
///     force[1] += scale * displacement[1];
/// };
///
/// let points = [0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
/// let tree = BarnesHutTree::<f32, u64, 2>::new_uniform(&points);
/// let mut forces = [0.0f32; 8];
/// tree.accumulate_all(&points, 0.5, &kernel, &mut forces);
/// assert!(forces.iter().all(|value| value.is_finite()));
/// ```
pub trait ForceKernel<T, const D: usize> {
    /// Adds the force from a cell of summed `mass` whose center of mass sits at
    /// `displacement` from the query (the vector `query - center_of_mass`, squared length
    /// `distance_squared`) into `force`. `distance_squared` is always strictly positive:
    /// the traversal skips zero-distance cells before calling the kernel. `is_leaf` is
    /// true for a leaf holding actual points, false for an internal summary. A kernel that
    /// treats near contact differently from the far field (as FM3 does, doubling the leaf
    /// charge) uses this flag.
    fn accumulate(
        &self,
        displacement: &[T; D],
        distance_squared: T,
        mass: T,
        is_leaf: bool,
        force: &mut [T; D],
    );
}

impl<T, const D: usize, F> ForceKernel<T, D> for F
where
    F: Fn(&[T; D], T, T, bool, &mut [T; D]),
{
    #[inline]
    fn accumulate(
        &self,
        displacement: &[T; D],
        distance_squared: T,
        mass: T,
        is_leaf: bool,
        force: &mut [T; D],
    ) {
        self(displacement, distance_squared, mass, is_leaf, force);
    }
}
