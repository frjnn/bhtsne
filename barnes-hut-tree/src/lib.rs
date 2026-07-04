#![doc = include_str!("../README.md")]

mod arena;
mod morton;

pub use arena::Arena;
pub use morton::{Dim, Morton, MortonWord};
