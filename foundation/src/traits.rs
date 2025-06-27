use crate::num::traits::{FixedWidthInteger, UnsignedInteger};

/// A trait for generating random values of various types.
///
/// `RandomNumberGenerator` defines a common interface for generating
/// random values, including unconstrained values and values within a
/// specified upper bound.
///
/// This trait is generic and can be implemented for different underlying
/// random number generation strategies.
pub trait RandomNumberGenerator {
    /// Generates a random value of type `T`.
    ///
    /// The type `T` must implement `Sized`, `Copy`, and `Default`.
    /// The implementation determines how the random bits are generated.
    fn next<T: Sized + Copy + Default>(&mut self) -> T;

    /// Generates a random value of type `T` less than the specified `upper_bound`.
    ///
    /// The output will be in the range `[0, upper_bound)`.
    /// The type `T` must implement `BinaryInteger` and `Default`.
    ///
    /// # Parameters
    ///
    /// - `upper_bound`: The exclusive upper bound for the random value.
    fn next_below<T>(&mut self, upper_bound: T) -> T
    where
        T: FixedWidthInteger + UnsignedInteger + Default;
}
