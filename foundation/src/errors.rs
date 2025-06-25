//! Error handling primitives modeled after Apple's Foundation framework.

use alloc::boxed::Box;

/// A custom error trait that extends the standard `core::error::Error` trait.
///
/// This trait serves as a shorthand or abstraction for any type that implements
/// the standard error trait from the Rust core library. It allows for easier
/// customization and potential extension in larger error-handling architectures.
pub trait Error: core::error::Error {}

/// A result type that represents either success or failure.
///
/// This enum is similar to the standard `Result<T, E>` type, but with the added
/// flexibility of allowing an optional default failure type.
///
/// # Examples
/// ```rust
/// use foundation::errors::{Result, Error};
///
/// fn do_work() -> Result<i32> {
///     // Simulate a successful computation
///     Result::Success(42)
/// }
///
/// match do_work() {
///     Result::Success(value) => println!("Success: {}", value),
///     Result::Failure(e) => eprintln!("Error: {:?}", e),
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub enum Result<Success, Failure = Box<dyn Error>> {
    /// Represents a successful outcome, containing the success value.
    Success(Success),
    /// Represents a failure outcome, containing the error value.
    Failure(Failure),
}
