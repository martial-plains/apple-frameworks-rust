/*!
  # Collections Module

  This module provides Rust implementations of commonly-used collection types inspired by the Swift Foundation framework. These collections aim to provide efficient and ergonomic data structures, with an API design that mirrors the style and functionality of Swift's collections.

  The collections in this module are designed to be:
  - **Efficient**: Built with Rust's performance characteristics in mind, ensuring fast access, insertion, and removal operations.
  - **Idiomatic**: While they draw inspiration from Swift, they adhere to Rust's conventions, making use of ownership, borrowing, and lifetime tracking.
  - **Safe**: The collections are designed with Rust’s strict safety guarantees, ensuring memory safety and preventing data races.

  ## Key Types
  - **`Array`**: A dynamic, growable array type similar to Swift's `Array`. Provides methods for adding, removing, and accessing elements efficiently.
  - **`Dictionary`**: A hash map-like collection inspired by Swift's `Dictionary`. Supports key-value pairs, with fast lookups and insertion.
  - **`Set`**: An unordered collection of unique elements, similar to Swift’s `Set`. Supports fast membership checking and element removal.

  These types are designed to make working with collections in Rust as ergonomic and flexible as possible, while providing the essential functionality that developers may expect from the Swift Foundation’s collection types.

*/

mod array;
mod default_indices;
mod sequences;
mod slices;
mod traits;

pub use array::*;
pub use default_indices::*;
pub use sequences::*;
pub use slices::*;
pub use traits::*;
