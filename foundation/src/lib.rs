/*!
  # Swift Foundation in Rust

  This crate is a Rust implementation of Apple's Foundation framework, inspired by the APIs and functionality provided by Swift Foundation. The goal of this library is to provide commonly-used utilities and abstractions found in the Foundation framework, such as data types, collections, and utilities for handling tasks like time, string manipulation, and networking, but with idiomatic Rust implementations.

  This crate is intended for Rust developers who need a Rust-native solution to tasks commonly handled by the Swift Foundation framework in Apple's ecosystem.

  ## Limitations
  While this crate strives to mirror the functionality of the Swift Foundation framework, it may not cover all the features of Apple's Foundation library. The focus is on implementing common and widely-used parts of the API, with an emphasis on performance, safety, and usability in Rust.
*/

#![no_std]
#![warn(
    clippy::nursery,
    clippy::pedantic,
    clippy::all,
    missing_debug_implementations,
    missing_copy_implementations,
    missing_docs
)]
#![debugger_visualizer(natvis_file = "../.natvis")]
#![feature(associated_type_defaults, cfg_select, new_range_api)]

extern crate alloc;

/// Alias for a platform-dependent signed integer (`isize`),
/// typically used for indexing and pointer arithmetic.
pub type Int = isize;

/// Alias for a platform-dependent unsigned integer (`usize`),
/// commonly used for indexing collections.
pub type UInt = usize;

/// Alias for an 8-bit signed integer (`i8`),
/// range: -128 to 127.
pub type Int8 = i8;

/// Alias for an 8-bit unsigned integer (`u8`),
/// range: 0 to 255.
pub type UInt8 = u8;

/// Alias for a 16-bit signed integer (`i16`),
/// range: -32,768 to 32,767.
pub type Int16 = i16;

/// Alias for a 16-bit unsigned integer (`u16`),
/// range: 0 to 65,535.
pub type UInt16 = u16;

/// Alias for a 32-bit signed integer (`i32`),
/// range: -2,147,483,648 to 2,147,483,647.
pub type Int32 = i32;

/// Alias for a 32-bit unsigned integer (`u32`),
/// range: 0 to 4,294,967,295.
pub type UInt32 = u32;

/// Alias for a 64-bit signed integer (`i64`),
/// range: −9,223,372,036,854,775,808 to 9,223,372,036,854,775,807.
pub type Int64 = i64;

/// Alias for a 64-bit unsigned integer (`u64`),
/// range: 0 to 18,446,744,073,709,551,615.
pub type UInt64 = u64;

pub mod collections;
pub mod errors;
pub mod num;

mod random;
mod traits;

pub use random::*;
pub use traits::*;
