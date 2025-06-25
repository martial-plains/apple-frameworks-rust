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

pub mod collections;
pub mod errors;
pub mod num;

mod random;
mod traits;

pub use random::*;
pub use traits::*;
