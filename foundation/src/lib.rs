#![no_std]
#![warn(
    clippy::nursery,
    clippy::pedantic,
    clippy::all,
    missing_debug_implementations,
    missing_copy_implementations
)]
#![debugger_visualizer(natvis_file = "../.natvis")]

extern crate alloc;

pub mod collections;
pub mod errors;
pub mod num;
