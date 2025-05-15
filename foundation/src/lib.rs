#![no_std]
#![warn(clippy::nursery, clippy::pedantic, clippy::all)]
#![feature(rustc_attrs, try_trait_v2)]
#![debugger_visualizer(natvis_file = "../.natvis")]

extern crate alloc;

pub mod collections;
pub mod errors;
pub mod num;
