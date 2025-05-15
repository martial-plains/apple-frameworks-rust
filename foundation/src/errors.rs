use alloc::boxed::Box;

pub trait Error: core::error::Error {}

#[derive(Debug, Clone, Copy)]
pub enum Result<Success, Failure = Box<dyn Error>> {
    Success(Success),
    Failure(Failure),
}
