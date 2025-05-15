use alloc::boxed::Box;

pub trait Error: core::error::Error {}

pub enum Result<Success, Failure = Box<dyn Error>> {
    Success(Success),
    Failure(Failure),
}
