#![allow(warnings)]
use rust_class_derive::rust_class;

#[derive(Debug)]
#[rust_class]
pub struct Animal {
    pub species: String,
    pub age: usize,
    pub name: String,
}

impl Animal {
    fn eat() {}

    fn sleep() {}

    fn make_sound() {}
}

#[derive(Debug)]
#[rust_class(Animal)]
pub struct Cat {}

impl Cat {
    fn new() -> Self {
        Self {
            _super: Some(AnimalImpl::default()),
        }
    }
}

fn main() {
    let mut cat = Cat::new();

    cat.name = "Buddy".to_owned();
    cat.age = 6;
    cat.species = "Orange cat".to_owned();

    println!("{}", cat.age)
}
