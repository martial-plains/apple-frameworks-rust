use rust_class_derive::rust_class;

#[derive(Debug)]
pub struct Base<T>(T);

#[derive(Debug, Clone, Copy)]
#[rust_class]
pub struct NSObject;

impl NSObject {
    #[must_use]
    pub const fn new() -> Self {
        NSObjectImpl { _super: Some(()) }
    }
    pub const fn super_class(self) -> Option<()> {
        self._super
    }
}

#[derive(Debug, Clone, Copy)]
#[rust_class(NSObject)]
pub struct NSArray;
