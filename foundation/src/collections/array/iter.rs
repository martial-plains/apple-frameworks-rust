use core::{
    alloc::Layout,
    ptr::{self, NonNull},
};

use alloc::alloc::dealloc;

pub struct Iter<T> {
    pub(super) ptr: NonNull<T>,
    pub(super) capacity: usize,
    pub(super) start: usize,
    pub(super) end: usize,
}

impl<T> Iterator for Iter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            return None;
        }

        unsafe {
            let item = ptr::read(self.ptr.as_ptr().add(self.start));
            self.start += 1;
            Some(item)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.end - self.start;
        (len, Some(len))
    }
}

impl<T> Drop for Iter<T> {
    fn drop(&mut self) {
        unsafe {
            for i in self.start..self.end {
                ptr::drop_in_place(self.ptr.as_ptr().add(i));
            }
            let layout = Layout::array::<T>(self.capacity).unwrap();
            dealloc(self.ptr.as_ptr().cast(), layout);
        }
    }
}
