use alloc::{
    alloc::{alloc, dealloc, realloc},
    boxed::Box,
    vec::Vec,
};
use iter::Iter;

use core::{
    alloc::Layout,
    ops::{Index, IndexMut, Range},
    ptr::{self, NonNull},
};

use super::sequences::Sequence;

mod iter;

#[derive(Debug, Clone)]
pub struct Array<T> {
    ptr: NonNull<T>,
    capacity: usize,
    length: usize,
}

impl<T> Default for Array<T> {
    fn default() -> Self {
        let capacity = 4;
        let layout = Layout::array::<T>(capacity).expect("Invalid layout");
        let raw_ptr = unsafe { alloc(layout).cast::<T>() };

        let ptr = NonNull::new(raw_ptr).expect("Memory allocation failed");

        Self {
            ptr,
            capacity,
            length: 0,
        }
    }
}

impl<T> Array<T> {
    /// Creates an array with the specified capacity, then calls the given closure with a buffer covering the array’s uninitialized memory.
    ///
    /// # Panics
    ///
    /// This function will panic in the following cases:
    ///
    /// - If the requested `capacity` causes an overflow when computing the memory layout.
    /// - If memory allocation fails (e.g., the allocator returns a null pointer).
    ///
    /// These panics prevent the creation of an instance with an invalid or null pointer.
    /// For fallible allocation, consider using a method that returns a `Result` instead.
    #[must_use]
    pub fn with_uninitialized<F>(capacity: usize, initializer: F) -> Self
    where
        F: FnOnce(*mut T, &mut usize),
    {
        let ptr = if capacity == 0 {
            NonNull::dangling()
        } else {
            let layout = Layout::array::<T>(capacity).expect("Invalid layout");
            let raw = unsafe { alloc(layout).cast::<T>() };
            NonNull::new(raw).expect("Memory allocation failed")
        };

        let mut length: usize = 0;
        initializer(ptr.as_ptr(), &mut length);

        Self {
            ptr,
            capacity,
            length,
        }
    }

    pub fn repeating(value: T, count: usize) -> Self
    where
        T: Copy,
    {
        Self::with_uninitialized(count, |ptr, len| unsafe {
            for i in 0..count {
                ptr::write(ptr.add(i), value);
            }
            *len = count;
        })
    }

    /// A `bool` value indicating whether the collection is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// The number of elements in the array.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.length
    }

    /// The total number of elements that the array can contain without allocating new storage.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns a reference to the first element, if available.
    #[must_use]
    pub fn first(&self) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            Some(&self[0])
        }
    }

    /// Returns a reference to the last element, if available.
    #[must_use]
    pub fn last(&self) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            Some(&self[self.length - 1])
        }
    }

    /// Adds a new element at the end of the array.
    pub fn append(&mut self, value: T) {
        if self.length == self.capacity {
            self.resize();
        }

        unsafe {
            ptr::write(self.ptr.as_ptr().add(self.length), value);
        }

        self.length += 1;
    }

    /// Inserts a new element at the specified position.
    ///
    /// Shifts all elements after the specified index to the right by one position.
    ///
    /// # Panics
    ///
    /// Panics if `at > self.length`, i.e., if the insertion index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use foundation::collections::array::Array;
    ///
    /// let mut array = Array::default();
    /// array.append(1);
    /// array.append(2);
    /// array.insert(3, 1); // Inserts 3 at index 1
    /// assert_eq!(array[1], 3);
    /// ```
    pub fn insert(&mut self, element: T, at: usize)
    where
        T: Copy,
    {
        assert!(at <= self.length, "Insert position out of bounds");

        self.insert_values_at(at, [element]);
    }

    /// Inserts the elements of a sequence into the collection at the specified position.
    pub fn insert_many<C>(&mut self, contents: C, at: usize)
    where
        C: IntoIterator<Item = T>,
        T: Copy,
    {
        self.insert_values_at(at, contents);
    }

    /// Replaces a range of elements with the elements from the specified collection.
    ///
    /// The elements in the given `range` will be removed, and the elements yielded by
    /// the iterator `with` will be inserted in their place. The array will resize
    /// automatically if the new elements exceed the current capacity.
    ///
    /// # Panics
    ///
    /// Panics if the range is invalid:
    ///
    /// - If `range.start > range.end`
    /// - If `range.end > self.count()`
    ///
    /// # Examples
    ///
    /// ```
    /// use foundation::collections::array::Array;
    ///
    /// let mut array = Array::default();
    /// array.append(1);
    /// array.append(2);
    /// array.append(3);
    /// array.replace_subrange(1..2, [9, 8]); // Replaces element at index 1 with 9 and 8
    /// assert_eq!(array[1], 9);
    /// assert_eq!(array[2], 8);
    /// ```
    pub fn replace_subrange<C>(&mut self, range: Range<usize>, with: C)
    where
        C: IntoIterator<Item = T>,
        T: Copy,
    {
        assert!(
            range.end <= self.count() && range.start <= range.end,
            "Invalid range"
        );

        let new_values: Vec<T> = with.into_iter().collect();
        let removed = range.end - range.start;

        assert!(
            removed <= self.length,
            "Attempt to remove more elements than present in the array"
        );

        assert!(
            (self.length >= removed),
            "Attempt to remove more elements than present"
        );

        let shift_amount = if new_values.len() > removed {
            new_values.len() - removed
        } else {
            0
        };

        while self.length - removed + new_values.len() > self.capacity {
            self.resize();
        }

        self.shift_right(range.end, shift_amount);
        self.write_slice(range.start, &new_values);

        self.length = self.length.saturating_sub(removed) + new_values.len();
    }

    /// Reserves enough space to store the specified number of elements.
    pub fn reserve_capacity(&mut self, new_capacity: usize) {
        if new_capacity > self.capacity {
            self.realloc(new_capacity);
        }
    }

    /// Removes and returns the element at the specified position.
    ///
    /// All elements following the removed one are shifted one position to the left.
    ///
    /// # Panics
    ///
    /// Panics if `at >= self.length`.
    ///
    /// # Examples
    ///
    /// ```
    /// use foundation::collections::array::Array;
    ///
    /// let mut array = Array::default();
    /// array.append(10);
    /// array.append(20);
    /// let value = array.remove(0);
    /// assert_eq!(value, 10);
    /// assert_eq!(array[0], 20);
    /// ```
    pub fn remove(&mut self, at: usize) -> T {
        assert!(at < self.length, "Remove index out of bounds");

        unsafe {
            let value = ptr::read(self.ptr.as_ptr().add(at));
            self.shift_left(at + 1, 1);
            self.length -= 1;
            value
        }
    }

    /// Removes and returns the first element of the array.
    ///
    /// Shifts all remaining elements to the left by one.
    ///
    /// # Panics
    ///
    /// Panics if the array is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use foundation::collections::array::Array;
    ///
    /// let mut array = Array::default();
    /// array.append(1);
    /// array.append(2);
    /// let first = array.remove_first();
    /// assert_eq!(first, 1);
    /// assert_eq!(array[0], 2);
    /// ```
    pub fn remove_first(&mut self) -> T {
        assert!((self.length != 0), "Cannot remove from an empty array");
        self.remove(0)
    }

    /// Removes the first `n` elements from the array.
    ///
    /// Shifts all remaining elements to the left by `n` positions.
    ///
    /// # Panics
    ///
    /// Panics if `n > self.length`.
    ///
    /// # Examples
    ///
    /// ```
    /// use foundation::collections::array::Array;
    ///
    /// let mut array = Array::default();
    /// array.append(1);
    /// array.append(2);
    /// array.append(3);
    /// array.remove_first_n(2);
    /// assert_eq!(array[0], 3);
    /// ```
    pub fn remove_first_n(&mut self, n: usize) {
        assert!(
            (n <= self.length),
            "Cannot remove more elements than present"
        );

        self.shift_left(n, n);

        self.length -= n;
    }

    /// Removes and returns the last element of the array.
    ///
    /// # Panics
    ///
    /// Panics if the array is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use foundation::collections::array::Array;
    ///
    /// let mut array = Array::default();
    /// array.append(1);
    /// array.append(2);
    /// let last = array.remove_last();
    /// assert_eq!(last, 2);
    /// ```
    pub fn remove_last(&mut self) -> T {
        assert!(self.length > 0, "Empty array");
        self.remove(self.length - 1)
    }

    /// Removes the last `n` elements from the array.
    ///
    /// # Panics
    ///
    /// Panics if `n > self.length`.
    ///
    /// # Examples
    ///
    /// ```
    /// use foundation::collections::array::Array;
    ///
    /// let mut array = Array::default();
    /// array.append(1);
    /// array.append(2);
    /// array.append(3);
    /// array.append(4);
    /// array.remove_last_n(2);
    /// assert_eq!(array.count(), 2);
    /// ```
    pub fn remove_last_n(&mut self, n: usize) {
        assert!(
            (n <= self.length),
            "Cannot remove more elements than present"
        );

        unsafe {
            for i in (self.length - n)..self.length {
                ptr::drop_in_place(self.ptr.as_ptr().add(i));
            }
        }

        self.length -= n;
    }

    /// Removes the elements in the specified subrange from the array.
    ///
    /// Elements after the range are shifted left to fill the gap.
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds, i.e., if
    /// `range.start >= self.length` or `range.end > self.length`.
    ///
    /// # Examples
    ///
    /// ```
    /// use foundation::collections::array::Array;
    ///
    /// let mut array = Array::default();
    /// array.append(10);
    /// array.append(20);
    /// array.append(30);
    /// array.append(40);
    /// array.remove_subrange(1..3);
    /// assert_eq!(array[0], 10);
    /// assert_eq!(array[1], 40);
    /// ```
    pub fn remove_subrange(&mut self, range: core::ops::Range<usize>) {
        assert!(
            !(range.start >= self.length || range.end > self.length),
            "Remove range out of bounds"
        );

        let count = range.end - range.start;

        self.shift_left(range.end, count);

        self.length -= count;
    }

    /// Removes all the elements that satisfy the given predicate.
    pub fn remove_all<F>(&mut self, predicate: F)
    where
        F: Fn(&T) -> bool,
    {
        let mut write = 0;
        unsafe {
            for read in 0..self.length {
                let item = &*self.ptr.as_ptr().add(read);
                if !predicate(item) {
                    if write != read {
                        ptr::write(
                            self.ptr.as_ptr().add(write),
                            ptr::read(self.ptr.as_ptr().add(read)),
                        );
                    }
                    write += 1;
                }
            }
        }
        self.length = write;
    }

    /// Removes all elements from the array.
    ///
    /// Optionally retains the capacity of the array if `keep_capacity` is true.
    ///
    /// # Panics
    ///
    /// Panics if memory layout is invalid when deallocating, which could happen if
    /// `self.capacity` is invalid or the deallocation process fails (e.g., due to a
    /// corrupted memory layout).
    ///
    /// # Examples
    ///
    /// ```
    /// use foundation::collections::array::Array;
    ///
    /// let mut array = Array::default();
    /// array.append(1);
    /// array.append(2);
    /// array.append(3);
    /// array.remove_all_with_capacity(false); // Removes all elements and deallocates memory
    /// assert_eq!(array.count(), 0);
    /// ```
    pub fn remove_all_with_capacity(&mut self, keep_capacity: bool) {
        if !keep_capacity && self.capacity > 0 {
            unsafe {
                let layout = Layout::array::<T>(self.capacity).unwrap();
                dealloc(self.ptr.as_ptr().cast(), layout);
            }
            self.ptr = NonNull::dangling();
            self.capacity = 0;
        }
        self.length = 0;
    }

    /// Removes and returns the last element of the collection.
    pub fn pop_last(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            Some(self.remove_last())
        }
    }

    pub fn first_index_of(&self, element: &T) -> Option<usize>
    where
        T: PartialEq,
    {
        (0..self.length).find(|&i| &self[i] == element)
    }

    pub fn index_of(&self, element: &T) -> Option<usize>
    where
        T: PartialEq,
    {
        self.first_index_of(element)
    }

    pub fn first_index_where<F>(&self, predicate: F) -> Option<usize>
    where
        F: Fn(&T) -> bool,
    {
        (0..self.length).find(|&i| predicate(&self[i]))
    }

    pub fn last_where<F>(&self, predicate: F) -> Option<&T>
    where
        F: Fn(&T) -> bool,
    {
        for i in (0..self.length).rev() {
            if predicate(&self[i]) {
                return Some(&self[i]);
            }
        }
        None
    }

    pub fn last_index_of(&self, element: &T) -> Option<usize>
    where
        T: PartialEq,
    {
        (0..self.length).rev().find(|&i| &self[i] == element)
    }

    pub fn last_index_where<F>(&self, predicate: F) -> Option<usize>
    where
        F: Fn(&T) -> bool, // The predicate takes a reference to an element and returns a bool
    {
        (0..self.length).rev().find(|&i| predicate(&self[i]))
    }

    #[must_use]
    pub fn prefix(&self, n: usize) -> Self
    where
        T: Copy,
    {
        let count = n.min(self.length);
        let mut result = Self::default();
        for i in 0..count {
            result.append(self[i]);
        }
        result
    }

    #[must_use]
    pub fn prefix_through(&self, index: usize) -> Array<T>
    where
        T: Copy,
    {
        assert!(index < self.length, "Index out of bounds");
        self.prefix(index + 1)
    }

    #[must_use]
    pub fn prefix_up_to(&self, index: usize) -> Array<T>
    where
        T: Copy,
    {
        assert!(index <= self.length, "Index out of bounds");
        self.prefix(index)
    }

    #[must_use]
    pub fn prefix_while<F>(&self, predicate: F) -> Self
    where
        T: Copy,
        F: Fn(&T) -> bool,
    {
        let mut result = Self::default();
        for i in 0..self.length {
            if predicate(&self[i]) {
                result.append(self[i]);
            } else {
                break;
            }
        }
        result
    }

    #[must_use]
    pub fn suffix(&self, n: usize) -> Self
    where
        T: Copy,
    {
        let start = self.length.saturating_sub(n);
        let mut result = Self::default();
        for i in start..self.length {
            result.append(self[i]);
        }
        result
    }

    #[must_use]
    pub fn suffix_from(&self, index: usize) -> Self
    where
        T: Copy,
    {
        assert!(index <= self.length, "Index out of bounds");
        let mut result = Self::default();
        for i in index..self.length {
            result.append(self[i]);
        }
        result
    }

    #[must_use]
    pub fn drop_first(&self, n: usize) -> Self
    where
        T: Copy,
    {
        let start = n.min(self.length);

        Self::with_uninitialized(self.length - start, |ptr: *mut T, len| unsafe {
            for i in start..self.length {
                ptr::write(ptr.add(i - start), self[i]);
            }
            *len = self.length - start;
        })
    }

    #[must_use]
    pub fn drop_last(&self, n: usize) -> Self
    where
        T: Copy,
    {
        let end = self.length.saturating_sub(n);

        Self::with_uninitialized(end, |ptr: *mut T, len| unsafe {
            for i in 0..end {
                ptr::write(ptr.add(i), self[i]);
            }
            *len = end;
        })
    }

    #[must_use]
    pub fn drop_while<F>(&self, predicate: F) -> Self
    where
        T: Copy,
        F: Fn(&T) -> bool,
    {
        let mut start = 0;
        while start < self.length && predicate(&self[start]) {
            start += 1;
        }

        Self::with_uninitialized(self.length - start, |ptr: *mut T, len| unsafe {
            for i in start..self.length {
                ptr::write(ptr.add(i - start), self[i]);
            }
            *len = self.length - start;
        })
    }

    fn insert_values_at<I>(&mut self, at: usize, contents: I)
    where
        I: IntoIterator<Item = T>,
        T: Copy,
    {
        let values: Vec<T> = contents.into_iter().collect();
        let count = values.len();
        assert!(at <= self.length, "Insert position out of bounds");

        while self.length + count > self.capacity {
            self.resize();
        }

        self.shift_right(at, count);
        self.write_slice(at, &values);

        self.length += count;
    }

    fn resize(&mut self) {
        let new_capacity = if self.capacity == 0 {
            1
        } else {
            self.capacity * 2
        };
        self.realloc(new_capacity);
    }

    fn realloc(&mut self, new_capacity: usize) {
        let new_layout = Layout::array::<T>(new_capacity).unwrap();
        let new_ptr = unsafe {
            if self.capacity == 0 {
                alloc(new_layout).cast()
            } else {
                let old_layout = Layout::array::<T>(self.capacity).unwrap();
                realloc(self.ptr.as_ptr().cast(), old_layout, new_layout.size()).cast()
            }
        };
        self.ptr = NonNull::new(new_ptr).expect("Reallocation failed");
        self.capacity = new_capacity;
    }

    fn shift_right(&mut self, from: usize, count: usize) {
        for i in (from..self.length).rev() {
            unsafe {
                ptr::write(
                    self.ptr.as_ptr().add(i + count),
                    ptr::read(self.ptr.as_ptr().add(i)),
                );
            };
        }
    }

    fn shift_left(&mut self, from: usize, count: usize) {
        for i in from..self.length {
            unsafe {
                ptr::write(
                    self.ptr.as_ptr().add(i - count),
                    ptr::read(self.ptr.as_ptr().add(i)),
                );
            };
        }
    }

    fn write_slice(&mut self, at: usize, values: &[T])
    where
        T: Copy,
    {
        for (i, &val) in values.iter().enumerate() {
            unsafe { ptr::write(self.ptr.as_ptr().add(at + i), val) };
        }
    }
}

impl<'a, T: 'a> Sequence for &'a Array<T> {
    type Item = &'a T;
    type Iterator = core::slice::Iter<'a, T>;

    fn iter(&self) -> Self::Iterator {
        self.into_iter()
    }

    fn underestimated_count(&self) -> usize {
        self.length
    }
}

impl<T> Index<usize> for Array<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        assert!((index < self.length), "Index out of bounds");
        unsafe { &*self.ptr.as_ptr().add(index) }
    }
}

impl<T> IndexMut<usize> for Array<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!((index < self.length), "Index out of bounds");
        unsafe { &mut *self.ptr.as_ptr().add(index) }
    }
}

impl<T> FromIterator<T> for Array<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut array = Self::default();
        array.reserve_capacity(lower);
        for item in iter {
            array.append(item);
        }
        array
    }
}

impl<T> IntoIterator for Array<T> {
    type Item = T;
    type IntoIter = Iter<T>;

    fn into_iter(self) -> Self::IntoIter {
        let iter = Iter {
            ptr: self.ptr,
            capacity: self.capacity,
            start: 0,
            end: self.length,
        };

        core::mem::forget(self);
        iter
    }
}

impl<'a, T> IntoIterator for &'a Array<T> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.length).iter() }
    }
}

impl<'a, T> IntoIterator for &'a mut Array<T> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.length).iter_mut() }
    }
}

impl<T> Drop for Array<T> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            unsafe {
                for i in 0..self.length {
                    ptr::drop_in_place(self.ptr.as_ptr().add(i));
                }
                let layout = Layout::array::<T>(self.capacity).expect("Invalid layout");
                dealloc(self.ptr.as_ptr().cast::<u8>(), layout);
            }
        }
    }
}

#[macro_export]
macro_rules! array {
    ($($elem:expr),* $(,)?) => {{
        let mut arr = $crate::collections::array::Array::default();
        $(arr.append($elem);)*
        arr
    }};

    ($elem:expr; $count:expr) => {{
        let arr = $crate::collections::array::Array::repeating($elem, $count);
        arr
    }};
}

#[cfg(test)]
mod tests {
    use crate::collections::array::Array;

    #[test]
    fn test_default_array() {
        let arr: Array<i32> = Array::default();
        assert_eq!(arr.count(), 0);
        assert_eq!(arr.capacity(), 4);
        assert!(arr.is_empty());
    }

    #[test]
    fn test_append_and_indexing() {
        let mut arr = Array::default();
        arr.append(10);
        arr.append(20);
        arr.append(30);
        assert_eq!(arr.count(), 3);
        assert_eq!(arr[0], 10);
        assert_eq!(arr[1], 20);
        assert_eq!(arr[2], 30);
    }

    #[test]
    fn test_first_last() {
        let arr = array![5, 10, 15];
        assert_eq!(arr.first(), Some(&5));
        assert_eq!(arr.last(), Some(&15));
    }

    #[test]
    fn test_insert() {
        let mut arr = array![1, 2, 4];
        arr.insert(3, 2);
        assert_eq!(arr[0], 1);
        assert_eq!(arr[1], 2);
        assert_eq!(arr[2], 3);
        assert_eq!(arr[3], 4);
    }

    #[test]
    fn test_insert_many() {
        let mut arr = array![1, 4];
        arr.insert_many([2, 3], 1);
        assert_eq!(arr.count(), 4);
        assert_eq!(arr[0], 1);
        assert_eq!(arr[1], 2);
        assert_eq!(arr[2], 3);
        assert_eq!(arr[3], 4);
    }

    #[test]
    fn test_replace_subrange() {
        let mut arr = array![1, 2, 3, 4, 5];
        (1..4).count();
        arr.replace_subrange(1..4, [10, 11]);
        assert_eq!(arr.count(), 4);
        assert_eq!(arr[0], 1);
        assert_eq!(arr[1], 10);
        assert_eq!(arr[2], 11);
    }

    #[test]
    fn test_remove_variants() {
        let mut arr = array![1, 2, 3, 4, 5];
        assert_eq!(arr.remove_first(), 1);
        assert_eq!(arr.remove_last(), 5);
        arr.remove_first_n(1);
        arr.remove_last_n(1);
        assert_eq!(arr.count(), 1);
    }

    #[test]
    fn test_remove_subrange() {
        let mut arr = array![1, 2, 3, 4, 5];
        arr.remove_subrange(1..4);
        assert_eq!(arr.count(), 2);
        assert_eq!(arr[0], 1);
        assert_eq!(arr[1], 5);
    }

    #[test]
    fn test_remove_all() {
        let mut arr = array![1, 2, 3, 4, 5];
        arr.remove_all(|x| x % 2 == 0);
        assert_eq!(arr.count(), 3);
        assert_eq!(arr[0], 1);
        assert_eq!(arr[1], 3);
        assert_eq!(arr[2], 5);
    }

    #[test]
    fn test_pop_last() {
        let mut arr = array![1, 2, 3];
        assert_eq!(arr.pop_last(), Some(3));
        assert_eq!(arr.pop_last(), Some(2));
        assert_eq!(arr.pop_last(), Some(1));
        assert_eq!(arr.pop_last(), None);
    }

    #[test]
    fn test_macro_repeat() {
        let arr = array![7; 4];

        assert_eq!(arr.count(), 4);
        for i in 0..4 {
            assert_eq!(arr[i], 7);
        }
    }

    #[test]
    fn test_reserve_capacity() {
        let mut arr = Array::default();
        arr.reserve_capacity(10);
        assert!(arr.capacity() >= 10);
        arr.append(1);
        assert_eq!(arr.count(), 1);
    }

    #[test]
    fn test_remove_all_with_capacity() {
        let mut arr = array![1, 2, 3];
        let old_capacity = arr.capacity();

        arr.remove_all_with_capacity(true);
        assert_eq!(arr.count(), 0);
        assert_eq!(arr.capacity(), old_capacity);

        arr.append(10);
        assert_eq!(arr[0], 10);

        arr.remove_all_with_capacity(false);
        assert_eq!(arr.count(), 0);
        assert_eq!(arr.capacity(), 0);
    }
}
