/*!
# Array Module

This module provides a custom implementation of a dynamic array (`Array<T>`) that is designed to be similar to Swift's `Array`. Unlike Rust's `Vec<T>`, which is the standard dynamic array type in Rust, this implementation focuses on manual memory management and closely mirrors the behavior and structure of Swift's `Array`.

The `Array` type is implemented using a raw pointer (`NonNull<T>`) to manage the underlying buffer of elements. This allows fine-grained control over memory allocation and deallocation, making it a great choice for low-level applications where custom memory management is desired.

## Key Features
- **Manual Memory Management**: Uses `NonNull<T>` to allocate and manage memory, providing a low-level implementation of dynamic arrays.
- **Dynamic Size**: The array can grow or shrink as elements are added or removed, with dynamic resizing based on the current capacity.
- **Swift-Like API**: Designed to offer an API and functionality similar to Swift’s `Array`, with methods for inserting, removing, and accessing elements.
- **Efficient Memory Use**: Includes custom allocation and deallocation routines to optimize memory usage and reduce overhead.

## Example Usage

```rust
use foundation::collections::Array;

let mut arr = Array::default();
arr.append(10);
arr.append(20);
arr.append(30);

println!("Array: {:?}", arr);

if let Some(value) = arr.pop_last() {
    println!("Popped value: {}", value);
}

let first_element = arr.first();
match first_element {
    Some(&val) => println!("First element: {}", val),
    None => println!("Array is empty"),
}
*/

use alloc::{
    alloc::{alloc, dealloc, realloc},
    vec::Vec,
};

use core::{
    alloc::Layout,
    clone::Clone,
    cmp::{Ordering, PartialEq},
    ops::{Index, IndexMut},
    ptr::{self, NonNull},
    range::Range,
};

use crate::{
    Int, UInt,
    collections::{Collection, DefaultIndices, MutableCollection, Sequence},
    errors::Result::Success,
};

use super::sequences::IndexingIterator;

/// A fixed-size inline array type alias.
///
/// This is simply a type alias for a fixed-size array `[T; COUNT]`.
/// It is useful when you want to express an array with a compile-time constant size.
///
/// # Type Parameters
/// - `COUNT`: The number of elements in the array (a compile-time constant).
/// - `T`: The element type stored in the array.
pub type InlineArray<const COUNT: usize, T> = [T; COUNT];

/// A dynamically sized array with manual memory management.
///
/// This struct represents a growable array that keeps track of a pointer to the elements,
/// the current capacity (how many elements it can hold without reallocating),
/// and the current length (how many elements are initialized).
///
/// It uses a `NonNull<T>` pointer internally to manage the buffer safely without
/// allowing null pointers.
///
/// # Type Parameters
/// - `T`: The element type stored in the array.
#[derive(Debug)]
pub struct Array<T> {
    ptr: NonNull<T>,
    capacity: usize,
    length: usize,
}

impl<T: Clone> Clone for Array<T> {
    fn clone(&self) -> Self {
        Self::with_uninitialized(self.capacity, |dst, length| {
            for i in 0..self.length {
                unsafe {
                    let src = self.ptr.as_ptr().add(i);
                    let dst_ptr = dst.add(i);
                    dst_ptr.write((*src).clone());
                    *length += 1;
                }
            }
        })
    }
}

impl<T: PartialEq> PartialEq for Array<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length {
            return false;
        }

        for i in 0..self.length {
            let a = unsafe { &*self.ptr.as_ptr().add(i) };
            let b = unsafe { &*other.ptr.as_ptr().add(i) };
            if a != b {
                return false;
            }
        }

        true
    }
}
impl<T: PartialOrd> PartialOrd for Array<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let min_len = self.length.min(other.length);
        for i in 0..min_len {
            let a = unsafe { self.ptr.as_ptr().add(i).read() };
            let b = unsafe { other.ptr.as_ptr().add(i).read() };
            match a.partial_cmp(&b) {
                Some(Ordering::Equal) => {}
                non_eq => return non_eq,
            }
        }
        self.length.partial_cmp(&other.length)
    }
}

impl<T: Ord> Ord for Array<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        let min_len = self.length.min(other.length);
        for i in 0..min_len {
            let a = unsafe { self.ptr.as_ptr().add(i).read() };
            let b = unsafe { other.ptr.as_ptr().add(i).read() };
            match a.cmp(&b) {
                Ordering::Equal => {}
                non_eq => return non_eq,
            }
        }
        self.length.cmp(&other.length)
    }
}

impl<T: Eq> Eq for Array<T> {}

unsafe impl<T> Send for Array<T> {}

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

    /// Initializes a new `Array` from a sequence of elements.
    ///
    /// Allocates enough memory to hold all elements of the given sequence,
    /// clones each element into the newly allocated memory, and returns
    /// an `Array` owning that memory.
    ///
    /// # Type Parameters
    /// - `S`: A sequence type that yields elements of type `T` and supports partial equality.
    ///
    /// # Panics
    /// - If the memory layout for the allocation is invalid.
    /// - If memory allocation fails.
    ///
    /// # Safety
    /// This function uses unsafe memory allocation and writes elements via raw pointers.
    pub fn init<S>(sequence: &S) -> Self
    where
        T: Clone,
        S: Sequence<Element = T> + PartialEq,
    {
        let mut iter = sequence.iter();
        let mut count = 0;

        while iter.next().is_some() {
            count += 1;
        }

        let layout = Layout::array::<T>(count).expect("Invalid layout");
        let ptr = unsafe { alloc(layout).cast::<T>() };
        let non_null_ptr = NonNull::new(ptr).expect("Failed to allocate memory");

        let iter = sequence.iter();
        let mut index = 0;

        unsafe {
            for element in iter {
                ptr::write(non_null_ptr.as_ptr().add(index), element.clone());
                index += 1;
            }
        }

        Self {
            ptr: non_null_ptr,
            capacity: count,
            length: count,
        }
    }

    /// Returns an iterator over immutable references to the elements.
    ///
    /// Allows read-only traversal of the array contents.
    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.into_iter()
    }

    /// Returns an iterator over mutable references to the elements.
    ///
    /// Allows modifying elements in place.
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.into_iter()
    }

    /// Creates an `Array` filled by repeating the given value `count` times.
    ///
    /// Allocates space for `count` elements and initializes each element with `value`.
    ///
    /// # Constraints
    /// - `T` must implement `Copy` because elements are copied repeatedly.
    ///
    /// # Safety
    /// Uses unsafe pointer writes to initialize memory.
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
    /// use foundation::collections::Array;
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
    /// #![feature(new_range_api)]
    /// use foundation::collections::Array;
    /// use core::range::Range;
    ///
    /// let mut array = Array::default();
    /// array.append(1);
    /// array.append(2);
    /// array.append(3);
    /// array.replace_subrange(Range::from(1..2), [9, 8]); // Replaces element at index 1 with 9 and 8
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
    /// use foundation::collections::Array;
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
    /// use foundation::collections::Array;
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
    /// use foundation::collections::Array;
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
    /// use foundation::collections::Array;
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
    /// use foundation::collections::Array;
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
    /// use foundation::collections::Array;
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
    /// use foundation::collections::Array;
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

    /// Returns the index of the first occurrence of `element` in the array.
    ///
    /// # Parameters
    /// - `element`: A reference to the element to search for.
    ///
    /// # Returns
    /// - `Some(index)` if the element is found.
    /// - `None` if the element is not found.
    pub fn first_index_of(&self, element: &T) -> Option<usize>
    where
        T: PartialEq,
    {
        (0..self.length).find(|&i| &self[i] == element)
    }

    /// Returns the index of the first occurrence of `element` in the array.
    #[deprecated]
    pub fn index_of(&self, element: &T) -> Option<usize>
    where
        T: PartialEq,
    {
        self.first_index_of(element)
    }

    /// Returns the index of the first element satisfying the predicate.
    ///
    /// # Parameters
    /// - `predicate`: A closure that returns `true` for the desired element.
    ///
    /// # Returns
    /// - `Some(index)` of the first element where `predicate` returns `true`.
    /// - `None` if no element satisfies the predicate.
    pub fn first_index_where<F>(&self, predicate: F) -> Option<usize>
    where
        F: Fn(&T) -> bool,
    {
        (0..self.length).find(|&i| predicate(&self[i]))
    }

    /// Returns a reference to the last element that satisfies the predicate.
    ///
    /// # Parameters
    /// - `predicate`: A closure that returns `true` for the desired element.
    ///
    /// # Returns
    /// - `Some(&T)` for the last matching element.
    /// - `None` if no element satisfies the predicate.
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

    /// Returns the index of the last occurrence of `element` in the array.
    ///
    /// # Parameters
    /// - `element`: A reference to the element to search for.
    ///
    /// # Returns
    /// - `Some(index)` if the element is found.
    /// - `None` if the element is not found.
    pub fn last_index_of(&self, element: &T) -> Option<usize>
    where
        T: PartialEq,
    {
        (0..self.length).rev().find(|&i| &self[i] == element)
    }

    /// Returns the index of the last element satisfying the predicate.
    ///
    /// # Parameters
    /// - `predicate`: A closure that returns `true` for the desired element.
    ///
    /// # Returns
    /// - `Some(index)` of the last element where `predicate` returns `true`.
    /// - `None` if no element satisfies the predicate.
    pub fn last_index_where<F>(&self, predicate: F) -> Option<usize>
    where
        F: Fn(&T) -> bool,
    {
        (0..self.length).rev().find(|&i| predicate(&self[i]))
    }

    /// Returns a new `Array` containing the first `n` elements (a prefix) of this array.
    ///
    /// If `n` is greater than the length of the array, returns the whole array.
    ///
    /// # Returns
    /// - A new `Array` containing the prefix elements.
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

    /// Returns the prefix of the collection through the given index (inclusive).
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than or equal to the length of the collection.
    #[must_use]
    pub fn prefix_through(&self, index: usize) -> Self
    where
        T: Copy,
    {
        assert!(index < self.length, "Index out of bounds");
        self.prefix(index + 1)
    }

    /// Returns the prefix of the collection up to the given index (exclusive).
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than the length of the collection.
    #[must_use]
    pub fn prefix_up_to(&self, index: usize) -> Self
    where
        T: Copy,
    {
        assert!(index <= self.length, "Index out of bounds");
        self.prefix(index)
    }

    /// Returns a new `Array` containing the longest prefix of elements
    /// that satisfy the given predicate.
    ///
    /// Iterates from the start and collects elements until the predicate returns false.
    ///
    /// # Parameters
    /// - `predicate`: Closure returning `true` for elements to include.
    ///
    /// # Returns
    /// - An `Array` containing the prefix of matching elements.
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

    /// Returns a new `Array` containing the last `n` elements (a suffix) of this array.
    ///
    /// If `n` is greater than the array length, returns the whole array.
    ///
    /// # Parameters
    /// - `n`: The number of elements to include from the end.
    ///
    /// # Returns
    /// - An `Array` with the last `n` elements.
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

    /// Returns the suffix of the collection starting at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than the length of the collection.
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

    /// Returns a new `Array` with the first `n` elements dropped.
    ///
    /// If `n` is greater than the array length, returns an empty array.
    ///
    /// # Parameters
    /// - `n`: Number of elements to drop from the start.
    ///
    /// # Returns
    /// - An `Array` with the first `n` elements removed.
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

    /// Returns a new `Array` with the last `n` elements dropped.
    ///
    /// If `n` is greater than the array length, returns an empty array.
    ///
    /// # Parameters
    /// - `n`: Number of elements to drop from the end.
    ///
    /// # Returns
    /// - An `Array` with the last `n` elements removed.
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

    /// Returns a new `Array` with the longest prefix of elements satisfying the predicate dropped.
    ///
    /// Iterates from the start, dropping elements while the predicate returns `true`,
    /// then returns the rest of the array.
    ///
    /// # Parameters
    /// - `predicate`: Closure that returns `true` for elements to drop.
    ///
    /// # Returns
    /// - An `Array` starting from the first element where `predicate` returns `false`.
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

impl<T> Sequence for Array<T>
where
    T: Clone,
{
    type Element = T;

    type Iterator = IndexingIterator<Self>;

    fn iter(&self) -> Self::Iterator {
        IndexingIterator::new(self.clone())
    }

    fn underestimated_count(&self) -> usize {
        self.length
    }
}

impl<T> Collection for Array<T>
where
    T: Clone,
{
    type Index = UInt;

    type Indices = DefaultIndices<Self>;

    type SubSequence = ArraySlice<T>;

    fn start_index(&self) -> Self::Index {
        0
    }

    fn end_index(&self) -> Self::Index {
        self.length
    }

    fn index_after(&self, after: Self::Index) -> Option<Self::Element> {
        self.iter().nth(after).cloned()
    }

    fn count(&self) -> usize {
        self.length
    }

    fn is_empty(&self) -> bool {
        self.length == 0
    }

    #[allow(clippy::cast_sign_loss)]
    fn index_offset_by(&self, index: Self::Index, offset_by: Int) -> Self::Index {
        let new_index = index.saturating_add(offset_by as usize);
        new_index.min(self.end_index())
    }

    #[allow(clippy::cast_sign_loss)]
    fn index_offset_by_limited_by(
        &self,
        index: Self::Index,
        offset_by: Int,
        limited_by: Self::Index,
    ) -> Option<Self::Index> {
        let new_index = index.saturating_add(offset_by as usize);
        if new_index <= limited_by && new_index < self.end_index() {
            Some(new_index)
        } else {
            None
        }
    }

    fn indices(&self) -> Self::Indices {
        DefaultIndices::new(self.clone(), self.start_index(), self.end_index())
    }
}

impl<T> MutableCollection for Array<T>
where
    T: Clone,
{
    type SubSequence = ArraySlice<T>;

    fn partition_by<F>(&mut self, mut predicate: F) -> Self::Index
    where
        F: FnMut(&Self::Element) -> crate::errors::Result<bool>,
    {
        let mut i = 0;
        let mut j = self.length;

        while i != j {
            if matches!(predicate(&self[i]), Success(true)) {
                i += 1;
            } else {
                j -= 1;
                self.swap_at(i, j);
            }
        }

        i
    }

    fn swap_at(&mut self, index1: Self::Index, index2: Self::Index) {
        assert!(index1 < self.length && index2 < self.length);
        unsafe {
            let a = self.ptr.as_ptr().add(index1);
            let b = self.ptr.as_ptr().add(index2);
            core::ptr::swap(a, b);
        }
    }

    fn with_contiguous_mutable_storage_if_available<R, F>(&mut self, f: F) -> Option<R>
    where
        F: FnOnce(&mut [Self::Element]) -> R,
    {
        unsafe {
            Some(f(core::slice::from_raw_parts_mut(
                self.ptr.as_ptr(),
                self.length,
            )))
        }
    }
}

impl<'a, T: 'a> Sequence for &'a Array<T> {
    type Element = &'a T;
    type Iterator = core::slice::Iter<'a, T>;

    fn iter(&self) -> Self::Iterator {
        self.into_iter()
    }

    fn underestimated_count(&self) -> usize {
        self.length
    }
}

impl<T> Index<UInt> for Array<T> {
    type Output = T;

    fn index(&self, index: UInt) -> &Self::Output {
        assert!((index < self.length), "Index out of bounds");
        unsafe { &*self.ptr.as_ptr().add(index) }
    }
}

impl<T> IndexMut<UInt> for Array<T> {
    fn index_mut(&mut self, index: UInt) -> &mut Self::Output {
        assert!((index < self.length), "Index out of bounds");
        unsafe { &mut *self.ptr.as_ptr().add(index) }
    }
}

impl<T> Index<Range<UInt>> for Array<T> {
    type Output = [T];

    fn index(&self, index: Range<UInt>) -> &Self::Output {
        assert!(
            index.start < self.length && index.end <= self.length,
            "Index out of bounds"
        );
        unsafe {
            core::slice::from_raw_parts(self.ptr.as_ptr().add(index.start), index.end - index.start)
        }
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

impl<T: Clone + PartialEq> IntoIterator for Array<T> {
    type Item = T;
    type IntoIter = IndexingIterator<Self>;

    fn into_iter(self) -> Self::IntoIter {
        IndexingIterator::new(self)
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

/// Macro to create an `Array` collection with various initialization options.
///
/// This macro supports the following forms:
///
/// - `array!()`
///
///   Creates a default, empty `Array`.
///
/// - `array!(elem; count)`
///
///   Creates an `Array` filled with `count` copies of `elem`.
///
/// - `array!(elem1, elem2, ..., elemN)`
///
///   Creates an `Array` containing the given elements in order.
///
/// # Examples
///
/// ```
/// use foundation::{array, collections::Array};
///
/// // Create an empty array
/// let a: Array<i32> = array![];
///
/// // Create an array with 5 copies of 42
/// let b = array![42; 5];
///
/// // Create an array with given elements
/// let c = array![1, 2, 3, 4];
/// ```
#[macro_export]
macro_rules! array {
    () => (
        $crate::collections::Array::default()
    );

    ($elem:expr; $count:expr) => {
        $crate::collections::Array::repeating($elem, $count)
    };

    ($($elem:expr),* $(,)?) => {{
        let mut arr = $crate::collections::Array::default();
        $(arr.append($elem);)*
        arr
    }};
}

/// A dynamically allocated slice-like collection managing a buffer of elements.
///
/// `ArraySlice` provides manual memory management, insertion, removal, and indexing.
/// It stores elements in a contiguous memory block and manages capacity explicitly.
#[derive(Debug)]
pub struct ArraySlice<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
}

impl<T: Clone> Clone for ArraySlice<T> {
    fn clone(&self) -> Self {
        let mut slice = Self::new(self.capacity);
        for i in 0..self.len {
            let value = unsafe { &*self.ptr.as_ptr().add(i) };
            slice.insert(value.clone(), slice.len);
        }
        slice
    }
}

impl<T> ArraySlice<T> {
    /// Constructs a new `ArraySlice` with the specified capacity.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero or if memory allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use foundation::collections::ArraySlice;
    ///
    /// let slice: ArraySlice<i32> = ArraySlice::new(10);
    /// ```
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be greater than zero");
        let layout = Layout::array::<T>(capacity).expect("Invalid layout");

        unsafe {
            let ptr = alloc(layout).cast::<T>();
            assert!(!ptr.is_null(), "Failed to allocate memory");
            Self {
                ptr: NonNull::new(ptr).expect("Failed to create NonNull"),
                len: 0,
                capacity,
            }
        }
    }

    /// Inserts an element at the specified index, shifting subsequent elements to the right.
    ///
    /// # Panics
    ///
    /// Panics if `at` is greater than the current length.
    ///
    /// # Safety
    ///
    /// This function uses unsafe code to manipulate raw pointers.
    ///
    /// # Examples
    ///
    /// ```
    /// use foundation::collections::ArraySlice;
    ///
    /// let mut slice = ArraySlice::new(5);
    /// slice.insert(42, 0);
    /// ```
    pub fn insert(&mut self, element: T, at: usize) {
        assert!((at <= self.len), "Index out of bounds");

        unsafe {
            ptr::copy(
                self.ptr.as_ptr().add(at),
                self.ptr.as_ptr().add(at + 1),
                self.len - at,
            );
            ptr::write(self.ptr.as_ptr().add(at), element);
            self.len += 1;
        }
    }

    /// Removes and returns the element at the specified index, shifting subsequent elements left.
    ///
    /// # Panics
    ///
    /// Panics if `at` is out of bounds.
    ///
    /// # Safety
    ///
    /// Uses unsafe pointer manipulation for efficient element removal.
    ///
    /// # Examples
    ///
    /// ```
    /// use foundation::collections::ArraySlice;
    ///
    /// let mut slice = ArraySlice::new(5);
    /// slice.insert(42, 0);
    /// let removed = slice.remove(0);
    /// assert_eq!(removed, 42);
    /// ```
    pub fn remove(&mut self, at: usize) -> T {
        assert!((at < self.len), "Index out of bounds");

        unsafe {
            let removed = ptr::read(self.ptr.as_ptr().add(at));
            ptr::copy(
                self.ptr.as_ptr().add(at + 1),
                self.ptr.as_ptr().add(at),
                self.len - at - 1,
            );
            self.len -= 1;
            removed
        }
    }

    /// Returns the total capacity of the `ArraySlice`.
    ///
    /// # Examples
    ///
    /// ```
    /// use foundation::collections::ArraySlice;
    ///
    /// let slice = ArraySlice::<i32>::new(10);
    /// assert_eq!(slice.capacity(), 10);
    /// ```
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Reserves additional capacity to accommodate at least `additional` more elements.
    ///
    /// This may reallocate the internal buffer and move existing elements to the new memory.
    ///
    /// # Panics
    ///
    /// Panics if memory allocation fails.
    ///
    /// # Safety
    ///
    /// Uses unsafe pointer operations and manual memory management.
    ///
    /// # Examples
    ///
    /// ```
    /// use foundation::collections::ArraySlice;
    ///
    /// let mut slice: ArraySlice<usize> = ArraySlice::new(2);
    /// slice.reserve_capacity(3); // total capacity becomes 5
    /// ```
    pub fn reserve_capacity(&mut self, additional: usize) {
        let new_capacity = self.capacity + additional;
        let new_layout = Layout::array::<T>(new_capacity).expect("Invalid layout");
        let old_layout = Layout::array::<T>(self.capacity).expect("Invalid layout");

        unsafe {
            let new_ptr = alloc(new_layout).cast::<T>();
            assert!(!new_ptr.is_null(), "Failed to allocate memory");

            ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr, self.len);
            dealloc(self.ptr.as_ptr().cast::<u8>(), old_layout);

            self.ptr = NonNull::new(new_ptr).expect("Failed to update NonNull");
            self.capacity = new_capacity;
        }
    }
}

impl<T> Index<UInt> for ArraySlice<T> {
    type Output = T;

    fn index(&self, idx: UInt) -> &Self::Output {
        assert!((idx < self.len), "Index out of bounds");
        unsafe { &*self.ptr.as_ptr().add(idx) }
    }
}

impl<T> IndexMut<UInt> for ArraySlice<T> {
    fn index_mut(&mut self, idx: UInt) -> &mut Self::Output {
        assert!((idx < self.len), "Index out of bounds");
        unsafe { &mut *self.ptr.as_ptr().add(idx) }
    }
}

impl<T> Index<Range<UInt>> for ArraySlice<T> {
    type Output = [T];

    fn index(&self, index: Range<UInt>) -> &Self::Output {
        assert!(
            index.start < self.len && index.end <= self.len,
            "Index out of bounds"
        );
        unsafe {
            core::slice::from_raw_parts(self.ptr.as_ptr().add(index.start), index.end - index.start)
        }
    }
}

impl<T> Sequence for ArraySlice<T>
where
    T: Clone,
{
    type Element = T;

    type Iterator = IndexingIterator<Self>;

    fn iter(&self) -> Self::Iterator {
        IndexingIterator::new(self.clone())
    }

    fn underestimated_count(&self) -> usize {
        self.len
    }
}

impl<T> Collection for ArraySlice<T>
where
    T: Clone,
{
    type Index = usize;

    type Indices = DefaultIndices<Self>;

    type SubSequence = ArraySlice<Self>;

    fn start_index(&self) -> Self::Index {
        0
    }

    fn end_index(&self) -> Self::Index {
        self.len
    }

    fn index_after(&self, i: Self::Index) -> Option<Self::Element> {
        if i < self.len {
            Some(self[i].clone())
        } else {
            None
        }
    }

    fn count(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[allow(clippy::cast_sign_loss)]
    fn index_offset_by(&self, index: Self::Index, offset_by: Int) -> Self::Index {
        let new_index = index.saturating_add(offset_by as usize);
        new_index.min(self.end_index())
    }

    #[allow(clippy::cast_sign_loss)]
    fn index_offset_by_limited_by(
        &self,
        index: Self::Index,
        offset_by: Int,
        limited_by: Self::Index,
    ) -> Option<Self::Index> {
        let new_index = index.saturating_add(offset_by as usize);
        if new_index <= limited_by && new_index < self.end_index() {
            Some(new_index)
        } else {
            None
        }
    }

    fn indices(&self) -> Self::Indices {
        DefaultIndices::new(self.clone(), self.start_index(), self.end_index())
    }
}

impl<T> MutableCollection for ArraySlice<T>
where
    T: Clone,
{
    type SubSequence = Self;

    fn partition_by<F>(&mut self, mut predicate: F) -> Self::Index
    where
        F: FnMut(&Self::Element) -> crate::errors::Result<bool>,
    {
        let mut i = 0;
        let mut j = self.len;

        while i != j {
            if matches!(predicate(&self[i]), Success(true)) {
                i += 1;
            } else {
                j -= 1;
                self.swap_at(i, j);
            }
        }

        i
    }

    fn swap_at(&mut self, index1: Self::Index, index2: Self::Index) {
        assert!(index1 < self.len && index2 < self.len);
        unsafe {
            let a = self.ptr.as_ptr().add(index1);
            let b = self.ptr.as_ptr().add(index2);
            core::ptr::swap(a, b);
        }
    }

    fn with_contiguous_mutable_storage_if_available<R, F>(&mut self, f: F) -> Option<R>
    where
        F: FnOnce(&mut [Self::Element]) -> R,
    {
        unsafe {
            Some(f(core::slice::from_raw_parts_mut(
                self.ptr.as_ptr(),
                self.len,
            )))
        }
    }
}

impl<T: Clone> Iterator for ArraySlice<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_empty() {
            return None;
        }

        let index = self.start_index();
        if index < self.end_index() {
            let item = self.index_after(index);
            self.index_offset_by(index, 1);
            item
        } else {
            None
        }
    }
}

impl<T> Drop for ArraySlice<T> {
    fn drop(&mut self) {
        unsafe {
            for i in 0..self.len {
                ptr::drop_in_place(self.ptr.as_ptr().add(i));
            }

            if self.capacity > 0 {
                let layout = Layout::array::<T>(self.capacity).expect("Invalid layout");
                dealloc(self.ptr.as_ptr().cast::<u8>(), layout);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use core::{ptr, range::Range};

    use alloc::vec::Vec;

    use crate::{
        collections::{ArraySlice, MutableCollection, array::Array},
        errors::Result::Success,
    };

    #[test]
    fn test_empty_array() {
        let array: Array<i32> = Array::with_uninitialized(0, |_, _| {});
        assert_eq!(array.capacity, 0);
        assert_eq!(array.length, 0);
    }

    #[test]
    fn test_default_array() {
        let arr: Array<i32> = Array::default();
        assert_eq!(arr.count(), 0);
        assert_eq!(arr.capacity(), 4);
        assert!(arr.is_empty());
    }

    #[test]
    fn test_array_from_initializer() {
        let values = [1, 2, 3, 4, 5];
        let array = Array::with_uninitialized(values.len(), |dst: *mut i32, len| unsafe {
            for (i, v) in values.iter().enumerate() {
                ptr::write(dst.add(i), *v);
                *len += 1;
            }
        });

        assert_eq!(array.length, values.len());
        for (i, value) in values.iter().enumerate() {
            assert_eq!(unsafe { *array.ptr.as_ptr().add(i) }, *value);
        }
    }

    #[test]
    fn test_array_clone() {
        let values = [10, 20, 30];
        let array = Array::with_uninitialized(values.len(), |dst: *mut i32, len| unsafe {
            for (i, v) in values.iter().enumerate() {
                ptr::write(dst.add(i), *v);
                *len += 1;
            }
        });

        let clone = array.clone();
        assert_eq!(array, clone);

        // Ensure data is copied, not shared
        assert_ne!(array.ptr, clone.ptr);
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
    fn test_array_equality() {
        let a = Array::with_uninitialized(3, |dst: *mut i32, len| unsafe {
            for (i, v) in [1, 2, 3].iter().enumerate() {
                dst.add(i).write(*v);
                *len += 1;
            }
        });

        let b = Array::with_uninitialized(3, |dst: *mut i32, len| unsafe {
            for (i, v) in [1, 2, 3].iter().enumerate() {
                dst.add(i).write(*v);
                *len += 1;
            }
        });

        assert_eq!(a, b);
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
        arr.replace_subrange(Range::from(1..4), [10, 11]);
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

    #[test]
    fn test_array_drop_safety() {
        use alloc::rc::Rc;
        use core::cell::RefCell;

        struct Tracker(Rc<RefCell<usize>>);

        impl Drop for Tracker {
            fn drop(&mut self) {
                *self.0.borrow_mut() += 1;
            }
        }

        let drop_count = Rc::new(RefCell::new(0));

        {
            let trackers: Vec<Tracker> = (0..5).map(|_| Tracker(drop_count.clone())).collect();
            let _array =
                Array::with_uninitialized(trackers.len(), |dst: *mut Tracker, len| unsafe {
                    for (i, tracker) in trackers.into_iter().enumerate() {
                        dst.add(i).write(tracker);
                        *len += 1;
                    }
                });
        }

        assert_eq!(*drop_count.borrow(), 5);
    }

    #[test]
    fn test_array_swap_at() {
        let mut array = array![1, 2, 3];
        array.swap_at(0, 2);
        assert_eq!(array, array![3, 2, 1]);
    }

    #[test]
    fn test_partition_by() {
        let mut array = array![1, 2, 3, 4, 5, 6];
        let result = array.partition_by(|x| Success(*x % 2 == 0)); // Partition evens first
        for i in 0..result {
            assert!(array[i] % 2 == 0);
        }
        for i in result..array.count() {
            assert!(array[i] % 2 != 0);
        }
    }

    #[test]
    fn test_with_contiguous_mutable_storage_if_available() {
        let mut array = array![10, 20, 30];
        let result = array.with_contiguous_mutable_storage_if_available(|slice| {
            slice[0] += 1;
            slice[2] += 2;
            Array::from_iter(slice.to_vec())
        });
        assert_eq!(result, Some(array![11, 20, 32]));
    }

    #[test]
    fn arrayslice_creates_new_slice_with_specified_capacity() {
        let slice = ArraySlice::<i32>::new(5);
        assert_eq!(slice.capacity(), 5);
    }

    #[test]
    fn arrayslice_inserts_element_at_start() {
        let mut slice = ArraySlice::new(3);
        slice.insert(10, 0);
        assert_eq!(slice[0], 10);
    }

    #[test]
    fn arrayslice_inserts_element_in_middle() {
        let mut slice = ArraySlice::new(3);
        slice.insert(1, 0);
        slice.insert(3, 1);
        slice.insert(2, 1);
        assert_eq!(slice[0], 1);
        assert_eq!(slice[1], 2);
        assert_eq!(slice[2], 3);
    }

    #[test]
    fn arrayslice_removes_element_and_shifts_remaining() {
        let mut slice = ArraySlice::new(3);
        slice.insert(100, 0);
        slice.insert(200, 1);
        slice.insert(300, 2);
        let removed = slice.remove(1);
        assert_eq!(removed, 200);
        assert_eq!(slice[0], 100);
        assert_eq!(slice[1], 300);
    }

    #[test]
    fn arrayslice_clones_correctly() {
        let mut slice = ArraySlice::new(2);
        slice.insert(5, 0);
        slice.insert(6, 1);
        let clone = slice.clone();
        assert_eq!(clone[0], 5);
        assert_eq!(clone[1], 6);
    }

    #[test]
    fn arrayslice_reserves_additional_capacity() {
        let mut slice = ArraySlice::new(2);
        slice.insert(1, 0);
        let old_capacity = slice.capacity();
        slice.reserve_capacity(3);
        let new_capacity = slice.capacity();
        assert!(new_capacity > old_capacity);
        assert_eq!(slice[0], 1);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn arrayslice_panics_on_out_of_bounds_insert() {
        let mut slice = ArraySlice::<i32>::new(1);
        slice.insert(99, 2);
    }
}
