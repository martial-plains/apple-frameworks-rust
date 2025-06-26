use core::{
    clone::Clone,
    cmp::{Ordering, PartialEq, min},
    marker::Sized,
    ops::Index,
};

use alloc::vec::Vec;

use crate::{
    Int,
    collections::{
        IndexingIterator,
        array::Array,
        sequences::{DropFirstSequence, EnumeratedSequence, PrefixSequence},
    },
    errors::Result::{self, Failure, Success},
};

/// A trait representing a collection of elements with indexed access and sequence behavior.
///
/// This trait extends the functionality of a `Sequence` by adding indexed access
/// (via the `Index` trait), cloning ability, and index manipulation methods.
/// It is designed to work with collections that have a notion of indices which
/// can be ordered and copied.
pub trait Collection: Sequence<Iterator = IndexingIterator<Self>> + Index<Self::Index>
where
    Self: Sized,
{
    /// The type used to index elements in the collection.
    type Index: PartialOrd + Copy;

    /// The type representing a collection of indices.
    type Indices: Collection;

    /// The type representing a contiguous subsequence of this collection.
    type SubSequence: Collection;

    /// Returns the first element in the collection, if any, without removing it.
    ///
    /// Returns `None` if the collection is empty.
    ///
    /// # Default Implementation
    ///
    /// Returns the element at `start_index()` if the collection is not empty.
    fn pop_first(&self) -> Option<Self::Element>
    where
        Self: Index<Self::Index, Output = Self::Element>,
        Self::Element: Clone,
    {
        if self.is_empty() {
            None
        } else {
            self.first().cloned()
        }
    }

    /// Returns the first element in the collection by cloning it.
    ///
    /// # Panics
    ///
    /// Panics if the collection is empty.
    fn remove_first(&self) -> Self::Element
    where
        Self::Element: Clone,
        Self: Index<Self::Index, Output = Self::Element>,
    {
        self[self.start_index()].clone()
    }

    /// Returns the starting index of the collection.
    fn start_index(&self) -> Self::Index;

    /// Returns the ending index of the collection (one past the last valid index).
    fn end_index(&self) -> Self::Index;

    /// Returns the element immediately after the specified index, if any.
    fn index_after(&self, i: Self::Index) -> Option<Self::Element>;

    /// Modifies the provided index by offsetting it by a given amount.
    ///
    /// This method updates `index` in-place by advancing it `offset_by` steps.
    fn form_index_offset_by(&self, index: &mut Self::Index, offset_by: Int)
    where
        Self::Index: Copy,
    {
        *index = self.index_offset_by(*index, offset_by);
    }

    /// Modifies the provided index by offsetting it by a given amount, bounded by `limited_by`.
    ///
    /// Returns `true` if the index was updated and remains less than `limited_by`.
    /// Returns `false` if the index was already `>= limited_by` and no update occurred.
    fn form_index_offset_by_limited_by(
        &self,
        index: &mut Self::Index,
        offset_by: isize,
        limited_by: Self::Index,
    ) -> bool
    where
        Self::Index: Copy,
    {
        if *index < limited_by {
            *index = self
                .index_offset_by_limited_by(*index, offset_by, limited_by)
                .unwrap_or(limited_by);
            true
        } else {
            false
        }
    }

    /// Returns the number of elements in the collection.
    fn count(&self) -> usize;

    /// Returns a reference to the first element, if any.
    ///
    /// Returns `None` if the collection is empty.
    fn first(&self) -> Option<&Self::Element>
    where
        Self::Index: Copy + PartialOrd,
        Self: Index<Self::Index, Output = Self::Element>,
    {
        if self.is_empty() {
            None
        } else {
            Some(&self[self.start_index()])
        }
    }

    /// Returns `true` if the collection contains no elements.
    fn is_empty(&self) -> bool;

    /// Returns the index offset by a specified number of steps from the given index.
    ///
    /// The returned index may be beyond `end_index()`.
    fn index_offset_by(&self, index: Self::Index, offset_by: Int) -> Self::Index;

    /// Returns the index offset by a specified number of steps from the given index,
    /// limited by a `limited_by` index.
    ///
    /// Returns `None` if the offset would exceed `limited_by`.
    fn index_offset_by_limited_by(
        &self,
        index: Self::Index,
        offset_by: isize,
        limited_by: Self::Index,
    ) -> Option<Self::Index>;

    /// Returns the index of the first occurrence of the specified value, if any.
    ///
    /// Returns `None` if the value is not found.
    fn index_of(&self, value: Self::Element) -> Option<Self::Index>
    where
        Self::Element: PartialEq,
    {
        let mut index = self.start_index();
        for element in self.iter() {
            if element == value {
                return Some(index);
            }
            index = self.index_offset_by(index, 1);
        }
        None
    }
}

/// A trait representing a mutable collection that extends the `Collection` trait.
///
/// This trait adds mutation capabilities to a collection, such as partitioning,
/// swapping elements, and accessing underlying mutable storage when available.
pub trait MutableCollection:
    Collection + Index<Self::Index> + Index<core::ops::Range<Self::Index>>
{
    /// The type of subsequence that this mutable collection can produce.
    ///
    /// This allows operations that return a portion of the collection
    /// while preserving the ability to mutate it.
    type SubSequence: MutableCollection;

    /// Partitions the collection in-place according to the given predicate.
    ///
    /// The elements for which the predicate returns `Ok(true)` will be moved
    /// to the front, and the rest to the back. The function returns the index
    /// that separates the two partitions.
    ///
    /// # Errors
    /// Returns an error if the predicate returns an error for any element.
    ///
    /// # Parameters
    /// - `predicate`: A function that takes a reference to an element and
    ///   returns a `Result<bool>`, used to determine partitioning.
    fn partition_by<F>(&mut self, predicate: F) -> Self::Index
    where
        F: FnMut(&Self::Element) -> Result<bool>;

    /// Swaps the elements at the specified indices.
    ///
    /// # Parameters
    /// - `index1`: The index of the first element to swap.
    /// - `index2`: The index of the second element to swap.
    fn swap_at(&mut self, index1: Self::Index, index2: Self::Index);

    /// Provides temporary mutable access to the underlying contiguous storage
    /// if it is available.
    ///
    /// This method allows direct access to the internal slice of the collection
    /// for optimized bulk operations when the storage is contiguous.
    ///
    /// # Parameters
    /// - `data`: A mutable reference to a vector containing the collection data.
    /// - `f`: A closure that takes a mutable slice of the data and returns a result.
    ///
    /// # Returns
    /// - `Some(result)` if contiguous storage is available and the function was called.
    /// - `None` if contiguous storage is not available.
    fn with_contiguous_mutable_storage_if_available<R, F>(&mut self, f: F) -> Option<R>
    where
        F: FnOnce(&mut [Self::Element]) -> R;
}

/// A trait representing a sequence of elements that can be iterated over.
///
/// Provides functionality for querying and operating on sequences using iterators,
/// including filtering, finding, comparison, and basic slicing.
pub trait Sequence {
    /// A type representing the sequence’s elements.
    type Element;

    /// The type of iterator used to iterate over the sequence.
    type Iterator: Iterator<Item = Self::Element>;

    /// Returns an iterator over the sequence’s elements.
    fn iter(&self) -> Self::Iterator;

    /// Returns `true` if the sequence contains the given item.
    fn contains(&self, item: &Self::Element) -> bool
    where
        Self::Element: PartialEq,
    {
        self.iter().any(|x| x == *item)
    }

    /// Returns `true` if the predicate returns `true` for any element in the sequence.
    ///
    /// Stops early on the first match or returns `Success(false)` if no match is found.
    fn contains_where<F>(&self, mut predicate: F) -> Result<bool>
    where
        F: FnMut(Self::Element) -> Result<bool>,
    {
        for element in self.iter() {
            if matches!(predicate(element), Success(true)) {
                return Success(true);
            }
        }
        Success(false)
    }

    /// Returns `true` if all elements in the sequence satisfy the given predicate.
    ///
    /// Stops early if the predicate returns `false` for any element.
    fn all_satisfy<F>(&self, mut predicate: F) -> Result<bool>
    where
        F: FnMut(Self::Element) -> Result<bool>,
    {
        for element in self.iter() {
            if matches!(predicate(element), Success(false)) {
                return Success(false);
            }
        }
        Success(true)
    }

    /// Returns the first element in the sequence that satisfies the given predicate.
    ///
    /// Returns `Success(None)` if no matching element is found.
    fn first_where<F>(&self, mut predicate: F) -> Result<Option<Self::Element>>
    where
        F: FnMut(&Self::Element) -> Result<bool>,
    {
        for element in self.iter() {
            if matches!(predicate(&element), Success(true)) {
                return Success(Some(element));
            }
        }
        Success(None)
    }

    /// Returns the minimum element in the sequence, according to the natural ordering.
    fn min(&self) -> Option<Self::Element>
    where
        Self::Element: Ord,
    {
        self.iter().min()
    }

    /// Returns the minimum element in the sequence, using the given comparator function.
    ///
    /// Returns `Success(None)` if the sequence is empty.
    fn min_by<F>(&self, compare: F) -> Result<Option<Self::Element>>
    where
        F: Fn(&Self::Element, &Self::Element) -> Result<Ordering>,
    {
        let mut min_element = None;
        for element in self.iter() {
            match min_element {
                None => min_element = Some(element),
                Some(ref min) => {
                    if matches!(compare(min, &element), Success(Ordering::Greater)) {
                        min_element = Some(element);
                    }
                }
            }
        }
        Success(min_element)
    }

    /// Returns the maximum element in the sequence, according to the natural ordering.
    fn max(&self) -> Option<Self::Element>
    where
        Self::Element: Ord,
    {
        self.iter().max()
    }

    /// Returns the maximum element in the sequence, using the given comparator function.
    ///
    /// Returns `Success(None)` if the sequence is empty.
    fn max_by<F>(&self, compare: F) -> Result<Option<Self::Element>>
    where
        F: Fn(&Self::Element, &Self::Element) -> Result<Ordering>,
    {
        let mut max_element = None;
        for element in self.iter() {
            match max_element {
                None => max_element = Some(element),
                Some(ref max) => {
                    if matches!(compare(max, &element), Success(Ordering::Less)) {
                        max_element = Some(element);
                    }
                }
            }
        }
        Success(max_element)
    }

    /// Returns a sequence containing at most `max_len` elements from the start of the sequence.
    fn prefix(self, max_len: usize) -> PrefixSequence<impl Iterator<Item = Self::Element>>
    where
        Self: Sized,
    {
        PrefixSequence::new(self.iter(), max_len)
    }

    /// Returns a sequence containing the leading elements that satisfy a predicate.
    ///
    /// Iteration stops at the first failure or error returned by the predicate.
    fn prefix_while<F>(&self, mut predicate: F) -> Result<Array<Self::Element>>
    where
        F: FnMut(Self::Element) -> Result<bool>,
        Self::Element: Copy,
    {
        let mut result = Array::default();

        for element in self.iter() {
            match predicate(element) {
                Success(true) => result.append(element),
                Success(false) => break,
                Failure(e) => return Failure(e),
            }
        }

        Success(result)
    }

    /// Returns the last `n` elements in the sequence.
    ///
    /// If `n` is greater than the number of elements, returns the entire sequence.
    fn suffix(&self, n: usize) -> Array<Self::Element>
    where
        Self::Element: Copy,
    {
        let start_index = if self.underestimated_count() <= n {
            0
        } else {
            self.underestimated_count() - n
        };

        self.iter().skip(start_index).collect::<Array<_>>()
    }

    /// Returns a new sequence by dropping the first `n` elements.
    ///
    /// # Note
    /// This does **not** modify the original sequence—it returns a new iterator-backed sequence.
    #[must_use]
    fn drop_first(self, n: usize) -> DropFirstSequence<impl Iterator<Item = Self::Element>>
    where
        Self: Sized,
    {
        DropFirstSequence::new(self.iter(), n)
    }

    /// Returns a new sequence with the last `k` elements removed.
    ///
    /// If `k` is greater than the sequence length, the result will be empty.
    #[must_use]
    fn drop_last(&self, k: usize) -> Array<Self::Element>
    where
        Self::Element: Clone,
    {
        let collected: Vec<Self::Element> = self.iter().collect();
        let dropped = collected[..collected.len().saturating_sub(k)].to_vec();

        Array::from_iter(dropped)
    }

    /// Returns a new sequence containing only elements for which the predicate returns `true`.
    fn filter<F>(&self, mut predicate: F) -> Array<Self::Element>
    where
        F: FnMut(&Self::Element) -> bool,
        Self::Element: Clone,
    {
        self.iter().filter(|x| predicate(x)).collect()
    }

    /// Applies a fallible transformation to each element, returning a result.
    ///
    /// If any transformation returns a failure, the whole operation fails.
    fn map<T, E, F>(&self, mut f: F) -> Result<Array<T>, E>
    where
        F: FnMut(Self::Element) -> Result<T, E>,
    {
        let mut result = Array::default();

        for element in self.iter() {
            match f(element) {
                Success(mapped) => result.append(mapped),
                Failure(e) => return Failure(e),
            }
        }

        Success(result)
    }

    /// Applies a transformation that returns `Option<B>` and collects non-`None` results.
    fn compact_map<B, F>(&self, f: F) -> Array<B>
    where
        F: FnMut(Self::Element) -> Option<B>,
    {
        self.iter().filter_map(f).collect()
    }

    /// Applies a transformation to each element that returns an iterator, and flattens the results.
    fn flat_map<B, I, F>(&self, f: F) -> Array<B>
    where
        F: FnMut(Self::Element) -> I,
        I: IntoIterator<Item = B>,
    {
        self.iter().flat_map(f).collect()
    }

    /// Reduces the sequence to a single value using an initial accumulator and a combining function.
    fn reduce<B, F>(&self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Element) -> B,
    {
        self.iter().fold(init, f)
    }

    /// Applies the given function to each element in the sequence for side effects.
    fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(Self::Element),
    {
        for x in self.iter() {
            f(x);
        }
    }

    /// Returns an enumerated sequence, pairing each element with its index.
    fn enumerated(self) -> EnumeratedSequence<Self>
    where
        Self: Sized,
    {
        EnumeratedSequence::new(&self)
    }

    /// A value less than or equal to the number of elements in the sequence, calculated non-destructively.
    fn underestimated_count(&self) -> usize;

    /// Returns a new sequence with the elements in reverse order.
    fn reversed(&self) -> Array<Self::Element>
    where
        Self::Element: Clone,
    {
        let mut reversed_array = Array::default();
        let len = self.underestimated_count();

        for (i, item) in self.iter().enumerate() {
            reversed_array[len - 1 - i] = item.clone();
        }

        reversed_array
    }

    /// Returns a sorted version of the sequence, using natural ordering.
    ///
    /// Internally uses a TimSort-like hybrid of insertion sort and merge sort.
    fn sorted(&self) -> Array<Self::Element>
    where
        Self::Element: Ord + Copy + Default,
    {
        #[allow(clippy::many_single_char_names)]
        fn merge<T: Default + Copy + PartialOrd>(arr: &mut Array<T>, l: usize, m: usize, r: usize) {
            let (mut x, mut y, mut i, mut j, mut k) = (0, 0, 0, 0, 0);
            let len1: usize = m - l + 1;
            let len2 = r - m;
            let mut left = Array::repeating(T::default(), len1);
            let mut right = Array::repeating(T::default(), len2);

            while x < len1 {
                left[x] = arr[l + x];
                x += 1;
            }

            while y < len2 {
                right[y] = arr[(m + 1) + y];
                y += 1;
            }

            while i < len1 && j < len2 {
                if left[i] <= right[j] {
                    arr[l + k] = left[i];
                    i += 1;
                } else {
                    arr[l + k] = right[j];
                    j += 1;
                }

                k += 1;
            }

            while i < len1 {
                arr[l + k] = left[i];
                k += 1;
                i += 1;
            }

            while j < len2 {
                arr[l + k] = right[j];
                k += 1;
                j += 1;
            }
        }

        fn insertion_sort<T: Copy + PartialOrd>(arr: &mut Array<T>, left: usize, right: usize) {
            for i in (left + 1)..=right {
                let tmp = arr[i];
                let mut j = i;

                while j > left && arr[j - 1] > tmp {
                    arr[j] = arr[j - 1];
                    j -= 1;
                }

                arr[j] = tmp;
            }
        }

        let n = self.underestimated_count();
        let mut arr = self.iter().collect::<Array<_>>();
        let run = 32;

        for i in (0..n).step_by(run) {
            let right = min(i + run - 1, n - 1);
            insertion_sort(&mut arr, i, right);
        }

        let mut size = run;
        while size < n {
            let mut left = 0;
            while left < n {
                let mid = left + size - 1;
                let right = min(left + 2 * size - 1, n - 1);

                if mid < right {
                    merge(&mut arr, left, mid, right);
                }

                left += 2 * size;
            }
            size *= 2;
        }

        arr
    }

    /// Returns a sorted version of the sequence, using a custom comparator.
    ///
    /// Internally uses a TimSort-like hybrid sort algorithm.
    fn sorted_by<F>(&self, mut cmp: F) -> Array<Self::Element>
    where
        Self::Element: Copy + Default,
        F: FnMut(&Self::Element, &Self::Element) -> core::cmp::Ordering,
    {
        #[allow(clippy::many_single_char_names)]
        fn merge<T, F>(arr: &mut Array<T>, l: usize, m: usize, r: usize, cmp: &mut F)
        where
            T: Copy + Default,
            F: FnMut(&T, &T) -> core::cmp::Ordering,
        {
            let len1 = m - l + 1;
            let len2 = r - m;
            let mut left = Array::repeating(T::default(), len1);
            let mut right = Array::repeating(T::default(), len2);

            for i in 0..len1 {
                left[i] = arr[l + i];
            }

            for j in 0..len2 {
                right[j] = arr[m + 1 + j];
            }

            let (mut i, mut j, mut k) = (0, 0, l);
            while i < len1 && j < len2 {
                if cmp(&left[i], &right[j]) == core::cmp::Ordering::Greater {
                    arr[k] = right[j];
                    j += 1;
                } else {
                    arr[k] = left[i];
                    i += 1;
                }
                k += 1;
            }

            while i < len1 {
                arr[k] = left[i];
                i += 1;
                k += 1;
            }

            while j < len2 {
                arr[k] = right[j];
                j += 1;
                k += 1;
            }
        }

        fn insertion_sort<T, F>(arr: &mut Array<T>, left: usize, right: usize, cmp: &mut F)
        where
            T: Copy,
            F: FnMut(&T, &T) -> core::cmp::Ordering,
        {
            for i in (left + 1)..=right {
                let tmp = arr[i];
                let mut j = i;

                while j > left && cmp(&arr[j - 1], &tmp) == core::cmp::Ordering::Greater {
                    arr[j] = arr[j - 1];
                    j -= 1;
                }

                arr[j] = tmp;
            }
        }

        let n = self.underestimated_count();
        let mut arr = self.iter().collect::<Array<_>>();
        let run = 32;

        for i in (0..n).step_by(run) {
            let right = core::cmp::min(i + run - 1, n - 1);
            insertion_sort(&mut arr, i, right, &mut cmp);
        }

        let mut size = run;
        while size < n {
            let mut left = 0;
            while left < n {
                let mid = left + size - 1;
                let right = core::cmp::min(left + 2 * size - 1, n - 1);

                if mid < right {
                    merge(&mut arr, left, mid, right, &mut cmp);
                }

                left += 2 * size;
            }
            size *= 2;
        }

        arr
    }

    /// Counts the number of elements for which the predicate returns `true`.
    ///
    /// If any predicate call fails, returns the error.
    fn count_where<F>(&self, mut predicate: F) -> Result<usize>
    where
        F: FnMut(Self::Element) -> Result<bool>,
    {
        let mut count = 0;

        for element in self.iter() {
            match predicate(element) {
                Success(true) => count += 1,
                Success(false) => {}
                Failure(e) => return Failure(e),
            }
        }

        Success(count)
    }
}
