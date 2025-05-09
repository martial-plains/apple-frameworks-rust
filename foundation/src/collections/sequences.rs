use core::cmp::{Ordering, min};

use alloc::vec::Vec;

use crate::errors::Result::{self, Failure, Success};

use super::array::Array;

pub trait Sequence {
    /// A type representing the sequence’s elements.
    type Item;

    type Iterator: Iterator<Item = Self::Item>;

    fn iter(&self) -> Self::Iterator;

    fn contains(&self, item: &Self::Item) -> bool
    where
        Self::Item: PartialEq,
    {
        self.iter().any(|x| x == *item)
    }

    fn contains_where<F>(&self, mut predicate: F) -> Result<bool>
    where
        F: FnMut(Self::Item) -> Result<bool>,
    {
        for element in self.iter() {
            if matches!(predicate(element), Success(true)) {
                return Success(true);
            }
        }
        Success(false)
    }

    fn all_satisfy<F>(&self, mut predicate: F) -> Result<bool>
    where
        F: FnMut(Self::Item) -> Result<bool>,
    {
        for element in self.iter() {
            if matches!(predicate(element), Success(false)) {
                return Success(false);
            }
        }
        Success(true)
    }

    fn first_where<F>(&self, mut predicate: F) -> Result<Option<Self::Item>>
    where
        F: FnMut(&Self::Item) -> Result<bool>,
    {
        for element in self.iter() {
            if matches!(predicate(&element), Success(true)) {
                return Success(Some(element));
            }
        }
        Success(None)
    }

    fn min(&self) -> Option<Self::Item>
    where
        Self::Item: Ord,
    {
        self.iter().min()
    }

    fn min_by<F>(&self, compare: F) -> Result<Option<Self::Item>>
    where
        F: Fn(&Self::Item, &Self::Item) -> Result<Ordering>,
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

    fn max(&self) -> Option<Self::Item>
    where
        Self::Item: Ord,
    {
        self.iter().max()
    }

    fn max_by<F>(&self, compare: F) -> Result<Option<Self::Item>>
    where
        F: Fn(&Self::Item, &Self::Item) -> Result<Ordering>,
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

    fn prefix(self, max_len: usize) -> PrefixSequence<impl Iterator<Item = Self::Item>>
    where
        Self: core::marker::Sized,
    {
        // Create a PrefixSequence using the iterator from `self` and the given `max_len`
        PrefixSequence::new(self.iter(), max_len)
    }

    fn prefix_while<F>(&self, mut predicate: F) -> Result<Array<Self::Item>>
    where
        F: FnMut(Self::Item) -> Result<bool>,
        Self::Item: Copy,
    {
        let mut result = Array::default();

        for element in self.iter() {
            // Apply the predicate and handle potential errors
            match predicate(element) {
                Success(true) => result.append(element),
                Success(false) => break, // Stop once the predicate fails
                Failure(e) => return Failure(e), // Return the error if predicate throws
            }
        }

        Success(result)
    }

    fn suffix(&self, n: usize) -> Array<Self::Item>
    where
        Self::Item: Copy,
    {
        let start_index = if self.underestimated_count() <= n {
            0
        } else {
            self.underestimated_count() - n
        };

        self.iter().skip(start_index).collect::<Array<_>>()
    }

    #[must_use]
    fn drop_first(self, n: usize) -> DropFirstSequence<impl Iterator<Item = Self::Item>>
    where
        Self: core::marker::Sized,
    {
        DropFirstSequence::new(self.iter(), n)
    }

    #[must_use]
    fn drop_last(&self, k: usize) -> Array<Self::Item>
    where
        Self::Item: Clone,
    {
        let collected: Vec<Self::Item> = self.iter().collect();
        let dropped = collected[..collected.len().saturating_sub(k)].to_vec();

        Array::from_iter(dropped)
    }

    fn filter<F>(&self, mut predicate: F) -> Array<Self::Item>
    where
        F: FnMut(&Self::Item) -> bool,
        Self::Item: Clone,
    {
        self.iter().filter(|x| predicate(x)).collect()
    }

    // The map function that applies the closure to each element of the sequence
    fn map<T, E, F>(&self, mut f: F) -> Result<Array<T>, E>
    where
        F: FnMut(Self::Item) -> Result<T, E>,
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

    fn compact_map<B, F>(&self, f: F) -> Array<B>
    where
        F: FnMut(Self::Item) -> Option<B>,
    {
        self.iter().filter_map(f).collect()
    }

    fn flat_map<B, I, F>(&self, f: F) -> Array<B>
    where
        F: FnMut(Self::Item) -> I,
        I: IntoIterator<Item = B>,
    {
        self.iter().flat_map(f).collect()
    }

    fn reduce<B, F>(&self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        self.iter().fold(init, f)
    }

    fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(Self::Item),
    {
        for x in self.iter() {
            f(x);
        }
    }

    fn enumerated(self) -> EnumeratedSequence<Self>
    where
        Self: core::marker::Sized + core::iter::Iterator,
    {
        EnumeratedSequence::new(self)
    }

    fn underestimated_count(&self) -> usize;

    fn reversed(&self) -> Array<Self::Item>
    where
        Self::Item: Clone,
    {
        let mut reversed_array = Array::default();
        let len = self.underestimated_count();

        for (i, item) in self.iter().enumerate() {
            reversed_array[len - 1 - i] = item.clone();
        }

        reversed_array
    }

    fn sorted(&self) -> Array<Self::Item>
    where
        Self::Item: Ord + Copy + Default,
    {
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

    fn sorted_by<F>(&self, mut cmp: F) -> Array<Self::Item>
    where
        Self::Item: Copy + Default,
        F: FnMut(&Self::Item, &Self::Item) -> core::cmp::Ordering,
    {
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

    fn count_where<F>(&self, mut predicate: F) -> Result<usize>
    where
        F: FnMut(Self::Item) -> Result<bool>,
    {
        let mut count = 0;

        for element in self.iter() {
            // Apply the predicate and handle the result
            match predicate(element) {
                Success(true) => count += 1, // Increment count if predicate is satisfied
                Success(false) => {}         // Do nothing if predicate is not satisfied
                Failure(e) => return Failure(e), // Propagate the error
            }
        }

        Success(count) // Return the total count
    }
}

// A struct representing the PrefixSequence, which limits the number of elements from the base iterator.
pub struct PrefixSequence<Base> {
    base: Base,
    limit: usize,
    count: usize,
}

impl<Base> PrefixSequence<Base>
where
    Base: Iterator,
{
    // Takes a base iterator and a limit `n`.
    pub const fn new(base: Base, n: usize) -> Self {
        Self {
            base,
            limit: n,
            count: 0,
        }
    }
}

impl<Base> Iterator for PrefixSequence<Base>
where
    Base: Iterator,
{
    type Item = Base::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count >= self.limit {
            return None;
        }

        self.count += 1;
        self.base.next()
    }
}

pub struct EnumeratedSequence<Base>
where
    Base: IntoIterator,
{
    base_iter: Base::IntoIter,
    index: usize,
}

impl<Base> EnumeratedSequence<Base>
where
    Base: IntoIterator,
{
    pub fn new(base: Base) -> Self {
        Self {
            base_iter: base.into_iter(),
            index: 0,
        }
    }
}

impl<Base> Iterator for EnumeratedSequence<Base>
where
    Base: IntoIterator,
{
    type Item = (usize, Base::Item);

    fn next(&mut self) -> Option<Self::Item> {
        self.base_iter.next().map(|item| {
            let result = (self.index, item);
            self.index += 1;
            result
        })
    }
}

pub struct DropFirstSequence<Base> {
    base: Base,
    dropping: usize,
}

impl<Base> DropFirstSequence<Base>
where
    Base: Iterator,
{
    pub const fn new(base: Base, dropping: usize) -> Self {
        Self { base, dropping }
    }

    #[must_use]
    pub fn drop_first(self, n: usize) -> Self {
        Self::new(self.base, n)
    }
}

impl<Base> Iterator for DropFirstSequence<Base>
where
    Base: Iterator,
{
    type Item = Base::Item;

    fn next(&mut self) -> Option<Self::Item> {
        for _ in 0..self.dropping {
            self.base.next();
        }

        self.base.next()
    }
}

pub struct DropWhileSequence<Base, F> {
    base: Base,
    predicate: F,
    dropped: bool,
}

impl<Base, F> DropWhileSequence<Base, F>
where
    Base: Iterator,
    F: FnMut(&Base::Item) -> Result<bool>,
{
    // Constructor for creating a DropWhileSequence
    pub const fn new(base: Base, predicate: F) -> Self {
        Self {
            base,
            predicate,
            dropped: false,
        }
    }
}

impl<Base, F> Iterator for DropWhileSequence<Base, F>
where
    Base: Iterator,
    F: FnMut(&Base::Item) -> bool,
{
    type Item = Base::Item;

    fn next(&mut self) -> Option<Self::Item> {
        for item in self.base.by_ref() {
            if (self.predicate)(&item) {
                continue;
            }
            self.dropped = true;
            return Some(item);
        }

        if self.dropped {
            return self.base.next();
        }

        None
    }
}
