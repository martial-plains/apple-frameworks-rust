use core::{
    clone::Clone,
    cmp::{Ordering, PartialEq, min},
    marker::Sized,
    ops::Index,
};

use alloc::vec::Vec;

use crate::{
    collections::{
        DefaultIndices, IndexingIterator, Slice,
        array::Array,
        sequences::{DropFirstSequence, EnumeratedSequence, PrefixSequence},
    },
    errors::Result::{self, Failure, Success},
};

pub trait Collection:
    Sequence<Iterator = IndexingIterator<Self>> + Index<Self::Index> + Sized + Clone
{
    type Index: PartialOrd + Copy;

    type Indices: Collection;

    type SubSequence: Collection;

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

    fn remove_first(&self) -> Self::Element
    where
        Self::Element: Clone,
        Self: Index<Self::Index, Output = Self::Element>,
    {
        self[self.start_index()].clone()
    }

    fn start_index(&self) -> Self::Index;

    fn end_index(&self) -> Self::Index;

    fn index_after(&self, i: Self::Index) -> Option<Self::Element>;

    fn form_index_offset_by(&self, index: &mut Self::Index, offset_by: usize)
    where
        Self::Index: Copy,
    {
        *index = self.index_offset_by(*index, offset_by);
    }

    fn form_index_offset_by_limited_by(
        &self,
        index: &mut Self::Index,
        offset_by: usize,
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

    fn count(&self) -> usize;

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

    fn is_empty(&self) -> bool;

    fn index_offset_by(&self, index: Self::Index, offset_by: usize) -> Self::Index;

    fn index_offset_by_limited_by(
        &self,
        index: Self::Index,
        offset_by: usize,
        limited_by: Self::Index,
    ) -> Option<Self::Index>;

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

pub trait Sequence {
    /// A type representing the sequence’s elements.
    type Element;

    type Iterator: Iterator<Item = Self::Element>;

    fn iter(&self) -> Self::Iterator;

    fn contains(&self, item: &Self::Element) -> bool
    where
        Self::Element: PartialEq,
    {
        self.iter().any(|x| x == *item)
    }

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

    fn min(&self) -> Option<Self::Element>
    where
        Self::Element: Ord,
    {
        self.iter().min()
    }

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

    fn max(&self) -> Option<Self::Element>
    where
        Self::Element: Ord,
    {
        self.iter().max()
    }

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

    fn prefix(self, max_len: usize) -> PrefixSequence<impl Iterator<Item = Self::Element>>
    where
        Self: Sized,
    {
        PrefixSequence::new(self.iter(), max_len)
    }

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

    #[must_use]
    fn drop_first(self, n: usize) -> DropFirstSequence<impl Iterator<Item = Self::Element>>
    where
        Self: Sized,
    {
        DropFirstSequence::new(self.iter(), n)
    }

    #[must_use]
    fn drop_last(&self, k: usize) -> Array<Self::Element>
    where
        Self::Element: Clone,
    {
        let collected: Vec<Self::Element> = self.iter().collect();
        let dropped = collected[..collected.len().saturating_sub(k)].to_vec();

        Array::from_iter(dropped)
    }

    fn filter<F>(&self, mut predicate: F) -> Array<Self::Element>
    where
        F: FnMut(&Self::Element) -> bool,
        Self::Element: Clone,
    {
        self.iter().filter(|x| predicate(x)).collect()
    }

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

    fn compact_map<B, F>(&self, f: F) -> Array<B>
    where
        F: FnMut(Self::Element) -> Option<B>,
    {
        self.iter().filter_map(f).collect()
    }

    fn flat_map<B, I, F>(&self, f: F) -> Array<B>
    where
        F: FnMut(Self::Element) -> I,
        I: IntoIterator<Item = B>,
    {
        self.iter().flat_map(f).collect()
    }

    fn reduce<B, F>(&self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Element) -> B,
    {
        self.iter().fold(init, f)
    }

    fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(Self::Element),
    {
        for x in self.iter() {
            f(x);
        }
    }

    fn enumerated(self) -> EnumeratedSequence<Self>
    where
        Self: Sized,
    {
        EnumeratedSequence::new(self)
    }

    fn underestimated_count(&self) -> usize;

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
