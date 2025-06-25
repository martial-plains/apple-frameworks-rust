use core::ops::Index;

use crate::collections::{Collection, IndexingIterator, Sequence};

/// Represents a range of indices within a collection.
///
/// `DefaultIndices` encapsulates a start and end index defining a half-open
/// range (`[start, end)`) over a given collection. It owns the collection and
/// provides utility methods to work with the index range.
///
/// # Type parameters
///
/// - `C`: The type of the underlying collection, which must implement
///   the `Collection` trait.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultIndices<C: Collection> {
    collection: C,
    start: C::Index,
    end: C::Index,
}

impl<C: Collection> DefaultIndices<C> {
    /// Creates a new `DefaultIndices` with the given collection and range bounds.
    ///
    /// # Parameters
    ///
    /// - `collection`: The collection to own.
    /// - `start`: The start index (inclusive) of the range.
    /// - `end`: The end index (exclusive) of the range.
    ///
    /// # Returns
    ///
    /// A new instance of `DefaultIndices`.
    pub const fn new(collection: C, start: C::Index, end: C::Index) -> Self {
        Self {
            collection,
            start,
            end,
        }
    }

    /// Checks whether the specified index lies within the range `[start, end)`.
    ///
    /// # Parameters
    ///
    /// - `index`: The index to check.
    ///
    /// # Returns
    ///
    /// `true` if `index` is within the range, `false` otherwise.
    pub fn contains(&self, index: C::Index) -> bool {
        index >= self.start && index < self.end
    }

    /// Returns the range `[start, end)` represented by this `DefaultIndices`.
    ///
    /// # Returns
    ///
    /// A `core::ops::Range` from `start` (inclusive) to `end` (exclusive).
    pub const fn range(&self) -> core::ops::Range<C::Index> {
        self.start..self.end
    }
}

impl<C> Sequence for DefaultIndices<C>
where
    C: Collection + Index<C::Index, Output = C::Element> + Clone,
{
    type Element = C::Index;

    type Iterator = IndexingIterator<Self>;

    fn iter(&self) -> Self::Iterator {
        IndexingIterator::new(self.clone())
    }

    fn underestimated_count(&self) -> usize {
        self.collection.count()
    }
}

impl<C> Index<C::Index> for DefaultIndices<C>
where
    C: Collection + Index<C::Index, Output = C::Element>,
{
    type Output = C::Element;

    fn index(&self, index: C::Index) -> &Self::Output {
        &self.collection[index]
    }
}

impl<C: Collection + Index<C::Index, Output = C::Element> + Clone> Collection
    for DefaultIndices<C>
{
    type Index = C::Index;

    type Indices = Self;

    type SubSequence = Self;

    fn start_index(&self) -> Self::Index {
        self.start
    }

    fn end_index(&self) -> Self::Index {
        self.end
    }

    fn index_after(&self, i: Self::Index) -> Option<Self::Element> {
        let next = self.collection.index_offset_by(i, 1);
        if self.contains(next) {
            Some(next)
        } else {
            None
        }
    }

    fn count(&self) -> usize {
        self.collection.count()
    }

    fn is_empty(&self) -> bool {
        self.collection.is_empty()
    }

    fn index_offset_by(&self, index: Self::Index, offset_by: usize) -> Self::Index {
        self.collection.index_offset_by(index, offset_by)
    }

    fn index_offset_by_limited_by(
        &self,
        index: Self::Index,
        offset_by: usize,
        limited_by: Self::Index,
    ) -> Option<Self::Index> {
        self.collection
            .index_offset_by_limited_by(index, offset_by, limited_by)
    }
}
