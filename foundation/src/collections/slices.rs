use core::{
    ops::{Deref, Index},
    range::Range,
};

use crate::{
    Int,
    collections::{Collection, IndexingIterator, Sequence},
};

/// A view into a contiguous range of elements within a collection.
///
/// `Slice` holds a reference to a collection and a range specifying
/// the subset of elements this slice represents. It does not own the
/// elements but provides a window into the original collection.
#[derive(Debug, Clone)]
pub struct Slice<'a, C: Collection> {
    collection: &'a C,
    range: Range<C::Index>,
}

impl<C: Collection> Index<C::Index> for Slice<'_, C>
where
    <C as Index<C::Index>>::Output: Deref<Target = C::Element>,
{
    type Output = C::Element;

    fn index(&self, index: C::Index) -> &Self::Output {
        &self.collection[index]
    }
}

impl<C> Iterator for Slice<'_, C>
where
    C: Collection + Iterator + Clone,
{
    type Item = C::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.collection.clone().next()
    }
}

impl<T, C> Sequence for Slice<'_, C>
where
    C: Collection<Element = T>,
    Self: Collection + Clone,
{
    type Element = C::Element;

    type Iterator = IndexingIterator<Self>;

    fn iter(&self) -> Self::Iterator {
        IndexingIterator::new(self.clone())
    }

    fn underestimated_count(&self) -> usize {
        self.collection.count()
    }
}

impl<C: Collection + Deref> Collection for Slice<'_, C>
where
    Self: Index<<C as Collection>::Index> + Clone,
{
    type Index = C::Index;

    type Indices = C::Indices;

    type SubSequence = Self;

    fn start_index(&self) -> Self::Index {
        self.range.start
    }

    fn end_index(&self) -> Self::Index {
        self.range.end
    }

    fn index_after(&self, i: Self::Index) -> Option<Self::Element> {
        self.collection.index_after(i)
    }

    fn count(&self) -> usize {
        self.collection.count()
    }

    fn is_empty(&self) -> bool {
        self.collection.is_empty()
    }

    fn index_offset_by(&self, index: Self::Index, offset_by: Int) -> Self::Index {
        self.collection.index_offset_by(index, offset_by)
    }

    fn index_offset_by_limited_by(
        &self,
        index: Self::Index,
        offset_by: Int,
        limited_by: Self::Index,
    ) -> Option<Self::Index> {
        self.collection
            .index_offset_by_limited_by(index, offset_by, limited_by)
    }
}
