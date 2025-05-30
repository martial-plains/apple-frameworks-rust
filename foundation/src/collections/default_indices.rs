use core::ops::Index;

use crate::collections::{Collection, IndexingIterator, Sequence};

#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultIndices<C: Collection> {
    collection: C,
    start: C::Index,
    end: C::Index,
}

impl<C: Collection> DefaultIndices<C> {
    pub const fn new(collection: C, start: C::Index, end: C::Index) -> Self {
        Self {
            collection,
            start,
            end,
        }
    }

    pub fn contains(&self, index: C::Index) -> bool {
        index >= self.start && index < self.end
    }

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
