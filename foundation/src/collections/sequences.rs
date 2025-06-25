use crate::{
    collections::{Sequence, traits::Collection},
    errors::Result::{self},
};

/// A struct representing the `PrefixSequence`, which limits the number of elements from the base iterator.
#[derive(Debug, Clone, Copy)]
pub struct PrefixSequence<Base> {
    base: Base,
    limit: usize,
    count: usize,
}

impl<Base> PrefixSequence<Base>
where
    Base: Iterator,
{
    /// Creates a new `PrefixSequence` wrapping the given base iterator,
    /// yielding at most `n` elements.
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

/// A struct representing a sequence that enumerates elements from a base sequence,
/// producing `(index, element)` pairs.
#[derive(Debug, Clone, Copy)]
pub struct EnumeratedSequence<Base>
where
    Base: Sequence,
{
    iter: Base::Iterator,
    index: usize,
}

impl<Base> EnumeratedSequence<Base>
where
    Base: Sequence,
{
    /// Creates a new `EnumeratedSequence` from a base sequence,
    /// which yields pairs of `(index, element)` from the underlying sequence.
    pub fn new(base: &Base) -> Self {
        Self {
            iter: base.iter(),
            index: 0,
        }
    }
}

impl<Base> Iterator for EnumeratedSequence<Base>
where
    Base: Sequence,
{
    type Item = (usize, Base::Element);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|item| {
            let idx = self.index;
            self.index += 1;
            (idx, item)
        })
    }
}

/// A struct representing a sequence that drops the first `n` elements from the base iterator.
#[derive(Debug, Clone, Copy)]
pub struct DropFirstSequence<Base> {
    base: Base,
    dropping: usize,
}

impl<Base> DropFirstSequence<Base>
where
    Base: Iterator,
{
    /// Creates a new `DropFirstSequence` that drops the first `dropping` elements from the base iterator.
    pub const fn new(base: Base, dropping: usize) -> Self {
        Self { base, dropping }
    }

    /// Returns a new `DropFirstSequence` that will drop the first `n` elements.
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

/// A struct representing a sequence that drops elements from the base iterator
/// while a predicate holds true, then yields the rest.
#[derive(Debug)]
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
    /// Creates a new `DropWhileSequence` that skips elements from the base iterator
    /// while the predicate returns `true`.
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

/// A struct representing an iterator over a collection, which uses
/// the collection's indexing to iterate elements.
#[derive(Debug, Clone, Copy)]
pub struct IndexingIterator<C>
where
    C: Collection,
{
    base: C,
    current_index: C::Index,
}

impl<C> IndexingIterator<C>
where
    C: Collection,
{
    /// Creates a new `IndexingIterator` starting from the collection's start index.
    pub fn new(base: C) -> Self where {
        let current_index = base.start_index();
        Self {
            base,
            current_index,
        }
    }
}

impl<C> Iterator for IndexingIterator<C>
where
    C: Collection,
{
    type Item = C::Element;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.base.end_index() {
            return None;
        }

        let result = self.base.index_after(self.current_index);
        self.current_index = self.base.index_offset_by(self.current_index, 1);
        result
    }
}

impl<C: Collection<Element = C>> Sequence for IndexingIterator<C> {
    type Element = C::Element;

    type Iterator = Self;

    fn iter(&self) -> Self::Iterator {
        Self::new(self.base.clone())
    }

    fn underestimated_count(&self) -> usize {
        self.base.count()
    }
}
