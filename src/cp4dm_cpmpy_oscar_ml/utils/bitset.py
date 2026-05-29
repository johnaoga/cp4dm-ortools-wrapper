"""
Reversible sparse bitset implementation.

Port of Oscar's ReversibleSparseBitSet2. Uses 64-bit word arrays with a sparse
index of non-zero words for efficient intersection and counting.

This implementation uses Python integers and arrays. For correctness first,
performance optimizations (numpy, ctypes) can follow.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def _bit_length(size: int) -> int:
    """Number of 64-bit words needed to represent `size` bits."""
    return (size + 63) >> 6


def _bit_offset(pos: int) -> int:
    """Word index for bit position."""
    return pos >> 6


def _bit_pos(pos: int) -> int:
    """Bit position within a word."""
    return pos & 63


def _popcount(x: int) -> int:
    """Count bits set in a 64-bit integer."""
    return bin(x & 0xFFFFFFFFFFFFFFFF).count("1")


class ImmutableBitSet:
    """
    An immutable bitset column (e.g., the coverage of one item).

    Inner class of ReversibleSparseBitset, analogous to Oscar's coverage.BitSet.
    """

    __slots__ = ("words", "n_words", "last_support")

    def __init__(self, n_words: int, values: Iterable[int]) -> None:
        self.n_words = n_words
        self.words: list[int] = [0] * n_words
        self.last_support = 0
        for v in values:
            offset = _bit_offset(v)
            self.words[offset] |= 1 << _bit_pos(v)
        # Mask to 64-bit
        self.words = [w & 0xFFFFFFFFFFFFFFFF for w in self.words]


class ReversibleSparseBitset:
    """
    A mutable sparse bitset with trail-based reversibility.

    Equivalent to Oscar's ReversibleSparseBitSet2.
    Maintains a sparse index of non-zero words for fast operations.

    Trail integration: call save() before modification at a new search level,
    and restore(snapshot) on backtrack.
    """

    def __init__(self, n: int, initial_values: Iterable[int]) -> None:
        """
        Args:
            n: Universe size (values in 0..n-1).
            initial_values: Initially present values.
        """
        self.n = n
        self.n_words = _bit_length(n)
        self.words: list[int] = [0] * self.n_words

        for v in initial_values:
            assert 0 <= v < n
            offset = _bit_offset(v)
            self.words[offset] |= 1 << _bit_pos(v)
        self.words = [w & 0xFFFFFFFFFFFFFFFF for w in self.words]

        # Sparse non-zero index
        self.non_zero_idx: list[int] = [i for i in range(self.n_words) if self.words[i] != 0]
        self.n_non_zero: int = len(self.non_zero_idx)

        # Temp mask for collected operations
        self._temp_mask: list[int] = [0] * self.n_words

    def create_column(self, values: Iterable[int]) -> ImmutableBitSet:
        """Create an immutable column bitset for intersection with this set."""
        return ImmutableBitSet(self.n_words, values)

    # --- Snapshot/Restore ---

    def save(self) -> tuple[list[int], list[int], int]:
        """Save current state for backtracking."""
        return (
            self.words[:],
            self.non_zero_idx[:self.n_non_zero],
            self.n_non_zero,
        )

    def restore(self, snapshot: tuple[list[int], list[int], int]) -> None:
        """Restore from a saved snapshot."""
        saved_words, saved_nz_idx, saved_n_nz = snapshot
        self.words = saved_words[:]
        self.n_non_zero = saved_n_nz
        self.non_zero_idx = saved_nz_idx[:] + [0] * (self.n_words - len(saved_nz_idx))

    # --- Collected operations ---

    def clear_collected(self) -> None:
        """Clear the temp mask."""
        for i in range(self.n_non_zero):
            self._temp_mask[self.non_zero_idx[i]] = 0

    def collect(self, col: ImmutableBitSet) -> None:
        """OR a column into the temp mask."""
        for i in range(self.n_non_zero):
            offset = self.non_zero_idx[i]
            self._temp_mask[offset] |= col.words[offset]

    def intersect_collected(self) -> bool:
        """Intersect self with the collected temp mask. Returns True if changed."""
        changed = False
        i = self.n_non_zero
        while i > 0:
            i -= 1
            offset = self.non_zero_idx[i]
            old_word = self.words[offset]
            new_word = old_word & self._temp_mask[offset]
            self.words[offset] = new_word
            if new_word == 0:
                self.n_non_zero -= 1
                self.non_zero_idx[i] = self.non_zero_idx[self.n_non_zero]
                self.non_zero_idx[self.n_non_zero] = offset
            changed |= (old_word != new_word)
        return changed

    # --- Direct operations ---

    def intersect_with(self, col: ImmutableBitSet) -> bool:
        """Intersect self with a column. Returns True if changed."""
        changed = False
        i = self.n_non_zero
        while i > 0:
            i -= 1
            offset = self.non_zero_idx[i]
            old_word = self.words[offset]
            set_word = col.words[offset]
            new_word = old_word & set_word if set_word != 0 else 0
            self.words[offset] = new_word
            if new_word == 0:
                self.n_non_zero -= 1
                self.non_zero_idx[i] = self.non_zero_idx[self.n_non_zero]
                self.non_zero_idx[self.n_non_zero] = offset
            changed |= (old_word != new_word)
        return changed

    def intersect_count(self, col: ImmutableBitSet, early_stop: int = 0) -> int:
        """
        Count |self & col|.

        If early_stop > 0, may return early with a partial count when it's clear
        the threshold cannot be reached.
        """
        count = 0
        if early_stop <= 0:
            for i in range(self.n_non_zero):
                offset = self.non_zero_idx[i]
                count += _popcount(self.words[offset] & col.words[offset])
        else:
            i = self.n_non_zero
            while i > 0 and (i * 64) >= (early_stop - count):
                i -= 1
                offset = self.non_zero_idx[i]
                set_word = col.words[offset]
                if set_word != 0:
                    count += _popcount(self.words[offset] & set_word)
        return count

    def intersect_count_all(self, columns: list[ImmutableBitSet], item_indices: list[int], limit: int) -> int:
        """
        Count |self & col[idx[0]] & col[idx[1]] & ... & col[idx[limit-1]]|.

        Used by CoverSize to compute lower bound on support.
        """
        count = 0
        for i in range(self.n_non_zero):
            offset = self.non_zero_idx[i]
            my_word = 0xFFFFFFFFFFFFFFFF
            j = limit
            while j > 0 and my_word != 0:
                j -= 1
                my_word &= columns[item_indices[j]].words[offset]
            if my_word != 0:
                count += _popcount(self.words[offset] & my_word)
        return count

    def is_subset_of(self, col: ImmutableBitSet) -> bool:
        """Check if self is a subset of col (all bits in self are also in col)."""
        for i in range(self.n_non_zero):
            offset = self.non_zero_idx[i]
            if (self.words[offset] & ~col.words[offset]) & 0xFFFFFFFFFFFFFFFF != 0:
                return False
        return True

    def count(self) -> int:
        """Count number of set bits."""
        total = 0
        for i in range(self.n_non_zero):
            total += _popcount(self.words[self.non_zero_idx[i]])
        return total

    def is_empty(self) -> bool:
        return self.n_non_zero == 0
