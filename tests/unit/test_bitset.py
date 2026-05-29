"""Tests for the reversible sparse bitset."""

import pytest

from cp4dm_cpmpy_oscar_ml.utils.bitset import ImmutableBitSet, ReversibleSparseBitset


class TestReversibleSparseBitset:
    def test_basic_count(self):
        bs = ReversibleSparseBitset(10, range(10))
        assert bs.count() == 10

    def test_empty(self):
        bs = ReversibleSparseBitset(10, [])
        assert bs.count() == 0
        assert bs.is_empty()

    def test_intersect_with(self):
        bs = ReversibleSparseBitset(10, range(10))
        col = bs.create_column([0, 2, 4, 6, 8])
        changed = bs.intersect_with(col)
        assert changed
        assert bs.count() == 5

    def test_intersect_count(self):
        bs = ReversibleSparseBitset(10, range(10))
        col = bs.create_column([1, 3, 5, 7])
        assert bs.intersect_count(col) == 4

    def test_intersect_count_with_early_stop(self):
        bs = ReversibleSparseBitset(100, range(100))
        col = bs.create_column(range(50))
        # With early_stop, should still return correct count
        count = bs.intersect_count(col, early_stop=30)
        assert count == 50

    def test_is_subset_of(self):
        bs = ReversibleSparseBitset(10, [0, 1, 2])
        col_superset = bs.create_column([0, 1, 2, 3, 4])
        col_partial = bs.create_column([0, 1])
        assert bs.is_subset_of(col_superset)
        assert not bs.is_subset_of(col_partial)

    def test_save_restore(self):
        bs = ReversibleSparseBitset(10, range(10))
        snap = bs.save()
        col = bs.create_column([0, 1, 2])
        bs.intersect_with(col)
        assert bs.count() == 3
        bs.restore(snap)
        assert bs.count() == 10

    def test_intersect_count_all(self):
        bs = ReversibleSparseBitset(5, range(5))
        # Item 0 covers {0, 1, 2}
        col0 = bs.create_column([0, 1, 2])
        # Item 1 covers {1, 2, 3}
        col1 = bs.create_column([1, 2, 3])
        columns = [col0, col1]
        # Intersection of col0 and col1 with bs: {1, 2}
        assert bs.intersect_count_all(columns, [0, 1], 2) == 2

    def test_collected_operations(self):
        bs = ReversibleSparseBitset(10, range(10))
        col1 = bs.create_column([0, 1, 2, 3])
        col2 = bs.create_column([2, 3, 4, 5])
        bs.clear_collected()
        bs.collect(col1)
        bs.collect(col2)
        # temp_mask = col1 | col2 = {0,1,2,3,4,5}
        changed = bs.intersect_collected()
        assert changed
        assert bs.count() == 6

    def test_large_bitset(self):
        """Test with more than 64 elements (multiple words)."""
        n = 200
        bs = ReversibleSparseBitset(n, range(n))
        assert bs.count() == n
        col = bs.create_column(range(0, n, 2))  # even numbers
        assert bs.intersect_count(col) == 100
        bs.intersect_with(col)
        assert bs.count() == 100
