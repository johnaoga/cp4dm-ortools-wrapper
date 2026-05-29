"""Tests for data loading and format parsing."""

from pathlib import Path

import pytest

from cp4dm_cpmpy_oscar_ml.core.data import PatternDataset, Transaction
from cp4dm_cpmpy_oscar_ml.pattern_mining.formats import TdbFormat

FIXTURES = Path(__file__).parent.parent / "fixtures"


class TestTdbFormat:
    def test_load_test_txt(self):
        ds = PatternDataset.from_file(FIXTURES / "fim" / "test.txt", format=TdbFormat())
        assert ds.n_transactions == 5
        # Items: 1,2,3,4 -> max is 4, so n_items = 5 (Oscar convention)
        assert ds.n_items == 5
        assert ds.records[0].data == (1, 2, 4)
        assert ds.records[1].data == (2, 3, 4)
        assert ds.records[4].data == (2, 3)

    def test_load_context_pasquier(self):
        ds = PatternDataset.from_file(FIXTURES / "fim" / "contextPasquier99.txt", format=TdbFormat())
        assert ds.n_transactions == 5
        assert ds.n_items == 6  # items 1..5, n_items = 6

    def test_vertical_representation(self):
        ds = PatternDataset.from_file(FIXTURES / "fim" / "test.txt", format=TdbFormat())
        vert = ds.as_vertical()
        # Item 0: not in any transaction
        assert vert[0] == set()
        # Item 1: in transactions 0, 2, 3
        assert vert[1] == {0, 2, 3}
        # Item 2: in all transactions
        assert vert[2] == {0, 1, 2, 3, 4}
        # Item 3: in transactions 1, 3, 4
        assert vert[3] == {1, 3, 4}
        # Item 4: in transactions 0, 1, 2, 3
        assert vert[4] == {0, 1, 2, 3}

    def test_from_transactions(self):
        raw = [[1, 2, 3], [2, 3], [1, 3]]
        ds = PatternDataset.from_transactions(raw)
        assert ds.n_transactions == 3
        assert ds.n_items == 4  # max item is 3, n_items = 4

    def test_density(self):
        ds = PatternDataset.from_file(FIXTURES / "fim" / "test.txt", format=TdbFormat())
        d = ds.density()
        # 5 transactions, items 1-4 (4 real items), total items = 3+3+3+4+2 = 15
        # density = 15 / (5 * 4) = 0.75
        assert abs(d - 0.75) < 0.01
