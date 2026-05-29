"""
Brute-force parity tests for FIM propagators.

Compare Oscar ML FIM/ClosedFIM results against brute-force enumeration on small data.
"""

from itertools import combinations
from pathlib import Path

import cpmpy as cp
import pytest

from cp4dm_cpmpy_oscar_ml.core.data import PatternDataset
from cp4dm_cpmpy_oscar_ml.cpmpy_integration.globals import ClosedFrequentItemset, FrequentItemset
from cp4dm_cpmpy_oscar_ml.cpmpy_integration.solver import CPM_oscar_ml
from cp4dm_cpmpy_oscar_ml.pattern_mining.formats import TdbFormat

FIXTURES = Path(__file__).parent.parent / "fixtures"


def brute_force_frequent_itemsets(data: PatternDataset, minsup: int) -> set[frozenset[int]]:
    """Enumerate all frequent itemsets by brute force."""
    vertical = data.as_vertical()
    real_items = [i for i in range(1, data.n_items) if len(vertical[i]) > 0]
    frequent = set()

    # Check all non-empty subsets of real items
    for size in range(1, len(real_items) + 1):
        for combo in combinations(real_items, size):
            # Compute support as intersection of coverages
            coverage = set(range(data.n_transactions))
            for item in combo:
                coverage &= vertical[item]
            if len(coverage) >= minsup:
                frequent.add(frozenset(combo))

    # Also include empty itemset (support = n_transactions)
    if data.n_transactions >= minsup:
        frequent.add(frozenset())

    return frequent


def brute_force_closed_itemsets(data: PatternDataset, minsup: int) -> set[frozenset[int]]:
    """Enumerate all closed frequent itemsets by brute force."""
    vertical = data.as_vertical()
    frequent = brute_force_frequent_itemsets(data, minsup)
    closed = set()

    for itemset in frequent:
        # Compute coverage
        coverage = set(range(data.n_transactions))
        for item in itemset:
            coverage &= vertical[item]
        support = len(coverage)

        # Check if closed: no proper superset has same support
        is_closed = True
        for other in frequent:
            if other != itemset and itemset < other:
                # Compute support of superset
                other_cov = set(range(data.n_transactions))
                for item in other:
                    other_cov &= vertical[item]
                if len(other_cov) == support:
                    is_closed = False
                    break
        if is_closed:
            closed.add(itemset)

    return closed


def solver_frequent_itemsets(data: PatternDataset, minsup: int) -> set[frozenset[int]]:
    """Use CPM_oscar_ml to enumerate all frequent itemsets."""
    items = cp.boolvar(shape=data.n_items, name="I")
    model = cp.Model()
    model += FrequentItemset(items, minsup=minsup, data=data)

    solver = CPM_oscar_ml(model)
    solutions: list[frozenset[int]] = []

    def on_sol():
        pattern = frozenset(i for i in range(data.n_items) if items[i].value() == 1)
        solutions.append(pattern)

    solver.solveAll(display=on_sol)
    return set(solutions)


def solver_closed_itemsets(data: PatternDataset, minsup: int) -> set[frozenset[int]]:
    """Use CPM_oscar_ml to enumerate all closed frequent itemsets."""
    items = cp.boolvar(shape=data.n_items, name="I")
    model = cp.Model()
    model += ClosedFrequentItemset(items, minsup=minsup, data=data)

    solver = CPM_oscar_ml(model)
    solutions: list[frozenset[int]] = []

    def on_sol():
        pattern = frozenset(i for i in range(data.n_items) if items[i].value() == 1)
        solutions.append(pattern)

    solver.solveAll(display=on_sol)
    return set(solutions)


class TestFIMParity:
    """Compare FIM propagator against brute force."""

    def test_frequent_test_txt_minsup2(self):
        data = PatternDataset.from_file(FIXTURES / "fim" / "test.txt", TdbFormat())
        minsup = 2
        expected = brute_force_frequent_itemsets(data, minsup)
        actual = solver_frequent_itemsets(data, minsup)
        # The solver finds non-empty itemsets (item vars include item 0 which is dummy)
        # Filter out empty-set solution and item-0 from comparison
        expected_no_empty = {fs for fs in expected if len(fs) > 0}
        actual_no_zero = {frozenset(i for i in fs if i > 0) for fs in actual if any(i > 0 for i in fs)}
        assert actual_no_zero == expected_no_empty

    def test_frequent_test_txt_minsup3(self):
        data = PatternDataset.from_file(FIXTURES / "fim" / "test.txt", TdbFormat())
        minsup = 3
        expected = brute_force_frequent_itemsets(data, minsup)
        actual = solver_frequent_itemsets(data, minsup)
        expected_no_empty = {fs for fs in expected if len(fs) > 0}
        actual_no_zero = {frozenset(i for i in fs if i > 0) for fs in actual if any(i > 0 for i in fs)}
        assert actual_no_zero == expected_no_empty

    def test_frequent_pasquier_minsup2(self):
        data = PatternDataset.from_file(FIXTURES / "fim" / "contextPasquier99.txt", TdbFormat())
        minsup = 2
        expected = brute_force_frequent_itemsets(data, minsup)
        actual = solver_frequent_itemsets(data, minsup)
        expected_no_empty = {fs for fs in expected if len(fs) > 0}
        actual_no_zero = {frozenset(i for i in fs if i > 0) for fs in actual if any(i > 0 for i in fs)}
        assert actual_no_zero == expected_no_empty

    def test_closed_test_txt_minsup2(self):
        data = PatternDataset.from_file(FIXTURES / "fim" / "test.txt", TdbFormat())
        minsup = 2
        expected = brute_force_closed_itemsets(data, minsup)
        actual = solver_closed_itemsets(data, minsup)
        expected_no_empty = {fs for fs in expected if len(fs) > 0}
        actual_no_zero = {frozenset(i for i in fs if i > 0) for fs in actual if any(i > 0 for i in fs)}
        assert actual_no_zero == expected_no_empty

    def test_closed_pasquier_minsup2(self):
        data = PatternDataset.from_file(FIXTURES / "fim" / "contextPasquier99.txt", TdbFormat())
        minsup = 2
        expected = brute_force_closed_itemsets(data, minsup)
        actual = solver_closed_itemsets(data, minsup)
        expected_no_empty = {fs for fs in expected if len(fs) > 0}
        actual_no_zero = {frozenset(i for i in fs if i > 0) for fs in actual if any(i > 0 for i in fs)}
        assert actual_no_zero == expected_no_empty


class TestFIMWithConstraints:
    """Test mixing Oscar globals with CPMpy primitive constraints."""

    def test_fim_with_cardinality_bound(self):
        """FrequentItemset + sum(items) <= 2: only itemsets of size <= 2."""
        data = PatternDataset.from_file(FIXTURES / "fim" / "test.txt", TdbFormat())
        minsup = 2
        items = cp.boolvar(shape=data.n_items, name="I")
        model = cp.Model()
        model += FrequentItemset(items, minsup=minsup, data=data)
        model += cp.sum(items) <= 2

        solver = CPM_oscar_ml(model)
        solutions: list[frozenset[int]] = []

        def on_sol():
            pattern = frozenset(i for i in range(data.n_items) if items[i].value() == 1)
            solutions.append(pattern)

        solver.solveAll(display=on_sol)

        # Verify all solutions have at most 2 items (including item 0 if selected)
        for sol in solutions:
            assert len(sol) <= 2

        # Verify all solutions are actually frequent
        vertical = data.as_vertical()
        for sol in solutions:
            if not sol:
                continue
            coverage = set(range(data.n_transactions))
            for item in sol:
                coverage &= vertical[item]
            assert len(coverage) >= minsup
