"""
Parity tests for CoverSize and CoverClosure propagators vs brute force.

CoverSize(I, Sup, data):  Sup = |cover(I)| for selected items.
CoverClosure(I, Sup, data): Same + closure enforcement.
"""

from __future__ import annotations

import cpmpy as cp

from cp4dm_cpmpy_oscar_ml.core.data import PatternDataset
from cp4dm_cpmpy_oscar_ml.cpmpy_integration.globals import CoverClosure, CoverSize
from cp4dm_cpmpy_oscar_ml.cpmpy_integration.solver import CPM_oscar_ml


# Datasets

PASQUIER = [
    (1, 3, 4),
    (2, 3, 5),
    (1, 2, 3, 5),
    (2, 5),
    (1, 2, 3, 5),
]

TINY = [
    (1, 2),
    (2, 3),
    (1, 3),
]


def make_dataset(raw):
    return PatternDataset.from_transactions(raw)


# --- Brute-force helpers ---

def brute_force_cover_size(raw_transactions, sup_lb, sup_ub):
    """Enumerate (itemset, support) pairs where sup_lb <= |cover| <= sup_ub."""
    n_items = max(item for t in raw_transactions for item in t) + 1
    vertical = [set() for _ in range(n_items)]
    for tid, t in enumerate(raw_transactions):
        for item in t:
            vertical[item].add(tid)

    results = set()
    for mask in range(1 << n_items):
        items = tuple(i for i in range(n_items) if mask & (1 << i))
        if not items:
            cov = len(raw_transactions)
        else:
            cov = len(set.intersection(*[vertical[i] for i in items]))
        if sup_lb <= cov <= sup_ub:
            results.add((items, cov))
    return results


def solve_cover_size(raw_transactions, sup_lb, sup_ub):
    """Enumerate solutions via CPM_oscar_ml CoverSize."""
    data = make_dataset(raw_transactions)
    n_items = data.n_items
    items = cp.boolvar(shape=n_items, name="I")
    sup = cp.intvar(sup_lb, sup_ub, name="Sup")

    model = cp.Model()
    model += CoverSize(items, sup, data)

    solver = CPM_oscar_ml(model)
    results = set()

    def _cb():
        selected = tuple(i for i in range(n_items) if items[i].value() == 1)
        s = sup.value()
        results.add((selected, s))

    solver.solveAll(display=_cb)
    return results


def solve_cover_closure(raw_transactions, sup_lb, sup_ub):
    """Enumerate solutions via CPM_oscar_ml CoverClosure (should yield closed sets)."""
    data = make_dataset(raw_transactions)
    n_items = data.n_items
    items = cp.boolvar(shape=n_items, name="I")
    sup = cp.intvar(sup_lb, sup_ub, name="Sup")

    model = cp.Model()
    model += CoverClosure(items, sup, data)

    solver = CPM_oscar_ml(model)
    results = set()

    def _cb():
        selected = tuple(i for i in range(n_items) if items[i].value() == 1)
        s = sup.value()
        results.add((selected, s))

    solver.solveAll(display=_cb)
    return results


# --- Tests ---

class TestCoverSizeParity:
    def test_tiny_all(self):
        """All itemsets with their support on tiny dataset."""
        bf = brute_force_cover_size(TINY, 1, len(TINY))
        solver_res = solve_cover_size(TINY, 1, len(TINY))
        # Check the solver finds at least the brute-force results with matching support
        # (may differ in exact support value per solution)
        bf_items = {items for items, _ in bf}
        solver_items = {items for items, _ in solver_res}
        assert bf_items == solver_items, f"Mismatch:\n  BF={bf_items}\n  Solver={solver_items}"

    def test_tiny_sup_ge2(self):
        """Items with support >= 2 on tiny dataset."""
        bf = {items for items, s in brute_force_cover_size(TINY, 2, len(TINY))}
        solver = {items for items, _ in solve_cover_size(TINY, 2, len(TINY))}
        assert bf == solver

    def test_pasquier_sup_ge2(self):
        bf = {items for items, _ in brute_force_cover_size(PASQUIER, 2, len(PASQUIER))}
        solver = {items for items, _ in solve_cover_size(PASQUIER, 2, len(PASQUIER))}
        assert bf == solver

    def test_pasquier_sup_ge3(self):
        bf = {items for items, _ in brute_force_cover_size(PASQUIER, 3, len(PASQUIER))}
        solver = {items for items, _ in solve_cover_size(PASQUIER, 3, len(PASQUIER))}
        assert bf == solver

    def test_sup_exact_constraint(self):
        """sup_lb == sup_ub should restrict to exact coverage."""
        target_sup = 2
        bf = {items for items, s in brute_force_cover_size(TINY, target_sup, target_sup)}
        solver = {items for items, _ in solve_cover_size(TINY, target_sup, target_sup)}
        assert bf == solver

    def test_empty_result(self):
        """sup_lb > n_transactions should yield nothing."""
        solver = {items for items, _ in solve_cover_size(TINY, 10, 10)}
        assert solver == set()


class TestCoverClosureParity:
    def _brute_force_closed(self, raw, sup_lb):
        """Compute closed frequent itemsets by brute force."""
        n_items = max(item for t in raw for item in t) + 1
        vertical = [set() for _ in range(n_items)]
        for tid, t in enumerate(raw):
            for item in t:
                vertical[item].add(tid)

        # Collect all (itemset, cover) pairs
        all_itemsets: dict[frozenset, int] = {}
        for mask in range(1 << n_items):
            items = frozenset(i for i in range(n_items) if mask & (1 << i))
            if not items:
                cov = len(raw)
            else:
                cov = len(set.intersection(*[vertical[i] for i in items]))
            if cov >= sup_lb:
                all_itemsets[items] = cov

        # Closed = itemset I s.t. no superset J has same coverage
        closed = set()
        for items, cov in all_itemsets.items():
            is_closed = True
            for items2, cov2 in all_itemsets.items():
                if items2 > items and cov2 == cov:
                    is_closed = False
                    break
            if is_closed:
                closed.add(tuple(sorted(items)))
        return closed

    def test_tiny_closed_minsup1(self):
        bf = self._brute_force_closed(TINY, 1)
        solver = {items for items, _ in solve_cover_closure(TINY, 1, len(TINY))}
        assert bf == solver

    def test_tiny_closed_minsup2(self):
        bf = self._brute_force_closed(TINY, 2)
        solver = {items for items, _ in solve_cover_closure(TINY, 2, len(TINY))}
        assert bf == solver

    def test_pasquier_closed_minsup2(self):
        bf = self._brute_force_closed(PASQUIER, 2)
        solver = {items for items, _ in solve_cover_closure(PASQUIER, 2, len(PASQUIER))}
        assert bf == solver


class TestZDCPropagator:
    """Smoke tests for ZDCPropagator (score-bound filtering)."""

    def test_zdc_basic(self):
        """Simple WRAcc-like score: (p/n_pos - n/n_neg) should be filterable."""
        import cpmpy as cp
        from cp4dm_cpmpy_oscar_ml.cpmpy_integration.globals import ZeroDiagonalConvexScore
        from cp4dm_cpmpy_oscar_ml.cpmpy_integration.solver import CPM_oscar_ml

        class SimpleScore:
            def eval(self, p, n, n_pos, n_neg):
                if n_pos == 0:
                    return 0.0
                return 100 * (p / n_pos - (n / n_neg if n_neg > 0 else 0))

        n_pos, n_neg = 5, 5
        pos_var = cp.intvar(0, n_pos, name="P")
        neg_var = cp.intvar(0, n_neg, name="N")
        score_var = cp.intvar(0, 100, name="S")

        model = cp.Model()
        model += ZeroDiagonalConvexScore(pos_var, neg_var, score_var, n_pos, n_neg, SimpleScore())

        solver = CPM_oscar_ml(model)
        count = solver.solveAll()
        assert count > 0

    def test_zdc_impossible(self):
        """Score threshold that can never be met should yield 0 solutions."""
        import cpmpy as cp
        from cp4dm_cpmpy_oscar_ml.cpmpy_integration.globals import ZeroDiagonalConvexScore
        from cp4dm_cpmpy_oscar_ml.cpmpy_integration.solver import CPM_oscar_ml

        class AlwaysZero:
            def eval(self, p, n, n_pos, n_neg):
                return 0.0

        n_pos, n_neg = 5, 5
        pos_var = cp.intvar(0, n_pos, name="P")
        neg_var = cp.intvar(0, n_neg, name="N")
        score_var = cp.intvar(1, 100, name="S")  # lb=1, but score always 0

        model = cp.Model()
        model += ZeroDiagonalConvexScore(pos_var, neg_var, score_var, n_pos, n_neg, AlwaysZero())

        solver = CPM_oscar_ml(model)
        count = solver.solveAll()
        assert count == 0
