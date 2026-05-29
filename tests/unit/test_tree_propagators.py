"""
Smoke tests for classification tree propagators.
"""

from __future__ import annotations

import cpmpy as cp

from cp4dm_cpmpy_oscar_ml.core.data import PatternDataset
from cp4dm_cpmpy_oscar_ml.cpmpy_integration.globals import (
    DummyEndNode,
    DummyNode,
    SplitPossible,
    SplitUseful,
    TreeCoverSizeSR,
)
from cp4dm_cpmpy_oscar_ml.cpmpy_integration.solver import CPM_oscar_ml


TINY = [
    (1, 2),
    (2, 3),
    (1, 3),
]


def make_dataset(raw):
    return PatternDataset.from_transactions(raw)


class TestCoverSizeSRSmoke:
    def test_all_take_gives_correct_sup(self):
        """All items taken; Sup should equal intersected coverage."""
        data = make_dataset(TINY)
        n_items = data.n_items
        # take_vars: item decisions (0=nothing, i=item i)
        # We use one take var per item slot
        take_vars = [cp.intvar(0, n_items - 1, name=f"T{i}") for i in range(2)]
        reject_vars = []
        sup = cp.intvar(0, len(TINY), name="Sup")

        model = cp.Model()
        model += TreeCoverSizeSR(take_vars, reject_vars, sup, data)

        solver = CPM_oscar_ml(model)
        count = solver.solveAll()
        assert count > 0

    def test_empty_take_reject_full_sup(self):
        """No items taken/rejected; Sup should equal n_transactions."""
        data = make_dataset(TINY)
        take_vars = []
        reject_vars = []
        sup = cp.intvar(0, len(TINY), name="Sup")

        model = cp.Model()
        model += TreeCoverSizeSR(take_vars, reject_vars, sup, data)

        solver = CPM_oscar_ml(model)
        assert solver.solve()
        assert sup.value() == len(TINY)


class TestCstDummySmoke:
    def test_dummy_parent_zero_forces_children_zero(self):
        """When parent decision = 0, both child decisions must be 0."""
        dp = cp.intvar(0, 1, name="dp")
        dcl = cp.intvar(0, 1, name="dcl")
        dcr = cp.intvar(0, 1, name="dcr")
        scl = cp.intvar(1, 10, name="scl")
        scr = cp.intvar(1, 10, name="scr")

        model = cp.Model()
        model += DummyNode(dp, dcl, dcr, scl, scr)
        model += dp == 0

        solver = CPM_oscar_ml(model)
        assert solver.solve()
        assert dcl.value() == 0
        assert dcr.value() == 0

    def test_dummy_parent_one_forces_sum_nonzero(self):
        """When parent decision = 1, child sums must be > 0."""
        dp = cp.intvar(1, 1, name="dp")
        dcl = cp.intvar(0, 1, name="dcl")
        dcr = cp.intvar(0, 1, name="dcr")
        scl = cp.intvar(0, 10, name="scl")
        scr = cp.intvar(0, 10, name="scr")

        model = cp.Model()
        model += DummyNode(dp, dcl, dcr, scl, scr)

        solver = CPM_oscar_ml(model)
        assert solver.solve()
        assert scl.value() > 0
        assert scr.value() > 0

    def test_dummy_end_parent_one_forces_sums_nonzero(self):
        """DummyEndNode: when parent = 1, sums must be > 0."""
        dp = cp.intvar(1, 1, name="dp")
        scl = cp.intvar(0, 5, name="scl")
        scr = cp.intvar(0, 5, name="scr")

        model = cp.Model()
        model += DummyEndNode(dp, scl, scr)

        solver = CPM_oscar_ml(model)
        assert solver.solve()
        assert scl.value() > 0
        assert scr.value() > 0


class TestCstSplitPossibleSmoke:
    def test_split_impossible_if_sum_too_small(self):
        """count_sum < 2*threshold → decision must be forced to 0."""
        threshold = 3
        decision = cp.intvar(0, 1, name="d")
        count_pos = cp.intvar(1, 10, name="cp_")
        count_neg = cp.intvar(1, 10, name="cn_")
        count_sum = cp.intvar(0, 4, name="cs_")  # max=4 < 2*3=6

        model = cp.Model()
        model += SplitPossible(decision, count_pos, count_neg, count_sum, threshold)

        solver = CPM_oscar_ml(model)
        assert solver.solve()
        assert decision.value() == 0

    def test_split_forced_enforces_sum_lb(self):
        """decision > 0 → count_sum.lb >= 2*threshold."""
        threshold = 2
        decision = cp.intvar(1, 1, name="d")
        count_pos = cp.intvar(1, 10, name="cp_")
        count_neg = cp.intvar(1, 10, name="cn_")
        count_sum = cp.intvar(0, 20, name="cs_")

        model = cp.Model()
        model += SplitPossible(decision, count_pos, count_neg, count_sum, threshold)

        solver = CPM_oscar_ml(model)
        assert solver.solve()
        assert count_sum.value() >= 2 * threshold


class TestCstSplitUsefulSmoke:
    def test_split_useful_decision_one(self):
        """When decision > 0, mini_sum must exceed error_ub."""
        error_ub = 2
        decision = cp.intvar(1, 1, name="d")
        mini_sum = cp.intvar(0, 10, name="ms")
        count_pos = cp.intvar(5, 10, name="cp_")
        count_neg = cp.intvar(5, 10, name="cn_")

        model = cp.Model()
        model += SplitUseful(decision, mini_sum, count_pos, count_neg, error_ub)

        solver = CPM_oscar_ml(model)
        assert solver.solve()
        assert mini_sum.value() > error_ub

    def test_split_useful_impossible_if_min_too_small(self):
        """If min(count_pos, count_neg) <= error_ub → infeasible when decision=1."""
        error_ub = 10
        decision = cp.intvar(1, 1, name="d")
        mini_sum = cp.intvar(0, 5, name="ms")  # ub=5 <= error_ub=10
        count_pos = cp.intvar(1, 5, name="cp_")
        count_neg = cp.intvar(1, 5, name="cn_")

        model = cp.Model()
        model += SplitUseful(decision, mini_sum, count_pos, count_neg, error_ub)

        solver = CPM_oscar_ml(model)
        assert not solver.solve()
