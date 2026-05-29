"""
Parity tests for SPM (PPIC) propagator vs brute-force subsequence enumeration.
"""

from __future__ import annotations

import cpmpy as cp

from cp4dm_cpmpy_oscar_ml.core.data import PatternDataset
from cp4dm_cpmpy_oscar_ml.cpmpy_integration.globals import SequentialPattern
from cp4dm_cpmpy_oscar_ml.cpmpy_integration.solver import CPM_oscar_ml


# Small sequence database: items 1-3, epsilon=0
SDB_RAW = [
    (1, 2, 3),
    (1, 3),
    (2, 3),
    (1, 2),
]


def make_dataset(raw):
    return PatternDataset.from_transactions(raw)


# --- Brute-force subsequence support ---

def is_subsequence(pattern, seq):
    pi = 0
    for item in seq:
        if pi < len(pattern) and item == pattern[pi]:
            pi += 1
    return pi == len(pattern)


def brute_force_frequent_sequences(raw, minsup, max_len):
    """All non-empty sequences of length <= max_len that appear in >= minsup sequences."""
    n_items = max(item for t in raw for item in t) + 1
    results = set()

    def _enumerate(prefix, start_item):
        sup = sum(1 for seq in raw if is_subsequence(prefix, seq))
        if sup < minsup:
            return
        if prefix:
            results.add(tuple(prefix))
        if len(prefix) < max_len:
            for item in range(1, n_items):
                _enumerate(prefix + [item], item)

    _enumerate([], 1)
    return results


def solve_ppic(raw, minsup, max_len):
    """Enumerate frequent sequences via CPM_oscar_ml PPIC."""
    data = make_dataset(raw)
    n_items = data.n_items
    P = [cp.intvar(0, n_items - 1, name=f"P{i}") for i in range(max_len)]

    model = cp.Model()
    model += SequentialPattern(P, minsup, data, method="ppic")

    solver = CPM_oscar_ml(model)
    results = set()

    def _cb():
        # Take only the prefix up to the first epsilon (0)
        vals = [v.value() for v in P]
        seq = []
        for v in vals:
            if v == 0:
                break
            seq.append(v)
        if seq:
            results.add(tuple(seq))

    solver.solveAll(display=_cb)
    return results


class TestPPICParity:
    def test_minsup1_len1(self):
        """All length-1 items with support >= 1."""
        bf = brute_force_frequent_sequences(SDB_RAW, 1, 1)
        solver = solve_ppic(SDB_RAW, 1, 1)
        assert bf == solver, f"BF={bf}  Solver={solver}"

    def test_minsup2_len1(self):
        bf = brute_force_frequent_sequences(SDB_RAW, 2, 1)
        solver = solve_ppic(SDB_RAW, 2, 1)
        assert bf == solver, f"BF={bf}  Solver={solver}"

    def test_minsup2_len2(self):
        bf = brute_force_frequent_sequences(SDB_RAW, 2, 2)
        solver = solve_ppic(SDB_RAW, 2, 2)
        assert bf == solver, f"BF={bf}  Solver={solver}"

    def test_minsup3_len2(self):
        bf = brute_force_frequent_sequences(SDB_RAW, 3, 2)
        solver = solve_ppic(SDB_RAW, 3, 2)
        assert bf == solver, f"BF={bf}  Solver={solver}"

    def test_minsup2_len3(self):
        bf = brute_force_frequent_sequences(SDB_RAW, 2, 3)
        solver = solve_ppic(SDB_RAW, 2, 3)
        assert bf == solver, f"BF={bf}  Solver={solver}"

    def test_high_minsup_empty(self):
        """minsup > n_sequences should yield nothing."""
        solver = solve_ppic(SDB_RAW, 10, 2)
        assert solver == set()
