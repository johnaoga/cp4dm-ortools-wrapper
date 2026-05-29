"""
Smoke tests for FEM (EpisodeSupport) propagator and high-level API classes.
"""

from __future__ import annotations

import cpmpy as cp

from cp4dm_cpmpy_oscar_ml.api import FIMConfig, FIMSolver, SPMConfig, SequentialPatternSolver
from cp4dm_cpmpy_oscar_ml.core.data import PatternDataset
from cp4dm_cpmpy_oscar_ml.core.result import PatternResult
from cp4dm_cpmpy_oscar_ml.cpmpy_integration.globals import FrequentEpisode
from cp4dm_cpmpy_oscar_ml.cpmpy_integration.solver import CPM_oscar_ml


# Long sequence for FEM: items 1-3
LS_RAW = [(1, 2, 3, 1, 2, 1, 3, 2, 1)]  # one long sequence

PASQUIER = [
    (1, 3, 4),
    (2, 3, 5),
    (1, 2, 3, 5),
    (2, 5),
    (1, 2, 3, 5),
]

SDB_RAW = [
    (1, 2, 3),
    (1, 3),
    (2, 3),
    (1, 2),
]


def make_dataset(raw):
    return PatternDataset.from_transactions(raw)


# --- FEM smoke tests ---

class TestEpisodeSupportSmoke:
    def test_single_items_frequent(self):
        """All single items should be frequent with minsup=1 in the long sequence."""
        data = make_dataset(LS_RAW)
        n_items = data.n_items
        E = [cp.intvar(0, n_items - 1, name=f"E{i}") for i in range(1)]

        model = cp.Model()
        model += FrequentEpisode(E, 1, data)

        solver = CPM_oscar_ml(model)
        results = set()

        def _cb():
            ep = tuple(v.value() for v in E if v.value() != 0)
            if ep:
                results.add(ep)

        solver.solveAll(display=_cb)
        # Items 1, 2, 3 must be found
        assert (1,) in results
        assert (2,) in results
        assert (3,) in results

    def test_len2_episodes(self):
        """Length-2 frequent episodes with minsup=2."""
        data = make_dataset(LS_RAW)
        n_items = data.n_items
        E = [cp.intvar(0, n_items - 1, name=f"E{i}") for i in range(2)]

        model = cp.Model()
        model += FrequentEpisode(E, 2, data)

        solver = CPM_oscar_ml(model)
        count = solver.solveAll()
        assert count > 0

    def test_impossible_minsup(self):
        """minsup higher than sequence length => no solutions."""
        data = make_dataset(LS_RAW)
        n_items = data.n_items
        E = [cp.intvar(0, n_items - 1, name=f"E{i}") for i in range(2)]

        model = cp.Model()
        model += FrequentEpisode(E, 100, data)

        solver = CPM_oscar_ml(model)
        count = solver.solveAll()
        assert count == 0


# --- High-level FIMSolver tests ---

class TestFIMSolverAPI:
    def test_fimsolver_returns_patterns(self):
        data = make_dataset(PASQUIER)
        cfg = FIMConfig(minsup=2)
        result = FIMSolver(cfg).solve(data)

        assert isinstance(result, PatternResult)
        assert len(result.patterns) > 0
        assert result.status == "completed"
        assert result.runtime_s >= 0

    def test_fimsolver_closed(self):
        data = make_dataset(PASQUIER)
        cfg = FIMConfig(minsup=2, closed=True)
        result = FIMSolver(cfg).solve(data)
        assert len(result.patterns) > 0

    def test_fimsolver_max_size(self):
        """Patterns limited to max 2 items."""
        data = make_dataset(PASQUIER)
        cfg = FIMConfig(minsup=2, max_size=2)
        result = FIMSolver(cfg).solve(data)
        for p in result.patterns:
            assert len(p.items) <= 2

    def test_fimsolver_support_correct(self):
        """Verify each reported pattern's support is correct."""
        data = make_dataset(PASQUIER)
        vertical = data.as_vertical()
        cfg = FIMConfig(minsup=2)
        result = FIMSolver(cfg).solve(data)

        for p in result.patterns:
            if p.items:
                cov = vertical[p.items[0]].copy()
                for item in p.items[1:]:
                    cov &= vertical[item]
                assert p.support == len(cov), (
                    f"Pattern {p.items}: expected {len(cov)}, got {p.support}"
                )

    def test_fimsolver_minsup_filter(self):
        """No pattern should have support < minsup."""
        data = make_dataset(PASQUIER)
        minsup = 3
        result = FIMSolver(FIMConfig(minsup=minsup)).solve(data)
        for p in result.patterns:
            assert p.support >= minsup

    def test_pattern_result_json(self):
        data = make_dataset(PASQUIER)
        result = FIMSolver(FIMConfig(minsup=3)).solve(data)
        j = result.to_json()
        assert isinstance(j, list)
        for entry in j:
            assert "items" in entry and "support" in entry


# --- High-level SequentialPatternSolver tests ---

class TestSequentialPatternSolverAPI:
    def test_spm_solver_returns_patterns(self):
        data = make_dataset(SDB_RAW)
        cfg = SPMConfig(minsup=2, max_pattern_length=2)
        result = SequentialPatternSolver(cfg).solve(data)
        assert isinstance(result, PatternResult)
        assert len(result.patterns) > 0

    def test_spm_solver_no_results_high_minsup(self):
        data = make_dataset(SDB_RAW)
        cfg = SPMConfig(minsup=10, max_pattern_length=2)
        result = SequentialPatternSolver(cfg).solve(data)
        assert len(result.patterns) == 0
