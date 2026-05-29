"""
High-level solver APIs for Oscar ML pattern mining.

These provide a simple one-call interface while still using the native
Oscar ML globals and CPM_oscar_ml solver under the hood.

Example usage::

    from cp4dm_cpmpy_oscar_ml import PatternDataset, FIMConfig, FIMSolver
    from cp4dm_cpmpy_oscar_ml.pattern_mining.formats import TdbFormat

    data = PatternDataset.from_file("retail.txt", format=TdbFormat())
    result = FIMSolver(FIMConfig(minsup=10, closed=True)).solve(data)
    for pattern in result.patterns:
        print(pattern.items, pattern.support)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import cpmpy as cp

from cp4dm_cpmpy_oscar_ml.core.data import PatternDataset
from cp4dm_cpmpy_oscar_ml.core.result import Pattern, PatternResult
from cp4dm_cpmpy_oscar_ml.cpmpy_integration.globals import (
    ClosedFrequentItemset,
    FrequentEpisode,
    FrequentItemset,
    SequentialPattern,
)
from cp4dm_cpmpy_oscar_ml.cpmpy_integration.solver import CPM_oscar_ml


@dataclass
class FIMConfig:
    """Configuration for Frequent Itemset Mining."""
    minsup: int = 1
    closed: bool = False
    max_size: int | None = None
    min_size: int | None = None
    extra_constraints: list = field(default_factory=list)


@dataclass
class SPMConfig:
    """Configuration for Sequential Pattern Mining."""
    minsup: int = 1
    max_pattern_length: int = 4
    method: str = "ppic"
    extra_constraints: list = field(default_factory=list)


@dataclass
class FEMConfig:
    """Configuration for Frequent Episode Mining."""
    minsup: int = 1
    max_episode_length: int = 4
    extra_constraints: list = field(default_factory=list)


class FIMSolver:
    """
    High-level solver for Frequent Itemset Mining.

    Uses Oscar ML's FrequentItemset or ClosedFrequentItemset global
    constraint with the CPM_oscar_ml engine.
    """

    def __init__(self, config: FIMConfig | None = None) -> None:
        self.config = config or FIMConfig()
        self._item_vars: cp.NDVarArray | None = None
        self._model: cp.Model | None = None
        self._solver: CPM_oscar_ml | None = None

    def build(self, data: PatternDataset) -> None:
        """Build the CPMpy model for the given dataset."""
        cfg = self.config
        n_items = data.n_items
        self._item_vars = cp.boolvar(shape=n_items, name="I")
        items = self._item_vars

        self._model = cp.Model()

        if cfg.closed:
            self._model += ClosedFrequentItemset(items, cfg.minsup, data)
        else:
            self._model += FrequentItemset(items, cfg.minsup, data)

        if cfg.max_size is not None:
            self._model += cp.sum(items) <= cfg.max_size
        if cfg.min_size is not None:
            self._model += cp.sum(items) >= cfg.min_size

        for c in cfg.extra_constraints:
            self._model += c

        self._solver = CPM_oscar_ml(self._model)

    def solve(self, data: PatternDataset) -> PatternResult:
        """Find all frequent itemsets and return a PatternResult."""
        self.build(data)
        result = PatternResult()
        patterns: list[Pattern] = []
        items = self._item_vars

        t0 = time.perf_counter()

        def _on_solution():
            selected = tuple(i for i in range(len(items)) if items[i].value() == 1)
            support = self._compute_support(selected, data)
            patterns.append(Pattern(items=selected, support=support))

        self._solver.solveAll(display=_on_solution)
        result.patterns = patterns
        result.runtime_s = time.perf_counter() - t0
        result.n_nodes = self._solver.stats.nodes
        result.n_failures = self._solver.stats.failures
        result.status = "completed"
        return result

    def _compute_support(self, selected: tuple[int, ...], data: PatternDataset) -> int:
        """Compute transaction support of the selected itemset."""
        if not selected:
            return data.n_transactions
        vertical = data.as_vertical()
        cov = vertical[selected[0]].copy()
        for item in selected[1:]:
            cov &= vertical[item]
        return len(cov)


class SequentialPatternSolver:
    """
    High-level solver for Sequential Pattern Mining (PPIC).

    Uses Oscar ML's SequentialPattern global constraint.
    """

    def __init__(self, config: SPMConfig | None = None) -> None:
        self.config = config or SPMConfig()
        self._pattern_vars: list | None = None
        self._model: cp.Model | None = None
        self._solver: CPM_oscar_ml | None = None

    def build(self, data: PatternDataset) -> None:
        """Build CPMpy model."""
        cfg = self.config
        n_items = data.n_items
        max_len = cfg.max_pattern_length
        # items 0=epsilon, 1..n_items-1 = real items
        self._pattern_vars = [cp.intvar(0, n_items - 1, name=f"P{i}") for i in range(max_len)]
        P = self._pattern_vars

        self._model = cp.Model()
        self._model += SequentialPattern(P, cfg.minsup, data, method=cfg.method)

        for c in cfg.extra_constraints:
            self._model += c

        self._solver = CPM_oscar_ml(self._model)

    def solve(self, data: PatternDataset) -> PatternResult:
        """Find all frequent sequences."""
        self.build(data)
        result = PatternResult()
        patterns: list[Pattern] = []
        P = self._pattern_vars

        t0 = time.perf_counter()

        def _on_solution():
            seq = tuple(v.value() for v in P if v.value() != 0)
            if seq:
                sup = self._compute_sequence_support(seq, data)
                if sup >= self.config.minsup:
                    patterns.append(Pattern(items=seq, support=sup))

        self._solver.solveAll(display=_on_solution)
        result.patterns = patterns
        result.runtime_s = time.perf_counter() - t0
        result.n_nodes = self._solver.stats.nodes
        result.n_failures = self._solver.stats.failures
        result.status = "completed"
        return result

    def _compute_sequence_support(self, pattern: tuple[int, ...], data: PatternDataset) -> int:
        """Count sequences containing pattern as subsequence."""
        count = 0
        for seq in data.get_data():
            if self._is_subsequence(pattern, seq):
                count += 1
        return count

    @staticmethod
    def _is_subsequence(pattern: tuple[int, ...], seq: tuple[int, ...]) -> bool:
        pi = 0
        for item in seq:
            if pi < len(pattern) and item == pattern[pi]:
                pi += 1
        return pi == len(pattern)


class FrequentEpisodeSolver:
    """
    High-level solver for Frequent Episode Mining (EpisodeSupport).

    Uses Oscar ML's FrequentEpisode global constraint.
    """

    def __init__(self, config: FEMConfig | None = None) -> None:
        self.config = config or FEMConfig()
        self._pattern_vars: list | None = None
        self._model: cp.Model | None = None
        self._solver: CPM_oscar_ml | None = None

    def build(self, data: PatternDataset) -> None:
        """Build CPMpy model."""
        cfg = self.config
        n_items = data.n_items
        max_len = cfg.max_episode_length
        self._pattern_vars = [cp.intvar(0, n_items - 1, name=f"E{i}") for i in range(max_len)]
        E = self._pattern_vars

        self._model = cp.Model()
        self._model += FrequentEpisode(E, cfg.minsup, data)

        for c in cfg.extra_constraints:
            self._model += c

        self._solver = CPM_oscar_ml(self._model)

    def solve(self, data: PatternDataset) -> PatternResult:
        """Find all frequent episodes."""
        self.build(data)
        result = PatternResult()
        patterns: list[Pattern] = []
        E = self._pattern_vars

        t0 = time.perf_counter()

        def _on_solution():
            ep = tuple(v.value() for v in E if v.value() != 0)
            if ep:
                patterns.append(Pattern(items=ep, support=self.config.minsup))

        self._solver.solveAll(display=_on_solution)
        result.patterns = patterns
        result.runtime_s = time.perf_counter() - t0
        result.n_nodes = self._solver.stats.nodes
        result.n_failures = self._solver.stats.failures
        result.status = "completed"
        return result
