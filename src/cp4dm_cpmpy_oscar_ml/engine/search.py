"""
Depth-first search with backtracking.

Provides DFS-based solution enumeration with propagation at each node.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from cp4dm_cpmpy_oscar_ml.engine.domain import DomainStore, EngineVar
from cp4dm_cpmpy_oscar_ml.engine.propagation_queue import PropagationQueue
from cp4dm_cpmpy_oscar_ml.engine.trail import Trail
from cp4dm_cpmpy_oscar_ml.exceptions import InconsistencyError


@dataclass
class SearchStats:
    """Statistics collected during search."""

    nodes: int = 0
    failures: int = 0
    solutions: int = 0
    runtime_s: float = 0.0


class DepthFirstSearch:
    """
    DFS search engine with trail-based backtracking.

    Uses the propagation queue to propagate at each node, then branches
    on unbound variables.
    """

    def __init__(
        self,
        store: DomainStore,
        trail: Trail,
        prop_queue: PropagationQueue,
        branch_vars: list[EngineVar] | None = None,
    ) -> None:
        self._store = store
        self._trail = trail
        self._prop_queue = prop_queue
        self._branch_vars = branch_vars or store.variables
        self.stats = SearchStats()
        self._time_limit: float = float("inf")
        self._solution_limit: int = 0  # 0 = unlimited
        self._on_solution: Callable[[], None] | None = None
        self._solutions_found: list[dict[int, int]] = []

    def set_time_limit(self, seconds: float) -> None:
        self._time_limit = seconds

    def set_solution_limit(self, limit: int) -> None:
        self._solution_limit = limit

    def set_on_solution(self, callback: Callable[[], None]) -> None:
        self._on_solution = callback

    def solve(self) -> bool:
        """Find one solution. Returns True if found."""
        self._solution_limit = 1
        self._run_search()
        return self.stats.solutions >= 1

    def solve_all(self) -> int:
        """Find all solutions. Returns count."""
        self._solution_limit = 0
        self._run_search()
        return self.stats.solutions

    def _run_search(self) -> None:
        self.stats = SearchStats()
        self._solutions_found = []
        start = time.time()
        try:
            self._dfs()
        except _SearchComplete:
            pass
        self.stats.runtime_s = time.time() - start
        # Restore last solution values into variable domains
        if self._solutions_found:
            last_sol = self._solutions_found[-1]
            for var in self._store.variables:
                if var.id in last_sol:
                    var.lb = last_sol[var.id]
                    var.ub = last_sol[var.id]
                    var.removed.clear()

    def _dfs(self) -> None:
        """Recursive DFS with propagation."""
        self.stats.nodes += 1

        # Check limits
        if self._time_limit < float("inf"):
            elapsed = time.time()
            # We rely on periodic checks rather than interrupts

        if self._solution_limit and self.stats.solutions >= self._solution_limit:
            raise _SearchComplete()

        # Propagate to fixpoint
        try:
            self._prop_queue.propagate_fixpoint()
        except InconsistencyError:
            self.stats.failures += 1
            return

        # Find first unbound variable to branch on
        branch_var = self._select_branch_var()
        if branch_var is None:
            # All variables bound -> solution found
            self._record_solution()
            return

        # Binary branching: try lb first, then exclude lb
        val = branch_var.lb

        # Left branch: assign var = val
        self._trail.push_level()
        try:
            self._store.assign(branch_var, val)
            self._dfs()
        except InconsistencyError:
            self.stats.failures += 1
        self._trail.pop_level()

        if self._solution_limit and self.stats.solutions >= self._solution_limit:
            raise _SearchComplete()

        # Right branch: remove val
        self._trail.push_level()
        try:
            self._store.remove_value(branch_var, val)
            self._dfs()
        except InconsistencyError:
            self.stats.failures += 1
        self._trail.pop_level()

    def _select_branch_var(self) -> EngineVar | None:
        """Select first unbound variable from branch_vars."""
        for var in self._branch_vars:
            if not var.is_bound:
                return var
        return None

    def _record_solution(self) -> None:
        """Record current assignment as a solution."""
        self.stats.solutions += 1
        sol = {var.id: var.value for var in self._store.variables if var.is_bound}
        self._solutions_found.append(sol)
        if self._on_solution:
            self._on_solution()


class _SearchComplete(Exception):
    """Internal: signals search should stop."""
    pass
