"""
FEM propagators: port of Oscar's EpisodeSupport.scala.

EpisodeSupport mines frequent episodes in a single long sequence.
It adapts PPIC to a long-sequence setting: positions are absolute indices
into the long sequence rather than (sid, pos) pairs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cp4dm_cpmpy_oscar_ml.engine.domain import DomainStore, EngineVar
from cp4dm_cpmpy_oscar_ml.engine.propagator import Propagator
from cp4dm_cpmpy_oscar_ml.engine.trail import Trail
from cp4dm_cpmpy_oscar_ml.exceptions import InconsistencyError

if TYPE_CHECKING:
    from cp4dm_cpmpy_oscar_ml.core.data import PatternDataset


class EpisodeSupportPropagator(Propagator):
    """
    Port of Oscar's EpisodeSupport.scala.

    Maintains a pseudo-projected database of positions in the long sequence.
    When pattern position i is bound to item v, the psdb is projected to
    positions just after the next occurrence of v. Infrequent extensions are
    pruned from the next pattern position.

    epsilon = 0; real items start at 1.
    """

    name = "oscar_episode_support"

    def __init__(self, pattern_vars: list[EngineVar], minsup: int, data: "PatternDataset") -> None:
        super().__init__()
        self.pattern_vars = pattern_vars
        self.minsup = minsup
        self.data = data
        self.variables = tuple(pattern_vars)

        # Long sequence: first (and only) transaction
        ls_tuple = data.get_data()[0] if data.n_transactions > 0 else ()
        self._ls: tuple[int, ...] = ls_tuple
        self._len_ls = len(self._ls)
        self._n_items = data.n_items
        self._len_pattern = len(pattern_vars)
        self._epsilon = 0

        # last position of each item in the long sequence (0-based, -1 if absent)
        last_pos_of_item = [-1] * self._n_items
        for pos, item in enumerate(self._ls):
            if item < self._n_items:
                last_pos_of_item[item] = pos
        self._last_pos_of_item = last_pos_of_item

        # sorted descending by last position: [(last_pos, item), ...]
        last_pos_list = sorted(
            ((lp, item) for item, lp in enumerate(last_pos_of_item) if lp >= 0),
            key=lambda x: -x[0],
        )
        self._last_pos_list_pos = [p for p, _ in last_pos_list]
        self._last_pos_list_item = [it for _, it in last_pos_list]

        # Pseudo-projected database: list of positions (0-based start for search)
        trail_size = self._len_ls * 5 + self._len_pattern * self._len_ls + 10
        self._psdb_pos: list[int] = list(range(self._len_ls)) + [0] * (trail_size - self._len_ls)
        self._trail_size = trail_size

        self._psdb_start: int = 0
        self._psdb_size: int = self._len_ls

        self._support_counter: list[int] = [0] * (self._n_items + 1)
        self._cur_pos: int = 0

        # Initial support: count items reachable from pos 0
        for item, lp in enumerate(last_pos_of_item):
            if lp >= 0:
                self._support_counter[item] = 1  # simplified: present in ls

    def setup(self, store: DomainStore, trail: Trail) -> None:
        # Recompute proper initial support
        self._recompute_support_from_all_positions()
        self._do_propagate(store, trail)
        for var in self.pattern_vars:
            self.watch_bind(var)

    def propagate(self, store: DomainStore, trail: Trail) -> None:
        self._do_propagate(store, trail)

    def _recompute_support_from_all_positions(self) -> None:
        """Compute initial item support counts over the full long sequence."""
        self._support_counter = [0] * (self._n_items + 1)
        seen: set[int] = set()
        for item in self._ls:
            if item not in seen and item < self._n_items:
                self._support_counter[item] += 1
                seen.add(item)

    def _do_propagate(self, store: DomainStore, trail: Trail) -> None:
        saved_start = self._psdb_start
        saved_size = self._psdb_size
        saved_cur = self._cur_pos
        saved_support = self._support_counter[:]

        def _restore():
            self._psdb_start = saved_start
            self._psdb_size = saved_size
            self._cur_pos = saved_cur
            self._support_counter = saved_support[:]

        trail.save_reversible(_restore)

        v = self._cur_pos

        if v < self._len_pattern and self.pattern_vars[v].is_bound and self.pattern_vars[v].value == self._epsilon:
            if v > 0:
                self._enforce_epsilon_from(v, store)
            return

        while (v < self._len_pattern and
               self.pattern_vars[v].is_bound and
               self.pattern_vars[v].value != self._epsilon):
            prefix = self.pattern_vars[v].value
            if not self._filter_prefix_projection(prefix, store):
                raise InconsistencyError(f"EpisodeSupport: prefix {prefix} at position {v} is infrequent")
            self._cur_pos += 1
            v = self._cur_pos

        if v > 0 and v < self._len_pattern and self.pattern_vars[v].is_bound and self.pattern_vars[v].value == self._epsilon:
            self._enforce_epsilon_from(v, store)

    def _enforce_epsilon_from(self, i: int, store: DomainStore) -> None:
        j = i
        while j < self._len_pattern:
            store.assign(self.pattern_vars[j], self._epsilon)
            j += 1

    def _filter_prefix_projection(self, prefix: int, store: DomainStore) -> bool:
        next_pos = self._cur_pos + 1
        if next_pos >= 2 and prefix == self._epsilon:
            return True
        sup = self._project_sdb(prefix)
        if sup < self.minsup:
            return False
        self._prune(next_pos, store)
        return True

    def _project_sdb(self, prefix: int) -> int:
        """Project pseudo-projected database on prefix in long sequence."""
        start_init = self._psdb_start
        size_init = self._psdb_size
        self._support_counter = [0] * (self._n_items + 1)
        cur_prefix_support = 0

        if prefix >= self._n_items or self._last_pos_of_item[prefix] < 0:
            self._psdb_start = start_init + size_init
            self._psdb_size = 0
            return 0

        # Grow trail if needed
        while start_init + size_init + size_init + 10 >= self._trail_size:
            self._psdb_pos = self._psdb_pos + [0] * self._trail_size
            self._trail_size *= 2

        matched_positions: list[int] = []
        i = start_init
        j = start_init + size_init
        previous_pos = -1

        while i < start_init + size_init:
            pos = self._psdb_pos[i]
            # Check initial position condition
            if self._cur_pos > 0 or (pos < self._len_ls and self._ls[pos] == prefix):
                if self._last_pos_of_item[prefix] < pos:
                    # Dominance rule: no more occurrences after pos
                    break
                # Find next occurrence of prefix at or after pos
                if previous_pos < pos:
                    while pos < self._len_ls and self._ls[pos] != prefix:
                        pos += 1
                    previous_pos = pos
                else:
                    pos = previous_pos

                if pos < self._len_ls:
                    matched_positions.append(pos + 1)
                    self._psdb_pos[j] = pos + 1
                    j += 1
                    cur_prefix_support += 1
            i += 1

        # Compute item supports from matched positions
        k = 0
        lp_pos = self._last_pos_list_pos
        lp_item = self._last_pos_list_item
        matched_positions.sort(reverse=True)
        mp_idx = 0
        mp_len = len(matched_positions)

        k = 0
        while k < len(lp_pos) and mp_idx < mp_len:
            while mp_idx < mp_len and matched_positions[mp_idx] > lp_pos[k]:
                mp_idx += 1
            remaining = mp_len - mp_idx
            if remaining > 0:
                self._support_counter[lp_item[k]] += remaining
            k += 1

        self._psdb_start = start_init + size_init
        self._psdb_size = cur_prefix_support
        return cur_prefix_support

    def _prune(self, i: int, store: DomainStore) -> None:
        if i >= self._len_pattern:
            return
        var = self.pattern_vars[i]
        domain = list(store.domain_values(var))
        for item in domain:
            if item != self._epsilon and (item >= self._n_items or self._support_counter[item] < self.minsup):
                store.remove_value(var, item)
