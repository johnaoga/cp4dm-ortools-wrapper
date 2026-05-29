"""
SPM propagators: port of Oscar's PPIC.scala (Prefix Projection with Item Constraints).

PPIC maintains a pseudo-projected database (psdb) as a trail of (sid, pos) pairs,
projecting it for each newly bound prefix item and pruning infrequent extensions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cp4dm_cpmpy_oscar_ml.engine.domain import DomainStore, EngineVar
from cp4dm_cpmpy_oscar_ml.engine.propagator import Propagator
from cp4dm_cpmpy_oscar_ml.engine.trail import Trail
from cp4dm_cpmpy_oscar_ml.exceptions import InconsistencyError

if TYPE_CHECKING:
    from cp4dm_cpmpy_oscar_ml.core.data import PatternDataset


def _get_item_first_pos_by_sequence(sdb: list[tuple[int, ...]], n_items: int) -> list[list[int]]:
    """
    firstPositionMap[sid][item] = first 1-based position of item in sequence sid, or 0.
    """
    result: list[list[int]] = []
    for seq in sdb:
        row = [0] * n_items
        for pos, item in enumerate(seq, start=1):
            if item < n_items and row[item] == 0:
                row[item] = pos
        result.append(row)
    return result


def _get_item_last_pos_by_sequence(sdb: list[tuple[int, ...]], n_items: int) -> list[list[int]]:
    """
    lastPositionMap[sid][item] = last 1-based position of item in sequence sid, or 0.
    """
    result: list[list[int]] = []
    for seq in sdb:
        row = [0] * n_items
        for pos, item in enumerate(seq, start=1):
            if item < n_items:
                row[item] = pos
        result.append(row)
    return result


def _get_sdb_last_pos(sdb: list[tuple[int, ...]], last_pos_map: list[list[int]]) -> list[list[int]]:
    """
    lastPositionList[sid] = sorted descending list of last positions of items in sid.
    """
    result: list[list[int]] = []
    for sid, seq in enumerate(sdb):
        seen: set[int] = set()
        last_positions: list[int] = []
        for item in seq:
            if item not in seen:
                lp = last_pos_map[sid][item]
                if lp > 0:
                    last_positions.append(lp)
                    seen.add(item)
        last_positions.sort(reverse=True)
        result.append(last_positions)
    return result


def _get_sdb_support(sdb: list[tuple[int, ...]], n_items: int) -> list[int]:
    """
    support[item] = number of sequences containing item (1-indexed item array, 0 = epsilon).
    Returns array of length n_items (index 0 = epsilon with support = len(sdb)).
    """
    support = [0] * n_items
    for seq in sdb:
        seen: set[int] = set()
        for item in seq:
            if item not in seen:
                support[item] += 1
                seen.add(item)
    return support


class PPICPropagator(Propagator):
    """
    Port of Oscar's PPIC.scala.

    Maintains a pseudo-projected database (psdb) as a trail. When pattern
    position i is bound to item v, the psdb is projected: only sequence
    entries where v still occurs after the current position are kept.
    Infrequent extensions are pruned from the next pattern position.

    epsilon = 0; real items start at 1.
    """

    name = "oscar_ppic"

    def __init__(self, pattern_vars: list[EngineVar], minsup: int, data: "PatternDataset") -> None:
        super().__init__()
        self.pattern_vars = pattern_vars
        self.minsup = minsup
        self.data = data
        self.variables = tuple(pattern_vars)

        sdb = list(data.get_data())
        self._sdb = sdb
        self._n_items = data.n_items
        self._len_sdb = len(sdb)
        self._len_pattern = len(pattern_vars)
        self._epsilon = 0

        # Precomputed maps
        self._first_pos_map = _get_item_first_pos_by_sequence(sdb, self._n_items)
        self._last_pos_map = _get_item_last_pos_by_sequence(sdb, self._n_items)
        self._last_pos_list = _get_sdb_last_pos(sdb, self._last_pos_map)

        # Initial support per item (index = item id, 0 = epsilon = all sequences)
        raw_support = _get_sdb_support(sdb, self._n_items)
        # raw_support[0] is epsilon (always 0 from helper); override to len_sdb
        raw_support[0] = self._len_sdb
        self._items_support: list[int] = raw_support

        # Pseudo-projected database as a flat trail of (sid, start_pos) pairs.
        # Initial window: sid[i] = i, pos[i] = -1 (meaning: scan full sequence)
        trail_size = max(self._len_sdb * 10, 64)
        self._psdb_seq_id: list[int] = list(range(self._len_sdb)) + [0] * (trail_size - self._len_sdb)
        self._psdb_pos: list[int] = [-1] * trail_size
        self._trail_size = trail_size

        # Reversible: psdb_start = start index of current window, psdb_size = size
        self._psdb_start: int = 0
        self._psdb_size: int = self._len_sdb

        # Support counters per item (reset each projection)
        self._support_counter: list[int] = list(self._items_support)

        # Current position in pattern
        self._cur_pos: int = 0

    def setup(self, store: DomainStore, trail: Trail) -> None:
        # Prune position 0 based on initial item supports (may raise InconsistencyError)
        if self.pattern_vars:
            self._prune(0, store)
        self._do_propagate(store, trail)
        for var in self.pattern_vars:
            self.watch_bind(var)

    def propagate(self, store: DomainStore, trail: Trail) -> None:
        self._do_propagate(store, trail)

    def _do_propagate(self, store: DomainStore, trail: Trail) -> None:
        # Save reversible state
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

        # If already at epsilon, enforce epsilon from here
        if v < self._len_pattern and self.pattern_vars[v].is_bound and self.pattern_vars[v].value == self._epsilon:
            if v > 0:
                self._enforce_epsilon_from(v, store)
            return

        # Advance over newly bound non-epsilon positions
        while (v < self._len_pattern and
               self.pattern_vars[v].is_bound and
               self.pattern_vars[v].value != self._epsilon):
            prefix = self.pattern_vars[v].value
            if not self._filter_prefix_projection(prefix, store):
                raise InconsistencyError(f"PPIC: prefix {prefix} at position {v} is infrequent")
            self._cur_pos += 1
            v = self._cur_pos

        # If next position is epsilon, enforce epsilon suffix
        if v > 0 and v < self._len_pattern and self.pattern_vars[v].is_bound and self.pattern_vars[v].value == self._epsilon:
            self._enforce_epsilon_from(v, store)

    def _enforce_epsilon_from(self, i: int, store: DomainStore) -> None:
        j = i
        while j < self._len_pattern:
            store.assign(self.pattern_vars[j], self._epsilon)
            j += 1

    def _filter_prefix_projection(self, prefix: int, store: DomainStore) -> bool:
        """Project psdb on prefix; returns True if support >= minsup."""
        next_pos = self._cur_pos + 1
        if next_pos >= 2 and prefix == self._epsilon:
            return True
        sup = self._project_sdb(prefix)
        if sup < self.minsup:
            return False
        self._prune(next_pos, store)
        return True

    def _project_sdb(self, prefix: int) -> int:
        """
        Project the current pseudo-projected database on prefix.

        For each (sid, pos) entry in the current window, find the first
        occurrence of prefix at or after pos. If found, record the
        projected (sid, pos_after_prefix) and accumulate support counts
        for items that still appear after that position.
        """
        start_init = self._psdb_start
        size_init = self._psdb_size
        new_support = [0] * self._n_items
        cur_sup = 0

        # Grow trail if needed
        needed = start_init + size_init + size_init + 10
        while needed >= self._trail_size:
            self._psdb_seq_id = self._psdb_seq_id + [0] * self._trail_size
            self._psdb_pos = self._psdb_pos + [-1] * self._trail_size
            self._trail_size *= 2

        j = start_init + size_init

        for i in range(start_init, start_init + size_init):
            sid = self._psdb_seq_id[i]
            seq = self._sdb[sid]
            start_pos = self._psdb_pos[i]

            # Find first occurrence of prefix at index >= start_pos
            found = -1
            if start_pos == -1:
                # Full sequence scan (initial state)
                for k, item in enumerate(seq):
                    if item == prefix:
                        found = k
                        break
            else:
                for k in range(start_pos, len(seq)):
                    if seq[k] == prefix:
                        found = k
                        break

            if found == -1:
                continue  # prefix not in this sequence after start_pos

            # Projected position: first index after the found prefix
            proj_pos = found + 1
            self._psdb_seq_id[j] = sid
            self._psdb_pos[j] = proj_pos
            j += 1
            cur_sup += 1

            # Count items that still appear at or after proj_pos
            seen_after: set[int] = set()
            for k in range(proj_pos, len(seq)):
                item = seq[k]
                if item < self._n_items and item not in seen_after:
                    new_support[item] += 1
                    seen_after.add(item)

        self._psdb_start = start_init + size_init
        self._psdb_size = cur_sup
        self._support_counter = new_support
        self._items_support = new_support[:]
        self._items_support[0] = cur_sup  # epsilon support = current projected size
        return cur_sup

    def _prune(self, i: int, store: DomainStore) -> None:
        """Remove infrequent items from domain of pattern_vars[i]."""
        if i >= self._len_pattern:
            return
        var = self.pattern_vars[i]
        domain = list(store.domain_values(var))
        for item in domain:
            if item != self._epsilon and (item >= self._n_items or self._support_counter[item] < self.minsup):
                store.remove_value(var, item)
        if i == 0:
            remaining = store.domain_values(var)
            if all(v == self._epsilon for v in remaining):
                raise InconsistencyError(
                    f"PPIC: no item meets minsup={self.minsup} at position {i}"
                )
