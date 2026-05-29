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

        # Initial support: index 0 = epsilon (all sequences)
        raw_support = _get_sdb_support(sdb, self._n_items)
        self._items_support: list[int] = [self._len_sdb] + raw_support[1:]

        # Pseudo-projected database as a flat trail
        # psdb_seq_id[i], psdb_pos[i] encode (sid, pos_after_prefix)
        trail_size = self._len_sdb * 5
        self._psdb_seq_id: list[int] = list(range(trail_size))
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
        """Compute next pseudo-projected database, update support counters."""
        start_init = self._psdb_start
        size_init = self._psdb_size
        self._support_counter = [0] * self._n_items
        cur_prefix_support = 0
        nb_added_target = self._items_support[prefix] if prefix < len(self._items_support) else 0

        i = start_init
        j = start_init + size_init
        nb_added = 0

        # Grow trail if needed
        while j + size_init >= self._trail_size:
            self._psdb_seq_id = self._psdb_seq_id + [0] * self._trail_size
            self._psdb_pos = self._psdb_pos + [-1] * self._trail_size
            self._trail_size *= 2

        while i < start_init + size_init and nb_added < nb_added_target:
            sid = self._psdb_seq_id[i]
            seq = self._sdb[sid]
            lti = len(seq)
            start = self._psdb_pos[i]
            pos = start

            last_pos_prefix = self._last_pos_map[sid][prefix] if prefix < len(self._last_pos_map[sid]) else 0
            if last_pos_prefix != 0 and last_pos_prefix - 1 >= pos:
                nb_added += 1
                # Find next occurrence of prefix
                if start == -1:
                    pos = (self._first_pos_map[sid][prefix] - 1) if prefix < len(self._first_pos_map[sid]) else 0
                else:
                    while pos < lti and seq[pos] != prefix:
                        pos += 1

                self._psdb_seq_id[j] = sid
                self._psdb_pos[j] = pos + 1
                j += 1
                cur_prefix_support += 1

                # Recompute support from last-position list
                ti_last = self._last_pos_list[sid]
                c = 0
                while c < len(ti_last) and ti_last[c] - 1 > pos:
                    item_at = seq[ti_last[c] - 1]
                    if item_at < self._n_items:
                        self._support_counter[item_at] += 1
                    c += 1
            i += 1

        self._psdb_start = start_init + size_init
        self._psdb_size = cur_prefix_support
        self._items_support = self._support_counter[:]
        # Restore epsilon support
        self._items_support[0] = cur_prefix_support
        return cur_prefix_support

    def _prune(self, i: int, store: DomainStore) -> None:
        """Remove infrequent items from domain of pattern_vars[i]."""
        if i >= self._len_pattern:
            return
        var = self.pattern_vars[i]
        # Collect current domain values
        domain = list(store.domain_values(var))
        for item in domain:
            if item != self._epsilon and (item >= self._n_items or self._support_counter[item] < self.minsup):
                store.remove_value(var, item)
