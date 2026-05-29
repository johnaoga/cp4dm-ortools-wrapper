"""
FIM propagators: direct port of Oscar's FIM.scala and ClosedFIM.scala.

These propagators maintain a reversible coverage bitset and prune items
that would violate the minimum frequency threshold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cp4dm_cpmpy_oscar_ml.engine.domain import DomainStore, EngineVar
from cp4dm_cpmpy_oscar_ml.engine.propagator import Propagator
from cp4dm_cpmpy_oscar_ml.engine.trail import Trail
from cp4dm_cpmpy_oscar_ml.exceptions import InconsistencyError
from cp4dm_cpmpy_oscar_ml.utils.bitset import ImmutableBitSet, ReversibleSparseBitset

if TYPE_CHECKING:
    from cp4dm_cpmpy_oscar_ml.core.data import PatternDataset


class FIMPropagator(Propagator):
    """
    Frequent Itemset Mining propagator.

    Port of Oscar's FIM.scala. Maintains coverage as the intersection of
    selected item columns, prunes items that would drop support below minsup.
    """

    name = "oscar_fim"

    def __init__(self, item_vars: list[EngineVar], minsup: int, data: PatternDataset) -> None:
        super().__init__()
        self.item_vars = item_vars
        self.minsup = minsup
        self.data = data
        self.variables = tuple(item_vars)
        self._n_items = data.n_items
        self._n_trans = data.n_transactions

    def setup(self, store: DomainStore, trail: Trail) -> None:
        # Build vertical representation and bitset columns
        vertical = self.data.as_vertical()
        self._coverage = ReversibleSparseBitset(self._n_trans, range(self._n_trans))
        self._columns: list[ImmutableBitSet] = [
            self._coverage.create_column(vertical[i]) for i in range(self._n_items)
        ]

        # Track unbound-not-in-closure indices (Oscar's unboundNotInClosureIndices)
        self._unbound_indices = list(range(len(self.item_vars)))
        self._n_unbound = len(self.item_vars)

        # Register watches: propagate when any item variable becomes bound
        for var in self.item_vars:
            self.watch_bind(var)

        # Save initial coverage state registration
        trail.save_reversible(self._make_restore_fn())

        # Initial propagation: handle any already-bound variables
        self._do_propagate(store, trail, is_setup=True)

    def propagate(self, store: DomainStore, trail: Trail) -> None:
        self._do_propagate(store, trail, is_setup=False)

    def _do_propagate(self, store: DomainStore, trail: Trail, is_setup: bool) -> None:
        # Save coverage state for backtracking
        saved = self._coverage.save()
        trail.save_reversible(self._make_restore_fn_from(saved))

        cover_changed = False
        n_u = self._n_unbound

        # Process newly bound variables
        i = n_u
        while i > 0:
            i -= 1
            idx = self._unbound_indices[i]
            var = self.item_vars[idx]
            if var.is_bound:
                n_u = self._remove_item(i, n_u, idx)
                # If bound to 1 (selected), intersect coverage with item column
                if var.value == 1:
                    cover_changed |= self._coverage.intersect_with(self._columns[idx])

        if not cover_changed and not is_setup:
            self._n_unbound = n_u
            return

        cardinality = self._coverage.count()

        # Failure check
        if cardinality < self.minsup:
            raise InconsistencyError(
                f"FIM: coverage {cardinality} < minsup {self.minsup}"
            )

        # Pruning
        if cardinality == self.minsup:
            n_u = self._prune_if_sup_eq_freq(n_u, store)
        else:
            n_u = self._prune(n_u, cardinality, store)

        self._n_unbound = n_u

    def _prune(self, n_u_bound: int, cardinality: int, store: DomainStore) -> int:
        """Prune items whose inclusion would drop support below minsup."""
        n_u = n_u_bound
        i = n_u
        while i > 0:
            i -= 1
            idx = self._unbound_indices[i]
            card_idx = self._coverage.intersect_count(self._columns[idx], self.minsup)

            # Condition 1: item is infrequent -> remove it
            if card_idx < self.minsup:
                n_u = self._remove_item(i, n_u, idx)
                store.assign(self.item_vars[idx], 0)
            # Condition 2: item is in closure (same support) -> just remove from unbound list
            elif card_idx == cardinality:
                n_u = self._remove_item(i, n_u, idx)
        return n_u

    def _prune_if_sup_eq_freq(self, n_u_bound: int, store: DomainStore) -> int:
        """Special pruning when support equals frequency exactly."""
        n_u = n_u_bound
        i = n_u
        while i > 0:
            i -= 1
            idx = self._unbound_indices[i]
            # If coverage is NOT a subset of item column -> item is infrequent
            if not self._coverage.is_subset_of(self._columns[idx]):
                n_u = self._remove_item(i, n_u, idx)
                store.assign(self.item_vars[idx], 0)
            else:
                # Coverage IS subset -> item in closure
                n_u = self._remove_item(i, n_u, idx)
        return n_u

    def _remove_item(self, position: int, n_u: int, idx: int) -> int:
        """Swap-remove from unbound list (Oscar's removeItem)."""
        last_u = n_u - 1
        self._unbound_indices[position] = self._unbound_indices[last_u]
        self._unbound_indices[last_u] = idx
        return last_u

    def _make_restore_fn(self) -> callable:
        saved_words = self._coverage.words[:]
        saved_nz = self._coverage.non_zero_idx[:self._coverage.n_non_zero]
        saved_n_nz = self._coverage.n_non_zero
        saved_unbound = self._unbound_indices[:]
        saved_n_u = self._n_unbound

        def restore():
            self._coverage.restore((saved_words, saved_nz, saved_n_nz))
            self._unbound_indices = saved_unbound[:]
            self._n_unbound = saved_n_u

        return restore

    def _make_restore_fn_from(self, coverage_snap: tuple) -> callable:
        saved_unbound = self._unbound_indices[:]
        saved_n_u = self._n_unbound

        def restore():
            self._coverage.restore(coverage_snap)
            self._unbound_indices = saved_unbound[:]
            self._n_unbound = saved_n_u

        return restore


class ClosedFIMPropagator(Propagator):
    """
    Closed Frequent Itemset Mining propagator.

    Port of Oscar's ClosedFIM.scala. Like FIM but also enforces closure:
    items in the closure of the current itemset must be included.
    """

    name = "oscar_closed_fim"

    def __init__(self, item_vars: list[EngineVar], minsup: int, data: PatternDataset) -> None:
        super().__init__()
        self.item_vars = item_vars
        self.minsup = minsup
        self.data = data
        self.variables = tuple(item_vars)
        self._n_items = data.n_items
        self._n_trans = data.n_transactions

    def setup(self, store: DomainStore, trail: Trail) -> None:
        vertical = self.data.as_vertical()
        self._coverage = ReversibleSparseBitset(self._n_trans, range(self._n_trans))
        self._columns = [self._coverage.create_column(vertical[i]) for i in range(self._n_items)]
        self._unbound_indices = list(range(len(self.item_vars)))
        self._n_unbound = len(self.item_vars)

        for var in self.item_vars:
            self.watch_bind(var)

        trail.save_reversible(self._make_restore_fn())
        self._do_propagate(store, trail)

    def propagate(self, store: DomainStore, trail: Trail) -> None:
        self._do_propagate(store, trail)

    def _do_propagate(self, store: DomainStore, trail: Trail) -> None:
        saved = self._coverage.save()
        trail.save_reversible(self._make_restore_fn_from(saved))

        self._coverage.clear_collected()
        cover_changed = False
        n_u = self._n_unbound

        # Process newly bound-to-true variables
        i = n_u
        while i > 0:
            i -= 1
            idx = self._unbound_indices[i]
            var = self.item_vars[idx]
            if var.is_bound and var.value == 1:
                cover_changed |= self._coverage.intersect_with(self._columns[idx])
                n_u = self._remove_item(i, n_u, idx)

        cardinality = self._coverage.count()

        if cardinality < self.minsup:
            raise InconsistencyError(
                f"ClosedFIM: coverage {cardinality} < minsup {self.minsup}"
            )

        # Prune: enforce closure
        n_u = self._prune_closed(n_u, cardinality, store)
        self._n_unbound = n_u

    def _prune_closed(self, n_u_bound: int, cardinality: int, store: DomainStore) -> int:
        """Prune and enforce closure."""
        n_u = n_u_bound
        i = n_u
        while i > 0:
            i -= 1
            idx = self._unbound_indices[i]
            var = self.item_vars[idx]
            card_idx = self._coverage.intersect_count(self._columns[idx], self.minsup)

            if var.is_bound and var.value == 0:
                # Item explicitly excluded: if it's in closure, fail
                if cardinality - card_idx <= 0:
                    raise InconsistencyError(
                        f"ClosedFIM: excluded item {idx} is in closure"
                    )
                elif card_idx < self.minsup:
                    n_u = self._remove_item(i, n_u, idx)
            else:
                # Unbound item
                if card_idx < self.minsup:
                    # Item infrequent -> exclude
                    n_u = self._remove_item(i, n_u, idx)
                    store.assign(var, 0)
                elif cardinality - card_idx <= 0:
                    # Item in closure -> must include
                    n_u = self._remove_item(i, n_u, idx)
                    store.assign(var, 1)
        return n_u

    def _remove_item(self, position: int, n_u: int, idx: int) -> int:
        last_u = n_u - 1
        self._unbound_indices[position] = self._unbound_indices[last_u]
        self._unbound_indices[last_u] = idx
        return last_u

    def _make_restore_fn(self) -> callable:
        saved_words = self._coverage.words[:]
        saved_nz = self._coverage.non_zero_idx[:self._coverage.n_non_zero]
        saved_n_nz = self._coverage.n_non_zero
        saved_unbound = self._unbound_indices[:]
        saved_n_u = self._n_unbound

        def restore():
            self._coverage.restore((saved_words, saved_nz, saved_n_nz))
            self._unbound_indices = saved_unbound[:]
            self._n_unbound = saved_n_u

        return restore

    def _make_restore_fn_from(self, coverage_snap: tuple) -> callable:
        saved_unbound = self._unbound_indices[:]
        saved_n_u = self._n_unbound

        def restore():
            self._coverage.restore(coverage_snap)
            self._unbound_indices = saved_unbound[:]
            self._n_unbound = saved_n_u

        return restore
