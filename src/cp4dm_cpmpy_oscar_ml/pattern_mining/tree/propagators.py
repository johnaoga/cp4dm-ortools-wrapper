"""
Classification tree propagators: ports of Oscar ML's CoverSizeSR,
CstDummy, CstDummyEnd, CstSplitPossible, and CstSplitUseful constraints.

Reference: "Learning Optimal Decision Trees Using CP"
  H. Verhaeghe, S. Nijssen, C-G Quimpert, G. Pesant, P. Schaus
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


class CoverSizeSRPropagator(Propagator):
    """
    Port of Oscar's CoverSizeSR.scala.

    Maintains coverage bitset for a binary decision tree node.
    `take_vars` (I)  — items to include in the node (coverage intersects).
    `reject_vars` (I2) — items to exclude from the node (coverage intersects complement).
    `sup_var`    (Sup) — current node coverage count.

    Propagation:
      - When a take/reject var is bound, intersect coverage accordingly.
      - Update Sup.ub = |coverage|.
      - When all vars are bound, fix Sup.lb = Sup.ub.
      - Prune items from unbound vars that would violate Sup bounds.
    """

    name = "oscar_tree_cover_size_sr"

    def __init__(
        self,
        take_vars: list[EngineVar],
        reject_vars: list[EngineVar],
        sup_var: EngineVar,
        data: "PatternDataset",
    ) -> None:
        super().__init__()
        self.take_vars = take_vars
        self.reject_vars = reject_vars
        self.sup_var = sup_var
        self.data = data
        self.variables = tuple(take_vars + reject_vars + [sup_var])

        n_trans = data.n_transactions
        n_items = data.n_items

        # Build vertical bitsets: columns[item] = set of transaction ids containing item
        # ImmutableBitSet(n_words, values): n_words = ceil(n_trans / 64)
        n_words = (n_trans + 63) // 64
        vertical = data.as_vertical()
        all_tids = set(range(n_trans))
        self._columns: list[ImmutableBitSet] = []
        self._columns_inv: list[ImmutableBitSet] = []  # complement = transactions NOT containing item
        for item in range(n_items):
            item_tids = vertical[item]
            self._columns.append(ImmutableBitSet(n_words, item_tids))
            self._columns_inv.append(ImmutableBitSet(n_words, all_tids - item_tids))

        self._n_trans = n_trans
        self._n_items = n_items

        # Reversible coverage: starts as all transactions
        self._coverage = ReversibleSparseBitset(n_trans, set(range(n_trans)))

    def setup(self, store: DomainStore, trail: Trail) -> None:
        for var in self.take_vars:
            self.watch_bind(var)
        for var in self.reject_vars:
            self.watch_bind(var)
        self.watch_bounds(self.sup_var)
        self._do_propagate(store, trail)

    def propagate(self, store: DomainStore, trail: Trail) -> None:
        self._do_propagate(store, trail)

    def _do_propagate(self, store: DomainStore, trail: Trail) -> None:
        # Save and restore coverage state via trail
        snap = self._coverage.save()

        def _restore():
            self._coverage.restore(snap)

        trail.save_reversible(_restore)

        # Process newly bound take vars (intersect with column)
        n_take_unbound = []
        for var in self.take_vars:
            if var.is_bound:
                item = var.value
                if 0 < item < self._n_items:
                    self._coverage.intersect_with(self._columns[item])
            else:
                n_take_unbound.append(var)

        # Process newly bound reject vars (intersect with complement column)
        n_reject_unbound = []
        for var in self.reject_vars:
            if var.is_bound:
                item = var.value
                if 0 < item < self._n_items:
                    self._coverage.intersect_with(self._columns_inv[item])
            else:
                n_reject_unbound.append(var)

        sup_ub = self._coverage.count()
        store.update_max(self.sup_var, sup_ub)

        # If all bound, fix lower bound too
        if not n_take_unbound and not n_reject_unbound:
            store.update_min(self.sup_var, sup_ub)
            return

        sup_lb = self.sup_var.lb

        # Prune: for each item, if adding it would drop coverage below sup_lb, remove it
        # from unbound take vars; if excluding it would drop coverage below sup_lb, remove
        # from unbound reject vars.
        for item in range(1, self._n_items):
            count = self._coverage.intersect_count(self._columns[item])
            left_count = sup_ub - count  # coverage if item is rejected

            # Adding item: coverage = count
            if count < sup_lb:
                for var in n_take_unbound:
                    store.remove_value(var, item)

            # Rejecting item: coverage = left_count
            if left_count < sup_lb:
                for var in n_reject_unbound:
                    store.remove_value(var, item)

        # Closure pruning when sup_ub == sup_lb: force items whose coverage == sup_ub into take
        if sup_ub == sup_lb:
            for item in range(1, self._n_items):
                # Item must be in all covered transactions (is a subset)
                col = self._columns[item]
                if not self._coverage.is_subset_of(col):
                    # Coverage is not subset of column → item not in all covered transactions
                    # → adding it would shrink coverage → must not add → remove from take
                    for var in n_take_unbound:
                        store.remove_value(var, item)
                # If coverage intersects column → some covered transactions have item → can't reject
                if self._coverage.intersect_count(col) > 0:
                    for var in n_reject_unbound:
                        store.remove_value(var, item)


class CstDummyPropagator(Propagator):
    """
    Port of Oscar's CstDummy.scala.

    When decision_parent is forced to 0 (leaf), both children decisions are forced to 0.
    When decision_parent > 0 (internal), both child sums must be > 0.
    """

    name = "oscar_dummy_node"

    def __init__(
        self,
        decision_parent: EngineVar,
        decision_child_left: EngineVar,
        decision_child_right: EngineVar,
        sum_child_left: EngineVar,
        sum_child_right: EngineVar,
    ) -> None:
        super().__init__()
        self.decision_parent = decision_parent
        self.decision_child_left = decision_child_left
        self.decision_child_right = decision_child_right
        self.sum_child_left = sum_child_left
        self.sum_child_right = sum_child_right
        self.variables = (decision_parent, decision_child_left, decision_child_right,
                          sum_child_left, sum_child_right)
        self._deactivated = False

    def setup(self, store: DomainStore, trail: Trail) -> None:
        self.watch_bounds(self.decision_parent)
        self._do_propagate(store, trail)

    def propagate(self, store: DomainStore, trail: Trail) -> None:
        if self._deactivated:
            return
        self._do_propagate(store, trail)

    def _do_propagate(self, store: DomainStore, trail: Trail) -> None:
        dp = self.decision_parent
        if dp.ub == 0:
            # Leaf: both children are also leaves (decision = 0)
            store.assign(self.decision_child_left, 0)
            store.assign(self.decision_child_right, 0)
            self._deactivated = True
        elif dp.lb > 0:
            # Internal node: children must have non-zero support
            store.remove_value(self.sum_child_right, 0)
            store.remove_value(self.sum_child_left, 0)
            self._deactivated = True


class CstDummyEndPropagator(Propagator):
    """
    Port of Oscar's CstDummyEnd.scala (leaf-level node).

    When decision_parent > 0, both child sums must be > 0.
    """

    name = "oscar_dummy_end_node"

    def __init__(
        self,
        decision_parent: EngineVar,
        sum_child_left: EngineVar,
        sum_child_right: EngineVar,
    ) -> None:
        super().__init__()
        self.decision_parent = decision_parent
        self.sum_child_left = sum_child_left
        self.sum_child_right = sum_child_right
        self.variables = (decision_parent, sum_child_left, sum_child_right)
        self._deactivated = False

    def setup(self, store: DomainStore, trail: Trail) -> None:
        self.watch_bounds(self.decision_parent)
        self._do_propagate(store, trail)

    def propagate(self, store: DomainStore, trail: Trail) -> None:
        if self._deactivated:
            return
        self._do_propagate(store, trail)

    def _do_propagate(self, store: DomainStore, trail: Trail) -> None:
        dp = self.decision_parent
        if dp.lb > 0:
            store.remove_value(self.sum_child_right, 0)
            store.remove_value(self.sum_child_left, 0)
            self._deactivated = True


class CstSplitPossiblePropagator(Propagator):
    """
    Port of Oscar's CstSplitPossible.scala.

    Enforces: a split is possible only if both partitions have >= threshold points,
    i.e., count_sum >= 2 * threshold.

    If count_pos.max == 0 or count_neg.max == 0: no split needed → decision = 0.
    If count_sum.max < 2*threshold: can't produce a valid split → decision = 0.
    If decision.lb > 0: split will happen → count_sum.lb >= 2*threshold.
    """

    name = "oscar_split_possible"

    def __init__(
        self,
        decision: EngineVar,
        count_pos: EngineVar,
        count_neg: EngineVar,
        count_sum: EngineVar,
        threshold: int,
    ) -> None:
        super().__init__()
        self.decision = decision
        self.count_pos = count_pos
        self.count_neg = count_neg
        self.count_sum = count_sum
        self.threshold = threshold
        self.threshold2 = 2 * threshold
        self.variables = (decision, count_pos, count_neg, count_sum)
        self._deactivated = False

    def setup(self, store: DomainStore, trail: Trail) -> None:
        self.watch_bounds(self.count_sum)
        self.watch_bounds(self.decision)
        self._do_propagate(store, trail)

    def propagate(self, store: DomainStore, trail: Trail) -> None:
        if self._deactivated:
            return
        self._do_propagate(store, trail)

    def _do_propagate(self, store: DomainStore, trail: Trail) -> None:
        if self.count_pos.ub == 0 or self.count_neg.ub == 0:
            store.assign(self.decision, 0)
            self._deactivated = True
        elif self.count_sum.ub < self.threshold2:
            store.assign(self.decision, 0)
            self._deactivated = True
        elif self.decision.lb > 0:
            store.update_min(self.count_sum, self.threshold2)
            self._deactivated = True


class CstSplitUsefulPropagator(Propagator):
    """
    Port of Oscar's CstSplitUseful.scala.

    Enforces: if a split decision is made, the minimum of the two partition sizes
    must exceed the current best error upper bound (i.e., the split must improve).

    If decision.lb > 0: min(count_pos, count_neg) > error_ub must hold.
    If decision.max == 0: deactivate.
    """

    name = "oscar_split_useful"

    def __init__(
        self,
        decision: EngineVar,
        mini_sum: EngineVar,
        count_pos: EngineVar,
        count_neg: EngineVar,
        error_ub: int,
    ) -> None:
        super().__init__()
        self.decision = decision
        self.mini_sum = mini_sum
        self.count_pos = count_pos
        self.count_neg = count_neg
        self.error_ub = error_ub
        self.variables = (decision, mini_sum, count_pos, count_neg)
        self._deactivated = False

    def setup(self, store: DomainStore, trail: Trail) -> None:
        self.watch_bounds(self.decision)
        self._do_propagate(store, trail)

    def propagate(self, store: DomainStore, trail: Trail) -> None:
        if self._deactivated:
            return
        self._do_propagate(store, trail)

    def _do_propagate(self, store: DomainStore, trail: Trail) -> None:
        if self.decision.lb > 0:
            # min(count_pos, count_neg) must be > error_ub
            # mini_sum.ub = min(count_pos.ub, count_neg.ub)
            mini_ub = min(self.count_pos.ub, self.count_neg.ub)
            store.update_max(self.mini_sum, mini_ub)
            # The split is only useful if mini_sum > error_ub
            if self.mini_sum.ub <= self.error_ub:
                raise InconsistencyError(
                    f"CstSplitUseful: min partition {self.mini_sum.ub} <= error_ub {self.error_ub}"
                )
            store.update_min(self.mini_sum, self.error_ub + 1)
            self._deactivated = True
        elif self.decision.ub == 0:
            self._deactivated = True
