"""
CPMpy GlobalConstraint classes for Oscar ML constraints.

These are valid CPMpy expressions that can be posted to a Model.
They require CPM_oscar_ml to solve natively; decomposition raises by default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import cpmpy as cp
from cpmpy.expressions.globalconstraints import GlobalConstraint

from cp4dm_cpmpy_oscar_ml.exceptions import NotSupportedError

if TYPE_CHECKING:
    from cp4dm_cpmpy_oscar_ml.core.data import PatternDataset


class FrequentItemset(GlobalConstraint):
    """
    Oscar ML FIM global constraint.

    Enforces that the selected items form a frequent itemset with support >= minsup.

    Args:
        item_vars: Array of BoolVars (one per item).
        minsup: Minimum support threshold.
        data: The PatternDataset.
    """

    def __init__(self, item_vars: Sequence, minsup: int, data: PatternDataset) -> None:
        self.item_vars = tuple(item_vars)
        self.minsup = minsup
        self.data = data
        super().__init__("oscar_fim", list(item_vars))

    def decompose(self) -> Any:
        raise NotSupportedError(
            "FrequentItemset is an Oscar ML native global. "
            "Use CPM_oscar_ml or explicitly request a reference decomposition."
        )

    def __repr__(self) -> str:
        return f"FrequentItemset(minsup={self.minsup}, n_items={len(self.item_vars)})"


class ClosedFrequentItemset(GlobalConstraint):
    """
    Oscar ML ClosedFIM global constraint.

    Enforces that the selected items form a closed frequent itemset.
    """

    def __init__(self, item_vars: Sequence, minsup: int, data: PatternDataset) -> None:
        self.item_vars = tuple(item_vars)
        self.minsup = minsup
        self.data = data
        super().__init__("oscar_closed_fim", list(item_vars))

    def decompose(self) -> Any:
        raise NotSupportedError(
            "ClosedFrequentItemset is an Oscar ML native global. "
            "Use CPM_oscar_ml or explicitly request a reference decomposition."
        )

    def __repr__(self) -> str:
        return f"ClosedFrequentItemset(minsup={self.minsup}, n_items={len(self.item_vars)})"


class CoverSize(GlobalConstraint):
    """
    Oscar ML CoverSize global constraint.

    Links item selection to a support variable: Sup = |coverage(selected items)|.
    """

    def __init__(self, item_vars: Sequence, support_var: Any, data: PatternDataset) -> None:
        self.item_vars = tuple(item_vars)
        self.support_var = support_var
        self.data = data
        super().__init__("oscar_cover_size", list(item_vars) + [support_var])

    def decompose(self) -> Any:
        raise NotSupportedError(
            "CoverSize is an Oscar ML native global. "
            "Use CPM_oscar_ml or explicitly request a reference decomposition."
        )

    def __repr__(self) -> str:
        return f"CoverSize(n_items={len(self.item_vars)})"


class CoverClosure(GlobalConstraint):
    """
    Oscar ML CoverClosure global constraint.

    Enforces closure: if adding an item doesn't reduce support, it must be included.
    """

    def __init__(self, item_vars: Sequence, support_var: Any, data: PatternDataset) -> None:
        self.item_vars = tuple(item_vars)
        self.support_var = support_var
        self.data = data
        super().__init__("oscar_cover_closure", list(item_vars) + [support_var])

    def decompose(self) -> Any:
        raise NotSupportedError(
            "CoverClosure is an Oscar ML native global. "
            "Use CPM_oscar_ml or explicitly request a reference decomposition."
        )

    def __repr__(self) -> str:
        return f"CoverClosure(n_items={len(self.item_vars)})"
