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


class ZeroDiagonalConvexScore(GlobalConstraint):
    """
    Oscar ML ZDC global constraint.

    Enforces a convex discriminative score f(pos_sup, neg_sup) >= score_var.lb.
    """

    def __init__(self, pos_var: Any, neg_var: Any, score_var: Any,
                 n_pos: int, n_neg: int, score_fn: Any) -> None:
        self.pos_var = pos_var
        self.neg_var = neg_var
        self.score_var = score_var
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.score_fn = score_fn
        super().__init__("oscar_zdc", [pos_var, neg_var, score_var])

    def decompose(self) -> Any:
        raise NotSupportedError(
            "ZeroDiagonalConvexScore is an Oscar ML native global. "
            "Use CPM_oscar_ml."
        )

    def __repr__(self) -> str:
        return f"ZeroDiagonalConvexScore(n_pos={self.n_pos}, n_neg={self.n_neg})"


class SequentialPattern(GlobalConstraint):
    """
    Oscar ML SPM global constraint (PPIC/PPDC/PPmixed).

    Enforces that the pattern variable array P forms a frequent sequence.
    """

    def __init__(self, pattern_vars: Sequence, minsup: int, data: "PatternDataset",
                 method: str = "ppic") -> None:
        self.pattern_vars = tuple(pattern_vars)
        self.minsup = minsup
        self.data = data
        self.method = method
        super().__init__("oscar_ppic", list(pattern_vars))

    def decompose(self) -> Any:
        raise NotSupportedError(
            "SequentialPattern is an Oscar ML native global. Use CPM_oscar_ml."
        )

    def __repr__(self) -> str:
        return f"SequentialPattern(minsup={self.minsup}, len={len(self.pattern_vars)}, method={self.method})"


class FrequentEpisode(GlobalConstraint):
    """
    Oscar ML FEM global constraint (EpisodeSupport).

    Enforces that the episode pattern array P is frequent in a long sequence.
    """

    def __init__(self, pattern_vars: Sequence, minsup: int, data: "PatternDataset") -> None:
        self.pattern_vars = tuple(pattern_vars)
        self.minsup = minsup
        self.data = data
        super().__init__("oscar_episode_support", list(pattern_vars))

    def decompose(self) -> Any:
        raise NotSupportedError(
            "FrequentEpisode is an Oscar ML native global. Use CPM_oscar_ml."
        )

    def __repr__(self) -> str:
        return f"FrequentEpisode(minsup={self.minsup}, len={len(self.pattern_vars)})"


class TreeCoverSizeSR(GlobalConstraint):
    """
    Oscar ML classification tree coverage constraint.

    Links take/reject decision arrays to a support variable.
    """

    def __init__(self, take_vars: Sequence, reject_vars: Sequence,
                 support_var: Any, data: "ClassificationDataset") -> None:
        self.take_vars = tuple(take_vars)
        self.reject_vars = tuple(reject_vars)
        self.support_var = support_var
        self.data = data
        super().__init__("oscar_tree_cover_size_sr",
                         list(take_vars) + list(reject_vars) + [support_var])

    def decompose(self) -> Any:
        raise NotSupportedError(
            "TreeCoverSizeSR is an Oscar ML native global. Use CPM_oscar_ml."
        )

    def __repr__(self) -> str:
        return f"TreeCoverSizeSR(n_take={len(self.take_vars)}, n_reject={len(self.reject_vars)})"


class SplitPossible(GlobalConstraint):
    """Oscar ML SplitPossible classification tree constraint."""

    def __init__(self, decision: Any, count_pos: Any, count_neg: Any,
                 count_sum: Any, threshold: int) -> None:
        self.decision = decision
        self.count_pos = count_pos
        self.count_neg = count_neg
        self.count_sum = count_sum
        self.threshold = threshold
        super().__init__("oscar_split_possible",
                         [decision, count_pos, count_neg, count_sum])

    def decompose(self) -> Any:
        raise NotSupportedError(
            "SplitPossible is an Oscar ML native global. Use CPM_oscar_ml."
        )

    def __repr__(self) -> str:
        return f"SplitPossible(threshold={self.threshold})"


class SplitUseful(GlobalConstraint):
    """Oscar ML SplitUseful classification tree constraint."""

    def __init__(self, decision: Any, mini_sum: Any, count_pos: Any,
                 count_neg: Any, error_upper_bound: float) -> None:
        self.decision = decision
        self.mini_sum = mini_sum
        self.count_pos = count_pos
        self.count_neg = count_neg
        self.error_upper_bound = error_upper_bound
        super().__init__("oscar_split_useful",
                         [decision, mini_sum, count_pos, count_neg])

    def decompose(self) -> Any:
        raise NotSupportedError(
            "SplitUseful is an Oscar ML native global. Use CPM_oscar_ml."
        )

    def __repr__(self) -> str:
        return f"SplitUseful(error_ub={self.error_upper_bound})"
