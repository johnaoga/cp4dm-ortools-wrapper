"""
Structured result dataclasses for Oscar ML solvers.

These replace raw string-based outputs from Oscar ML examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Pattern:
    """A single frequent pattern (itemset or sequence)."""
    items: tuple[int, ...]
    support: int
    score: float | None = None

    def __repr__(self) -> str:
        score_str = f", score={self.score:.4f}" if self.score is not None else ""
        return f"Pattern(items={list(self.items)}, support={self.support}{score_str})"


@dataclass
class PatternResult:
    """Collection of patterns with search statistics."""
    patterns: list[Pattern] = field(default_factory=list)
    n_nodes: int = 0
    n_failures: int = 0
    runtime_s: float = 0.0
    status: str = "unknown"

    def __repr__(self) -> str:
        return (
            f"PatternResult({len(self.patterns)} patterns, "
            f"nodes={self.n_nodes}, failures={self.n_failures}, "
            f"runtime={self.runtime_s:.3f}s)"
        )

    def to_dataframe(self) -> Any:
        """Convert to pandas DataFrame (requires pandas)."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")
        return pd.DataFrame([
            {"items": list(p.items), "support": p.support, "score": p.score}
            for p in self.patterns
        ])

    def to_json(self) -> list[dict]:
        """Convert patterns to JSON-serialisable list."""
        return [
            {"items": list(p.items), "support": p.support, "score": p.score}
            for p in self.patterns
        ]


@dataclass
class DecisionTreeResult:
    """Result of optimal decision tree search."""
    tree: Any = None          # DecisionTreeModel instance
    cost: float = float("inf")
    completed: bool = False
    runtime_s: float = 0.0
    n_nodes: int = 0

    def predict(self, features: tuple[int, ...]) -> int:
        """Predict label for a feature vector."""
        if self.tree is None:
            raise ValueError("No tree available")
        return self.tree.predict(features)

    def to_string(self) -> str:
        if self.tree is None:
            return "<no tree>"
        return str(self.tree)
