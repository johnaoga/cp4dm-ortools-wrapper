"""
cp4dm_cpmpy_oscar_ml - Oscar ML constraint propagation integrated with CPMpy.

This package provides:
- CPMpy global constraint classes for Oscar ML constraints (FIM, SPM, FEM, classification trees).
- A dedicated solver/search engine (CPM_oscar_ml) that handles Oscar ML globals natively.
- High-level solver classes for common pattern mining and classification tree workflows.
- Data loading and format handling for transaction databases, sequences, and episodes.

Oscar ML globals are NOT decomposed by default. They require the CPM_oscar_ml solver.
"""

from cp4dm_cpmpy_oscar_ml.api import FEMConfig, FIMConfig, FIMSolver, FrequentEpisodeSolver, SequentialPatternSolver, SPMConfig
from cp4dm_cpmpy_oscar_ml.core.data import PatternDataset, Transaction
from cp4dm_cpmpy_oscar_ml.core.result import Pattern, PatternResult
from cp4dm_cpmpy_oscar_ml.cpmpy_integration.globals import (
    ClosedFrequentItemset,
    CoverClosure,
    CoverSize,
    FrequentEpisode,
    FrequentItemset,
    SequentialPattern,
    ZeroDiagonalConvexScore,
)
from cp4dm_cpmpy_oscar_ml.cpmpy_integration.solver import CPM_oscar_ml
from cp4dm_cpmpy_oscar_ml.pattern_mining.formats import TdbFormat

__version__ = "0.1.0"

__all__ = [
    "Transaction",
    "PatternDataset",
    "Pattern",
    "PatternResult",
    "TdbFormat",
    "CPM_oscar_ml",
    "FrequentItemset",
    "ClosedFrequentItemset",
    "CoverSize",
    "CoverClosure",
    "ZeroDiagonalConvexScore",
    "SequentialPattern",
    "FrequentEpisode",
    "FIMConfig",
    "FIMSolver",
    "SPMConfig",
    "SequentialPatternSolver",
    "FEMConfig",
    "FrequentEpisodeSolver",
]
