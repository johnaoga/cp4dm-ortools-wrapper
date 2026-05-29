"""
Core data structures for transaction databases.

Mirrors Oscar's Dataset/Transaction classes while preserving item indexing conventions:
- Item 0 is epsilon/dummy where relevant.
- Real items start at 1.
- nbItem = max_item_id + 1 (one past the largest real item).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from cp4dm_cpmpy_oscar_ml.pattern_mining.formats import BaseDataFormat


@dataclass(frozen=True, slots=True)
class Transaction:
    """A single transaction (itemset or sequence)."""

    data: tuple[int, ...] = ()
    label: int = 1
    time: tuple[int, ...] = ()


class PatternDataset:
    """
    Generic representation of a transaction database (TDB).

    Attributes:
        name: Dataset name (typically derived from filename).
        records: All transactions.
        n_transactions: Number of transactions.
        n_items: One past the largest item id (Oscar's nbItem convention).
        item_names: Human-readable item names indexed by item id.
        used_format: The format used to load the data.
    """

    def __init__(self, name: str, records: Sequence[Transaction], n_items: int | None = None) -> None:
        self.name = name
        self.records = tuple(records)
        self.n_transactions = len(self.records)
        if n_items is not None:
            self.n_items = n_items
        else:
            max_item = 0
            for t in self.records:
                if t.data:
                    max_item = max(max_item, max(t.data))
            self.n_items = max_item + 1
        self.item_names: list[str] = [str(i) for i in range(self.n_items)]
        self.used_format: BaseDataFormat | None = None

    # --- Factory methods ---

    @classmethod
    def from_file(cls, path: str | Path, format: BaseDataFormat | None = None) -> PatternDataset:
        """Load a dataset from a file using the given format."""
        from cp4dm_cpmpy_oscar_ml.pattern_mining.formats import TdbFormat as _TdbFormat

        if format is None:
            format = _TdbFormat()
        path = Path(path)
        lines = path.read_text().splitlines()
        records = format.read_lines(lines)
        name = path.stem
        ds = cls(name, records)
        ds.used_format = format
        if format.with_item_names_header:
            ds.item_names = format.read_header(lines, ds.n_items)
        return ds

    @classmethod
    def from_transactions(cls, raw: Sequence[Sequence[int]], n_items: int | None = None) -> PatternDataset:
        """Create dataset from raw integer arrays."""
        records = [Transaction(data=tuple(r)) for r in raw]
        return cls("Dataset", records, n_items=n_items)

    # --- Query methods ---

    def as_vertical(self) -> list[set[int]]:
        """
        Convert to vertical representation.

        Returns list indexed by item_id where each entry is the set of
        transaction indices containing that item.
        """
        vertical: list[set[int]] = [set() for _ in range(self.n_items)]
        for tid, trans in enumerate(self.records):
            for item in trans.data:
                vertical[item].add(tid)
        return vertical

    def get_data(self) -> list[tuple[int, ...]]:
        """Return list of item arrays per transaction."""
        return [t.data for t in self.records]

    def get_time(self) -> list[tuple[int, ...]]:
        """Return time arrays; generate positional indices if no times provided."""
        result = []
        for t in self.records:
            if t.time:
                result.append(t.time)
            elif t.data:
                result.append(tuple(range(1, len(t.data) + 1)))
            else:
                result.append(())
        return result

    def get_labels(self) -> list[int]:
        """Return labels per transaction."""
        return [t.label for t in self.records]

    def split_by_label(self) -> tuple[PatternDataset, PatternDataset]:
        """Split into positive (label=1) and negative (label=0) datasets."""
        pos = [t for t in self.records if t.label == 1]
        neg = [t for t in self.records if t.label == 0]
        ds_pos = PatternDataset(f"{self.name}:1", pos, n_items=self.n_items)
        ds_neg = PatternDataset(f"{self.name}:0", neg, n_items=self.n_items)
        return ds_pos, ds_neg

    def density(self) -> float:
        """Compute dataset density."""
        if self.n_transactions == 0 or self.n_items <= 1:
            return 0.0
        total_items = sum(len(t.data) for t in self.records)
        return total_items / (self.n_transactions * (self.n_items - 1))

    def pattern_to_string(self, pattern: Sequence[int], sep: str = " ") -> str:
        """Convert a pattern (list of item ids) to a human-readable string."""
        return sep.join(self.item_names[i] for i in pattern)

    def __repr__(self) -> str:
        return f"PatternDataset(name={self.name!r}, n_trans={self.n_transactions}, n_items={self.n_items})"
