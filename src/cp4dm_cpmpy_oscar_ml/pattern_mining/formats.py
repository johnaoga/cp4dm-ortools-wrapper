"""
File format readers for pattern mining datasets.

Ports Oscar's FileFormat hierarchy:
- TdbFormat: space-separated items per line.
- TdbWithLabelFormat: items + trailing label.
- SpadeFormat: SPADE columnar format with timestamps.
- BSpadeFormat: SPADE with header.
- SpmfFormat: SPMF sequential format.
- SpmfWithHeaderFormat: SPMF with @ITEM header lines.
- SpmfWithTimeFormat: SPMF with <time> annotations.
- LongSequenceFormat: single long sequence split across lines.
- LongSequenceTimeFormat: item-time columns.
- ProteinLongSequenceFormat: FASTA protein sequences.
- LongSequenceWithNameAndTimeFormat: named symbols with timestamps.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Sequence

from cp4dm_cpmpy_oscar_ml.core.data import Transaction
from cp4dm_cpmpy_oscar_ml.exceptions import InvalidFormatError


class BaseDataFormat(ABC):
    """Abstract base for file format readers."""

    extension: str = ".txt"
    separator: str = r"\s+"
    with_label: bool = False
    with_item_names_header: bool = False
    n_skip: int = 0

    def read_lines(self, lines: list[str]) -> list[Transaction]:
        """Parse all lines into Transactions."""
        data_lines = lines[self.n_skip:]
        return [self.read_line(line) for line in data_lines if line.strip()]

    @abstractmethod
    def read_line(self, line: str) -> Transaction:
        ...

    def read_header(self, lines: list[str], n_items: int) -> list[str]:
        """Parse header to extract item names. Override in subclasses."""
        return [str(i) for i in range(n_items)]


class SparseFormat(BaseDataFormat):
    """Space-separated item ids per line, optionally with trailing label."""

    extension = ".txt"
    separator = r"\s+"

    def read_line(self, line: str) -> Transaction:
        parts = re.split(self.separator, line.strip())
        if not parts or parts == [""]:
            return Transaction()
        values = [int(x) for x in parts if x]
        if self.with_label:
            return Transaction(data=tuple(sorted(values[:-1])), label=values[-1])
        return Transaction(data=tuple(values))


class TdbFormat(SparseFormat):
    """Transaction database format: space-separated item ids."""
    pass


class TdbWithLabelFormat(SparseFormat):
    """Transaction database with trailing class label."""

    def __init__(self) -> None:
        super().__init__()
        self.with_label = True


class SpadeFormat(BaseDataFormat):
    """SPADE columnar format: sid time size item."""

    extension = ".sp"
    separator = r"\s+"

    POS_SID = 0
    POS_EID = 1
    POS_SIZ = 2
    POS_ITM = 3

    def read_line(self, line: str) -> Transaction:
        raise InvalidFormatError("SpadeFormat uses read_lines, not read_line")

    def read_lines(self, lines: list[str]) -> list[Transaction]:
        data_lines = [line.strip() for line in lines[self.n_skip:] if line.strip()]
        rows = [list(map(int, re.split(self.separator, line))) for line in data_lines]
        if not rows:
            return []
        max_sid = rows[-1][self.POS_SID]
        transactions = []
        for sid in range(1, max_sid + 1):
            sid_rows = [r for r in rows if r[self.POS_SID] == sid]
            items = tuple(r[self.POS_ITM] for r in sid_rows)
            times = tuple(r[self.POS_EID] for r in sid_rows)
            transactions.append(Transaction(data=items, time=times))
        return transactions


class BSpadeFormat(BaseDataFormat):
    """SPADE format with header containing item names."""

    extension = ".b"
    separator = r"\s+"
    with_item_names_header = True
    n_skip = 2

    POS_SID = 0
    POS_EID = 1
    POS_SIZ = 2
    POS_ITM = 3

    def read_line(self, line: str) -> Transaction:
        raise InvalidFormatError("BSpadeFormat uses read_lines, not read_line")

    def read_lines(self, lines: list[str]) -> list[Transaction]:
        data_lines = [line.strip() for line in lines[self.n_skip:] if line.strip()]
        rows = [list(map(int, re.split(self.separator, line))) for line in data_lines]
        if not rows:
            return []
        max_sid = rows[-1][self.POS_SID]
        transactions = []
        for sid in range(1, max_sid + 1):
            sid_rows = [r for r in rows if r[self.POS_SID] == sid]
            items = tuple(r[self.POS_ITM] for r in sid_rows)
            times = tuple(r[self.POS_EID] for r in sid_rows)
            transactions.append(Transaction(data=items, time=times))
        return transactions

    def read_header(self, lines: list[str], n_items: int) -> list[str]:
        header_line = lines[1].strip()
        # Format: [0, A, B, C, D]
        inner = header_line.lstrip("[").rstrip("]")
        parts = [p.strip() for p in inner.split(",")]
        return parts[:n_items] if len(parts) >= n_items else parts + [""] * (n_items - len(parts))


class SpmfFormat(BaseDataFormat):
    """SPMF sequential format: items separated by ' -1 ', ending with ' -1 -2'."""

    extension = ".spmf"
    separator = " -1 "

    def read_line(self, line: str) -> Transaction:
        cleaned = line.strip().removesuffix("-2").strip().removesuffix("-1").strip()
        if not cleaned:
            return Transaction()
        parts = cleaned.split(" -1 ")
        items = tuple(int(p.strip()) for p in parts if p.strip())
        return Transaction(data=items)


class SpmfWithHeaderFormat(BaseDataFormat):
    """SPMF format with @ITEM header lines."""

    extension = ".spmf"
    separator = " -1 "
    with_item_names_header = True

    def read_line(self, line: str) -> Transaction:
        raise InvalidFormatError("SpmfWithHeaderFormat uses read_lines, not read_line")

    def read_lines(self, lines: list[str]) -> list[Transaction]:
        header_lines = [l for l in lines if l.strip().startswith("@")]
        data_lines = [l for l in lines if not l.strip().startswith("@") and l.strip()]
        transactions = []
        for line in data_lines:
            cleaned = line.strip().removesuffix("-2").strip().removesuffix("-1").strip()
            if not cleaned:
                continue
            parts = cleaned.split(" -1 ")
            # Each part may contain "<time> item" or just "item"
            items = []
            times = []
            for p in parts:
                nums = re.findall(r"\d+", p)
                if len(nums) >= 2:
                    times.append(int(nums[0]))
                    items.append(int(nums[-1]))
                elif len(nums) == 1:
                    items.append(int(nums[0]))
            t = Transaction(data=tuple(items), time=tuple(times) if times else ())
            transactions.append(t)
        return transactions

    def read_header(self, lines: list[str], n_items: int) -> list[str]:
        out = [""] * (n_items + 1)
        for line in lines:
            line = line.strip()
            if line.startswith("@ITEM") or line.startswith("@ ITEM"):
                parts = line.replace("@ ITEM", "@ITEM").split("=")
                if len(parts) >= 3:
                    idx = int(parts[1])
                    name = parts[2]
                    if 0 <= idx < len(out):
                        out[idx] = name
        return out[:n_items]


class SpmfWithTimeFormat(BaseDataFormat):
    """SPMF format with <time> annotations: <2> 1 -1 <5> 2 -1 -2."""

    extension = ".spmf"
    separator = " -1 "

    def read_line(self, line: str) -> Transaction:
        raise InvalidFormatError("SpmfWithTimeFormat uses read_lines, not read_line")

    def read_lines(self, lines: list[str]) -> list[Transaction]:
        data_lines = [l for l in lines[self.n_skip:] if l.strip()]
        transactions = []
        for line in data_lines:
            cleaned = line.strip().removesuffix("-2").strip().removesuffix("-1").strip()
            if not cleaned:
                continue
            parts = cleaned.split(" -1 ")
            items = []
            times = []
            for p in parts:
                nums = re.findall(r"\d+", p.strip())
                if len(nums) >= 2:
                    times.append(int(nums[0]))
                    items.append(int(nums[1]))
                elif len(nums) == 1:
                    items.append(int(nums[0]))
            transactions.append(Transaction(data=tuple(items), time=tuple(times) if times else ()))
        return transactions


class LongSequenceFormat(BaseDataFormat):
    """Long sequence: all items across all lines form one sequence."""

    extension = ".txt"
    separator = r"\s+"

    def read_line(self, line: str) -> Transaction:
        raise InvalidFormatError("LongSequenceFormat uses read_lines, not read_line")

    def read_lines(self, lines: list[str]) -> list[Transaction]:
        all_items: list[int] = []
        for line in lines:
            parts = re.split(self.separator, line.strip())
            for p in parts:
                if p:
                    all_items.append(int(p))
        return [Transaction(data=tuple(all_items))]


class LongSequenceTimeFormat(BaseDataFormat):
    """Long sequence with time: each line is 'item time'."""

    extension = ".txt"
    separator = r"\s+"

    def read_line(self, line: str) -> Transaction:
        raise InvalidFormatError("LongSequenceTimeFormat uses read_lines, not read_line")

    def read_lines(self, lines: list[str]) -> list[Transaction]:
        items: list[int] = []
        times: list[int] = []
        for line in lines:
            parts = re.split(self.separator, line.strip())
            if len(parts) >= 2:
                items.append(int(parts[0]))
                times.append(int(parts[1]))
        return [Transaction(data=tuple(items), time=tuple(times))]


class ProteinLongSequenceFormat(BaseDataFormat):
    """FASTA protein sequences: characters mapped to integers (A=1, B=2, ...)."""

    extension = ".fasta"
    separator = ""
    with_item_names_header = True
    n_skip = 1

    def read_line(self, line: str) -> Transaction:
        raise InvalidFormatError("ProteinLongSequenceFormat uses read_lines, not read_line")

    def read_lines(self, lines: list[str]) -> list[Transaction]:
        all_items: list[int] = []
        for line in lines:
            for ch in line.strip():
                if ch.isalpha():
                    all_items.append(ord(ch.upper()) - ord("A") + 1)
        return [Transaction(data=tuple(all_items))]

    def read_header(self, lines: list[str], n_items: int) -> list[str]:
        names = ["eps."] + [chr(ord("A") + i) for i in range(26)]
        return names[:n_items]


class LongSequenceWithNameAndTimeFormat(BaseDataFormat):
    """Named symbols with timestamps: each line is 'symbol time'."""

    extension = ".txt"
    separator = " "
    with_item_names_header = True
    n_skip = 1

    def __init__(self) -> None:
        super().__init__()
        self._rev_map: dict[str, int] = {}
        self._counter = 1

    def read_line(self, line: str) -> Transaction:
        raise InvalidFormatError("LongSequenceWithNameAndTimeFormat uses read_lines, not read_line")

    def read_lines(self, lines: list[str]) -> list[Transaction]:
        items: list[int] = []
        times: list[int] = []
        for line in lines:
            parts = line.strip().split(self.separator)
            if len(parts) >= 2:
                symbol = parts[0]
                timestamp = int(parts[1])
                if symbol not in self._rev_map:
                    self._rev_map[symbol] = self._counter
                    self._counter += 1
                items.append(self._rev_map[symbol])
                times.append(timestamp)
        return [Transaction(data=tuple(items), time=tuple(times))]

    def read_header(self, lines: list[str], n_items: int) -> list[str]:
        inv = {v: k for k, v in self._rev_map.items()}
        return ["eps."] + [inv.get(i, str(i)) for i in range(1, n_items)]
