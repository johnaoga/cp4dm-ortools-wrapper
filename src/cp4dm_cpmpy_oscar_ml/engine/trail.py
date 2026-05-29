"""
Trail (backtracking stack) for reversible state.

Implements push/pop semantics needed by DFS search: on each branch point we push
a level, on backtrack we pop and restore all saved state.
"""

from __future__ import annotations

from typing import Any, Callable

from cp4dm_cpmpy_oscar_ml.engine.domain import DomainStore, EngineVar


class Trail:
    """
    Trail-based reversible state manager.

    Each level saves domain snapshots and arbitrary reversible values.
    pop_level() restores everything saved at the current level.
    """

    def __init__(self, store: DomainStore) -> None:
        self._store = store
        # Stack of levels: each level is a list of (var, snapshot) pairs + reversible entries
        self._levels: list[_TrailLevel] = []
        self._magic: int = 0

    @property
    def magic(self) -> int:
        """Counter that increments on each push; used by reversible structures."""
        return self._magic

    @property
    def depth(self) -> int:
        return len(self._levels)

    def push_level(self) -> None:
        """Save state before branching."""
        self._magic += 1
        level = _TrailLevel()
        # Save all variable domains
        for var in self._store.variables:
            level.var_snapshots.append((var, self._store.snapshot_var(var)))
        self._levels.append(level)

    def pop_level(self) -> None:
        """Restore state on backtrack."""
        if not self._levels:
            raise RuntimeError("Trail is empty; cannot pop")
        level = self._levels.pop()
        # Restore variable domains
        for var, snap in level.var_snapshots:
            self._store.restore_var(var, snap)
        # Restore reversible values
        for restore_fn in reversed(level.reversible_entries):
            restore_fn()
        # Clear pending changes since we've restored
        self._store.drain_changes()

    def save_reversible(self, restore_fn: Callable[[], None]) -> None:
        """Register a restore callback at the current level."""
        if self._levels:
            self._levels[-1].reversible_entries.append(restore_fn)


class _TrailLevel:
    """Internal: one level of saved state."""

    __slots__ = ("var_snapshots", "reversible_entries")

    def __init__(self) -> None:
        self.var_snapshots: list[tuple[EngineVar, tuple[int, int, frozenset[int]]]] = []
        self.reversible_entries: list[Callable[[], None]] = []


# --- Reversible primitive types ---


class ReversibleInt:
    """An integer value that restores automatically on backtrack."""

    def __init__(self, trail: Trail, initial: int = 0) -> None:
        self._trail = trail
        self._value = initial
        self._saved_magic = -1

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, new_val: int) -> None:
        if new_val != self._value:
            self._save_if_needed()
            self._value = new_val

    def _save_if_needed(self) -> None:
        if self._trail.depth > 0 and self._saved_magic != self._trail.magic:
            old = self._value
            self._trail.save_reversible(lambda: self._restore(old))
            self._saved_magic = self._trail.magic

    def _restore(self, old_val: int) -> None:
        self._value = old_val
        self._saved_magic = -1


class ReversibleBool:
    """A boolean value that restores automatically on backtrack."""

    def __init__(self, trail: Trail, initial: bool = False) -> None:
        self._trail = trail
        self._value = initial
        self._saved_magic = -1

    @property
    def value(self) -> bool:
        return self._value

    @value.setter
    def value(self, new_val: bool) -> None:
        if new_val != self._value:
            self._save_if_needed()
            self._value = new_val

    def _save_if_needed(self) -> None:
        if self._trail.depth > 0 and self._saved_magic != self._trail.magic:
            old = self._value
            self._trail.save_reversible(lambda: self._restore(old))
            self._saved_magic = self._trail.magic

    def _restore(self, old_val: bool) -> None:
        self._value = old_val
        self._saved_magic = -1
