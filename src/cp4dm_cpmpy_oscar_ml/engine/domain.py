"""
Domain store for engine variables.

Manages mutable domains for Boolean and integer variables during search.
Tracks changes for event scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any

from cp4dm_cpmpy_oscar_ml.exceptions import InconsistencyError


class VarType(IntEnum):
    BOOL = auto()
    INT = auto()


@dataclass(slots=True)
class EngineVar:
    """An engine-level variable with a mutable domain."""

    id: int
    name: str
    var_type: VarType
    lb: int  # current lower bound
    ub: int  # current upper bound
    # For sparse domains (not needed for booleans or simple intervals):
    removed: set[int] = field(default_factory=set)
    # Reference back to the CPMpy variable (set by solver interface)
    cpmpy_var: Any = None

    @property
    def is_bound(self) -> bool:
        return self.lb == self.ub

    @property
    def value(self) -> int:
        if not self.is_bound:
            raise ValueError(f"Variable {self.name} is not bound")
        return self.lb

    @property
    def size(self) -> int:
        return self.ub - self.lb + 1 - len(self.removed)

    def contains(self, val: int) -> bool:
        return self.lb <= val <= self.ub and val not in self.removed


class DomainEvent(IntEnum):
    BIND = auto()
    BOUNDS = auto()
    DOMAIN = auto()


class DomainStore:
    """
    Maps engine variables to their current domains.
    Tracks domain changes for propagation event scheduling.
    """

    def __init__(self) -> None:
        self._vars: list[EngineVar] = []
        self._var_map: dict[int, EngineVar] = {}
        self._changes: list[tuple[EngineVar, DomainEvent]] = []
        self._next_id = 0

    def new_bool_var(self, name: str = "") -> EngineVar:
        """Create a Boolean variable (domain {0, 1})."""
        v = EngineVar(id=self._next_id, name=name or f"b{self._next_id}", var_type=VarType.BOOL, lb=0, ub=1)
        self._vars.append(v)
        self._var_map[v.id] = v
        self._next_id += 1
        return v

    def new_int_var(self, lb: int, ub: int, name: str = "") -> EngineVar:
        """Create an integer variable with interval domain [lb, ub]."""
        v = EngineVar(id=self._next_id, name=name or f"x{self._next_id}", var_type=VarType.INT, lb=lb, ub=ub)
        self._vars.append(v)
        self._var_map[v.id] = v
        self._next_id += 1
        return v

    @property
    def variables(self) -> list[EngineVar]:
        return self._vars

    def get_var(self, var_id: int) -> EngineVar:
        return self._var_map[var_id]

    # --- Domain operations ---

    def assign(self, var: EngineVar, value: int) -> None:
        """Assign variable to a single value."""
        if not var.contains(value):
            raise InconsistencyError(f"Cannot assign {var.name}={value}: not in domain")
        old_lb, old_ub = var.lb, var.ub
        var.lb = value
        var.ub = value
        var.removed.clear()
        if old_lb != value or old_ub != value:
            self._changes.append((var, DomainEvent.BIND))

    def remove_value(self, var: EngineVar, value: int) -> None:
        """Remove a single value from the domain."""
        if not var.contains(value):
            return  # already absent
        if var.is_bound and var.lb == value:
            raise InconsistencyError(f"Cannot remove last value {value} from {var.name}")
        if value == var.lb:
            var.lb += 1
            while var.lb <= var.ub and var.lb in var.removed:
                var.removed.discard(var.lb)
                var.lb += 1
            if var.lb > var.ub:
                raise InconsistencyError(f"Domain of {var.name} is empty after removing {value}")
            event = DomainEvent.BIND if var.is_bound else DomainEvent.BOUNDS
        elif value == var.ub:
            var.ub -= 1
            while var.ub >= var.lb and var.ub in var.removed:
                var.removed.discard(var.ub)
                var.ub -= 1
            if var.lb > var.ub:
                raise InconsistencyError(f"Domain of {var.name} is empty after removing {value}")
            event = DomainEvent.BIND if var.is_bound else DomainEvent.BOUNDS
        else:
            var.removed.add(value)
            event = DomainEvent.BIND if var.size == 1 else DomainEvent.DOMAIN
        self._changes.append((var, event))

    def update_min(self, var: EngineVar, new_lb: int) -> None:
        """Tighten lower bound."""
        if new_lb <= var.lb:
            return
        if new_lb > var.ub:
            raise InconsistencyError(f"update_min({var.name}, {new_lb}) exceeds ub={var.ub}")
        var.lb = new_lb
        # Remove invalidated sparse entries
        var.removed = {v for v in var.removed if v > var.lb}
        while var.lb <= var.ub and var.lb in var.removed:
            var.removed.discard(var.lb)
            var.lb += 1
        if var.lb > var.ub:
            raise InconsistencyError(f"Domain of {var.name} is empty")
        event = DomainEvent.BIND if var.is_bound else DomainEvent.BOUNDS
        self._changes.append((var, event))

    def update_max(self, var: EngineVar, new_ub: int) -> None:
        """Tighten upper bound."""
        if new_ub >= var.ub:
            return
        if new_ub < var.lb:
            raise InconsistencyError(f"update_max({var.name}, {new_ub}) below lb={var.lb}")
        var.ub = new_ub
        var.removed = {v for v in var.removed if v < var.ub}
        while var.ub >= var.lb and var.ub in var.removed:
            var.removed.discard(var.ub)
            var.ub -= 1
        if var.lb > var.ub:
            raise InconsistencyError(f"Domain of {var.name} is empty")
        event = DomainEvent.BIND if var.is_bound else DomainEvent.BOUNDS
        self._changes.append((var, event))

    # --- Change tracking ---

    def drain_changes(self) -> list[tuple[EngineVar, DomainEvent]]:
        """Return and clear pending changes."""
        changes = self._changes
        self._changes = []
        return changes

    def has_changes(self) -> bool:
        return len(self._changes) > 0

    # --- Snapshot for trail ---

    def snapshot_var(self, var: EngineVar) -> tuple[int, int, frozenset[int]]:
        """Take a snapshot of a variable's domain."""
        return (var.lb, var.ub, frozenset(var.removed))

    def restore_var(self, var: EngineVar, snap: tuple[int, int, frozenset[int]]) -> None:
        """Restore a variable's domain from a snapshot."""
        var.lb, var.ub = snap[0], snap[1]
        var.removed = set(snap[2])
