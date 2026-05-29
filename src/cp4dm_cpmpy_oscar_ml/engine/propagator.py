"""
Propagator base class and event-based scheduling.

Each propagator declares which variables it watches and which events trigger it.
The propagation queue calls propagate() until fixpoint or failure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from cp4dm_cpmpy_oscar_ml.engine.domain import DomainEvent, DomainStore, EngineVar
from cp4dm_cpmpy_oscar_ml.engine.trail import Trail

if TYPE_CHECKING:
    pass


class Propagator(ABC):
    """
    Abstract propagator base class.

    Subclasses implement setup() (called once) and propagate() (called on events).
    """

    name: str = "propagator"

    def __init__(self) -> None:
        self.variables: tuple[EngineVar, ...] = ()
        self._watch_events: dict[int, set[DomainEvent]] = {}  # var_id -> events

    def watch(self, var: EngineVar, event: DomainEvent) -> None:
        """Register interest in an event on a variable."""
        if var.id not in self._watch_events:
            self._watch_events[var.id] = set()
        self._watch_events[var.id].add(event)

    def watch_bind(self, var: EngineVar) -> None:
        """Watch for when variable becomes bound."""
        self.watch(var, DomainEvent.BIND)

    def watch_bounds(self, var: EngineVar) -> None:
        """Watch for bounds changes."""
        self.watch(var, DomainEvent.BOUNDS)
        self.watch(var, DomainEvent.BIND)

    def is_triggered_by(self, var_id: int, event: DomainEvent) -> bool:
        """Check if this propagator should fire for the given var/event."""
        events = self._watch_events.get(var_id)
        if events is None:
            return False
        # BIND triggers anything that watches BIND, BOUNDS, or DOMAIN
        if event == DomainEvent.BIND:
            return bool(events)
        # BOUNDS triggers BOUNDS and DOMAIN watchers
        if event == DomainEvent.BOUNDS:
            return DomainEvent.BOUNDS in events or DomainEvent.DOMAIN in events
        # DOMAIN triggers only DOMAIN watchers
        return DomainEvent.DOMAIN in events

    @abstractmethod
    def setup(self, store: DomainStore, trail: Trail) -> None:
        """Called once when the propagator is posted. Register watches and do initial propagation."""
        ...

    @abstractmethod
    def propagate(self, store: DomainStore, trail: Trail) -> None:
        """Called when a watched event fires. Must raise InconsistencyError on failure."""
        ...

    @property
    def is_idempotent(self) -> bool:
        """If True, need not be re-queued after own propagation."""
        return True
