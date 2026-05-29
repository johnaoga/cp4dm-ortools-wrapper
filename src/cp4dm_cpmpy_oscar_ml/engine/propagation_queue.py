"""
Propagation queue: schedules propagators based on domain events, runs to fixpoint.
"""

from __future__ import annotations

from collections import deque

from cp4dm_cpmpy_oscar_ml.engine.domain import DomainEvent, DomainStore, EngineVar
from cp4dm_cpmpy_oscar_ml.engine.propagator import Propagator
from cp4dm_cpmpy_oscar_ml.engine.trail import Trail
from cp4dm_cpmpy_oscar_ml.exceptions import InconsistencyError


class PropagationQueue:
    """
    Fixed-point propagation engine.

    Propagators are scheduled when variables they watch change.
    Propagation runs until no more propagators are pending or failure is detected.
    """

    def __init__(self, store: DomainStore, trail: Trail) -> None:
        self._store = store
        self._trail = trail
        self._propagators: list[Propagator] = []
        self._queue: deque[Propagator] = deque()
        self._in_queue: set[int] = set()  # propagator indices in queue

    def add_propagator(self, prop: Propagator) -> None:
        """Register a propagator and run its setup."""
        idx = len(self._propagators)
        self._propagators.append(prop)
        # Run setup (may raise InconsistencyError)
        prop.setup(self._store, self._trail)
        # Process any changes from setup
        self._schedule_from_changes()

    def propagate_fixpoint(self) -> None:
        """
        Run propagation to fixpoint.

        Raises InconsistencyError if any propagator detects failure.
        """
        self._schedule_from_changes()
        while self._queue:
            prop = self._queue.popleft()
            prop_idx = self._propagators.index(prop)
            self._in_queue.discard(prop_idx)
            prop.propagate(self._store, self._trail)
            self._schedule_from_changes()

    def _schedule_from_changes(self) -> None:
        """Check domain changes and schedule triggered propagators."""
        changes = self._store.drain_changes()
        for var, event in changes:
            for idx, prop in enumerate(self._propagators):
                if idx not in self._in_queue and prop.is_triggered_by(var.id, event):
                    self._queue.append(prop)
                    self._in_queue.add(idx)
