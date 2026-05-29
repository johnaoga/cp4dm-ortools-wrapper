"""Tests for the constraint propagation engine (domains, trail, search)."""

import pytest

from cp4dm_cpmpy_oscar_ml.engine.domain import DomainStore, EngineVar, VarType
from cp4dm_cpmpy_oscar_ml.engine.propagation_queue import PropagationQueue
from cp4dm_cpmpy_oscar_ml.engine.propagator import Propagator
from cp4dm_cpmpy_oscar_ml.engine.search import DepthFirstSearch
from cp4dm_cpmpy_oscar_ml.engine.trail import Trail, ReversibleInt
from cp4dm_cpmpy_oscar_ml.exceptions import InconsistencyError


class TestDomainStore:
    def test_bool_var(self):
        store = DomainStore()
        v = store.new_bool_var("x")
        assert v.lb == 0
        assert v.ub == 1
        assert not v.is_bound
        assert v.contains(0)
        assert v.contains(1)

    def test_assign(self):
        store = DomainStore()
        v = store.new_bool_var("x")
        store.assign(v, 1)
        assert v.is_bound
        assert v.value == 1

    def test_remove_value(self):
        store = DomainStore()
        v = store.new_bool_var("x")
        store.remove_value(v, 0)
        assert v.is_bound
        assert v.value == 1

    def test_remove_last_value_fails(self):
        store = DomainStore()
        v = store.new_bool_var("x")
        store.assign(v, 1)
        with pytest.raises(InconsistencyError):
            store.remove_value(v, 1)

    def test_int_var_bounds(self):
        store = DomainStore()
        v = store.new_int_var(0, 10, "y")
        store.update_min(v, 3)
        assert v.lb == 3
        store.update_max(v, 7)
        assert v.ub == 7

    def test_empty_domain_fails(self):
        store = DomainStore()
        v = store.new_int_var(5, 5, "z")
        with pytest.raises(InconsistencyError):
            store.remove_value(v, 5)


class TestTrail:
    def test_push_pop(self):
        store = DomainStore()
        trail = Trail(store)
        v = store.new_bool_var("x")

        trail.push_level()
        store.assign(v, 1)
        assert v.value == 1

        trail.pop_level()
        assert not v.is_bound
        assert v.lb == 0 and v.ub == 1

    def test_multiple_levels(self):
        store = DomainStore()
        trail = Trail(store)
        v = store.new_int_var(0, 10, "y")

        trail.push_level()
        store.update_min(v, 3)

        trail.push_level()
        store.update_min(v, 7)
        assert v.lb == 7

        trail.pop_level()
        assert v.lb == 3

        trail.pop_level()
        assert v.lb == 0

    def test_reversible_int(self):
        store = DomainStore()
        trail = Trail(store)
        ri = ReversibleInt(trail, 0)

        trail.push_level()
        ri.value = 5
        assert ri.value == 5

        trail.push_level()
        ri.value = 10
        assert ri.value == 10

        trail.pop_level()
        assert ri.value == 5

        trail.pop_level()
        assert ri.value == 0


class TestSearch:
    def test_solve_trivial(self):
        """Solve a model with just one boolean variable, no constraints."""
        store = DomainStore()
        trail = Trail(store)
        pq = PropagationQueue(store, trail)
        v = store.new_bool_var("x")

        search = DepthFirstSearch(store, trail, pq, [v])
        assert search.solve()
        assert v.is_bound

    def test_solve_all_booleans(self):
        """Enumerate all assignments of 3 booleans (should be 8)."""
        store = DomainStore()
        trail = Trail(store)
        pq = PropagationQueue(store, trail)
        vars_ = [store.new_bool_var(f"x{i}") for i in range(3)]

        search = DepthFirstSearch(store, trail, pq, vars_)
        count = search.solve_all()
        assert count == 8
