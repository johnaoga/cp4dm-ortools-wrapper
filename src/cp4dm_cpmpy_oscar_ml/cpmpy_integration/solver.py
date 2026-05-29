"""
CPM_oscar_ml: dedicated CPMpy solver interface for Oscar ML globals.

This solver owns the propagation engine and search. It translates CPMpy
variables and constraints into engine primitives, handles Oscar ML globals
natively (without decomposition), and populates CPMpy variable values.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import cpmpy as cp
from cpmpy.expressions.core import BoolVal, Comparison, Operator
from cpmpy.expressions.globalconstraints import GlobalConstraint
from cpmpy.expressions.variables import NDVarArray, _BoolVarImpl, _IntVarImpl, boolvar, cpm_array, intvar
from cpmpy.transformations.get_variables import get_variables as cpmpy_get_variables

from cp4dm_cpmpy_oscar_ml.cpmpy_integration.globals import (
    ClosedFrequentItemset,
    CoverClosure,
    CoverSize,
    FrequentItemset,
)
from cp4dm_cpmpy_oscar_ml.engine.domain import DomainStore, EngineVar, VarType
from cp4dm_cpmpy_oscar_ml.engine.propagation_queue import PropagationQueue
from cp4dm_cpmpy_oscar_ml.engine.propagator import Propagator
from cp4dm_cpmpy_oscar_ml.engine.search import DepthFirstSearch, SearchStats
from cp4dm_cpmpy_oscar_ml.engine.trail import Trail
from cp4dm_cpmpy_oscar_ml.exceptions import InconsistencyError, UnsupportedExpressionError
from cp4dm_cpmpy_oscar_ml.pattern_mining.fim.propagators import ClosedFIMPropagator, FIMPropagator


class CPM_oscar_ml:
    """
    Oscar ML CPMpy solver interface.

    Handles Oscar ML global constraints natively using the built-in
    propagation engine and DFS search.

    Supported globals:
        oscar_fim, oscar_closed_fim, oscar_cover_size, oscar_cover_closure

    Supported primitive constraints:
        sum comparisons, boolean implications, literal bounds
    """

    supported_global_constraints = frozenset({
        "oscar_fim",
        "oscar_closed_fim",
        "oscar_cover_size",
        "oscar_cover_closure",
        "oscar_zdc",
        "oscar_ppic",
        "oscar_ppict",
        "oscar_ppdc",
        "oscar_ppmixed",
        "oscar_episode_support",
        "oscar_episode_support_t",
        "oscar_tree_cover_size_sr",
        "oscar_split_possible",
        "oscar_split_useful",
        "oscar_dummy_node",
        "oscar_dummy_end_node",
    })

    def __init__(self, model: cp.Model | None = None) -> None:
        self._cpmpy_model = model
        self._store = DomainStore()
        self._trail = Trail(self._store)
        self._prop_queue = PropagationQueue(self._store, self._trail)
        self._var_map: dict[int, EngineVar] = {}  # CPMpy var id -> engine var
        self._engine_to_cpmpy: dict[int, Any] = {}  # engine var id -> CPMpy var
        self._search: DepthFirstSearch | None = None
        self._stats = SearchStats()
        self._built = False

    def solve(
        self,
        time_limit: float = 0,
        display: Callable[[], None] | None = None,
    ) -> bool:
        """Solve the model, finding one solution."""
        self._build()
        search = self._create_search()
        if time_limit > 0:
            search.set_time_limit(time_limit)
        if display:
            search.set_on_solution(self._wrap_callback(display))
        else:
            search.set_on_solution(self._populate_cpmpy_values)
        result = search.solve()
        self._stats = search.stats
        if result:
            self._populate_cpmpy_values()
        return result

    def solveAll(
        self,
        time_limit: float = 0,
        solution_limit: int = 0,
        display: Callable[[], None] | None = None,
    ) -> int:
        """Enumerate all solutions."""
        self._build()
        search = self._create_search()
        if time_limit > 0:
            search.set_time_limit(time_limit)
        if solution_limit > 0:
            search.set_solution_limit(solution_limit)
        if display:
            search.set_on_solution(self._wrap_callback(display))
        else:
            search.set_on_solution(self._populate_cpmpy_values)
        count = search.solve_all()
        self._stats = search.stats
        # Populate values from last solution if any
        if search._solutions_found:
            self._populate_values_from(search._solutions_found[-1])
        return count

    @property
    def stats(self) -> SearchStats:
        return self._stats

    # --- Internal build ---

    def _build(self) -> None:
        """Translate CPMpy model into engine propagators."""
        if self._built:
            return
        if self._cpmpy_model is None:
            raise ValueError("No model provided to CPM_oscar_ml")

        # Collect all CPMpy variables from constraints
        all_vars = cpmpy_get_variables(self._cpmpy_model.constraints)
        for cv in all_vars:
            self._get_or_create_engine_var(cv)

        # Translate constraints
        for constraint in self._cpmpy_model.constraints:
            self._post_constraint(constraint)

        self._built = True

    def _get_or_create_engine_var(self, cpmpy_var: Any) -> EngineVar:
        """Map a CPMpy variable to an engine variable."""
        vid = id(cpmpy_var)
        if vid in self._var_map:
            return self._var_map[vid]

        if isinstance(cpmpy_var, _BoolVarImpl):
            ev = self._store.new_bool_var(name=cpmpy_var.name)
        elif isinstance(cpmpy_var, _IntVarImpl):
            ev = self._store.new_int_var(cpmpy_var.lb, cpmpy_var.ub, name=cpmpy_var.name)
        else:
            raise UnsupportedExpressionError(f"Unsupported variable type: {type(cpmpy_var)}")

        ev.cpmpy_var = cpmpy_var
        self._var_map[vid] = ev
        self._engine_to_cpmpy[ev.id] = cpmpy_var
        return ev

    def _post_constraint(self, constraint: Any) -> None:
        """Translate a CPMpy constraint into an engine propagator."""
        if isinstance(constraint, FrequentItemset):
            self._post_fim(constraint)
        elif isinstance(constraint, ClosedFrequentItemset):
            self._post_closed_fim(constraint)
        elif isinstance(constraint, CoverSize):
            self._post_cover_size(constraint)
        elif isinstance(constraint, CoverClosure):
            self._post_cover_closure(constraint)
        elif isinstance(constraint, GlobalConstraint):
            if constraint.name in self.supported_global_constraints:
                raise UnsupportedExpressionError(
                    f"Global '{constraint.name}' is declared supported but not yet implemented"
                )
            raise UnsupportedExpressionError(
                f"Unsupported global constraint: {constraint.name}. "
                "Oscar ML globals require CPM_oscar_ml."
            )
        elif isinstance(constraint, Comparison):
            self._post_comparison(constraint)
        elif isinstance(constraint, Operator):
            self._post_operator(constraint)
        elif isinstance(constraint, BoolVal):
            if not constraint.value():
                raise InconsistencyError("Model contains False literal")
        else:
            raise UnsupportedExpressionError(
                f"Unsupported constraint type: {type(constraint).__name__}. "
                "Either implement a propagator or enable check-only fallback."
            )

    def _post_fim(self, constraint: FrequentItemset) -> None:
        """Post FIM propagator."""
        engine_vars = [self._get_or_create_engine_var(v) for v in constraint.item_vars]
        prop = FIMPropagator(engine_vars, constraint.minsup, constraint.data)
        self._prop_queue.add_propagator(prop)

    def _post_closed_fim(self, constraint: ClosedFrequentItemset) -> None:
        """Post ClosedFIM propagator."""
        engine_vars = [self._get_or_create_engine_var(v) for v in constraint.item_vars]
        prop = ClosedFIMPropagator(engine_vars, constraint.minsup, constraint.data)
        self._prop_queue.add_propagator(prop)

    def _post_cover_size(self, constraint: CoverSize) -> None:
        """Post CoverSize propagator (TODO: implement CoverSizePropagator)."""
        raise UnsupportedExpressionError("CoverSize propagator not yet implemented")

    def _post_cover_closure(self, constraint: CoverClosure) -> None:
        """Post CoverClosure propagator (TODO: implement)."""
        raise UnsupportedExpressionError("CoverClosure propagator not yet implemented")

    def _post_comparison(self, comp: Comparison) -> None:
        """Handle simple comparisons like sum(vars) <= k."""
        prop = _ComparisonPropagator(comp, self)
        self._prop_queue.add_propagator(prop)

    def _post_operator(self, op: Operator) -> None:
        """Handle operators like and/or at top level."""
        if op.name == "and":
            for arg in op.args:
                self._post_constraint(arg)
        elif op.name == "or":
            # Simple check-only: verify at solution time
            prop = _CheckOnlyPropagator(op, self)
            self._prop_queue.add_propagator(prop)
        else:
            raise UnsupportedExpressionError(f"Unsupported operator: {op.name}")

    def _create_search(self) -> DepthFirstSearch:
        """Create the DFS search engine."""
        # Branch on all variables that are Boolean (item vars typically)
        branch_vars = [v for v in self._store.variables if v.var_type == VarType.BOOL]
        if not branch_vars:
            branch_vars = self._store.variables
        search = DepthFirstSearch(self._store, self._trail, self._prop_queue, branch_vars)
        return search

    def _wrap_callback(self, user_callback: Callable[[], None]) -> Callable[[], None]:
        """Wrap user callback to populate CPMpy values before calling it."""
        def wrapped():
            self._populate_cpmpy_values()
            user_callback()
        return wrapped

    def _populate_cpmpy_values(self) -> None:
        """Set CPMpy variable values from current engine state."""
        for ev in self._store.variables:
            if ev.cpmpy_var is not None and ev.is_bound:
                ev.cpmpy_var._value = ev.value

    def _populate_values_from(self, solution: dict[int, int]) -> None:
        """Set CPMpy variable values from a solution dict."""
        for ev in self._store.variables:
            if ev.cpmpy_var is not None and ev.id in solution:
                ev.cpmpy_var._value = solution[ev.id]


class _ComparisonPropagator(Propagator):
    """
    Simple propagator for linear comparisons (e.g., sum(vars) <= k).

    For now: bounds propagation on sum constraints, check-only for others.
    """

    name = "comparison"

    def __init__(self, comp: Comparison, solver: CPM_oscar_ml) -> None:
        super().__init__()
        self._comp = comp
        self._solver = solver
        self._engine_vars: list[EngineVar] = []

    def setup(self, store: DomainStore, trail: Trail) -> None:
        # Extract variables from the comparison
        cpmpy_vars = cpmpy_get_variables(self._comp)
        self._engine_vars = [self._solver._get_or_create_engine_var(v) for v in cpmpy_vars]
        self.variables = tuple(self._engine_vars)
        for ev in self._engine_vars:
            self.watch_bind(ev)
            self.watch_bounds(ev)
        # Try initial propagation
        self._propagate_bounds(store)

    def propagate(self, store: DomainStore, trail: Trail) -> None:
        self._propagate_bounds(store)

    def _propagate_bounds(self, store: DomainStore) -> None:
        """Attempt bounds propagation for sum comparisons."""
        comp = self._comp
        # Handle: sum(bvars) <= k or sum(bvars) >= k
        lhs = comp.args[0]
        rhs = comp.args[1]

        # Try to evaluate bounds
        try:
            # Get current bounds of lhs
            lb, ub = self._eval_bounds(lhs)
            rhs_val = self._eval_const(rhs)

            if comp.name == "<=":
                if lb > rhs_val:
                    raise InconsistencyError(f"Comparison {comp}: lb={lb} > rhs={rhs_val}")
                # If at upper bound, fix remaining unbound vars to 0 for sum constraints
                if hasattr(lhs, "name") and lhs.name == "sum":
                    self._propagate_sum_leq(lhs, rhs_val, store)
            elif comp.name == ">=":
                if ub < rhs_val:
                    raise InconsistencyError(f"Comparison {comp}: ub={ub} < rhs={rhs_val}")
                if hasattr(lhs, "name") and lhs.name == "sum":
                    self._propagate_sum_geq(lhs, rhs_val, store)
            elif comp.name == "<":
                if lb >= rhs_val:
                    raise InconsistencyError(f"Comparison {comp}: lb={lb} >= rhs={rhs_val}")
            elif comp.name == ">":
                if ub <= rhs_val:
                    raise InconsistencyError(f"Comparison {comp}: ub={ub} <= rhs={rhs_val}")
            elif comp.name == "==":
                if lb > rhs_val or ub < rhs_val:
                    raise InconsistencyError(f"Comparison {comp}: [{lb},{ub}] excludes {rhs_val}")
        except (TypeError, AttributeError):
            # Cannot evaluate -> check-only at solution
            self._check_at_solution()

    def _propagate_sum_leq(self, sum_expr: Any, rhs: int, store: DomainStore) -> None:
        """If sum of booleans <= k and current min already at k, assign remaining to 0."""
        args = sum_expr.args
        current_min = 0
        unbound = []
        for arg in args:
            ev = self._solver._var_map.get(id(arg))
            if ev is None:
                continue
            if ev.is_bound:
                current_min += ev.value
            else:
                unbound.append(ev)

        if current_min > rhs:
            raise InconsistencyError(f"sum >= {current_min} > {rhs}")

        # If adding all unbound at max would still be <= rhs, no pruning needed
        max_possible = current_min + len(unbound)
        if max_possible <= rhs:
            return

        # If current_min == rhs, all remaining must be 0
        if current_min == rhs:
            for ev in unbound:
                store.assign(ev, 0)

    def _propagate_sum_geq(self, sum_expr: Any, rhs: int, store: DomainStore) -> None:
        """If sum of booleans >= k and current max can't reach k, fail."""
        args = sum_expr.args
        current_max = 0
        current_min = 0
        unbound = []
        for arg in args:
            ev = self._solver._var_map.get(id(arg))
            if ev is None:
                continue
            if ev.is_bound:
                current_min += ev.value
                current_max += ev.value
            else:
                current_max += ev.ub
                current_min += ev.lb
                unbound.append(ev)

        if current_max < rhs:
            raise InconsistencyError(f"sum max={current_max} < {rhs}")

        # If we need all unbound to be 1
        if current_min + len(unbound) == rhs and current_min < rhs:
            for ev in unbound:
                store.assign(ev, 1)

    def _eval_bounds(self, expr: Any) -> tuple[int, int]:
        """Evaluate lower and upper bounds of an expression."""
        if isinstance(expr, (int, float)):
            return int(expr), int(expr)
        if isinstance(expr, (_BoolVarImpl, _IntVarImpl)):
            ev = self._solver._var_map.get(id(expr))
            if ev:
                return ev.lb, ev.ub
            return expr.lb, expr.ub
        if hasattr(expr, "name") and expr.name == "sum":
            lb = 0
            ub = 0
            for arg in expr.args:
                a_lb, a_ub = self._eval_bounds(arg)
                lb += a_lb
                ub += a_ub
            return lb, ub
        raise TypeError(f"Cannot eval bounds of {type(expr)}")

    def _eval_const(self, expr: Any) -> int:
        """Evaluate a constant expression."""
        if isinstance(expr, (int, float)):
            return int(expr)
        if isinstance(expr, (_BoolVarImpl, _IntVarImpl)):
            ev = self._solver._var_map.get(id(expr))
            if ev and ev.is_bound:
                return ev.value
        raise TypeError(f"Cannot eval constant: {expr}")

    def _check_at_solution(self) -> None:
        """Check constraint on full assignment (check-only fallback)."""
        # Only check if all vars are bound
        all_bound = all(ev.is_bound for ev in self._engine_vars)
        if not all_bound:
            return
        # Evaluate the constraint
        # This is a simplified check - set var values and evaluate
        for ev in self._engine_vars:
            if ev.cpmpy_var is not None:
                ev.cpmpy_var._value = ev.value
        try:
            result = self._comp.value()
            if result is False:
                raise InconsistencyError(f"Check-only constraint violated: {self._comp}")
        except Exception:
            pass


class _CheckOnlyPropagator(Propagator):
    """Check-only propagator: verifies constraint at full assignment only."""

    name = "check_only"

    def __init__(self, expr: Any, solver: CPM_oscar_ml) -> None:
        super().__init__()
        self._expr = expr
        self._solver = solver
        self._engine_vars: list[EngineVar] = []

    def setup(self, store: DomainStore, trail: Trail) -> None:
        cpmpy_vars = cpmpy_get_variables(self._expr)
        self._engine_vars = [self._solver._get_or_create_engine_var(v) for v in cpmpy_vars]
        self.variables = tuple(self._engine_vars)
        for ev in self._engine_vars:
            self.watch_bind(ev)

    def propagate(self, store: DomainStore, trail: Trail) -> None:
        all_bound = all(ev.is_bound for ev in self._engine_vars)
        if not all_bound:
            return
        for ev in self._engine_vars:
            if ev.cpmpy_var is not None:
                ev.cpmpy_var._value = ev.value
        try:
            result = self._expr.value()
            if result is False:
                raise InconsistencyError(f"Constraint violated: {self._expr}")
        except InconsistencyError:
            raise
        except Exception:
            pass
