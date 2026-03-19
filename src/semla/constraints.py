"""Nonlinear parameter constraints for SEM estimation.

Supports inequality and equality constraints on labeled parameters:
    a > 0       — parameter 'a' must be positive
    a < 1       — parameter 'a' must be less than 1
    a > b       — 'a' must be greater than 'b'
    a*b == c    — product of 'a' and 'b' must equal 'c'

Constraints are enforced during optimization via scipy's SLSQP method.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .syntax import FormulaToken
from .specification import ModelSpecification, ParamInfo


# Constraint operators
_CONSTRAINT_OPS = {">", "<", ">=", "<=", "=="}


@dataclass
class Constraint:
    """A single nonlinear constraint."""

    lhs_expr: str  # left-hand side expression (e.g., "a", "a*b")
    op: str  # ">", "<", ">=", "<=", "=="
    rhs_expr: str  # right-hand side expression (e.g., "0", "c")

    @property
    def is_equality(self) -> bool:
        return self.op == "=="

    @property
    def is_inequality(self) -> bool:
        return self.op in (">", "<", ">=", "<=")


def extract_constraints(tokens: list[FormulaToken]) -> list[Constraint]:
    """Extract constraint definitions from parsed tokens.

    Returns
    -------
    list[Constraint]
        List of constraints found in the model syntax.
    """
    constraints = []
    for tok in tokens:
        if tok.op in _CONSTRAINT_OPS:
            constraints.append(Constraint(
                lhs_expr=tok.lhs,
                op=tok.op,
                rhs_expr=tok.rhs[0].var,
            ))
    return constraints


def _build_label_to_theta_map(
    spec: ModelSpecification,
) -> dict[str, int]:
    """Map parameter labels to their indices in the theta vector."""
    label_map = {}
    for p in spec.params:
        if p.free and p.label:
            idx = spec.param_theta_index(p)
            if idx is not None:
                label_map[p.label] = idx
    return label_map


def _eval_constraint_expr(
    expr: str,
    label_values: dict[str, float],
) -> float:
    """Evaluate a constraint expression given current label values.

    Substitutes label names with their values and evaluates the expression.
    """
    eval_expr = expr
    # Sort labels longest-first to avoid partial replacements
    for label in sorted(label_values.keys(), key=len, reverse=True):
        eval_expr = re.sub(
            rf'\b{re.escape(label)}\b',
            str(label_values[label]),
            eval_expr,
        )
    try:
        return float(eval(eval_expr, {"__builtins__": {}}, {"abs": abs}))
    except Exception:
        return float("nan")


def build_scipy_constraints(
    constraints: list[Constraint],
    spec: ModelSpecification,
) -> list[dict]:
    """Convert constraints to scipy constraint dicts for SLSQP.

    Parameters
    ----------
    constraints : list[Constraint]
        Parsed constraints from model syntax.
    spec : ModelSpecification
        Model specification with labeled parameters.

    Returns
    -------
    list[dict]
        Scipy constraint dicts for ``optimize.minimize(constraints=...)``.
    """
    label_to_idx = _build_label_to_theta_map(spec)

    scipy_constraints = []
    for c in constraints:
        if c.is_equality:
            # h(theta) = lhs - rhs = 0
            def eq_func(theta, _c=c, _map=label_to_idx):
                vals = {lb: theta[idx] for lb, idx in _map.items()}
                lhs_val = _eval_constraint_expr(_c.lhs_expr, vals)
                rhs_val = _eval_constraint_expr(_c.rhs_expr, vals)
                return lhs_val - rhs_val

            scipy_constraints.append({
                "type": "eq",
                "fun": eq_func,
            })

        elif c.op in (">", ">="):
            # g(theta) = lhs - rhs >= 0
            def ineq_func_gt(theta, _c=c, _map=label_to_idx):
                vals = {lb: theta[idx] for lb, idx in _map.items()}
                lhs_val = _eval_constraint_expr(_c.lhs_expr, vals)
                rhs_val = _eval_constraint_expr(_c.rhs_expr, vals)
                return lhs_val - rhs_val

            scipy_constraints.append({
                "type": "ineq",
                "fun": ineq_func_gt,
            })

        elif c.op in ("<", "<="):
            # g(theta) = rhs - lhs >= 0
            def ineq_func_lt(theta, _c=c, _map=label_to_idx):
                vals = {lb: theta[idx] for lb, idx in _map.items()}
                lhs_val = _eval_constraint_expr(_c.lhs_expr, vals)
                rhs_val = _eval_constraint_expr(_c.rhs_expr, vals)
                return rhs_val - lhs_val

            scipy_constraints.append({
                "type": "ineq",
                "fun": ineq_func_lt,
            })

    return scipy_constraints
