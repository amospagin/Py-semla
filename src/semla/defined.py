"""Evaluate user-defined parameters and compute delta method standard errors.

Supports expressions like:
    indirect := a*b
    total := a*b + c
    diff := a - b
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

from .syntax import FormulaToken


def extract_defined_params(tokens: list[FormulaToken]) -> list[tuple[str, str]]:
    """Extract defined parameter definitions from parsed tokens.

    Returns
    -------
    list[tuple[str, str]]
        List of (name, expression) pairs.
    """
    return [
        (tok.lhs, tok.rhs[0].var)  # expression stored in rhs[0].var
        for tok in tokens
        if tok.op == ":="
    ]


def _get_label_values(estimates_df: pd.DataFrame) -> dict[str, float]:
    """Extract a mapping of parameter labels to their estimated values."""
    label_map = {}

    # For each free parameter with a label, store label -> estimate
    for _, row in estimates_df.iterrows():
        if not row.get("free", False):
            continue
        # Check if the parameter has a label (stored in the spec params)
        # We'll match by looking for labeled params
        pass

    return label_map


def evaluate_defined_params(
    defined: list[tuple[str, str]],
    label_values: dict[str, float],
) -> list[dict]:
    """Evaluate defined parameter expressions.

    Parameters
    ----------
    defined : list[tuple[str, str]]
        (name, expression) pairs from extract_defined_params().
    label_values : dict[str, float]
        Mapping of parameter labels to estimated values.

    Returns
    -------
    list[dict]
        Each dict has: name, expression, est.
    """
    results = []
    for name, expr in defined:
        try:
            # Replace label names with their values in the expression
            eval_expr = expr
            # Sort labels longest-first to avoid partial replacements
            for label in sorted(label_values.keys(), key=len, reverse=True):
                eval_expr = re.sub(
                    rf'\b{re.escape(label)}\b',
                    str(label_values[label]),
                    eval_expr
                )
            value = eval(eval_expr, {"__builtins__": {}}, {})
            results.append({"name": name, "expression": expr, "est": float(value)})
        except Exception:
            results.append({"name": name, "expression": expr, "est": np.nan})
    return results


def compute_defined_se(
    defined: list[tuple[str, str]],
    label_values: dict[str, float],
    label_se: dict[str, float],
    label_vcov: dict[tuple[str, str], float] | None = None,
) -> list[float]:
    """Compute standard errors for defined parameters via the delta method.

    For f(a, b) = a*b:
        Var(f) = (df/da)^2 * Var(a) + (df/db)^2 * Var(b) + 2*(df/da)*(df/db)*Cov(a,b)

    Uses numerical differentiation of the expression.

    Parameters
    ----------
    defined : list[tuple[str, str]]
        (name, expression) pairs.
    label_values : dict[str, float]
        Label -> estimated value.
    label_se : dict[str, float]
        Label -> standard error.
    label_vcov : dict, optional
        (label_i, label_j) -> covariance. If None, assumes independence.

    Returns
    -------
    list[float]
        Standard error for each defined parameter.
    """
    eps = 1e-6
    labels = sorted(label_values.keys())
    ses = []

    for name, expr in defined:
        # Find which labels appear in this expression
        used_labels = [lb for lb in labels if re.search(rf'\b{re.escape(lb)}\b', expr)]

        if not used_labels:
            ses.append(np.nan)
            continue

        # Compute partial derivatives numerically
        def eval_expr(vals: dict[str, float]) -> float:
            e = expr
            for lb in sorted(vals.keys(), key=len, reverse=True):
                e = re.sub(rf'\b{re.escape(lb)}\b', str(vals[lb]), e)
            try:
                return float(eval(e, {"__builtins__": {}}, {}))
            except Exception:
                return np.nan

        partials = {}
        for lb in used_labels:
            vals_plus = dict(label_values)
            vals_plus[lb] += eps
            vals_minus = dict(label_values)
            vals_minus[lb] -= eps
            fp = eval_expr(vals_plus)
            fm = eval_expr(vals_minus)
            partials[lb] = (fp - fm) / (2 * eps)

        # Delta method variance
        var_f = 0.0
        for lb in used_labels:
            se_lb = label_se.get(lb, 0.0)
            var_f += partials[lb] ** 2 * se_lb ** 2

        # Cross terms (covariance between labels)
        if label_vcov:
            for i, lb_i in enumerate(used_labels):
                for j, lb_j in enumerate(used_labels):
                    if i < j:
                        cov_ij = label_vcov.get((lb_i, lb_j), label_vcov.get((lb_j, lb_i), 0.0))
                        var_f += 2 * partials[lb_i] * partials[lb_j] * cov_ij

        ses.append(np.sqrt(var_f) if var_f > 0 else np.nan)

    return ses
