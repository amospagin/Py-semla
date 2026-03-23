"""Automated measurement invariance testing.

Fits the full invariance hierarchy (configural → metric → scalar → strict)
and tests each step with chi-square difference tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .model import MultiGroupModel, cfa


_LEVELS = ["configural", "metric", "scalar", "strict"]


def _quick_chisq(est_result) -> float:
    """Compute chi-square from a MultiGroupEstimationResult without full results."""
    from .multigroup import multigroup_ml_objective
    mg = est_result.mg_spec
    n_total = mg.n_total
    fmin = est_result.fmin
    return sum(
        (mg.group_n_obs[g] - 1) * _group_fmin(est_result, g)
        for g in range(len(mg.group_names))
    )


def _group_fmin(est_result, g: int) -> float:
    """Compute per-group F_ML."""
    from .estimation import ml_objective
    mg = est_result.mg_spec
    theta_g = est_result.theta_combined[mg.theta_mapping[g]]
    sample_mean_g = mg.group_sample_means[g] if mg.group_specs[g].meanstructure else None
    return ml_objective(
        theta_g, mg.group_specs[g], mg.group_sample_covs[g],
        mg.group_n_obs[g], sample_mean_g,
    )


def _make_partial_fit(results, mg_spec):
    """Wrap MultiGroupResults in a lightweight fit object."""

    class _PartialFit:
        def __init__(self, results, mg_spec):
            self.results = results
            self.mg_spec = mg_spec

        def fit_indices(self):
            return self.results.fit_indices()

        def estimates(self):
            return self.results.estimates()

        def summary(self):
            return self.results.summary()

    return _PartialFit(results, mg_spec)


class InvarianceResult:
    """Results of an automated measurement invariance test."""

    def __init__(
        self,
        fits: dict[str, MultiGroupModel],
        table: pd.DataFrame,
        syntax: str = "",
        data: pd.DataFrame = None,
        group: str = "",
    ):
        self.fits = fits
        self._table = table
        self._syntax = syntax
        self._data = data
        self._group = group

    def __repr__(self) -> str:
        return self.summary()

    def __getitem__(self, level: str) -> MultiGroupModel:
        """Access a specific invariance level fit, e.g. result['metric']."""
        return self.fits[level]

    def summary(self) -> str:
        """Print and return a formatted invariance testing summary."""
        lines = []
        lines.append("Measurement Invariance Tests")
        lines.append("=" * 72)
        lines.append("")

        # Header
        lines.append(
            f"{'Level':<12s} {'χ²':>8s} {'df':>5s} {'CFI':>7s} {'RMSEA':>7s} "
            f"{'Δχ²':>8s} {'Δdf':>5s} {'p(Δχ²)':>8s} {'Decision':>10s}"
        )
        lines.append("-" * 72)

        for _, row in self._table.iterrows():
            level = row["level"]
            chi = f"{row['chisq']:.3f}"
            df = f"{int(row['df'])}"
            cfi = f"{row['cfi']:.3f}"
            rmsea = f"{row['rmsea']:.3f}"

            if pd.isna(row["delta_chisq"]):
                d_chi = ""
                d_df = ""
                d_p = ""
                decision = "baseline"
            else:
                d_chi = f"{row['delta_chisq']:.3f}"
                d_df = f"{int(row['delta_df'])}"
                d_p = f"{row['delta_p']:.3f}"
                # Decision based on chi-square diff test AND delta-CFI
                if row["delta_p"] > 0.05 and abs(row["delta_cfi"]) < 0.01:
                    decision = "PASS"
                elif row["delta_p"] > 0.05 or abs(row["delta_cfi"]) < 0.01:
                    decision = "marginal"
                else:
                    decision = "FAIL"

            lines.append(
                f"{level:<12s} {chi:>8s} {df:>5s} {cfi:>7s} {rmsea:>7s} "
                f"{d_chi:>8s} {d_df:>5s} {d_p:>8s} {decision:>10s}"
            )

        lines.append("")
        lines.append("Decision criteria: PASS if Δχ² p > .05 AND ΔCFI < .01")
        lines.append("=" * 72)

        output = "\n".join(lines)
        print(output)
        return output

    def table(self) -> pd.DataFrame:
        """Return the invariance test results as a DataFrame."""
        return self._table.copy()

    def partial(self, level: str = None, max_freed: int = 5) -> PartialInvarianceResult:
        """Achieve partial invariance by iteratively freeing parameters.

        Identifies and frees the most non-invariant parameters until the
        chi-square difference test vs the previous level passes.

        Parameters
        ----------
        level : str, optional
            The level to achieve partial invariance for.
            Default: the first failing level.
        max_freed : int
            Maximum number of parameters to free (default: 5).

        Returns
        -------
        PartialInvarianceResult
        """
        from .model import MultiGroupModel
        from .multigroup import (
            free_constraints, estimate_multigroup,
        )
        from .multigroup_results import MultiGroupResults

        if level is None:
            # Find first failing level
            highest = self.highest_level
            level_idx = _LEVELS.index(highest)
            if level_idx >= len(_LEVELS) - 1:
                raise ValueError("All invariance levels pass — nothing to free.")
            level = _LEVELS[level_idx + 1]

        if level not in self.fits:
            raise ValueError(
                f"Level '{level}' not in results. "
                f"Available: {list(self.fits.keys())}"
            )

        # Previous (passing) level
        level_idx = _LEVELS.index(level)
        prev_level = _LEVELS[level_idx - 1]
        if prev_level not in self.fits:
            raise ValueError(f"Previous level '{prev_level}' not available.")

        prev_fit = self.fits[prev_level]
        # If the failing level has mean structure but the previous doesn't,
        # refit the previous level with mean structure for fair comparison
        if (level in ("scalar", "strict") and
                not prev_fit.mg_spec.group_specs[0].meanstructure):
            from .model import cfa as _cfa
            prev_fit = _cfa(
                self.fits[level].syntax_str,
                data=self._data, group=self._group,
                invariance=prev_level, meanstructure=True,
            )
        prev_fi = prev_fit.fit_indices()

        # Get the constrained model's multi-group spec
        constrained_fit = self.fits[level]
        mg_spec = constrained_fit.mg_spec

        # Identify constrained parameters that differ between this
        # level and the previous one (e.g., intercepts for scalar vs metric)
        n_groups = len(mg_spec.group_names)
        k_per_group = len(mg_spec.theta_mapping[0])

        # Find params constrained in this level but NOT in the previous
        prev_mg = prev_fit.mg_spec
        constrained_indices = []
        for i in range(k_per_group):
            # Constrained in current level?
            curr_shared = len(set(mg_spec.theta_mapping[g][i]
                                  for g in range(n_groups))) == 1
            # Also constrained in previous level?
            if i < len(prev_mg.theta_mapping[0]):
                prev_shared = len(set(prev_mg.theta_mapping[g][i]
                                      for g in range(n_groups))) == 1
            else:
                prev_shared = False
            # Only include params that are NEWLY constrained at this level
            if curr_shared and not prev_shared:
                constrained_indices.append(i)

        freed_indices = []
        freed_names = []
        constrained_chisq = constrained_fit.fit_indices()["chi_square"]

        for iteration in range(max_freed):
            # Try freeing each remaining candidate, pick largest chi-sq drop
            remaining = [i for i in constrained_indices if i not in freed_indices]
            if not remaining:
                break

            best_idx = None
            best_drop = -1.0

            for param_idx in remaining:
                trial = free_constraints(mg_spec, freed_indices + [param_idx])
                trial_result = estimate_multigroup(trial)
                trial_chisq = _quick_chisq(trial_result)
                drop = constrained_chisq - trial_chisq
                if drop > best_drop:
                    best_drop = drop
                    best_idx = param_idx

            if best_idx is None or best_drop < 0.01:
                break

            freed_indices.append(best_idx)

            # Get parameter name
            spec0 = mg_spec.group_specs[0]
            free_params = [p for p in spec0.params if p.free]
            if best_idx < len(free_params):
                p = free_params[best_idx]
                name = f"{p.lhs} {p.op}" if p.op == "~1" else f"{p.lhs} {p.op} {p.rhs}"
                freed_names.append(name)
            else:
                freed_names.append(f"theta[{best_idx}]")

            # Free this parameter and refit
            partial_mg = free_constraints(mg_spec, freed_indices)
            est_result = estimate_multigroup(partial_mg)

            # Build a MultiGroupModel-like object for the result
            partial_results = MultiGroupResults(est_result)

            # Test vs previous level
            partial_fi = partial_results.fit_indices()
            d_chi = partial_fi["chi_square"] - prev_fi["chi_square"]
            d_df = partial_fi["df"] - prev_fi["df"]
            d_p = 1.0 - stats.chi2.cdf(d_chi, d_df) if d_df > 0 else np.nan

            if d_p > 0.05:
                # Partial invariance achieved
                # Wrap in a lightweight object
                partial_fit = _make_partial_fit(partial_results, partial_mg)
                return PartialInvarianceResult(
                    level=level, freed_params=freed_names,
                    fit=partial_fit, prev_fit=prev_fit,
                    delta_chisq=d_chi, delta_df=d_df, delta_p=d_p,
                )

        # Hit max_freed without passing — return best attempt
        partial_mg = free_constraints(mg_spec, freed_indices)
        est_result = estimate_multigroup(partial_mg)
        partial_results = MultiGroupResults(est_result)
        partial_fi = partial_results.fit_indices()
        d_chi = partial_fi["chi_square"] - prev_fi["chi_square"]
        d_df = partial_fi["df"] - prev_fi["df"]
        d_p = 1.0 - stats.chi2.cdf(d_chi, d_df) if d_df > 0 else np.nan

        partial_fit = _make_partial_fit(partial_results, partial_mg)
        return PartialInvarianceResult(
            level=level, freed_params=freed_names,
            fit=partial_fit, prev_fit=prev_fit,
            delta_chisq=d_chi, delta_df=d_df, delta_p=d_p,
        )

    @property
    def highest_level(self) -> str:
        """Return the highest invariance level that holds.

        Uses both chi-square difference test (p > .05) and delta-CFI (< .01).
        """
        passed = "configural"
        for _, row in self._table.iterrows():
            if pd.isna(row["delta_chisq"]):
                continue  # baseline
            if row["delta_p"] > 0.05 and abs(row["delta_cfi"]) < 0.01:
                passed = row["level"]
            else:
                break
        return passed


class PartialInvarianceResult:
    """Results of partial invariance testing."""

    def __init__(self, level: str, freed_params: list[str],
                 fit, prev_fit, delta_chisq: float,
                 delta_df: int, delta_p: float):
        self.level = level
        self.freed_params = freed_params
        self.fit = fit
        self._prev_fit = prev_fit
        self.delta_chisq = delta_chisq
        self.delta_df = delta_df
        self.delta_p = delta_p

    @property
    def passed(self) -> bool:
        return self.delta_p > 0.05

    def summary(self) -> str:
        lines = []
        lines.append(f"Partial {self.level.title()} Invariance")
        lines.append("=" * 60)
        lines.append(f"  Freed parameters: {', '.join(self.freed_params)}")
        lines.append(f"  Δχ² = {self.delta_chisq:.3f}, "
                     f"Δdf = {self.delta_df}, p = {self.delta_p:.3f}")
        lines.append(f"  Result: {'PASS' if self.passed else 'FAIL'}")
        fi = self.fit.fit_indices()
        lines.append(f"  CFI = {fi['cfi']:.3f}, RMSEA = {fi['rmsea']:.3f}")
        lines.append("=" * 60)
        output = "\n".join(lines)
        print(output)
        return output


def measurementInvariance(
    model: str,
    data: pd.DataFrame,
    group: str,
    levels: list[str] | None = None,
    **kwargs,
) -> InvarianceResult:
    """Run automated measurement invariance testing.

    Fits the full invariance hierarchy and tests each successive level
    with chi-square difference tests and delta-CFI.

    Parameters
    ----------
    model : str
        Model syntax in lavaan format (measurement model only).
    data : pd.DataFrame
        Data with a grouping column.
    group : str
        Column name defining groups.
    levels : list[str], optional
        Invariance levels to test. Default: all four
        (configural, metric, scalar, strict).
    **kwargs
        Additional arguments passed to ``cfa()`` (e.g., estimator).

    Returns
    -------
    InvarianceResult
        Object with .summary(), .table(), .fits dict, .highest_level.

    Examples
    --------
    >>> from semla import measurementInvariance
    >>> from semla.datasets import HolzingerSwineford1939
    >>> df = HolzingerSwineford1939()
    >>> result = measurementInvariance('''
    ...     visual  =~ x1 + x2 + x3
    ...     textual =~ x4 + x5 + x6
    ...     speed   =~ x7 + x8 + x9
    ... ''', data=df, group="school")
    >>> result.highest_level
    'strict'
    """
    if levels is None:
        levels = list(_LEVELS)

    # Validate levels
    for lv in levels:
        if lv not in _LEVELS:
            raise ValueError(
                f"Unknown invariance level '{lv}'. "
                f"Must be one of: {_LEVELS}"
            )

    # Ensure levels are in correct order
    levels = [lv for lv in _LEVELS if lv in levels]

    # Fit each level
    fits: dict[str, MultiGroupModel] = {}
    for lv in levels:
        fits[lv] = cfa(model, data=data, group=group, invariance=lv, **kwargs)

    # Build comparison table
    rows = []
    prev_level = None
    for lv in levels:
        fi = fits[lv].fit_indices()
        row = {
            "level": lv,
            "chisq": fi["chi_square"],
            "df": fi["df"],
            "cfi": fi["cfi"],
            "tli": fi["tli"],
            "rmsea": fi["rmsea"],
            "srmr": fi["srmr"],
        }

        if prev_level is not None:
            prev_fi = fits[prev_level].fit_indices()
            d_chi = fi["chi_square"] - prev_fi["chi_square"]
            d_df = fi["df"] - prev_fi["df"]
            row["delta_chisq"] = d_chi
            row["delta_df"] = d_df
            row["delta_p"] = (
                1.0 - stats.chi2.cdf(d_chi, d_df) if d_df > 0 else np.nan
            )
            row["delta_cfi"] = fi["cfi"] - prev_fi["cfi"]
        else:
            row["delta_chisq"] = np.nan
            row["delta_df"] = np.nan
            row["delta_p"] = np.nan
            row["delta_cfi"] = np.nan

        rows.append(row)
        prev_level = lv

    table = pd.DataFrame(rows)

    return InvarianceResult(fits=fits, table=table, syntax=model, data=data, group=group)
