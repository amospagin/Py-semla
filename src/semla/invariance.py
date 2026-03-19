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


class InvarianceResult:
    """Results of an automated measurement invariance test."""

    def __init__(
        self,
        fits: dict[str, MultiGroupModel],
        table: pd.DataFrame,
    ):
        self.fits = fits
        self._table = table

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

    return InvarianceResult(fits=fits, table=table)
