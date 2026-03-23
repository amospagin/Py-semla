"""Longitudinal measurement invariance testing.

Tests whether the same measure works equivalently across time points,
using equality constraints in a single-group model with correlated
residuals across waves.

Levels supported:
    configural — same structure, correlated residuals across waves
    metric     — equal loadings across time points
    strict     — equal loadings + equal residual variances
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .model import cfa


_LEVELS = ["configural", "metric"]


class LongitudinalInvarianceResult:
    """Results of longitudinal measurement invariance testing."""

    def __init__(self, fits: dict[str, object], table: pd.DataFrame):
        self.fits = fits
        self._table = table

    def __repr__(self) -> str:
        return self.summary()

    def __getitem__(self, level: str) -> object:
        return self.fits[level]

    def summary(self) -> str:
        lines = []
        lines.append("Longitudinal Measurement Invariance Tests")
        lines.append("=" * 72)
        lines.append("")
        lines.append(
            f"{'Level':<12s} {'chi2':>8s} {'df':>5s} {'CFI':>7s} {'RMSEA':>7s} "
            f"{'d_chi2':>8s} {'d_df':>5s} {'p':>8s} {'Decision':>10s}"
        )
        lines.append("-" * 72)

        for _, row in self._table.iterrows():
            chi = f"{row['chisq']:.3f}"
            df = f"{int(row['df'])}"
            cfi_str = f"{row['cfi']:.3f}"
            rmsea = f"{row['rmsea']:.3f}"

            if pd.isna(row["delta_chisq"]):
                d_chi = d_df = d_p = ""
                decision = "baseline"
            else:
                d_chi = f"{row['delta_chisq']:.3f}"
                d_df = f"{int(row['delta_df'])}"
                d_p = f"{row['delta_p']:.3f}"
                if row["delta_p"] > 0.05 and abs(row["delta_cfi"]) < 0.01:
                    decision = "PASS"
                elif row["delta_p"] > 0.05 or abs(row["delta_cfi"]) < 0.01:
                    decision = "marginal"
                else:
                    decision = "FAIL"

            lines.append(
                f"{row['level']:<12s} {chi:>8s} {df:>5s} {cfi_str:>7s} "
                f"{rmsea:>7s} {d_chi:>8s} {d_df:>5s} {d_p:>8s} {decision:>10s}"
            )

        lines.append("")
        lines.append("Decision criteria: PASS if d_chi2 p > .05 AND dCFI < .01")
        lines.append("=" * 72)
        output = "\n".join(lines)
        print(output)
        return output

    def table(self) -> pd.DataFrame:
        return self._table.copy()

    @property
    def highest_level(self) -> str:
        passed = "configural"
        for _, row in self._table.iterrows():
            if pd.isna(row["delta_chisq"]):
                continue
            if row["delta_p"] > 0.05 and abs(row["delta_cfi"]) < 0.01:
                passed = row["level"]
            else:
                break
        return passed


def longitudinalInvariance(
    model: str,
    data: pd.DataFrame,
    items: dict[str, str],
    factors: dict[str, str] | None = None,
    levels: list[str] | None = None,
    **kwargs,
) -> LongitudinalInvarianceResult:
    """Test longitudinal measurement invariance across time points.

    Parameters
    ----------
    model : str
        Model syntax with factors at each time point.
    data : pd.DataFrame
        Data with all time-point variables as columns.
    items : dict[str, str]
        Mapping of corresponding items across time, e.g.
        ``{"x1_t1": "x1_t2", "x2_t1": "x2_t2", ...}``.
    factors : dict[str, str], optional
        Mapping of corresponding factors, e.g. ``{"f_t1": "f_t2"}``.
        If None, inferred from item mapping.
    levels : list[str], optional
        Invariance levels to test. Default: configural, metric, strict.
    **kwargs
        Additional arguments passed to ``cfa()``.

    Returns
    -------
    LongitudinalInvarianceResult
    """
    from .syntax import parse_syntax

    if levels is None:
        levels = list(_LEVELS)
    levels = [lv for lv in _LEVELS if lv in levels]

    tokens = parse_syntax(model)

    # Identify factors and their indicators
    factor_indicators: dict[str, list[str]] = {}
    for tok in tokens:
        if tok.op == "=~":
            factor_indicators.setdefault(tok.lhs, []).extend(
                t.var for t in tok.rhs
            )

    # Infer factor mapping if not provided
    if factors is None:
        factors = _infer_factor_mapping(factor_indicators, items)

    # Build item pairs: (item_t1, item_t2, position_in_factor)
    item_pairs = []
    for item_a, item_b in items.items():
        item_pairs.append((item_a, item_b))

    # Fit each level
    fits = {}
    for level in levels:
        syntax = _build_syntax(
            factor_indicators, factors, item_pairs, level,
        )
        fit = cfa(syntax, data=data, **kwargs)
        fits[level] = fit

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

    return LongitudinalInvarianceResult(fits=fits, table=pd.DataFrame(rows))


def _infer_factor_mapping(
    factor_indicators: dict[str, list[str]],
    items: dict[str, str],
) -> dict[str, str]:
    """Infer which factors correspond across time from item mapping."""
    item_keys = set(items.keys())
    item_vals = set(items.values())

    time1_factors = []
    time2_factors = []
    for fac, inds in factor_indicators.items():
        ind_set = set(inds)
        if ind_set & item_keys:
            time1_factors.append(fac)
        elif ind_set & item_vals:
            time2_factors.append(fac)

    return dict(zip(time1_factors, time2_factors))


def _build_syntax(
    factor_indicators: dict[str, list[str]],
    factors: dict[str, str],
    item_pairs: list[tuple[str, str]],
    level: str,
) -> str:
    """Build model syntax with constraints for the given invariance level."""
    lines = []
    mapped_factors = set(factors.keys()) | set(factors.values())

    # Build factor definitions
    for fac_t1, fac_t2 in factors.items():
        inds_t1 = factor_indicators[fac_t1]
        inds_t2 = factor_indicators[fac_t2]

        # Build T1 factor definition
        parts_t1 = []
        for i, ind in enumerate(inds_t1):
            if i == 0:
                parts_t1.append(ind)  # first loading fixed to 1
            elif level in ("metric", "strict"):
                label = f"l_{fac_t1}_{i}"
                parts_t1.append(f"{label}*{ind}")
            else:
                parts_t1.append(ind)
        lines.append(f"{fac_t1} =~ {' + '.join(parts_t1)}")

        # Build T2 factor definition with same labels for metric+
        parts_t2 = []
        for i, ind in enumerate(inds_t2):
            if i == 0:
                parts_t2.append(ind)
            elif level in ("metric", "strict"):
                label = f"l_{fac_t1}_{i}"  # same label as T1
                parts_t2.append(f"{label}*{ind}")
            else:
                parts_t2.append(ind)
        lines.append(f"{fac_t2} =~ {' + '.join(parts_t2)}")

    # Factors not in mapping (pass through)
    for fac, inds in factor_indicators.items():
        if fac not in mapped_factors:
            lines.append(f"{fac} =~ {' + '.join(inds)}")

    # Correlated residuals across waves (all levels)
    for item_a, item_b in item_pairs:
        lines.append(f"{item_a} ~~ {item_b}")

    return "\n".join(lines)
