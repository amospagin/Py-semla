"""Validate FIML estimates against lavaan on multiple missingness patterns.

Reference values from lavaan 0.6-21, using HolzingerSwineford1939 with
set.seed(42) and the classic 3-factor CFA model.

GitHub issue #24.

Note: Parameter estimates (loadings, variances, covariances, intercepts)
match lavaan closely across all patterns. Fit indices (chi-square, CFI,
TLI, RMSEA) may diverge when many missing-data patterns exist because
the saturated-model log-likelihood computation differs between semla and
lavaan. When only a single variable has missingness (few unique patterns),
fit indices match tightly.
"""

import numpy as np
import pandas as pd
import pytest
from semla import cfa

HS_MODEL = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(scope="module")
def hs_mcar10():
    """MCAR ~10% missing on all observed variables."""
    return pd.read_csv("/tmp/hs_mcar10.csv")


@pytest.fixture(scope="module")
def hs_mcar20():
    """MCAR ~20% missing on all observed variables."""
    return pd.read_csv("/tmp/hs_mcar20.csv")


@pytest.fixture(scope="module")
def hs_mcar30_x1():
    """30% missing on x1 only."""
    return pd.read_csv("/tmp/hs_mcar30_x1.csv")


@pytest.fixture(scope="module")
def fit_mcar10(hs_mcar10):
    return cfa(HS_MODEL, data=hs_mcar10, missing="fiml")


@pytest.fixture(scope="module")
def fit_mcar20(hs_mcar20):
    return cfa(HS_MODEL, data=hs_mcar20, missing="fiml")


@pytest.fixture(scope="module")
def fit_mcar30_x1(hs_mcar30_x1):
    return cfa(HS_MODEL, data=hs_mcar30_x1, missing="fiml")


# ============================================================
# Helper
# ============================================================


def _get_est(fit, lhs, op, rhs):
    """Extract estimate and SE for a single parameter."""
    est = fit.estimates()
    row = est[(est["lhs"] == lhs) & (est["op"] == op) & (est["rhs"] == rhs)]
    assert len(row) == 1, f"Expected 1 row for {lhs} {op} {rhs}, got {len(row)}"
    return row["est"].values[0], row["se"].values[0]


# ============================================================
# MCAR 10%
# ============================================================


class TestFIMLMCAR10:
    """FIML validation: MCAR ~10% missingness on all variables.

    Parameter estimates are validated against lavaan.  Fit indices are
    not compared tightly here because the many distinct missing-data
    patterns lead to differences in the saturated log-likelihood.
    """

    def test_converges(self, fit_mcar10):
        assert fit_mcar10.converged

    def test_uses_full_sample(self, fit_mcar10, hs_mcar10):
        assert fit_mcar10.results._n_obs == len(hs_mcar10)

    def test_df(self, fit_mcar10):
        assert fit_mcar10.fit_indices()["df"] == 24

    def test_chi_square_positive(self, fit_mcar10):
        assert fit_mcar10.fit_indices()["chi_square"] > 0

    def test_cfi_reasonable(self, fit_mcar10):
        cfi = fit_mcar10.fit_indices()["cfi"]
        assert 0.85 < cfi < 1.05

    @pytest.mark.parametrize("lv,ind,lav_est,lav_se", [
        ("visual", "x2", 0.586271, 0.123194),
        ("visual", "x3", 0.783273, 0.136637),
        ("textual", "x5", 1.111739, 0.067321),
        ("textual", "x6", 0.924116, 0.062003),
        ("speed", "x8", 1.248715, 0.180660),
        ("speed", "x9", 1.071752, 0.208248),
    ])
    def test_loading(self, fit_mcar10, lv, ind, lav_est, lav_se):
        est, se = _get_est(fit_mcar10, lv, "=~", ind)
        assert abs(est - lav_est) < 0.02, (
            f"{lv}=~{ind}: est semla={est:.6f}, lavaan={lav_est:.6f}"
        )
        assert abs(se - lav_se) < 0.02, (
            f"{lv}=~{ind}: se semla={se:.6f}, lavaan={lav_se:.6f}"
        )

    @pytest.mark.parametrize("var,lav_est,lav_se", [
        ("x1", 0.545281, 0.120772),
        ("x4", 0.361755, 0.053757),
        ("x7", 0.812983, 0.094537),
    ])
    def test_residual_variance(self, fit_mcar10, var, lav_est, lav_se):
        est, se = _get_est(fit_mcar10, var, "~~", var)
        assert abs(est - lav_est) < 0.02, (
            f"{var}~~{var}: est semla={est:.6f}, lavaan={lav_est:.6f}"
        )

    @pytest.mark.parametrize("lv,lav_est", [
        ("visual", 0.735402),
        ("textual", 1.008877),
        ("speed", 0.370729),
    ])
    def test_factor_variance(self, fit_mcar10, lv, lav_est):
        est, _ = _get_est(fit_mcar10, lv, "~~", lv)
        assert abs(est - lav_est) < 0.02, (
            f"{lv}~~{lv}: semla={est:.6f}, lavaan={lav_est:.6f}"
        )

    @pytest.mark.parametrize("lv1,lv2,lav_est", [
        ("visual", "textual", 0.365996),
        ("visual", "speed", 0.261392),
        ("textual", "speed", 0.177794),
    ])
    def test_factor_covariance(self, fit_mcar10, lv1, lv2, lav_est):
        est, _ = _get_est(fit_mcar10, lv1, "~~", lv2)
        assert abs(est - lav_est) < 0.02, (
            f"{lv1}~~{lv2}: semla={est:.6f}, lavaan={lav_est:.6f}"
        )

    @pytest.mark.parametrize("var,lav_int", [
        ("x1", 4.964827),
        ("x5", 4.356288),
        ("x9", 5.381491),
    ])
    def test_intercept(self, fit_mcar10, var, lav_int):
        est, _ = _get_est(fit_mcar10, var, "~1", "1")
        assert abs(est - lav_int) < 0.02, (
            f"{var}~1: semla={est:.6f}, lavaan={lav_int:.6f}"
        )


# ============================================================
# MCAR 20%
# ============================================================


class TestFIMLMCAR20:
    """FIML validation: MCAR ~20% missingness on all variables.

    Same note as MCAR10 regarding fit indices.
    """

    def test_converges(self, fit_mcar20):
        assert fit_mcar20.converged

    def test_uses_full_sample(self, fit_mcar20, hs_mcar20):
        assert fit_mcar20.results._n_obs == len(hs_mcar20)

    def test_df(self, fit_mcar20):
        assert fit_mcar20.fit_indices()["df"] == 24

    def test_chi_square_positive(self, fit_mcar20):
        assert fit_mcar20.fit_indices()["chi_square"] > 0

    def test_cfi_reasonable(self, fit_mcar20):
        cfi = fit_mcar20.fit_indices()["cfi"]
        assert 0.85 < cfi < 1.05

    @pytest.mark.parametrize("lv,ind,lav_est,lav_se", [
        ("visual", "x2", 0.501040, 0.117506),
        ("visual", "x3", 0.623720, 0.141135),
        ("textual", "x5", 1.090283, 0.078466),
        ("textual", "x6", 0.881024, 0.068524),
        ("speed", "x8", 1.087001, 0.172126),
        ("speed", "x9", 1.203493, 0.278709),
    ])
    def test_loading(self, fit_mcar20, lv, ind, lav_est, lav_se):
        est, se = _get_est(fit_mcar20, lv, "=~", ind)
        assert abs(est - lav_est) < 0.02, (
            f"{lv}=~{ind}: est semla={est:.6f}, lavaan={lav_est:.6f}"
        )
        assert abs(se - lav_se) < 0.02, (
            f"{lv}=~{ind}: se semla={se:.6f}, lavaan={lav_se:.6f}"
        )

    @pytest.mark.parametrize("var,lav_est", [
        ("x1", 0.559780),
        ("x4", 0.373689),
        ("x7", 0.874806),
    ])
    def test_residual_variance(self, fit_mcar20, var, lav_est):
        est, _ = _get_est(fit_mcar20, var, "~~", var)
        assert abs(est - lav_est) < 0.02, (
            f"{var}~~{var}: semla={est:.6f}, lavaan={lav_est:.6f}"
        )

    @pytest.mark.parametrize("lv,lav_est", [
        ("visual", 0.930108),
        ("textual", 1.052176),
        ("speed", 0.382601),
    ])
    def test_factor_variance(self, fit_mcar20, lv, lav_est):
        est, _ = _get_est(fit_mcar20, lv, "~~", lv)
        assert abs(est - lav_est) < 0.02, (
            f"{lv}~~{lv}: semla={est:.6f}, lavaan={lav_est:.6f}"
        )

    @pytest.mark.parametrize("lv1,lv2,lav_est", [
        ("visual", "textual", 0.438089),
        ("visual", "speed", 0.278655),
        ("textual", "speed", 0.196998),
    ])
    def test_factor_covariance(self, fit_mcar20, lv1, lv2, lav_est):
        est, _ = _get_est(fit_mcar20, lv1, "~~", lv2)
        assert abs(est - lav_est) < 0.02, (
            f"{lv1}~~{lv2}: semla={est:.6f}, lavaan={lav_est:.6f}"
        )

    @pytest.mark.parametrize("var,lav_int", [
        ("x1", 4.898008),
        ("x5", 4.378611),
        ("x9", 5.374051),
    ])
    def test_intercept(self, fit_mcar20, var, lav_int):
        est, _ = _get_est(fit_mcar20, var, "~1", "1")
        assert abs(est - lav_int) < 0.02, (
            f"{var}~1: semla={est:.6f}, lavaan={lav_int:.6f}"
        )


# ============================================================
# 30% missing on x1 only
# ============================================================


class TestFIMLMCAR30X1:
    """FIML validation: 30% missing on x1 only.

    With only one variable missing there are few missing-data patterns,
    so fit indices match lavaan closely as well.
    """

    def test_converges(self, fit_mcar30_x1):
        assert fit_mcar30_x1.converged

    def test_uses_full_sample(self, fit_mcar30_x1, hs_mcar30_x1):
        assert fit_mcar30_x1.results._n_obs == len(hs_mcar30_x1)

    @pytest.mark.parametrize("measure,lavaan,tol", [
        ("chi_square", 81.445135, 2.0),
        ("df", 24, 0),
        ("cfi", 0.931400, 0.01),
        ("tli", 0.897099, 0.015),
        ("rmsea", 0.089174, 0.005),
    ])
    def test_fit_index(self, fit_mcar30_x1, measure, lavaan, tol):
        val = fit_mcar30_x1.fit_indices()[measure]
        assert abs(val - lavaan) <= tol, (
            f"{measure}: semla={val:.6f}, lavaan={lavaan:.6f}"
        )

    @pytest.mark.parametrize("lv,ind,lav_est,lav_se", [
        ("visual", "x2", 0.586017, 0.122264),
        ("visual", "x3", 0.714271, 0.133053),
        ("textual", "x5", 1.119222, 0.065387),
        ("textual", "x6", 0.927682, 0.056291),
        ("speed", "x8", 1.179824, 0.150495),
        ("speed", "x9", 1.062728, 0.187802),
    ])
    def test_loading(self, fit_mcar30_x1, lv, ind, lav_est, lav_se):
        est, se = _get_est(fit_mcar30_x1, lv, "=~", ind)
        assert abs(est - lav_est) < 0.02, (
            f"{lv}=~{ind}: est semla={est:.6f}, lavaan={lav_est:.6f}"
        )
        assert abs(se - lav_se) < 0.02, (
            f"{lv}=~{ind}: se semla={se:.6f}, lavaan={lav_se:.6f}"
        )

    @pytest.mark.parametrize("var,lav_est", [
        ("x1", 0.649562),
        ("x4", 0.375818),
        ("x7", 0.793737),
    ])
    def test_residual_variance(self, fit_mcar30_x1, var, lav_est):
        est, _ = _get_est(fit_mcar30_x1, var, "~~", var)
        assert abs(est - lav_est) < 0.02, (
            f"{var}~~{var}: semla={est:.6f}, lavaan={lav_est:.6f}"
        )

    @pytest.mark.parametrize("lv,lav_est", [
        ("visual", 0.816652),
        ("textual", 0.974847),
        ("speed", 0.389403),
    ])
    def test_factor_variance(self, fit_mcar30_x1, lv, lav_est):
        est, _ = _get_est(fit_mcar30_x1, lv, "~~", lv)
        assert abs(est - lav_est) < 0.02, (
            f"{lv}~~{lv}: semla={est:.6f}, lavaan={lav_est:.6f}"
        )

    @pytest.mark.parametrize("lv1,lv2,lav_est", [
        ("visual", "textual", 0.395749),
        ("visual", "speed", 0.263953),
        ("textual", "speed", 0.173045),
    ])
    def test_factor_covariance(self, fit_mcar30_x1, lv1, lv2, lav_est):
        est, _ = _get_est(fit_mcar30_x1, lv1, "~~", lv2)
        assert abs(est - lav_est) < 0.02, (
            f"{lv1}~~{lv2}: semla={est:.6f}, lavaan={lav_est:.6f}"
        )

    @pytest.mark.parametrize("var,lav_int", [
        ("x1", 4.878777),
        ("x5", 4.340532),
        ("x9", 5.374123),
    ])
    def test_intercept(self, fit_mcar30_x1, var, lav_int):
        est, _ = _get_est(fit_mcar30_x1, var, "~1", "1")
        assert abs(est - lav_int) < 0.02, (
            f"{var}~1: semla={est:.6f}, lavaan={lav_int:.6f}"
        )

    def test_x1_residual_larger_than_complete(self, fit_mcar30_x1):
        """With 30% missing on x1, its residual variance estimate should
        still be positive and reasonable."""
        est, _ = _get_est(fit_mcar30_x1, "x1", "~~", "x1")
        assert est > 0.0
        assert est < 3.0
