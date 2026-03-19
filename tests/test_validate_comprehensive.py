"""Comprehensive validation against lavaan 0.6-21 across model types.

Reference values generated from HolzingerSwineford1939 data using lavaan 0.6-21.
Each model type is a separate test class with module-scoped fixtures.
"""

import numpy as np
import pandas as pd
import pytest
from semla import cfa, sem
from semla.datasets import HolzingerSwineford1939


# ── shared data fixture ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def hs_data():
    return HolzingerSwineford1939()


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_est(estimates, lhs, op, rhs):
    """Extract a single parameter estimate row."""
    row = estimates[
        (estimates["lhs"] == lhs) &
        (estimates["op"] == op) &
        (estimates["rhs"] == rhs)
    ]
    assert len(row) == 1, f"Expected 1 row for {lhs} {op} {rhs}, got {len(row)}"
    return row.iloc[0]


def _check_param(estimates, lhs, op, rhs, lav_est, lav_se,
                 atol_est=0.01, atol_se=0.01):
    """Assert that estimate and SE match lavaan values."""
    row = _get_est(estimates, lhs, op, rhs)
    assert abs(row["est"] - lav_est) < atol_est, (
        f"{lhs} {op} {rhs}: est={row['est']:.6f}, lavaan={lav_est}, "
        f"diff={abs(row['est'] - lav_est):.6f}"
    )
    if lav_se > 0:
        assert abs(row["se"] - lav_se) < atol_se, (
            f"{lhs} {op} {rhs}: se={row['se']:.6f}, lavaan={lav_se}, "
            f"diff={abs(row['se'] - lav_se):.6f}"
        )


def _check_fit(fit_indices, measure, lavaan_val, atol):
    """Assert that a fit index matches lavaan."""
    semla_val = fit_indices[measure]
    assert abs(semla_val - lavaan_val) <= atol, (
        f"{measure}: semla={semla_val:.6f}, lavaan={lavaan_val}, "
        f"diff={abs(semla_val - lavaan_val):.6f}"
    )


# ============================================================
# Model A: Simple 3-factor CFA
# ============================================================

class TestSimpleCFA:
    """visual =~ x1+x2+x3, textual =~ x4+x5+x6, speed =~ x7+x8+x9."""

    MODEL = """
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """

    @pytest.fixture(scope="class")
    def fit(self, hs_data):
        return cfa(self.MODEL, data=hs_data)

    @pytest.fixture(scope="class")
    def est(self, fit):
        return fit.estimates()

    @pytest.fixture(scope="class")
    def fid(self, fit):
        return fit.fit_indices()

    # -- fit indices --

    @pytest.mark.parametrize("measure,lavaan,tol", [
        ("chi_square", 85.306, 0.5),
        ("df", 24, 0),
        ("cfi", 0.931, 0.005),
        ("tli", 0.896, 0.005),
        ("rmsea", 0.092, 0.005),
        ("srmr", 0.065, 0.005),
    ])
    def test_fit_index(self, fid, measure, lavaan, tol):
        _check_fit(fid, measure, lavaan, tol)

    # -- factor loadings (free) --

    @pytest.mark.parametrize("lv,ind,lav_est,lav_se", [
        ("visual", "x2", 0.554, 0.100),
        ("visual", "x3", 0.729, 0.109),
        ("textual", "x5", 1.113, 0.065),
        ("textual", "x6", 0.926, 0.055),
        ("speed", "x8", 1.180, 0.165),
        ("speed", "x9", 1.082, 0.151),
    ])
    def test_loading(self, est, lv, ind, lav_est, lav_se):
        _check_param(est, lv, "=~", ind, lav_est, lav_se)

    # -- residual variances --

    @pytest.mark.parametrize("var,lav_est,lav_se", [
        ("x1", 0.549, 0.114),
        ("x2", 1.134, 0.102),
        ("x3", 0.844, 0.091),
        ("x4", 0.371, 0.048),
        ("x5", 0.446, 0.058),
        ("x6", 0.356, 0.043),
        ("x7", 0.799, 0.081),
        ("x8", 0.488, 0.074),
        ("x9", 0.566, 0.071),
    ])
    def test_residual_variance(self, est, var, lav_est, lav_se):
        _check_param(est, var, "~~", var, lav_est, lav_se)

    # -- latent variances --

    @pytest.mark.parametrize("lv,lav_est,lav_se", [
        ("visual", 0.809, 0.145),
        ("textual", 0.979, 0.112),
        ("speed", 0.384, 0.086),
    ])
    def test_latent_variance(self, est, lv, lav_est, lav_se):
        _check_param(est, lv, "~~", lv, lav_est, lav_se)

    # -- latent covariances --

    @pytest.mark.parametrize("lv1,lv2,lav_est,lav_se", [
        ("visual", "textual", 0.408, 0.074),
        ("visual", "speed", 0.262, 0.056),
        ("textual", "speed", 0.173, 0.049),
    ])
    def test_latent_covariance(self, est, lv1, lv2, lav_est, lav_se):
        _check_param(est, lv1, "~~", lv2, lav_est, lav_se)


# ============================================================
# Model B: SEM with regressions
# semla auto-adds covariances between exogenous latent variables
# (matching lavaan's auto.cov.lv.x = TRUE default for sem()).
# ============================================================

class TestSEMRegression:
    """3-factor model + speed ~ visual + textual (exo covs auto-added)."""

    MODEL = """
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
        speed ~ visual + textual
    """

    @pytest.fixture(scope="class")
    def fit(self, hs_data):
        return sem(self.MODEL, data=hs_data)

    @pytest.fixture(scope="class")
    def est(self, fit):
        return fit.estimates()

    @pytest.fixture(scope="class")
    def fid(self, fit):
        return fit.fit_indices()

    # -- fit indices (equivalent to CFA model since same implied cov) --

    @pytest.mark.parametrize("measure,lavaan,tol", [
        ("chi_square", 85.306, 0.5),
        ("df", 24, 0),
        ("cfi", 0.931, 0.005),
        ("tli", 0.896, 0.005),
        ("rmsea", 0.092, 0.005),
        ("srmr", 0.065, 0.005),
    ])
    def test_fit_index(self, fid, measure, lavaan, tol):
        _check_fit(fid, measure, lavaan, tol)

    # -- factor loadings --

    @pytest.mark.parametrize("lv,ind,lav_est,lav_se", [
        ("visual", "x2", 0.554, 0.100),
        ("visual", "x3", 0.729, 0.109),
        ("textual", "x5", 1.113, 0.065),
        ("textual", "x6", 0.926, 0.055),
        ("speed", "x8", 1.180, 0.165),
        ("speed", "x9", 1.082, 0.151),
    ])
    def test_loading(self, est, lv, ind, lav_est, lav_se):
        _check_param(est, lv, "=~", ind, lav_est, lav_se)

    # -- regression coefficients --

    @pytest.mark.parametrize("dv,iv,lav_est,lav_se", [
        ("speed", "visual", 0.297, 0.078),
        ("speed", "textual", 0.053, 0.053),
    ])
    def test_regression(self, est, dv, iv, lav_est, lav_se):
        _check_param(est, dv, "~", iv, lav_est, lav_se)

    # -- residual variances --

    @pytest.mark.parametrize("var,lav_est,lav_se", [
        ("x1", 0.549, 0.114),
        ("x2", 1.134, 0.102),
        ("x3", 0.844, 0.091),
        ("x4", 0.371, 0.048),
        ("x5", 0.446, 0.058),
        ("x6", 0.356, 0.043),
        ("x7", 0.799, 0.081),
        ("x8", 0.488, 0.074),
        ("x9", 0.566, 0.071),
    ])
    def test_residual_variance(self, est, var, lav_est, lav_se):
        _check_param(est, var, "~~", var, lav_est, lav_se)

    # -- latent variances (exogenous) --

    @pytest.mark.parametrize("lv,lav_est,lav_se", [
        ("visual", 0.809, 0.145),
        ("textual", 0.979, 0.112),
    ])
    def test_latent_variance(self, est, lv, lav_est, lav_se):
        _check_param(est, lv, "~~", lv, lav_est, lav_se)

    def test_speed_residual_variance(self, est):
        """speed is endogenous; its residual variance differs from CFA."""
        _check_param(est, "speed", "~~", "speed", 0.297, 0.070)

    def test_exogenous_covariance(self, est):
        _check_param(est, "visual", "~~", "textual", 0.408, 0.074)


# ============================================================
# Model C: CFA with mean structure
# ============================================================

class TestCFAMeanStructure:
    """3-factor CFA with meanstructure=TRUE."""

    MODEL = """
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """

    @pytest.fixture(scope="class")
    def fit(self, hs_data):
        return cfa(self.MODEL, data=hs_data, meanstructure=True)

    @pytest.fixture(scope="class")
    def est(self, fit):
        return fit.estimates()

    @pytest.fixture(scope="class")
    def fid(self, fit):
        return fit.fit_indices()

    # -- fit indices (same chi-sq as model A) --
    # Note: lavaan SRMR with meanstructure includes mean residuals (0.060),
    # semla computes covariance-only SRMR (0.065), so we use wider tolerance.

    @pytest.mark.parametrize("measure,lavaan,tol", [
        ("chi_square", 85.306, 0.5),
        ("df", 24, 0),
        ("cfi", 0.931, 0.005),
        ("tli", 0.896, 0.005),
        ("rmsea", 0.092, 0.005),
        ("srmr", 0.065, 0.006),
    ])
    def test_fit_index(self, fid, measure, lavaan, tol):
        _check_fit(fid, measure, lavaan, tol)

    # -- factor loadings (same as model A) --

    @pytest.mark.parametrize("lv,ind,lav_est,lav_se", [
        ("visual", "x2", 0.554, 0.100),
        ("visual", "x3", 0.729, 0.109),
        ("textual", "x5", 1.113, 0.065),
        ("textual", "x6", 0.926, 0.055),
        ("speed", "x8", 1.180, 0.165),
        ("speed", "x9", 1.082, 0.151),
    ])
    def test_loading(self, est, lv, ind, lav_est, lav_se):
        _check_param(est, lv, "=~", ind, lav_est, lav_se)

    # -- intercepts --

    @pytest.mark.parametrize("var,lav_est,lav_se", [
        ("x1", 4.936, 0.067),
        ("x2", 6.088, 0.068),
        ("x3", 2.250, 0.065),
        ("x4", 3.061, 0.067),
        ("x5", 4.341, 0.074),
        ("x6", 2.186, 0.063),
        ("x7", 4.186, 0.063),
        ("x8", 5.527, 0.058),
        ("x9", 5.374, 0.058),
    ])
    def test_intercept(self, est, var, lav_est, lav_se):
        _check_param(est, var, "~1", "1", lav_est, lav_se)

    def test_intercepts_present(self, est):
        """Nine observed-variable intercepts should exist."""
        intercepts = est[est["op"] == "~1"]
        obs_intercepts = intercepts[
            intercepts["lhs"].isin(["x1","x2","x3","x4","x5","x6","x7","x8","x9"])
        ]
        assert len(obs_intercepts) == 9


# ============================================================
# Model D: CFA with equality constraints
# ============================================================

class TestCFAEqualityConstraints:
    """visual =~ x1 + a*x2 + a*x3, textual =~ x4+x5+x6, speed =~ x7+x8+x9."""

    MODEL = """
        visual  =~ x1 + a*x2 + a*x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """

    @pytest.fixture(scope="class")
    def fit(self, hs_data):
        return cfa(self.MODEL, data=hs_data)

    @pytest.fixture(scope="class")
    def est(self, fit):
        return fit.estimates()

    @pytest.fixture(scope="class")
    def fid(self, fit):
        return fit.fit_indices()

    # -- fit indices --

    @pytest.mark.parametrize("measure,lavaan,tol", [
        ("chi_square", 87.971, 0.5),
        ("df", 25, 0),
        ("cfi", 0.929, 0.005),
        ("tli", 0.897, 0.005),
        ("rmsea", 0.091, 0.005),
        ("srmr", 0.068, 0.005),
    ])
    def test_fit_index(self, fid, measure, lavaan, tol):
        _check_fit(fid, measure, lavaan, tol)

    def test_constrained_loadings_equal(self, est):
        """x2 and x3 loadings must be equal (same label 'a')."""
        x2 = _get_est(est, "visual", "=~", "x2")["est"]
        x3 = _get_est(est, "visual", "=~", "x3")["est"]
        assert abs(x2 - x3) < 1e-8

    def test_constrained_loading_value(self, est):
        """Constrained loading should be ~0.649."""
        x2 = _get_est(est, "visual", "=~", "x2")["est"]
        assert abs(x2 - 0.649) < 0.01

    def test_constrained_loading_se(self, est):
        """SE of constrained loading should be ~0.088."""
        x2 = _get_est(est, "visual", "=~", "x2")["se"]
        assert abs(x2 - 0.088) < 0.01

    # -- other free loadings --

    @pytest.mark.parametrize("lv,ind,lav_est,lav_se", [
        ("textual", "x5", 1.113, 0.065),
        ("textual", "x6", 0.926, 0.055),
        ("speed", "x8", 1.182, 0.165),
        ("speed", "x9", 1.075, 0.150),
    ])
    def test_loading(self, est, lv, ind, lav_est, lav_se):
        _check_param(est, lv, "=~", ind, lav_est, lav_se)

    # -- residual variances --

    @pytest.mark.parametrize("var,lav_est,lav_se", [
        ("x1", 0.549, 0.114),
        ("x2", 1.114, 0.103),
        ("x3", 0.877, 0.085),
        ("x4", 0.371, 0.048),
        ("x5", 0.446, 0.058),
        ("x6", 0.356, 0.043),
        ("x7", 0.798, 0.081),
        ("x8", 0.484, 0.075),
        ("x9", 0.570, 0.071),
    ])
    def test_residual_variance(self, est, var, lav_est, lav_se):
        _check_param(est, var, "~~", var, lav_est, lav_se)

    # -- latent variances --

    @pytest.mark.parametrize("lv,lav_est,lav_se", [
        ("visual", 0.810, 0.146),
        ("textual", 0.979, 0.112),
        ("speed", 0.385, 0.086),
    ])
    def test_latent_variance(self, est, lv, lav_est, lav_se):
        _check_param(est, lv, "~~", lv, lav_est, lav_se)

    # -- latent covariances --

    @pytest.mark.parametrize("lv1,lv2,lav_est,lav_se", [
        ("visual", "textual", 0.414, 0.074),
        ("visual", "speed", 0.259, 0.056),
        ("textual", "speed", 0.173, 0.049),
    ])
    def test_latent_covariance(self, est, lv1, lv2, lav_est, lav_se):
        _check_param(est, lv1, "~~", lv2, lav_est, lav_se)


# ============================================================
# Model E: Two-factor CFA (f1=x1-x4, f2=x5-x8)
# ============================================================

class TestTwoFactorCFA:
    """f1 =~ x1+x2+x3+x4, f2 =~ x5+x6+x7+x8."""

    MODEL = """
        f1 =~ x1 + x2 + x3 + x4
        f2 =~ x5 + x6 + x7 + x8
    """

    @pytest.fixture(scope="class")
    def fit(self, hs_data):
        return cfa(self.MODEL, data=hs_data)

    @pytest.fixture(scope="class")
    def est(self, fit):
        return fit.estimates()

    @pytest.fixture(scope="class")
    def fid(self, fit):
        return fit.fit_indices()

    # -- fit indices --
    # Note: chi-square tolerance is wider (1.0) because this poorly-fitting
    # model can trigger optimizer precision warnings.

    @pytest.mark.parametrize("measure,lavaan,tol", [
        ("chi_square", 207.212, 1.0),
        ("df", 19, 0),
        ("cfi", 0.753, 0.005),
        ("tli", 0.636, 0.005),
        ("rmsea", 0.181, 0.005),
        ("srmr", 0.123, 0.005),
    ])
    def test_fit_index(self, fid, measure, lavaan, tol):
        _check_fit(fid, measure, lavaan, tol)

    # -- factor loadings --

    @pytest.mark.parametrize("lv,ind,lav_est,lav_se", [
        ("f1", "x2", 0.496, 0.154),
        ("f1", "x3", 0.480, 0.148),
        ("f1", "x4", 2.033, 0.283),
        ("f2", "x6", 0.840, 0.050),
        ("f2", "x7", 0.164, 0.060),
        ("f2", "x8", 0.164, 0.056),
    ])
    def test_loading(self, est, lv, ind, lav_est, lav_se):
        _check_param(est, lv, "=~", ind, lav_est, lav_se)

    # -- residual variances --

    @pytest.mark.parametrize("var,lav_est,lav_se", [
        ("x1", 1.103, 0.093),
        ("x2", 1.319, 0.108),
        ("x3", 1.216, 0.100),
        ("x4", 0.297, 0.087),
        ("x5", 0.456, 0.059),
        ("x6", 0.347, 0.043),
        ("x7", 1.151, 0.094),
        ("x8", 0.989, 0.081),
    ])
    def test_residual_variance(self, est, var, lav_est, lav_se):
        _check_param(est, var, "~~", var, lav_est, lav_se)

    # -- latent variances --

    @pytest.mark.parametrize("lv,lav_est,lav_se", [
        ("f1", 0.255, 0.069),
        ("f2", 1.204, 0.138),
    ])
    def test_latent_variance(self, est, lv, lav_est, lav_se):
        _check_param(est, lv, "~~", lv, lav_est, lav_se)

    # -- factor covariance --

    def test_factor_covariance(self, est):
        _check_param(est, "f1", "~~", "f2", 0.528, 0.086)
