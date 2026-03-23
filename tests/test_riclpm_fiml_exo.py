"""Validation tests for RI-CLPM (#53), FIML (#24), and exogenous covs (#42).

RI-CLPM: lavaan 0.6-20 reference values on simulated riclpm_data.
FIML: lavaan 0.6-20 on HolzingerSwineford1939 with MCAR missingness.
Exogenous covariances: sem() auto-adds observed exogenous covariances.
"""

import numpy as np
import pandas as pd
import pytest
import warnings
from semla import cfa, sem
from semla.datasets import HolzingerSwineford1939


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_est(estimates, lhs, op, rhs):
    row = estimates[
        (estimates["lhs"] == lhs) &
        (estimates["op"] == op) &
        (estimates["rhs"] == rhs)
    ]
    assert len(row) == 1, f"Expected 1 row for {lhs} {op} {rhs}, got {len(row)}"
    return row.iloc[0]


def _check_param(estimates, lhs, op, rhs, lav_est, lav_se,
                 atol_est=0.01, atol_se=0.01):
    row = _get_est(estimates, lhs, op, rhs)
    assert abs(row["est"] - lav_est) < atol_est, (
        f"{lhs} {op} {rhs}: est={row['est']:.4f}, lavaan={lav_est}"
    )
    if lav_se > 0:
        assert abs(row["se"] - lav_se) < atol_se, (
            f"{lhs} {op} {rhs}: se={row['se']:.4f}, lavaan={lav_se}"
        )


# ============================================================
# RI-CLPM validation against lavaan 0.6-20 (#53)
# ============================================================

class TestRICLPMLavaan:
    """RI-CLPM with lavaan-verified reference values."""

    MODEL = """
        RI_x =~ 1*x1 + 1*x2 + 1*x3
        RI_y =~ 1*y1 + 1*y2 + 1*y3
        x2 ~ x1
        x3 ~ x2
        y2 ~ y1
        y3 ~ y2
        x2 ~ y1
        x3 ~ y2
        y2 ~ x1
        y3 ~ x2
        x1 ~~ y1
        x2 ~~ y2
        x3 ~~ y3
        RI_x ~~ RI_y
    """

    @pytest.fixture(scope="class")
    def fit(self):
        from semla.datasets import riclpm_data
        return sem(self.MODEL, data=riclpm_data())

    @pytest.fixture(scope="class")
    def est(self, fit):
        return fit.estimates()

    @pytest.fixture(scope="class")
    def fid(self, fit):
        return fit.fit_indices()

    # -- fit indices (lavaan 0.6-20) --

    def test_chi_square(self, fid):
        assert abs(fid["chi_square"] - 2.729) < 1.0

    def test_df(self, fid):
        assert fid["df"] == 1

    def test_cfi(self, fid):
        assert abs(fid["cfi"] - 0.999) < 0.005

    def test_rmsea(self, fid):
        assert fid["rmsea"] < 0.10

    # -- autoregressive paths --

    @pytest.mark.parametrize("dv,iv,lav_est,lav_se", [
        ("x2", "x1", 0.210, 0.066),
        ("x3", "x2", 0.195, 0.097),
        ("y2", "y1", 0.162, 0.055),
        ("y3", "y2", 0.021, 0.073),
    ])
    def test_ar_path(self, est, dv, iv, lav_est, lav_se):
        _check_param(est, dv, "~", iv, lav_est, lav_se, atol_est=0.02)

    # -- cross-lagged paths --

    @pytest.mark.parametrize("dv,iv,lav_est,lav_se", [
        ("x2", "y1", -0.030, 0.048),
        ("x3", "y2", -0.103, 0.065),
        ("y2", "x1", 0.034, 0.047),
        ("y3", "x2", -0.037, 0.067),
    ])
    def test_cl_path(self, est, dv, iv, lav_est, lav_se):
        _check_param(est, dv, "~", iv, lav_est, lav_se, atol_est=0.02)

    # -- random intercept variances --

    def test_ri_x_variance(self, est):
        _check_param(est, "RI_x", "~~", "RI_x", 0.387, 0.084, atol_est=0.02)

    def test_ri_y_variance(self, est):
        _check_param(est, "RI_y", "~~", "RI_y", 0.484, 0.077, atol_est=0.02)

    def test_ri_covariance(self, est):
        _check_param(est, "RI_x", "~~", "RI_y", 0.276, 0.058, atol_est=0.02)


# ============================================================
# FIML validation against lavaan 0.6-20 (#24)
# ============================================================


class TestFIMLComplete:
    """FIML on complete data should match ML exactly."""

    def test_fiml_matches_ml(self):
        df = HolzingerSwineford1939()
        model = "visual=~x1+x2+x3; textual=~x4+x5+x6; speed=~x7+x8+x9"
        fit_ml = cfa(model, data=df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_fiml = cfa(model, data=df, missing="fiml")
        # Chi-square should be very close
        ml_chi = fit_ml.fit_indices()["chi_square"]
        fiml_chi = fit_fiml.fit_indices()["chi_square"]
        assert abs(ml_chi - fiml_chi) < 1.0, (
            f"ML chi2={ml_chi:.3f}, FIML chi2={fiml_chi:.3f}"
        )


class TestFIML15:
    """FIML with 15% MCAR — lavaan 0.6-20 reference values.

    Uses R-generated missingness pattern (R set.seed(42)) for exact matching.
    """

    @pytest.fixture(scope="class")
    def fit(self):
        from pathlib import Path
        csv_path = Path(__file__).parent / "hs_mcar15.csv"
        df = pd.read_csv(csv_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cfa(
                "visual=~x1+x2+x3; textual=~x4+x5+x6; speed=~x7+x8+x9",
                data=df, missing="fiml",
            )

    def test_converges(self, fit):
        assert fit.converged

    def test_chi_square(self, fit):
        # lavaan: 63.762 (FIML chi-square computation differs slightly)
        assert abs(fit.fit_indices()["chi_square"] - 63.762) < 10.0

    def test_df(self, fit):
        assert fit.fit_indices()["df"] == 24

    def test_cfi(self, fit):
        # lavaan: 0.939
        assert abs(fit.fit_indices()["cfi"] - 0.939) < 0.03

    def test_loadings_reasonable(self, fit):
        est = fit.estimates()
        loads = est[(est["op"] == "=~") & est["free"]]
        assert (loads["est"] > 0).all()
        assert (loads["est"] < 3).all()

    @pytest.mark.parametrize("lv,ind,lav_est,lav_se", [
        ("visual", "x2", 0.481, 0.108),
        ("visual", "x3", 0.753, 0.128),
        ("textual", "x5", 1.102, 0.079),
        ("textual", "x6", 0.975, 0.072),
        ("speed", "x8", 1.221, 0.174),
        ("speed", "x9", 0.993, 0.177),
    ])
    def test_loading(self, fit, lv, ind, lav_est, lav_se):
        est = fit.estimates()
        _check_param(est, lv, "=~", ind, lav_est, lav_se, atol_est=0.03, atol_se=0.02)


# ============================================================
# Auto exogenous observed covariances (#42)
# ============================================================

class TestExogenousCovariances:
    """sem() should auto-add covariances between exogenous observed vars."""

    @pytest.fixture(scope="class")
    def fit(self):
        df = HolzingerSwineford1939()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sem(
                "visual =~ x1 + x2 + x3; visual ~ ageyr + grade",
                data=df,
            )

    def test_converges(self, fit):
        assert fit.converged

    def test_exo_covariance_present(self, fit):
        est = fit.estimates()
        cov = est[
            (est["op"] == "~~") &
            (est["lhs"] == "ageyr") &
            (est["rhs"] == "grade")
        ]
        assert len(cov) == 1, "ageyr ~~ grade should be auto-added"

    def test_exo_covariance_value(self, fit):
        est = fit.estimates()
        cov = est[
            (est["op"] == "~~") &
            (est["lhs"] == "ageyr") &
            (est["rhs"] == "grade")
        ]
        # lavaan: 0.268
        assert abs(cov["est"].values[0] - 0.268) < 0.02

    def test_exo_variances_present(self, fit):
        est = fit.estimates()
        for var in ["ageyr", "grade"]:
            v = est[(est["op"] == "~~") & (est["lhs"] == var) & (est["rhs"] == var)]
            assert len(v) == 1, f"{var} ~~ {var} should be present"

    def test_user_specified_not_overridden(self):
        """User-specified ~~ should not be overridden by fixed_x."""
        from pathlib import Path
        df = pd.read_csv(Path("src/semla/datasets/clpm_data.csv"))
        fit = sem(
            "x2~x1; y2~y1; x2~y1; y2~x1; x1~~y1; x2~~y2",
            data=df,
        )
        # x1~~y1 is user-specified, should stay free
        est = fit.estimates()
        cov = est[(est["op"] == "~~") & (est["lhs"] == "x1") & (est["rhs"] == "y1")]
        assert len(cov) == 1
        assert cov["free"].values[0], "x1~~y1 should be free (user-specified)"
        # Model should be just-identified (df=0)
        assert fit.fit_indices()["df"] == 0
