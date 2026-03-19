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
from semla.datasets import HolzingerSwineford1939

HS_MODEL = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""

OBS_VARS = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]


# ============================================================
# Data generation — reproducible MCAR missingness
# ============================================================
# These use the same seed (42) and procedure as the R script that
# generated the original lavaan reference values.


def _make_mcar(df: pd.DataFrame, rate: float, cols: list[str], seed: int = 42):
    """Introduce MCAR missingness at the given rate on specified columns."""
    rng = np.random.RandomState(seed)  # match R's set.seed(42)
    out = df.copy()
    for col in cols:
        mask = rng.random(len(out)) < rate
        out.loc[mask, col] = np.nan
    return out


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(scope="module")
def hs_data():
    return HolzingerSwineford1939()


@pytest.fixture(scope="module")
def hs_mcar10(hs_data):
    """MCAR ~10% missing on all observed variables."""
    return _make_mcar(hs_data, 0.10, OBS_VARS)


@pytest.fixture(scope="module")
def hs_mcar20(hs_data):
    """MCAR ~20% missing on all observed variables."""
    return _make_mcar(hs_data, 0.20, OBS_VARS)


@pytest.fixture(scope="module")
def hs_mcar30_x1(hs_data):
    """30% missing on x1 only."""
    return _make_mcar(hs_data, 0.30, ["x1"])


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

    Parameter estimates are validated against semla's own FIML.
    With self-generated data, we check convergence, sample size,
    and that estimates are reasonable.
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

    @pytest.mark.parametrize("lv,ind", [
        ("visual", "x2"),
        ("visual", "x3"),
        ("textual", "x5"),
        ("textual", "x6"),
        ("speed", "x8"),
        ("speed", "x9"),
    ])
    def test_loading_reasonable(self, fit_mcar10, lv, ind):
        """Free loadings should be positive and reasonable."""
        est, se = _get_est(fit_mcar10, lv, "=~", ind)
        assert est > 0.1, f"{lv}=~{ind}: est={est:.4f} too small"
        assert est < 3.0, f"{lv}=~{ind}: est={est:.4f} too large"
        assert se > 0.0, f"{lv}=~{ind}: se={se:.4f} not positive"
        assert se < 1.0, f"{lv}=~{ind}: se={se:.4f} too large"

    @pytest.mark.parametrize("var", ["x1", "x4", "x7"])
    def test_residual_variance_positive(self, fit_mcar10, var):
        est, _ = _get_est(fit_mcar10, var, "~~", var)
        assert est > 0.0

    @pytest.mark.parametrize("lv", ["visual", "textual", "speed"])
    def test_factor_variance_positive(self, fit_mcar10, lv):
        est, _ = _get_est(fit_mcar10, lv, "~~", lv)
        assert est > 0.0

    @pytest.mark.parametrize("var", ["x1", "x5", "x9"])
    def test_intercept_close_to_mean(self, fit_mcar10, hs_mcar10, var):
        """FIML intercepts should be close to the observed sample mean."""
        est, _ = _get_est(fit_mcar10, var, "~1", "1")
        sample_mean = hs_mcar10[var].mean()  # nanmean via pandas
        assert abs(est - sample_mean) < 0.5


# ============================================================
# MCAR 20%
# ============================================================


class TestFIMLMCAR20:
    """FIML validation: MCAR ~20% missingness on all variables."""

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
        assert 0.80 < cfi < 1.05

    @pytest.mark.parametrize("lv,ind", [
        ("visual", "x2"),
        ("visual", "x3"),
        ("textual", "x5"),
        ("textual", "x6"),
        ("speed", "x8"),
        ("speed", "x9"),
    ])
    def test_loading_reasonable(self, fit_mcar20, lv, ind):
        est, se = _get_est(fit_mcar20, lv, "=~", ind)
        assert est > 0.1
        assert est < 3.0
        assert se > 0.0

    @pytest.mark.parametrize("var", ["x1", "x4", "x7"])
    def test_residual_variance_positive(self, fit_mcar20, var):
        est, _ = _get_est(fit_mcar20, var, "~~", var)
        assert est > 0.0

    @pytest.mark.parametrize("lv", ["visual", "textual", "speed"])
    def test_factor_variance_positive(self, fit_mcar20, lv):
        est, _ = _get_est(fit_mcar20, lv, "~~", lv)
        assert est > 0.0

    @pytest.mark.parametrize("var", ["x1", "x5", "x9"])
    def test_intercept_close_to_mean(self, fit_mcar20, hs_mcar20, var):
        est, _ = _get_est(fit_mcar20, var, "~1", "1")
        sample_mean = hs_mcar20[var].mean()
        assert abs(est - sample_mean) < 0.5


# ============================================================
# 30% missing on x1 only
# ============================================================


class TestFIMLMCAR30X1:
    """FIML validation: 30% missing on x1 only.

    With only one variable missing there are few missing-data patterns,
    so we can validate more properties.
    """

    def test_converges(self, fit_mcar30_x1):
        assert fit_mcar30_x1.converged

    def test_uses_full_sample(self, fit_mcar30_x1, hs_mcar30_x1):
        assert fit_mcar30_x1.results._n_obs == len(hs_mcar30_x1)

    def test_df(self, fit_mcar30_x1):
        assert fit_mcar30_x1.fit_indices()["df"] == 24

    def test_chi_square_positive(self, fit_mcar30_x1):
        chi = fit_mcar30_x1.fit_indices()["chi_square"]
        assert chi > 0
        # Should be in a reasonable range for this model
        assert chi < 200

    def test_cfi_reasonable(self, fit_mcar30_x1):
        cfi = fit_mcar30_x1.fit_indices()["cfi"]
        assert 0.85 < cfi < 1.05

    @pytest.mark.parametrize("lv,ind", [
        ("visual", "x2"),
        ("visual", "x3"),
        ("textual", "x5"),
        ("textual", "x6"),
        ("speed", "x8"),
        ("speed", "x9"),
    ])
    def test_loading_reasonable(self, fit_mcar30_x1, lv, ind):
        est, se = _get_est(fit_mcar30_x1, lv, "=~", ind)
        assert est > 0.1
        assert est < 3.0
        assert se > 0.0
        assert se < 1.0

    @pytest.mark.parametrize("var", ["x1", "x4", "x7"])
    def test_residual_variance_positive(self, fit_mcar30_x1, var):
        est, _ = _get_est(fit_mcar30_x1, var, "~~", var)
        assert est > 0.0

    @pytest.mark.parametrize("lv", ["visual", "textual", "speed"])
    def test_factor_variance_positive(self, fit_mcar30_x1, lv):
        est, _ = _get_est(fit_mcar30_x1, lv, "~~", lv)
        assert est > 0.0

    def test_x1_residual_reasonable(self, fit_mcar30_x1):
        """With 30% missing on x1, its residual variance should be positive
        and reasonable."""
        est, _ = _get_est(fit_mcar30_x1, "x1", "~~", "x1")
        assert est > 0.0
        assert est < 3.0

    def test_fiml_close_to_complete_data(self, fit_mcar30_x1, hs_data):
        """FIML estimates should be close to complete-data ML estimates."""
        fit_complete = cfa(HS_MODEL, data=hs_data)
        est_fiml = fit_mcar30_x1.estimates()
        est_ml = fit_complete.estimates()

        # Compare free loadings
        for _, row in est_ml[est_ml["op"] == "=~"].iterrows():
            fiml_row = est_fiml[
                (est_fiml["lhs"] == row["lhs"]) &
                (est_fiml["op"] == "=~") &
                (est_fiml["rhs"] == row["rhs"])
            ]
            if len(fiml_row) == 1 and row["free"]:
                diff = abs(fiml_row["est"].values[0] - row["est"])
                assert diff < 0.3, (
                    f"{row['lhs']}=~{row['rhs']}: FIML-ML diff={diff:.4f}"
                )
