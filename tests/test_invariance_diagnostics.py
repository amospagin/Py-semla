"""Tests for scalar/strict invariance and diagnostics."""

import numpy as np
import pytest
from semla import cfa, chi_square_diff_test, mardia_test
from semla.datasets import HolzingerSwineford1939


HS_MODEL = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""


@pytest.fixture(scope="module")
def hs_data():
    return HolzingerSwineford1939()


class TestScalarInvariance:
    def test_converges(self, hs_data):
        fit = cfa(HS_MODEL, data=hs_data, group="school", invariance="scalar")
        assert fit.converged

    def test_df_greater_than_metric(self, hs_data):
        fit_m = cfa(HS_MODEL, data=hs_data, group="school", invariance="metric")
        fit_s = cfa(HS_MODEL, data=hs_data, group="school", invariance="scalar")
        assert fit_s.fit_indices()["df"] > fit_m.fit_indices()["df"]

    def test_forces_meanstructure(self, hs_data):
        fit = cfa(HS_MODEL, data=hs_data, group="school", invariance="scalar")
        assert fit.mg_spec.group_specs[0].meanstructure


class TestStrictInvariance:
    def test_converges(self, hs_data):
        fit = cfa(HS_MODEL, data=hs_data, group="school", invariance="strict")
        assert fit.converged

    def test_df_greater_than_scalar(self, hs_data):
        fit_s = cfa(HS_MODEL, data=hs_data, group="school", invariance="scalar")
        fit_st = cfa(HS_MODEL, data=hs_data, group="school", invariance="strict")
        assert fit_st.fit_indices()["df"] > fit_s.fit_indices()["df"]

    def test_progressive_chi_square(self, hs_data):
        """Chi-square should increase: configural < metric < scalar < strict."""
        fits = {}
        for inv in ["configural", "metric", "scalar", "strict"]:
            fits[inv] = cfa(HS_MODEL, data=hs_data, group="school", invariance=inv)
        chi = {k: v.fit_indices()["chi_square"] for k, v in fits.items()}
        assert chi["configural"] <= chi["metric"] + 0.1
        assert chi["metric"] <= chi["scalar"] + 0.1
        assert chi["scalar"] <= chi["strict"] + 0.1

    def test_diff_test_works(self, hs_data):
        fit_s = cfa(HS_MODEL, data=hs_data, group="school", invariance="scalar")
        fit_st = cfa(HS_MODEL, data=hs_data, group="school", invariance="strict")
        diff = chi_square_diff_test(fit_st, fit_s)
        assert diff["df_diff"] > 0
        assert diff["chi_sq_diff"] > 0


class TestResiduals:
    def test_raw_residuals(self, hs_data):
        fit = cfa(HS_MODEL, data=hs_data)
        resid = fit.residuals()
        assert resid.shape == (9, 9)
        # Diagonal should be near zero (variances explained)
        assert np.abs(np.diag(resid)).max() < 0.1

    def test_standardized_residuals(self, hs_data):
        fit = cfa(HS_MODEL, data=hs_data)
        resid = fit.residuals(type="standardized")
        assert resid.shape == (9, 9)
        assert np.abs(resid).max() < 0.5  # should be small for a reasonable model

    def test_invalid_type_raises(self, hs_data):
        fit = cfa(HS_MODEL, data=hs_data)
        with pytest.raises(ValueError):
            fit.residuals(type="invalid")


class TestMardiaTest:
    def test_returns_dict(self, hs_data):
        result = mardia_test(hs_data[["x1", "x2", "x3"]])
        assert "skewness" in result
        assert "kurtosis" in result
        assert "recommendation" in result

    def test_recommendation_is_valid(self, hs_data):
        result = mardia_test(hs_data[["x1", "x2", "x3"]])
        assert result["recommendation"] in ("ML", "MLR")

    def test_p_values_between_0_and_1(self, hs_data):
        result = mardia_test(hs_data[["x1", "x2", "x3"]])
        assert 0 <= result["skewness_p"] <= 1
        assert 0 <= result["kurtosis_p"] <= 1

    def test_normal_data_recommends_ml(self):
        """Truly normal data should not reject normality."""
        rng = np.random.default_rng(42)
        data = rng.multivariate_normal([0, 0, 0], np.eye(3), 2000)
        result = mardia_test(data)
        assert result["recommendation"] == "ML"

    def test_accepts_dataframe(self, hs_data):
        result = mardia_test(hs_data)
        assert "skewness" in result
