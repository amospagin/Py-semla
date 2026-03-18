"""Tests for robust ML (MLR) estimation."""

import numpy as np
import pytest
from semla import cfa
from semla.datasets import HolzingerSwineford1939


@pytest.fixture(scope="module")
def hs_data():
    return HolzingerSwineford1939()


@pytest.fixture(scope="module")
def fit_ml(hs_data):
    return cfa("""
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """, data=hs_data)


@pytest.fixture(scope="module")
def fit_mlr(hs_data):
    return cfa("""
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """, data=hs_data, estimator="MLR")


class TestMLR:
    def test_converged(self, fit_mlr):
        assert fit_mlr.converged

    def test_same_estimates_as_ml(self, fit_ml, fit_mlr):
        """MLR should give identical parameter estimates to ML."""
        est_ml = fit_ml.estimates()
        est_mlr = fit_mlr.estimates()
        free_ml = est_ml[est_ml["free"]]["est"].values
        free_mlr = est_mlr[est_mlr["free"]]["est"].values
        np.testing.assert_allclose(free_ml, free_mlr, atol=0.001)

    def test_scaled_chi_square(self, fit_ml, fit_mlr):
        """Scaled chi-square should differ from ML chi-square."""
        chi_ml = fit_ml.fit_indices()["chi_square"]
        chi_mlr = fit_mlr.fit_indices()["chi_square"]
        # Should be different (unless data is perfectly normal)
        assert chi_ml != chi_mlr

    def test_scaled_chi_square_positive(self, fit_mlr):
        assert fit_mlr.fit_indices()["chi_square"] > 0

    def test_same_df(self, fit_ml, fit_mlr):
        assert fit_ml.fit_indices()["df"] == fit_mlr.fit_indices()["df"]

    def test_robust_se_positive(self, fit_mlr):
        est = fit_mlr.estimates()
        free_se = est[est["free"]]["se"]
        assert (free_se > 0).all()

    def test_robust_se_different_from_ml(self, fit_ml, fit_mlr):
        """Robust SEs should differ from standard ML SEs."""
        se_ml = fit_ml.estimates()[fit_ml.estimates()["free"]]["se"].values
        se_mlr = fit_mlr.estimates()[fit_mlr.estimates()["free"]]["se"].values
        assert not np.allclose(se_ml, se_mlr, atol=0.001)

    def test_estimator_in_summary(self, fit_mlr):
        summary = fit_mlr.summary()
        assert "MLR" in summary

    def test_fit_indices_reasonable(self, fit_mlr):
        idx = fit_mlr.fit_indices()
        assert 0 < idx["cfi"] < 1.1
        assert idx["rmsea"] > 0
        assert idx["srmr"] > 0
