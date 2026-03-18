"""Tests for FIML (Full Information Maximum Likelihood) missing data handling."""

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


@pytest.fixture(scope="module")
def hs_data():
    return HolzingerSwineford1939()


@pytest.fixture(scope="module")
def hs_missing(hs_data):
    """Holzinger-Swineford with 15% MCAR missingness."""
    rng = np.random.default_rng(42)
    df = hs_data.copy()
    for col in ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]:
        df.loc[rng.random(len(df)) < 0.15, col] = np.nan
    return df


class TestFIMLComplete:
    """FIML on complete data should match ML."""

    def test_converges(self, hs_data):
        fit = cfa(HS_MODEL, data=hs_data, missing="fiml")
        assert fit.converged

    def test_chi_square_close_to_ml(self, hs_data):
        fit_fiml = cfa(HS_MODEL, data=hs_data, missing="fiml")
        fit_ml = cfa(HS_MODEL, data=hs_data, meanstructure=True)
        chi_fiml = fit_fiml.fit_indices()["chi_square"]
        chi_ml = fit_ml.fit_indices()["chi_square"]
        assert abs(chi_fiml - chi_ml) < 1.0

    def test_same_df(self, hs_data):
        fit_fiml = cfa(HS_MODEL, data=hs_data, missing="fiml")
        fit_ml = cfa(HS_MODEL, data=hs_data, meanstructure=True)
        assert fit_fiml.fit_indices()["df"] == fit_ml.fit_indices()["df"]


class TestFIMLMissing:
    """FIML with missing data."""

    def test_converges(self, hs_missing):
        fit = cfa(HS_MODEL, data=hs_missing, missing="fiml")
        assert fit.converged

    def test_uses_all_observations(self, hs_missing):
        fit = cfa(HS_MODEL, data=hs_missing, missing="fiml")
        # N should be full sample, not listwise-deleted
        assert fit.results._n_obs == len(hs_missing)

    def test_chi_square_positive(self, hs_missing):
        fit = cfa(HS_MODEL, data=hs_missing, missing="fiml")
        assert fit.fit_indices()["chi_square"] > 0

    def test_cfi_reasonable(self, hs_missing):
        fit = cfa(HS_MODEL, data=hs_missing, missing="fiml")
        assert 0.8 < fit.fit_indices()["cfi"] < 1.05

    def test_loadings_significant(self, hs_missing):
        fit = cfa(HS_MODEL, data=hs_missing, missing="fiml")
        est = fit.estimates()
        free_loadings = est[(est["op"] == "=~") & (est["free"])]
        assert (free_loadings["z"].abs() > 2).all()

    def test_loadings_close_to_complete_data(self, hs_data, hs_missing):
        """FIML estimates should be close to complete-data estimates."""
        fit_complete = cfa(HS_MODEL, data=hs_data)
        fit_missing = cfa(HS_MODEL, data=hs_missing, missing="fiml")

        est_c = fit_complete.estimates()
        est_m = fit_missing.estimates()

        for ind in ["x2", "x3", "x5", "x6", "x8", "x9"]:
            val_c = est_c[(est_c["op"] == "=~") & (est_c["rhs"] == ind)]["est"].values[0]
            val_m = est_m[(est_m["op"] == "=~") & (est_m["rhs"] == ind)]["est"].values[0]
            assert abs(val_c - val_m) < 0.3, (
                f"{ind}: complete={val_c:.3f}, FIML={val_m:.3f}"
            )

    def test_forces_meanstructure(self, hs_missing):
        fit = cfa(HS_MODEL, data=hs_missing, missing="fiml")
        assert fit.spec.meanstructure

    def test_intercepts_in_estimates(self, hs_missing):
        fit = cfa(HS_MODEL, data=hs_missing, missing="fiml")
        est = fit.estimates()
        intercepts = est[est["op"] == "~1"]
        assert len(intercepts) == 9


class TestMissingWarning:
    def test_warns_on_listwise_deletion(self, hs_missing):
        with pytest.warns(RuntimeWarning, match="missing values"):
            cfa(HS_MODEL, data=hs_missing)
