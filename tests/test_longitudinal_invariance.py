"""Tests for longitudinal measurement invariance (#66)."""

import numpy as np
import pandas as pd
import pytest
from semla import longitudinalInvariance


def _generate_longitudinal_data(n=500, seed=42, invariant=True):
    """Generate 2-wave longitudinal data from a known CFA model."""
    rng = np.random.default_rng(seed)
    loadings = [1.0, 0.8, 0.7, 0.6]
    n_items = 4
    resid_var = 0.4
    resid_cov = 0.1
    factor_var = 1.0
    factor_cov = 0.5

    p = 2 * n_items
    Sigma = np.zeros((p, p))
    for i in range(n_items):
        for j in range(n_items):
            lam_t2 = loadings if invariant else [1.0, 1.5, 0.3, 0.6]
            Sigma[i, j] = loadings[i] * loadings[j] * factor_var
            Sigma[n_items+i, n_items+j] = lam_t2[i] * lam_t2[j] * factor_var
            Sigma[i, n_items+j] = loadings[i] * lam_t2[j] * factor_cov
            Sigma[n_items+j, i] = lam_t2[i] * loadings[j] * factor_cov
    for i in range(p):
        Sigma[i, i] += resid_var
    for i in range(n_items):
        Sigma[i, n_items+i] += resid_cov
        Sigma[n_items+i, i] += resid_cov

    data = rng.multivariate_normal(np.zeros(p), Sigma, n)
    cols = ([f"y{i+1}_t1" for i in range(n_items)]
            + [f"y{i+1}_t2" for i in range(n_items)])
    return pd.DataFrame(data, columns=cols)


MODEL = """
    f_t1 =~ y1_t1 + y2_t1 + y3_t1 + y4_t1
    f_t2 =~ y1_t2 + y2_t2 + y3_t2 + y4_t2
"""

ITEMS = {
    "y1_t1": "y1_t2", "y2_t1": "y2_t2",
    "y3_t1": "y3_t2", "y4_t1": "y4_t2",
}


class TestInvariantData:
    """Data generated with equal loadings across waves should pass metric."""

    @pytest.fixture(scope="class")
    def result(self):
        df = _generate_longitudinal_data(invariant=True)
        return longitudinalInvariance(MODEL, data=df, items=ITEMS)

    def test_returns_result(self, result):
        assert hasattr(result, "highest_level")
        assert hasattr(result, "fits")

    def test_metric_passes(self, result):
        assert result.highest_level == "metric"

    def test_configural_converges(self, result):
        assert result["configural"].converged

    def test_metric_converges(self, result):
        assert result["metric"].converged

    def test_table_has_correct_levels(self, result):
        t = result.table()
        assert list(t["level"]) == ["configural", "metric"]

    def test_metric_df_greater_than_configural(self, result):
        t = result.table()
        assert t.iloc[1]["df"] > t.iloc[0]["df"]

    def test_summary_returns_string(self, result):
        s = result.summary()
        assert isinstance(s, str)
        assert "PASS" in s

    def test_delta_chisq_positive(self, result):
        t = result.table()
        metric_row = t[t["level"] == "metric"].iloc[0]
        assert metric_row["delta_chisq"] > 0
        assert metric_row["delta_p"] > 0.05


class TestNonInvariantData:
    """Data with different loadings should fail metric."""

    @pytest.fixture(scope="class")
    def result(self):
        df = _generate_longitudinal_data(invariant=False)
        return longitudinalInvariance(MODEL, data=df, items=ITEMS)

    def test_highest_is_configural(self, result):
        assert result.highest_level == "configural"

    def test_metric_fails(self, result):
        t = result.table()
        metric_row = t[t["level"] == "metric"].iloc[0]
        assert metric_row["delta_p"] < 0.05


class TestFactorInference:
    """Factor mapping should be auto-inferred from item mapping."""

    def test_infer_works(self):
        df = _generate_longitudinal_data()
        # Don't pass factors= — should infer
        result = longitudinalInvariance(MODEL, data=df, items=ITEMS)
        assert "configural" in result.fits

    def test_explicit_factors(self):
        df = _generate_longitudinal_data()
        result = longitudinalInvariance(
            MODEL, data=df, items=ITEMS,
            factors={"f_t1": "f_t2"},
        )
        assert result.highest_level == "metric"


class TestCorrelatedResiduals:
    """Configural model should include cross-wave residual correlations."""

    def test_correlated_residuals_in_estimates(self):
        df = _generate_longitudinal_data()
        result = longitudinalInvariance(MODEL, data=df, items=ITEMS)
        est = result["configural"].estimates()
        # Check for cross-wave residual covariances
        cross_covs = est[
            (est["op"] == "~~") &
            (est["lhs"] != est["rhs"]) &
            ~est["lhs"].isin(["f_t1", "f_t2"]) &
            ~est["rhs"].isin(["f_t1", "f_t2"])
        ]
        assert len(cross_covs) == 4  # one per item pair
