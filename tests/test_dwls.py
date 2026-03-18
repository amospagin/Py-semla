"""Tests for DWLS estimation and polychoric correlations."""

import numpy as np
import pandas as pd
import pytest
from semla import cfa
from semla.datasets import HolzingerSwineford1939
from semla.polychoric import polychoric_corr_pair, polychoric_correlation_matrix


@pytest.fixture(scope="module")
def ordinal_data():
    """Create ordinal version of Holzinger-Swineford data."""
    df = HolzingerSwineford1939()
    df_ord = df.copy()
    for col in ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]:
        df_ord[col] = pd.cut(df[col], bins=5, labels=[1, 2, 3, 4, 5]).astype(float)
    return df_ord


@pytest.fixture(scope="module")
def dwls_fit(ordinal_data):
    return cfa("""
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """, data=ordinal_data, estimator="DWLS")


class TestPolychoricCorrelation:
    def test_recovers_known_correlation(self):
        """Polychoric should recover true correlation from discretized normal data."""
        rng = np.random.default_rng(42)
        n = 1000
        rho_true = 0.6
        z = rng.multivariate_normal([0, 0], [[1, rho_true], [rho_true, 1]], n)
        # Discretize into 5 categories
        x = np.digitize(z[:, 0], [-1.5, -0.5, 0.5, 1.5])
        y = np.digitize(z[:, 1], [-1.5, -0.5, 0.5, 1.5])
        rho_est, info = polychoric_corr_pair(x.astype(float), y.astype(float))
        assert abs(rho_est - rho_true) < 0.1, f"Polychoric {rho_est:.3f} too far from {rho_true}"

    def test_polychoric_matrix_is_pd(self, ordinal_data):
        obs = ordinal_data[["x1", "x2", "x3", "x4", "x5", "x6"]].values
        R, _, _ = polychoric_correlation_matrix(obs)
        eigvals = np.linalg.eigvalsh(R)
        assert eigvals[0] > 0, "Polychoric matrix not positive definite"

    def test_polychoric_diagonal_is_one(self, ordinal_data):
        obs = ordinal_data[["x1", "x2", "x3"]].values
        R, _, _ = polychoric_correlation_matrix(obs)
        np.testing.assert_allclose(np.diag(R), 1.0)

    def test_polychoric_symmetric(self, ordinal_data):
        obs = ordinal_data[["x1", "x2", "x3"]].values
        R, _, _ = polychoric_correlation_matrix(obs)
        np.testing.assert_allclose(R, R.T)

    def test_polychoric_close_to_pearson(self, ordinal_data):
        """With 5 categories, polychoric should be close to Pearson."""
        obs = ordinal_data[["x1", "x2", "x3"]].values
        R_poly, _, _ = polychoric_correlation_matrix(obs)
        R_pearson = np.corrcoef(obs, rowvar=False)
        assert np.abs(R_poly - R_pearson).max() < 0.15


class TestDWLSEstimation:
    def test_converged(self, dwls_fit):
        assert dwls_fit.converged

    def test_chi_square_reasonable(self, dwls_fit):
        chi = dwls_fit.fit_indices()["chi_square"]
        # Should be positive and finite
        assert 0 < chi < 500

    def test_cfi_reasonable(self, dwls_fit):
        cfi = dwls_fit.fit_indices()["cfi"]
        assert 0.7 < cfi < 1.0

    def test_loadings_significant(self, dwls_fit):
        est = dwls_fit.estimates()
        free_loadings = est[(est["op"] == "=~") & (est["free"])]
        # All free loadings should be significant (z > 2)
        assert (free_loadings["z"].abs() > 2).all()

    def test_loadings_positive(self, dwls_fit):
        est = dwls_fit.estimates()
        free_loadings = est[(est["op"] == "=~") & (est["free"])]
        assert (free_loadings["est"] > 0).all()

    def test_estimator_in_summary(self, dwls_fit):
        summary = dwls_fit.summary()
        assert "DWLS" in summary


class TestDWLSAPI:
    def test_estimator_parameter(self, ordinal_data):
        fit = cfa("f1 =~ x1 + x2 + x3", data=ordinal_data, estimator="DWLS")
        assert fit.converged

    def test_invalid_estimator(self, ordinal_data):
        with pytest.raises(ValueError, match="Unknown estimator"):
            cfa("f1 =~ x1 + x2 + x3", data=ordinal_data, estimator="GLS")

    def test_dwls_on_continuous_data(self):
        """DWLS should also work on continuous data (uses Pearson as polychoric)."""
        df = HolzingerSwineford1939()
        fit = cfa("""
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
        """, data=df, estimator="DWLS")
        assert fit.converged
