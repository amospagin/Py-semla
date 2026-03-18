"""Tests for input validation and error messages."""

import warnings

import numpy as np
import pandas as pd
import pytest
from semla import cfa


@pytest.fixture
def simple_data():
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "x3": rng.normal(0, 1, n),
    })


class TestMissingVariables:
    def test_single_missing_var(self, simple_data):
        with pytest.raises(ValueError, match="Variable.*not found in data.*x99"):
            cfa("f1 =~ x1 + x2 + x99", data=simple_data)

    def test_multiple_missing_vars(self, simple_data):
        with pytest.raises(ValueError, match="Variable.*not found"):
            cfa("f1 =~ x1 + x88 + x99", data=simple_data)

    def test_shows_available_columns(self, simple_data):
        with pytest.raises(ValueError, match="Available columns"):
            cfa("f1 =~ x1 + x2 + x99", data=simple_data)


class TestConstantColumns:
    def test_constant_column_raises(self):
        df = pd.DataFrame({
            "x1": np.random.normal(0, 1, 100),
            "x2": np.random.normal(0, 1, 100),
            "x3": np.ones(100),
        })
        with pytest.raises(ValueError, match="zero variance"):
            cfa("f1 =~ x1 + x2 + x3", data=df)


class TestDuplicateIndicators:
    def test_duplicate_in_same_factor(self, simple_data):
        with pytest.raises(ValueError, match="Duplicate indicator.*x1"):
            cfa("f1 =~ x1 + x1 + x2 + x3", data=simple_data)

    def test_cross_loading_is_allowed(self):
        """Same variable loading on two factors is valid (cross-loading)."""
        rng = np.random.default_rng(42)
        n = 300
        f1 = rng.normal(0, 1, n)
        f2 = rng.normal(0, 1, n)
        df = pd.DataFrame({
            "x1": f1 + rng.normal(0, 0.5, n),
            "x2": f1 + rng.normal(0, 0.5, n),
            "x3": f1 + f2 + rng.normal(0, 0.5, n),  # cross-loading
            "x4": f2 + rng.normal(0, 0.5, n),
            "x5": f2 + rng.normal(0, 0.5, n),
        })
        # x3 loads on both f1 and f2 — this should NOT raise
        fit = cfa("f1 =~ x1 + x2 + x3\nf2 =~ x3 + x4 + x5", data=df)
        assert fit.converged


class TestSingleIndicator:
    def test_warns_single_indicator(self, simple_data):
        with pytest.warns(RuntimeWarning, match="only 1 indicator"):
            cfa("f1 =~ x1", data=simple_data)


class TestSmallSample:
    def test_warns_very_small_sample(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "x1": rng.normal(0, 1, 8),
            "x2": rng.normal(0, 1, 8),
            "x3": rng.normal(0, 1, 8),
        })
        with pytest.warns(RuntimeWarning, match="small sample size"):
            cfa("f1 =~ x1 + x2 + x3", data=df)


class TestHeywoodCase:
    def test_warns_negative_variance(self):
        """Data designed to produce a Heywood case."""
        rng = np.random.default_rng(999)
        f = rng.normal(0, 1, 50)
        df = pd.DataFrame({
            "x1": 1.0 * f + rng.normal(0, 0.01, 50),
            "x2": 1.0 * f + rng.normal(0, 0.01, 50),
            "x3": 1.0 * f + rng.normal(0, 0.5, 50),
        })
        with pytest.warns(RuntimeWarning, match="Negative variance.*Heywood"):
            cfa("f1 =~ x1 + x2 + x3", data=df)
