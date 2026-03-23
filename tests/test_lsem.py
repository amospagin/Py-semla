"""Tests for Local SEM (#68)."""

import numpy as np
import pandas as pd
import pytest
from semla import lsem


def _generate_lsem_data(n=800, seed=42, varying_loading=True):
    """Generate CFA data with optional age-varying loading on x2."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(20, 60, n)
    factor = rng.normal(0, 1, n)

    data = {}
    data["x1"] = 1.0 * factor + rng.normal(0, 0.5, n)
    if varying_loading:
        data["x2"] = (0.5 + 0.015 * (age - 40)) * factor + rng.normal(0, 0.5, n)
    else:
        data["x2"] = 0.7 * factor + rng.normal(0, 0.5, n)
    data["x3"] = 0.7 * factor + rng.normal(0, 0.5, n)
    data["x4"] = 0.6 * factor + rng.normal(0, 0.5, n)
    data["age"] = age
    return pd.DataFrame(data)


MODEL = "f =~ x1 + x2 + x3 + x4"


class TestLSEMBasic:
    """Basic LSEM functionality."""

    @pytest.fixture(scope="class")
    def result(self):
        df = _generate_lsem_data()
        return lsem(MODEL, data=df, moderator="age", focal_points=10)

    def test_returns_lsem_result(self, result):
        assert hasattr(result, "trajectories")
        assert hasattr(result, "trajectory")
        assert hasattr(result, "convergence_rate")

    def test_convergence_high(self, result):
        assert result.convergence_rate > 0.7

    def test_trajectories_shape(self, result):
        t = result.trajectories()
        assert len(t) == 10  # 10 focal points
        assert "focal_point" in t.columns
        assert "effective_n" in t.columns

    def test_param_names(self, result):
        assert "f =~ x2" in result.param_names
        assert "f =~ x3" in result.param_names

    def test_summary_returns_string(self, result):
        s = result.summary()
        assert isinstance(s, str)
        assert "Local SEM" in s

    def test_repr(self, result):
        r = repr(result)
        assert "LSEMResult" in r
        assert "age" in r


class TestVaryingLoading:
    """x2 loading should increase with age when simulated that way."""

    @pytest.fixture(scope="class")
    def result(self):
        df = _generate_lsem_data(n=1000)
        return lsem(MODEL, data=df, moderator="age", focal_points=10)

    def test_x2_loading_increases(self, result):
        traj = result.trajectory("f =~ x2")
        valid = traj.dropna(subset=["est"])
        if len(valid) >= 5:
            # First half average should be lower than second half
            mid = len(valid) // 2
            early = valid.iloc[:mid]["est"].mean()
            late = valid.iloc[mid:]["est"].mean()
            assert late > early, f"Expected increasing: early={early:.3f}, late={late:.3f}"

    def test_x3_loading_stable(self, result):
        traj = result.trajectory("f =~ x3")
        valid = traj.dropna(subset=["est"])
        if len(valid) >= 3:
            # x3 loading range should be small (no age effect)
            assert valid["est"].max() - valid["est"].min() < 0.3


class TestInvariantData:
    """When all loadings are constant, trajectories should be flat."""

    @pytest.fixture(scope="class")
    def result(self):
        df = _generate_lsem_data(varying_loading=False)
        return lsem(MODEL, data=df, moderator="age", focal_points=10)

    def test_all_loadings_stable(self, result):
        for param in ["f =~ x2", "f =~ x3", "f =~ x4"]:
            traj = result.trajectory(param)
            valid = traj.dropna(subset=["est"])
            if len(valid) >= 3:
                rng = valid["est"].max() - valid["est"].min()
                assert rng < 0.4, f"{param} range too large: {rng:.3f}"


class TestTrajectory:
    """trajectory() method should return correct structure."""

    @pytest.fixture(scope="class")
    def result(self):
        df = _generate_lsem_data()
        return lsem(MODEL, data=df, moderator="age", focal_points=10)

    def test_has_ci_columns(self, result):
        traj = result.trajectory("f =~ x2")
        assert "ci_lower" in traj.columns
        assert "ci_upper" in traj.columns
        assert "se" in traj.columns

    def test_invalid_param_raises(self, result):
        with pytest.raises(ValueError, match="not found"):
            result.trajectory("nonexistent")

    def test_ci_contains_estimate(self, result):
        traj = result.trajectory("f =~ x2")
        valid = traj.dropna(subset=["ci_lower"])
        assert (valid["ci_lower"] <= valid["est"]).all()
        assert (valid["ci_upper"] >= valid["est"]).all()


class TestOptions:
    """Test bandwidth and focal_points options."""

    def test_auto_bandwidth(self):
        df = _generate_lsem_data(n=300)
        result = lsem(MODEL, data=df, moderator="age", focal_points=5)
        assert result._bandwidth > 0

    def test_explicit_bandwidth(self):
        df = _generate_lsem_data(n=300)
        result = lsem(MODEL, data=df, moderator="age",
                      focal_points=5, bandwidth=5.0)
        assert result._bandwidth == 5.0

    def test_explicit_focal_array(self):
        df = _generate_lsem_data(n=300)
        fp = np.array([25.0, 35.0, 45.0, 55.0])
        result = lsem(MODEL, data=df, moderator="age", focal_points=fp)
        assert len(result._focal) == 4

    def test_missing_moderator_raises(self):
        df = _generate_lsem_data(n=100)
        with pytest.raises(ValueError, match="not found"):
            lsem(MODEL, data=df, moderator="nonexistent")
