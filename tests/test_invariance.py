"""Tests for automated measurement invariance testing."""

import numpy as np
import pytest
from semla import measurementInvariance
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
def result(hs_data):
    return measurementInvariance(HS_MODEL, data=hs_data, group="school")


class TestInvarianceResult:
    def test_all_levels_present(self, result):
        assert set(result.fits.keys()) == {"configural", "metric", "scalar", "strict"}

    def test_all_converged(self, result):
        for level, fit in result.fits.items():
            assert fit.converged, f"{level} did not converge"

    def test_table_has_correct_columns(self, result):
        table = result.table()
        assert "level" in table.columns
        assert "chisq" in table.columns
        assert "delta_chisq" in table.columns
        assert "delta_p" in table.columns
        assert "delta_cfi" in table.columns

    def test_table_has_four_rows(self, result):
        assert len(result.table()) == 4

    def test_chi_square_increases(self, result):
        table = result.table()
        chisqs = table["chisq"].values
        for i in range(1, len(chisqs)):
            assert chisqs[i] >= chisqs[i - 1] - 0.1

    def test_df_increases(self, result):
        table = result.table()
        dfs = table["df"].values
        for i in range(1, len(dfs)):
            assert dfs[i] > dfs[i - 1]

    def test_delta_chisq_positive(self, result):
        table = result.table()
        for _, row in table.iterrows():
            if not np.isnan(row["delta_chisq"]):
                assert row["delta_chisq"] >= -0.1

    def test_delta_p_between_0_and_1(self, result):
        table = result.table()
        for _, row in table.iterrows():
            if not np.isnan(row["delta_p"]):
                assert 0.0 <= row["delta_p"] <= 1.0

    def test_configural_is_baseline(self, result):
        table = result.table()
        row0 = table.iloc[0]
        assert row0["level"] == "configural"
        assert np.isnan(row0["delta_chisq"])


class TestInvarianceDecision:
    def test_highest_level_is_string(self, result):
        assert isinstance(result.highest_level, str)

    def test_highest_level_in_valid_set(self, result):
        assert result.highest_level in {"configural", "metric", "scalar", "strict"}

    def test_metric_holds_for_hs_data(self, result):
        """For HS data, metric invariance should hold."""
        assert result.highest_level in {"metric", "scalar", "strict"}

    def test_scalar_fails_for_hs_data(self, result):
        """For HS data, scalar invariance typically fails."""
        table = result.table()
        scalar_row = table[table["level"] == "scalar"].iloc[0]
        # The delta chi-square for scalar should be significant
        assert scalar_row["delta_p"] < 0.05


class TestInvarianceAPI:
    def test_getitem(self, result):
        fit = result["metric"]
        assert fit.converged

    def test_summary_returns_string(self, result):
        s = result.summary()
        assert isinstance(s, str)
        assert "Measurement Invariance" in s

    def test_custom_levels(self, hs_data):
        result = measurementInvariance(
            HS_MODEL, data=hs_data, group="school",
            levels=["configural", "metric"],
        )
        assert len(result.fits) == 2
        assert len(result.table()) == 2

    def test_invalid_level_raises(self, hs_data):
        with pytest.raises(ValueError, match="Unknown invariance level"):
            measurementInvariance(
                HS_MODEL, data=hs_data, group="school",
                levels=["configural", "partial"],
            )
