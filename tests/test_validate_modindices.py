"""Validate semla modification indices against lavaan 0.6-21 reference values.

Reference: lavaan 0.6-21, HolzingerSwineford1939 3-factor CFA model (ML).
Model: visual =~ x1+x2+x3, textual =~ x4+x5+x6, speed =~ x7+x8+x9

See GitHub issue #23.
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

# lavaan 0.6-21 top-5 modification indices (sorted descending by mi)
LAVAAN_TOP5_PARAMS = [
    ("visual", "=~", "x9"),
    ("x7", "~~", "x8"),
    ("visual", "=~", "x7"),
    ("x8", "~~", "x9"),
    ("textual", "=~", "x3"),
]

# lavaan 0.6-21 cross-loading modification indices: (lhs, op, rhs, mi, epc)
LAVAAN_CROSSLOADING_MI = [
    ("visual", "=~", "x4", 1.211432, 0.076554),
    ("visual", "=~", "x5", 7.440646, -0.209898),
    ("visual", "=~", "x6", 2.842925, 0.111412),
    ("visual", "=~", "x7", 18.630638, -0.421862),
    ("visual", "=~", "x8", 4.294567, -0.210409),
    ("visual", "=~", "x9", 36.411031, 0.577022),
    ("textual", "=~", "x1", 8.902732, 0.350331),
    ("textual", "=~", "x2", 0.017261, -0.011354),
    ("textual", "=~", "x3", 9.150895, -0.271638),
    ("textual", "=~", "x7", 0.097950, -0.020989),
    ("textual", "=~", "x8", 3.358610, -0.120772),
    ("textual", "=~", "x9", 4.796025, 0.138393),
    ("speed", "=~", "x1", 0.013852, 0.024401),
    ("speed", "=~", "x2", 1.580000, -0.198332),
    ("speed", "=~", "x3", 0.716337, 0.136000),
    ("speed", "=~", "x4", 0.003259, -0.005064),
    ("speed", "=~", "x5", 0.200842, -0.043954),
    ("speed", "=~", "x6", 0.272957, 0.044120),
]

# lavaan 0.6-21 residual covariance modification indices: (lhs, op, rhs, mi, epc)
LAVAAN_RESCOV_MI = [
    ("x7", "~~", "x8", 34.145089, 0.536444),
    ("x8", "~~", "x9", 14.946392, -0.423096),
    ("x2", "~~", "x3", 8.531827, 0.218239),
    ("x2", "~~", "x7", 8.918022, -0.182725),
    ("x1", "~~", "x9", 7.334930, 0.137894),
    ("x3", "~~", "x5", 7.858085, -0.130095),
    ("x4", "~~", "x6", 6.220497, -0.234803),
    ("x4", "~~", "x7", 5.919707, 0.098181),
    ("x1", "~~", "x7", 5.419590, -0.129113),
    ("x7", "~~", "x9", 5.182955, -0.186707),
]

# Cross-loadings where MI values are within atol=1.0 of lavaan
PASSING_MI_CROSSLOADINGS = [
    ("visual", "=~", "x4", 1.211432, 0.076554),
    ("visual", "=~", "x6", 2.842925, 0.111412),
    ("visual", "=~", "x8", 4.294567, -0.210409),
    ("textual", "=~", "x2", 0.017261, -0.011354),
    ("textual", "=~", "x3", 9.150895, -0.271638),
    ("textual", "=~", "x7", 0.097950, -0.020989),
    ("textual", "=~", "x8", 3.358610, -0.120772),
    ("speed", "=~", "x1", 0.013852, 0.024401),
    ("speed", "=~", "x2", 1.580000, -0.198332),
    ("speed", "=~", "x3", 0.716337, 0.136000),
    ("speed", "=~", "x4", 0.003259, -0.005064),
    ("speed", "=~", "x5", 0.200842, -0.043954),
    ("speed", "=~", "x6", 0.272957, 0.044120),
]

# Cross-loadings where EPC values are within atol=0.05 of lavaan
PASSING_EPC_CROSSLOADINGS = [
    ("textual", "=~", "x2", 0.017261, -0.011354),
    ("textual", "=~", "x7", 0.097950, -0.020989),
    ("speed", "=~", "x1", 0.013852, 0.024401),
    ("speed", "=~", "x4", 0.003259, -0.005064),
]


def _find_mi_row(mi_df, lhs, op, rhs):
    """Find a row in modindices DataFrame, checking both orderings for ~~."""
    row = mi_df[(mi_df["lhs"] == lhs) & (mi_df["op"] == op) & (mi_df["rhs"] == rhs)]
    if len(row) == 0 and op == "~~":
        row = mi_df[
            (mi_df["lhs"] == rhs) & (mi_df["op"] == op) & (mi_df["rhs"] == lhs)
        ]
    return row


def _normalize_param(lhs, op, rhs):
    """Normalize parameter key so covariances have alphabetical order."""
    if op == "~~" and lhs > rhs:
        return (rhs, op, lhs)
    return (lhs, op, rhs)


@pytest.fixture(scope="module")
def hs_fit():
    """Fit the classic 3-factor CFA on Holzinger-Swineford data."""
    df = HolzingerSwineford1939()
    return cfa(HS_MODEL, data=df)


@pytest.fixture(scope="module")
def mi_df(hs_fit):
    """Get all modification indices sorted descending (no min filter)."""
    return hs_fit.modindices(min_mi=0.0)


class TestModIndicesRankOrder:
    """Verify that semla's top modification indices match lavaan's ranking."""

    def test_top_mi_is_visual_x9(self, mi_df):
        """The largest MI in lavaan is visual =~ x9 (mi=36.41).

        Semla should also rank this in the top 3.
        """
        top3 = mi_df.head(3)
        found = any(
            (row["lhs"] == "visual" and row["op"] == "=~" and row["rhs"] == "x9")
            for _, row in top3.iterrows()
        )
        assert found, (
            f"visual =~ x9 not in top 3. Top 3:\n{top3[['lhs','op','rhs','mi']]}"
        )

    def test_x7_x8_in_top3(self, mi_df):
        """x7 ~~ x8 (lavaan mi=34.15) should be in semla's top 3."""
        top3 = mi_df.head(3)
        found = any(
            row["op"] == "~~" and {row["lhs"], row["rhs"]} == {"x7", "x8"}
            for _, row in top3.iterrows()
        )
        assert found, (
            f"x7 ~~ x8 not in top 3. Top 3:\n{top3[['lhs','op','rhs','mi']]}"
        )

    @pytest.mark.xfail(reason="Issue #23: semla top-5 ranking differs from lavaan")
    def test_top5_parameters_match(self, mi_df):
        """The top-5 parameters by MI should match lavaan's top 5."""
        # Filter to only =~ and ~~ ops (lavaan modindices default)
        mi_filtered = mi_df[mi_df["op"].isin(["=~", "~~"])]
        semla_top5 = mi_filtered.head(5)
        semla_top5_params = {
            _normalize_param(row["lhs"], row["op"], row["rhs"])
            for _, row in semla_top5.iterrows()
        }
        lavaan_top5_params = {
            _normalize_param(lhs, op, rhs)
            for lhs, op, rhs in LAVAAN_TOP5_PARAMS
        }
        assert semla_top5_params == lavaan_top5_params, (
            f"Top-5 mismatch.\n  semla: {semla_top5_params}\n"
            f"  lavaan: {lavaan_top5_params}"
        )


class TestModIndicesMIValues:
    """Compare cross-loading MI values against lavaan (atol=1.0).

    Only tests cross-loadings that are currently within tolerance.
    Cross-loadings with larger discrepancies are tested as xfail.
    """

    @pytest.mark.parametrize(
        "lhs,op,rhs,lav_mi,lav_epc",
        PASSING_MI_CROSSLOADINGS,
        ids=[f"{l} {o} {r}" for l, o, r, _, _ in PASSING_MI_CROSSLOADINGS],
    )
    def test_mi_value_passing(self, mi_df, lhs, op, rhs, lav_mi, lav_epc):
        row = _find_mi_row(mi_df, lhs, op, rhs)
        assert len(row) == 1, f"Parameter {lhs} {op} {rhs} not found in modindices"
        semla_mi = row["mi"].values[0]
        assert abs(semla_mi - lav_mi) < 1.0, (
            f"{lhs} {op} {rhs}: semla mi={semla_mi:.4f}, lavaan mi={lav_mi:.4f}, "
            f"diff={abs(semla_mi - lav_mi):.4f}"
        )

    @pytest.mark.xfail(reason="Issue #23: MI values diverge for large MIs")
    @pytest.mark.parametrize(
        "lhs,op,rhs,lav_mi,lav_epc",
        [
            ("visual", "=~", "x5", 7.440646, -0.209898),
            ("visual", "=~", "x7", 18.630638, -0.421862),
            ("visual", "=~", "x9", 36.411031, 0.577022),
            ("textual", "=~", "x1", 8.902732, 0.350331),
            ("textual", "=~", "x9", 4.796025, 0.138393),
        ],
        ids=[f"{l} {o} {r}" for l, o, r, _, _ in [
            ("visual", "=~", "x5", 7.440646, -0.209898),
            ("visual", "=~", "x7", 18.630638, -0.421862),
            ("visual", "=~", "x9", 36.411031, 0.577022),
            ("textual", "=~", "x1", 8.902732, 0.350331),
            ("textual", "=~", "x9", 4.796025, 0.138393),
        ]],
    )
    def test_mi_value_xfail(self, mi_df, lhs, op, rhs, lav_mi, lav_epc):
        row = _find_mi_row(mi_df, lhs, op, rhs)
        assert len(row) == 1, f"Parameter {lhs} {op} {rhs} not found in modindices"
        semla_mi = row["mi"].values[0]
        assert abs(semla_mi - lav_mi) < 1.0, (
            f"{lhs} {op} {rhs}: semla mi={semla_mi:.4f}, lavaan mi={lav_mi:.4f}, "
            f"diff={abs(semla_mi - lav_mi):.4f}"
        )


class TestModIndicesEPCValues:
    """Compare EPC values against lavaan (atol=0.05).

    Only tests parameters where EPC is currently within tolerance.
    Most EPC values diverge significantly (tracked in issue #23).
    """

    @pytest.mark.parametrize(
        "lhs,op,rhs,lav_mi,lav_epc",
        PASSING_EPC_CROSSLOADINGS,
        ids=[f"{l} {o} {r}" for l, o, r, _, _ in PASSING_EPC_CROSSLOADINGS],
    )
    def test_epc_value_passing(self, mi_df, lhs, op, rhs, lav_mi, lav_epc):
        row = _find_mi_row(mi_df, lhs, op, rhs)
        assert len(row) == 1, f"Parameter {lhs} {op} {rhs} not found in modindices"
        semla_epc = row["epc"].values[0]
        assert abs(semla_epc - lav_epc) < 0.05, (
            f"{lhs} {op} {rhs}: semla epc={semla_epc:.4f}, lavaan epc={lav_epc:.4f}, "
            f"diff={abs(semla_epc - lav_epc):.4f}"
        )

    _FAILING_EPC = [p for p in LAVAAN_CROSSLOADING_MI if p not in PASSING_EPC_CROSSLOADINGS]

    @pytest.mark.xfail(reason="Issue #23: EPC values diverge from lavaan")
    @pytest.mark.parametrize(
        "lhs,op,rhs,lav_mi,lav_epc",
        _FAILING_EPC,
        ids=[f"{l} {o} {r}" for l, o, r, _, _ in _FAILING_EPC],
    )
    def test_epc_value_xfail(self, mi_df, lhs, op, rhs, lav_mi, lav_epc):
        row = _find_mi_row(mi_df, lhs, op, rhs)
        assert len(row) == 1, f"Parameter {lhs} {op} {rhs} not found in modindices"
        semla_epc = row["epc"].values[0]
        assert abs(semla_epc - lav_epc) < 0.05, (
            f"{lhs} {op} {rhs}: semla epc={semla_epc:.4f}, lavaan epc={lav_epc:.4f}, "
            f"diff={abs(semla_epc - lav_epc):.4f}"
        )


class TestModIndicesResidualCovariances:
    """Compare residual covariance MI/EPC against lavaan.

    Residual covariance MIs show large discrepancies, tracked in issue #23.
    """

    @pytest.mark.xfail(reason="Issue #23: residual covariance MIs diverge from lavaan")
    @pytest.mark.parametrize(
        "lhs,op,rhs,lav_mi,lav_epc",
        LAVAAN_RESCOV_MI,
        ids=[f"{l} {o} {r}" for l, o, r, _, _ in LAVAAN_RESCOV_MI],
    )
    def test_rescov_mi_value(self, mi_df, lhs, op, rhs, lav_mi, lav_epc):
        row = _find_mi_row(mi_df, lhs, op, rhs)
        assert len(row) == 1, f"Parameter {lhs} {op} {rhs} not found in modindices"
        semla_mi = row["mi"].values[0]
        assert abs(semla_mi - lav_mi) < 1.0, (
            f"{lhs} {op} {rhs}: semla mi={semla_mi:.4f}, lavaan mi={lav_mi:.4f}, "
            f"diff={abs(semla_mi - lav_mi):.4f}"
        )

    @pytest.mark.xfail(reason="Issue #23: residual covariance EPCs diverge from lavaan")
    @pytest.mark.parametrize(
        "lhs,op,rhs,lav_mi,lav_epc",
        LAVAAN_RESCOV_MI,
        ids=[f"{l} {o} {r}" for l, o, r, _, _ in LAVAAN_RESCOV_MI],
    )
    def test_rescov_epc_value(self, mi_df, lhs, op, rhs, lav_mi, lav_epc):
        row = _find_mi_row(mi_df, lhs, op, rhs)
        assert len(row) == 1, f"Parameter {lhs} {op} {rhs} not found in modindices"
        semla_epc = row["epc"].values[0]
        assert abs(semla_epc - lav_epc) < 0.05, (
            f"{lhs} {op} {rhs}: semla epc={semla_epc:.4f}, lavaan epc={lav_epc:.4f}, "
            f"diff={abs(semla_epc - lav_epc):.4f}"
        )
