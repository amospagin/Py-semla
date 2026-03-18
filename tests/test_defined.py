"""Tests for defined parameters (:= operator) and indirect effects."""

import numpy as np
import pandas as pd
import pytest
from semla import sem, cfa
from semla.syntax import parse_syntax


@pytest.fixture(scope="module")
def mediation_data():
    rng = np.random.default_rng(42)
    n = 300
    X = rng.normal(0, 1, n)
    M = 0.5 * X + rng.normal(0, 0.8, n)
    Y = 0.4 * M + 0.2 * X + rng.normal(0, 0.7, n)
    return pd.DataFrame({"X": X, "M": M, "Y": Y})


@pytest.fixture(scope="module")
def mediation_fit(mediation_data):
    return sem("""
        M ~ a*X
        Y ~ b*M + c*X
        indirect := a*b
        total := a*b + c
    """, data=mediation_data)


class TestParsing:
    def test_parse_defined_param(self):
        tokens = parse_syntax("indirect := a*b")
        assert len(tokens) == 1
        assert tokens[0].op == ":="
        assert tokens[0].lhs == "indirect"
        assert tokens[0].rhs[0].var == "a*b"

    def test_parse_mixed_model_with_defined(self):
        tokens = parse_syntax("""
            M ~ a*X
            Y ~ b*M
            indirect := a*b
        """)
        assert len(tokens) == 3
        ops = [t.op for t in tokens]
        assert ":=" in ops

    def test_missing_expression_raises(self):
        with pytest.raises(SyntaxError, match="Missing expression"):
            parse_syntax("indirect :=")


class TestDefinedEstimates:
    def test_returns_dataframe(self, mediation_fit):
        df = mediation_fit.defined_estimates()
        assert isinstance(df, pd.DataFrame)
        assert "name" in df.columns
        assert "est" in df.columns
        assert "se" in df.columns

    def test_indirect_effect_computed(self, mediation_fit):
        df = mediation_fit.defined_estimates()
        indirect = df[df["name"] == "indirect"]
        assert len(indirect) == 1
        assert indirect["est"].values[0] > 0

    def test_total_effect_computed(self, mediation_fit):
        df = mediation_fit.defined_estimates()
        total = df[df["name"] == "total"]
        assert len(total) == 1

    def test_indirect_equals_a_times_b(self, mediation_fit):
        """indirect := a*b should equal the product of a and b estimates."""
        est = mediation_fit.estimates()
        a = est[(est["rhs"] == "X") & (est["lhs"] == "M")]["est"].values[0]
        b = est[(est["rhs"] == "M") & (est["lhs"] == "Y")]["est"].values[0]
        expected = a * b

        df = mediation_fit.defined_estimates()
        indirect = df[df["name"] == "indirect"]["est"].values[0]
        assert abs(indirect - expected) < 1e-6

    def test_total_equals_indirect_plus_direct(self, mediation_fit):
        df = mediation_fit.defined_estimates()
        indirect = df[df["name"] == "indirect"]["est"].values[0]
        total = df[df["name"] == "total"]["est"].values[0]

        est = mediation_fit.estimates()
        c = est[(est["rhs"] == "X") & (est["lhs"] == "Y")]["est"].values[0]
        assert abs(total - (indirect + c)) < 1e-6

    def test_se_is_positive(self, mediation_fit):
        df = mediation_fit.defined_estimates()
        assert (df["se"] > 0).all()

    def test_z_and_pvalue_computed(self, mediation_fit):
        df = mediation_fit.defined_estimates()
        assert not df["z"].isna().any()
        assert not df["pvalue"].isna().any()

    def test_indirect_significant(self, mediation_fit):
        """With true a=0.5, b=0.4, indirect should be significant."""
        df = mediation_fit.defined_estimates()
        indirect = df[df["name"] == "indirect"]
        assert indirect["pvalue"].values[0] < 0.05

    def test_in_summary(self, mediation_fit):
        summary = mediation_fit.summary()
        assert "Defined Parameters:" in summary
        assert "indirect" in summary

    def test_no_defined_returns_empty(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x1": rng.normal(0,1,100), "x2": rng.normal(0,1,100), "x3": rng.normal(0,1,100)})
        fit = cfa("f1 =~ x1 + x2 + x3", data=df)
        defined = fit.defined_estimates()
        assert len(defined) == 0


class TestDefinedWithCFA:
    def test_difference_of_loadings(self):
        """Test := with CFA model comparing loadings."""
        from semla.datasets import HolzingerSwineford1939
        df = HolzingerSwineford1939()
        fit = cfa("""
            visual =~ x1 + a*x2 + b*x3
            textual =~ x4 + x5 + x6
            speed =~ x7 + x8 + x9
            diff := a - b
        """, data=df)
        defined = fit.defined_estimates()
        assert len(defined) == 1

        est = fit.estimates()
        a = est[(est["rhs"] == "x2") & (est["op"] == "=~")]["est"].values[0]
        b = est[(est["rhs"] == "x3") & (est["op"] == "=~")]["est"].values[0]
        diff = defined[defined["name"] == "diff"]["est"].values[0]
        assert abs(diff - (a - b)) < 1e-6
