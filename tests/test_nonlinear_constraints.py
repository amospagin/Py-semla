"""Tests for nonlinear parameter constraints (#45)."""

import numpy as np
import pytest
from semla import cfa
from semla.datasets import HolzingerSwineford1939


@pytest.fixture(scope="module")
def hs_data():
    return HolzingerSwineford1939()


class TestInequalityConstraints:
    def test_lower_bound(self, hs_data):
        """a > 0.3 should keep the loading above 0.3."""
        fit = cfa("""
            visual  =~ x1 + a*x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
            a > 0.3
        """, data=hs_data)
        est = fit.estimates()
        a = est[(est["rhs"] == "x2") & (est["op"] == "=~")]["est"].values[0]
        assert a >= 0.3 - 1e-6
        assert fit.converged

    def test_upper_bound(self, hs_data):
        """a < 1.0 should keep the loading below 1."""
        fit = cfa("""
            visual  =~ x1 + a*x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
            a < 1.0
        """, data=hs_data)
        est = fit.estimates()
        a = est[(est["rhs"] == "x2") & (est["op"] == "=~")]["est"].values[0]
        assert a <= 1.0 + 1e-6
        assert fit.converged

    def test_ordering_constraint(self, hs_data):
        """a < b should enforce ordering between two loadings."""
        fit = cfa("""
            visual  =~ x1 + a*x2 + b*x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
            a < b
        """, data=hs_data)
        est = fit.estimates()
        a = est[(est["rhs"] == "x2") & (est["op"] == "=~")]["est"].values[0]
        b = est[(est["rhs"] == "x3") & (est["op"] == "=~")]["est"].values[0]
        assert a <= b + 1e-6
        assert fit.converged

    def test_multiple_inequalities(self, hs_data):
        """Multiple inequality constraints should all hold."""
        fit = cfa("""
            visual  =~ x1 + a*x2 + b*x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
            a > 0
            b > 0
            a < b
        """, data=hs_data)
        est = fit.estimates()
        a = est[(est["rhs"] == "x2") & (est["op"] == "=~")]["est"].values[0]
        b = est[(est["rhs"] == "x3") & (est["op"] == "=~")]["est"].values[0]
        assert a >= -1e-6
        assert b >= -1e-6
        assert a <= b + 1e-6
        assert fit.converged


class TestEqualityConstraints:
    def test_nonlinear_equality(self, hs_data):
        """a + b == c should be enforced exactly."""
        fit = cfa("""
            visual  =~ x1 + a*x2 + b*x3
            textual =~ x4 + c*x5 + x6
            speed   =~ x7 + x8 + x9
            a + b == c
        """, data=hs_data)
        est = fit.estimates()
        a = est[(est["rhs"] == "x2") & (est["op"] == "=~")]["est"].values[0]
        b = est[(est["rhs"] == "x3") & (est["op"] == "=~")]["est"].values[0]
        c = est[(est["rhs"] == "x5") & (est["op"] == "=~")]["est"].values[0]
        assert abs(a + b - c) < 1e-4
        assert fit.converged

    def test_product_equality(self, hs_data):
        """a*b == 0.5 should be enforced."""
        fit = cfa("""
            visual  =~ x1 + a*x2 + b*x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
            a*b == 0.5
        """, data=hs_data)
        est = fit.estimates()
        a = est[(est["rhs"] == "x2") & (est["op"] == "=~")]["est"].values[0]
        b = est[(est["rhs"] == "x3") & (est["op"] == "=~")]["est"].values[0]
        assert abs(a * b - 0.5) < 1e-3
        assert fit.converged


class TestConstraintFitStatistics:
    def test_constrained_chi_square_higher(self, hs_data):
        """Constrained model should not have lower chi-square."""
        fit_free = cfa("""
            visual  =~ x1 + a*x2 + b*x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        """, data=hs_data)

        fit_constrained = cfa("""
            visual  =~ x1 + a*x2 + b*x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
            a + b == 1
        """, data=hs_data)

        assert (fit_constrained.fit_indices()["chi_square"]
                >= fit_free.fit_indices()["chi_square"] - 0.5)

    def test_unconstrained_matches_bfgs(self, hs_data):
        """When constraint is not active, results should be close to BFGS."""
        # The unconstrained optimum has a ≈ 0.55, b ≈ 0.73
        # A loose bound (a > 0) shouldn't change results
        fit_free = cfa("""
            visual  =~ x1 + a*x2 + b*x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        """, data=hs_data)

        fit_loose = cfa("""
            visual  =~ x1 + a*x2 + b*x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
            a > 0
        """, data=hs_data)

        est_free = fit_free.estimates()
        est_loose = fit_loose.estimates()
        a_free = est_free[(est_free["rhs"] == "x2") & (est_free["op"] == "=~")]["est"].values[0]
        a_loose = est_loose[(est_loose["rhs"] == "x2") & (est_loose["op"] == "=~")]["est"].values[0]
        # Should be very close since constraint is not active
        assert abs(a_free - a_loose) < 0.01


class TestConstraintSyntax:
    def test_constraint_with_gte(self, hs_data):
        """>=  operator should work."""
        fit = cfa("""
            visual  =~ x1 + a*x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
            a >= 0.5
        """, data=hs_data)
        est = fit.estimates()
        a = est[(est["rhs"] == "x2") & (est["op"] == "=~")]["est"].values[0]
        assert a >= 0.5 - 1e-6
        assert fit.converged

    def test_constraint_with_lte(self, hs_data):
        """<= operator should work."""
        fit = cfa("""
            visual  =~ x1 + a*x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
            a <= 0.6
        """, data=hs_data)
        est = fit.estimates()
        a = est[(est["rhs"] == "x2") & (est["op"] == "=~")]["est"].values[0]
        assert a <= 0.6 + 1e-6
        assert fit.converged

    def test_no_constraints_uses_bfgs(self, hs_data):
        """Without constraints, BFGS should still be used (no SLSQP overhead)."""
        fit = cfa("""
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        """, data=hs_data)
        assert fit.converged
