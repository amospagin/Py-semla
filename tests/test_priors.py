"""Tests for semla.priors — Prior distribution classes for Bayesian SEM."""

import pytest

numpyro = pytest.importorskip("numpyro")
import jax
import jax.numpy as jnp
import numpyro.distributions as npdist

from semla.priors import (
    Prior,
    Normal,
    StudentT,
    Cauchy,
    Uniform,
    Laplace,
    HalfCauchy,
    HalfNormal,
    InverseGamma,
    Exponential,
    Gamma,
    LogNormal,
    LKJ,
    Beta,
)


class TestPriorBase:
    """Tests for the Prior base class."""

    def test_base_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Prior().to_numpyro()


class TestUnboundedPriors:
    """Tests for priors with support on the real line."""

    def test_normal_defaults(self):
        p = Normal()
        assert p.mu == 0.0
        assert p.sigma == 1.0
        d = p.to_numpyro()
        assert isinstance(d, npdist.Normal)

    def test_normal_custom(self):
        p = Normal(mu=5.0, sigma=2.0)
        d = p.to_numpyro()
        sample = d.sample(jax.random.PRNGKey(0))
        assert sample.shape == ()

    def test_student_t_defaults(self):
        p = StudentT()
        d = p.to_numpyro()
        assert isinstance(d, npdist.StudentT)

    def test_student_t_custom(self):
        p = StudentT(df=5.0, loc=1.0, scale=2.0)
        d = p.to_numpyro()
        assert d.sample(jax.random.PRNGKey(1)).shape == ()

    def test_cauchy(self):
        p = Cauchy(loc=0.0, scale=2.5)
        d = p.to_numpyro()
        assert isinstance(d, npdist.Cauchy)

    def test_uniform(self):
        p = Uniform(low=-1.0, high=1.0)
        d = p.to_numpyro()
        assert isinstance(d, npdist.Uniform)

    def test_laplace(self):
        p = Laplace(loc=0.0, scale=1.0)
        d = p.to_numpyro()
        assert isinstance(d, npdist.Laplace)


class TestPositivePriors:
    """Tests for priors with support on positive reals."""

    def test_half_cauchy(self):
        p = HalfCauchy(scale=2.5)
        d = p.to_numpyro()
        assert isinstance(d, npdist.HalfCauchy)

    def test_half_normal(self):
        p = HalfNormal(scale=1.0)
        d = p.to_numpyro()
        assert isinstance(d, npdist.HalfNormal)

    def test_inverse_gamma(self):
        p = InverseGamma(concentration=2.0, rate=1.0)
        d = p.to_numpyro()
        assert isinstance(d, npdist.InverseGamma)

    def test_exponential(self):
        p = Exponential(rate=0.5)
        d = p.to_numpyro()
        assert isinstance(d, npdist.Exponential)

    def test_gamma(self):
        p = Gamma(concentration=2.0, rate=1.0)
        d = p.to_numpyro()
        assert isinstance(d, npdist.Gamma)

    def test_log_normal(self):
        p = LogNormal(loc=0.0, scale=0.5)
        d = p.to_numpyro()
        assert isinstance(d, npdist.LogNormal)


class TestCorrelationPriors:
    """Tests for correlation-related priors."""

    def test_lkj_defaults(self):
        p = LKJ()
        assert p.dim == 2
        assert p.concentration == 1.0
        d = p.to_numpyro()
        assert isinstance(d, npdist.LKJCholesky)

    def test_lkj_custom(self):
        p = LKJ(dim=4, concentration=2.0)
        d = p.to_numpyro()
        sample = d.sample(jax.random.PRNGKey(42))
        assert sample.shape == (4, 4)

    def test_beta(self):
        p = Beta(concentration1=2.0, concentration0=5.0)
        d = p.to_numpyro()
        assert isinstance(d, npdist.Beta)


class TestDataclassBehavior:
    """Tests that priors behave correctly as dataclasses."""

    def test_equality(self):
        assert Normal(0, 1) == Normal(0, 1)
        assert Normal(0, 1) != Normal(0, 2)

    def test_repr(self):
        r = repr(Normal(mu=0.0, sigma=1.0))
        assert "Normal" in r
        assert "mu=0.0" in r

    def test_all_priors_are_dataclasses(self):
        import dataclasses

        for cls in [
            Normal, StudentT, Cauchy, Uniform, Laplace,
            HalfCauchy, HalfNormal, InverseGamma, Exponential, Gamma, LogNormal,
            LKJ, Beta,
        ]:
            assert dataclasses.is_dataclass(cls), f"{cls.__name__} is not a dataclass"

    def test_all_priors_subclass_prior(self):
        for cls in [
            Normal, StudentT, Cauchy, Uniform, Laplace,
            HalfCauchy, HalfNormal, InverseGamma, Exponential, Gamma, LogNormal,
            LKJ, Beta,
        ]:
            assert issubclass(cls, Prior), f"{cls.__name__} does not inherit Prior"


class TestLogProb:
    """Verify that the numpyro distributions produce valid log-probabilities."""

    @pytest.mark.parametrize("prior,value", [
        (Normal(0, 1), 0.5),
        (StudentT(3, 0, 1), 0.5),
        (Cauchy(0, 1), 0.5),
        (Uniform(0, 1), 0.5),
        (Laplace(0, 1), 0.5),
        (HalfCauchy(1), 0.5),
        (HalfNormal(1), 0.5),
        (InverseGamma(1, 1), 0.5),
        (Exponential(1), 0.5),
        (Gamma(1, 1), 0.5),
        (LogNormal(0, 1), 0.5),
        (Beta(2, 2), 0.5),
    ])
    def test_finite_log_prob(self, prior, value):
        d = prior.to_numpyro()
        lp = d.log_prob(jnp.array(value))
        assert jnp.isfinite(lp)


class TestImportPath:
    """Verify the public import path works."""

    def test_import_from_semla_priors(self):
        from semla.priors import Normal, HalfCauchy, LKJ
        assert Normal is not None

    def test_import_priors_module(self):
        import semla
        assert hasattr(semla, "priors")
        assert hasattr(semla.priors, "Normal")
