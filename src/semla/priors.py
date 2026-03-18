"""Prior distribution classes for Bayesian SEM estimation.

Semla-native dataclasses that wrap NumPyro distributions internally.
Each prior has a ``to_numpyro()`` method that returns the corresponding
NumPyro distribution object for use in probabilistic models.

Example
-------
>>> from semla.priors import Normal, HalfCauchy, LKJ
>>> Normal(mu=0, sigma=10).to_numpyro()
>>> HalfCauchy(scale=2.5).to_numpyro()
>>> LKJ(dim=3, concentration=2.0).to_numpyro()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _import_numpyro_dist():
    """Lazy-import numpyro.distributions, raising a clear error if missing."""
    try:
        import numpyro.distributions as dist

        return dist
    except ImportError:
        raise ImportError(
            "numpyro is required for Bayesian estimation. "
            "Install it with:  pip install semla[bayes]"
        ) from None


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class Prior:
    """Base class for all prior distributions."""

    def to_numpyro(self) -> Any:
        """Return the corresponding ``numpyro.distributions`` object."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Unbounded (support on the real line)
# ---------------------------------------------------------------------------

@dataclass
class Normal(Prior):
    """Normal (Gaussian) prior.  Support: (-inf, inf)."""

    mu: float = 0.0
    sigma: float = 1.0

    def to_numpyro(self):
        dist = _import_numpyro_dist()
        return dist.Normal(self.mu, self.sigma)


@dataclass
class StudentT(Prior):
    """Student-t prior.  Support: (-inf, inf)."""

    df: float = 3.0
    loc: float = 0.0
    scale: float = 1.0

    def to_numpyro(self):
        dist = _import_numpyro_dist()
        return dist.StudentT(self.df, self.loc, self.scale)


@dataclass
class Cauchy(Prior):
    """Cauchy prior.  Support: (-inf, inf)."""

    loc: float = 0.0
    scale: float = 1.0

    def to_numpyro(self):
        dist = _import_numpyro_dist()
        return dist.Cauchy(self.loc, self.scale)


@dataclass
class Uniform(Prior):
    """Uniform prior.  Support: [low, high]."""

    low: float = 0.0
    high: float = 1.0

    def to_numpyro(self):
        dist = _import_numpyro_dist()
        return dist.Uniform(self.low, self.high)


@dataclass
class Laplace(Prior):
    """Laplace prior.  Support: (-inf, inf)."""

    loc: float = 0.0
    scale: float = 1.0

    def to_numpyro(self):
        dist = _import_numpyro_dist()
        return dist.Laplace(self.loc, self.scale)


# ---------------------------------------------------------------------------
# Positive (support on the positive reals)
# ---------------------------------------------------------------------------

@dataclass
class HalfCauchy(Prior):
    """Half-Cauchy prior.  Support: (0, inf)."""

    scale: float = 1.0

    def to_numpyro(self):
        dist = _import_numpyro_dist()
        return dist.HalfCauchy(self.scale)


@dataclass
class HalfNormal(Prior):
    """Half-Normal prior.  Support: (0, inf)."""

    scale: float = 1.0

    def to_numpyro(self):
        dist = _import_numpyro_dist()
        return dist.HalfNormal(self.scale)


@dataclass
class InverseGamma(Prior):
    """Inverse-Gamma prior.  Support: (0, inf)."""

    concentration: float = 1.0
    rate: float = 1.0

    def to_numpyro(self):
        dist = _import_numpyro_dist()
        return dist.InverseGamma(self.concentration, self.rate)


@dataclass
class Exponential(Prior):
    """Exponential prior.  Support: (0, inf)."""

    rate: float = 1.0

    def to_numpyro(self):
        dist = _import_numpyro_dist()
        return dist.Exponential(self.rate)


@dataclass
class Gamma(Prior):
    """Gamma prior.  Support: (0, inf)."""

    concentration: float = 1.0
    rate: float = 1.0

    def to_numpyro(self):
        dist = _import_numpyro_dist()
        return dist.Gamma(self.concentration, self.rate)


@dataclass
class LogNormal(Prior):
    """Log-Normal prior.  Support: (0, inf)."""

    loc: float = 0.0
    scale: float = 1.0

    def to_numpyro(self):
        dist = _import_numpyro_dist()
        return dist.LogNormal(self.loc, self.scale)


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

@dataclass
class LKJ(Prior):
    """LKJ prior for correlation matrices.

    Parameters
    ----------
    dim : int
        Dimension of the correlation matrix.
    concentration : float
        Concentration parameter (eta).  ``concentration=1`` is uniform
        over correlation matrices; larger values concentrate toward the
        identity matrix.
    """

    dim: int = 2
    concentration: float = 1.0

    def to_numpyro(self):
        dist = _import_numpyro_dist()
        return dist.LKJCholesky(self.dim, self.concentration)


@dataclass
class Beta(Prior):
    """Beta prior.  Support: (0, 1).  Useful for correlations via transform."""

    concentration1: float = 1.0
    concentration0: float = 1.0

    def to_numpyro(self):
        dist = _import_numpyro_dist()
        return dist.Beta(self.concentration1, self.concentration0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "Prior",
    # Unbounded
    "Normal",
    "StudentT",
    "Cauchy",
    "Uniform",
    "Laplace",
    # Positive
    "HalfCauchy",
    "HalfNormal",
    "InverseGamma",
    "Exponential",
    "Gamma",
    "LogNormal",
    # Correlations
    "LKJ",
    "Beta",
]
