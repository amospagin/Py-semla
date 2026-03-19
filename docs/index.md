# semla

**Latent variable modeling and SEM in Python, with lavaan syntax.**

!!! note "Early Development"
    semla is in early development (v0.1.0). The API may change, and results should be validated against established tools like lavaan before use in published research.

semla is a Python package for structural equation modeling, confirmatory factor analysis, latent growth curves, IRT, and other latent variable models. It uses [lavaan](https://lavaan.ugent.be/)-style syntax for model specification, so if you know lavaan, you already know the syntax.

## Quick Example

```python
from semla import cfa
from semla.datasets import HolzingerSwineford1939

df = HolzingerSwineford1939()

model = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""

fit = cfa(model, data=df)
fit.summary()
```

## Features

- **lavaan syntax** -- same `=~`, `~`, `~~` operators you already know
- **Five estimators** -- ML, MLR (robust), DWLS (ordinal), FIML (missing data), Bayesian MCMC
- **Model types** -- CFA, SEM, mediation, growth curves (linear/nonlinear), higher-order factor models, cross-lagged panel models, IRT (1PL, 2PL, GRM)
- **Bayesian estimation** -- NumPyro NUTS sampler with adaptive priors, adaptive convergence, WAIC, and PSIS-LOO
- **Batch estimation** -- run many Bayesian models in parallel across CPU cores + GPU with `batch_bayes()`
- **GPU-accelerated** -- Bayesian estimation runs on NVIDIA GPUs via JAX
- **Multi-group analysis** -- configural, metric, scalar, and strict invariance with chi-square difference testing
- **Model comparison** -- `compare_models()` for anova-style multi-model comparison tables
- **Fit indices** -- chi-square, CFI, TLI, RMSEA (with CI), SRMR, AIC, BIC, WAIC, LOO
- **Diagnostics** -- modification indices, standardized solutions, residuals, model-implied matrices (`fitted()`), parameter covariance matrix (`vcov()`), Mardia's normality test, R-squared, reliability
- **Validated against lavaan** -- parameter estimates and standard errors match within 0.005

## Installation

```bash
pip install semla              # core (ML, MLR, DWLS, FIML, IRT, multi-group)
pip install semla[bayes]       # + Bayesian estimation (CPU)
pip install semla[bayes-cuda]  # + Bayesian estimation (NVIDIA GPU)
```

Or from source:

```bash
git clone https://github.com/amospagin/semla.git
cd semla
pip install -e ".[bayes]"
```

## Why semla?

Python lacks a mature package for specifying latent variable models with familiar lavaan syntax and running both frequentist and Bayesian estimation from the same interface. semla fills this gap -- with the added ability to run batches of Bayesian models in parallel across CPU cores and GPU.

## Guides

- [Getting Started](getting-started.md) -- installation, first model, syntax overview
- [CFA](guide/cfa.md) -- confirmatory factor analysis
- [SEM](guide/sem.md) -- structural models, mediation, indirect effects
- [Bayesian Estimation](guide/bayesian.md) -- MCMC, priors, GPU, WAIC/LOO
- [IRT Models](guide/irt.md) -- 1PL, 2PL, Graded Response Model
- [Multi-Group Analysis](guide/multigroup.md) -- measurement invariance
- [Ordinal Data (DWLS)](guide/dwls.md) -- polychoric correlations
- [Coming from R?](lavaan-migration.md) -- lavaan/blavaan/mirt mapping
- [API Reference](api.md) -- full function and class documentation
- [Roadmap](roadmap.md) -- what's shipped and what's next
