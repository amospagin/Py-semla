# semla

**Structural Equation Modeling with lavaan-style syntax for Python.**

!!! note "Early Development"
    semla is in early development (v0.1.0). The API may change, and results should be validated against established tools like lavaan before use in published research.

semla brings the familiar [lavaan](https://lavaan.ugent.be/) model syntax from R to Python. If you know lavaan, you already know semla.

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

- **lavaan syntax** — same `=~`, `~`, `~~` operators you already know
- **Five estimators** — ML, MLR (robust), DWLS (ordinal), FIML (missing data), Bayesian MCMC
- **Bayesian estimation** — NumPyro NUTS sampler with adaptive priors, adaptive convergence, WAIC, and PSIS-LOO
- **GPU-accelerated** — Bayesian estimation runs on NVIDIA GPUs via JAX for large models and datasets
- **IRT models** — 1PL (Rasch), 2PL, and Graded Response Model with ICC, information functions, and ability estimation
- **Multi-group analysis** — configural, metric, scalar, and strict invariance with chi-square difference testing
- **Fit indices** — chi-square, CFI, TLI, RMSEA (with CI), SRMR, AIC, BIC, WAIC, LOO
- **Diagnostics** — modification indices, standardized solutions, residuals, Mardia's normality test, R-squared, reliability
- **Validated against lavaan** — parameter estimates and standard errors match within 0.005

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

There's no mature Python package that lets you specify SEM models with lavaan syntax and run both frequentist and Bayesian estimation from the same interface. If you're a researcher who uses lavaan or blavaan in R but works in Python, semla lets you stay in one language — with the added benefit of GPU-accelerated Bayesian inference via JAX for models that would take hours on CPU.
