# Getting Started

## Installation

```bash
pip install semla              # core (ML, MLR, DWLS, FIML, IRT, multi-group)
pip install semla[bayes]       # + Bayesian estimation (CPU)
pip install semla[bayes-cuda]  # + Bayesian estimation (NVIDIA GPU)
```

## Your First CFA Model

A Confirmatory Factor Analysis tests whether observed variables load on hypothesized latent factors.

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

## Examining Results

### Summary

```python
fit.summary()
```

Prints a lavaan-style summary with fit indices, parameter estimates, standard errors, and p-values.

### Fit Indices

```python
fit.fit_indices()
# {'chi_square': 85.0, 'df': 24, 'cfi': 0.931, 'tli': 0.896, ...}
```

| Index | Good Fit |
|-------|----------|
| CFI | > .95 |
| TLI | > .95 |
| RMSEA | < .06 |
| SRMR | < .08 |

### Parameter Estimates

```python
fit.estimates()                          # unstandardized
fit.standardized_estimates()             # fully standardized (std.all)
fit.standardized_estimates("std.lv")     # standardized by LV SD only
```

### More Results

```python
fit.modindices(min_mi=5.0)               # modification indices
fit.r_squared()                          # R-squared for endogenous variables
fit.reliability()                        # McDonald's omega and Cronbach's alpha
fit.residuals(type="standardized")       # residual covariance matrix
fit.predict()                            # factor scores (regression method)
fit.bootstrap(nboot=1000)                # bootstrap confidence intervals
fit.defined_estimates()                  # indirect effects (:= operator)
```

## Choosing an Estimator

```python
fit = cfa(model, data=df)                          # ML (default)
fit = cfa(model, data=df, estimator="MLR")         # robust ML (non-normal data)
fit = cfa(model, data=df, estimator="DWLS")        # ordinal/categorical data
fit = cfa(model, data=df, missing="fiml")          # missing data (FIML)
fit = cfa(model, data=df, estimator="bayes")       # Bayesian MCMC
```

## Model Syntax

semla uses the same operators as lavaan:

| Operator | Meaning | Example |
|----------|---------|---------|
| `=~` | Latent variable definition | `visual =~ x1 + x2 + x3` |
| `~` | Regression | `y ~ x1 + x2` |
| `~~` | (Co)variance | `x1 ~~ x2` |
| `~1` | Intercept | `y ~1` |
| `:=` | Defined parameter | `indirect := a*b` |

### Modifiers

```python
"f1 =~ 1*x1 + x2 + x3"        # fix a loading to a specific value
"f1 =~ NA*x1 + x2 + x3"       # free the first loading
"f1 =~ x1 + a*x2 + a*x3"      # equality constraint (same label)
```

## Next Steps

- [CFA Guide](guide/cfa.md) -- factor analysis in depth
- [SEM Guide](guide/sem.md) -- structural models and mediation
- [Bayesian Estimation](guide/bayesian.md) -- MCMC with NumPyro
- [IRT Models](guide/irt.md) -- item response theory
- [Multi-Group Analysis](guide/multigroup.md) -- measurement invariance
- [Ordinal Data](guide/dwls.md) -- DWLS for Likert-scale data
