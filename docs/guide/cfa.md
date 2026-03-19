# Confirmatory Factor Analysis

CFA tests whether a set of observed variables loads on hypothesized latent factors.

## Basic CFA

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

## What `cfa()` Does Automatically

Compared to `sem()`, the `cfa()` function adds:

- **Covariances between all latent variables** -- all factors are allowed to correlate freely
- **First loading fixed to 1.0** per factor -- for identification (same as lavaan's default)
- **Residual variances** -- automatically added for all observed and latent variables

## Standardized Loadings

Standardized loadings are essential for interpretation. A loading of 0.7 means the factor explains 49% of the indicator's variance.

```python
std = fit.standardized_estimates("std.all")
loadings = std[std["op"] == "=~"]
print(loadings[["lhs", "rhs", "est", "est.std"]])
```

| std.all | Interpretation |
|---------|---------------|
| > 0.7 | Strong loading |
| 0.5-0.7 | Moderate |
| 0.3-0.5 | Weak |
| < 0.3 | Consider removing |

## Fixing and Freeing Parameters

```python
# Fix all loadings to specific values
"f1 =~ 1*x1 + 0.8*x2 + 0.6*x3"

# Free the first loading (estimate it instead of fixing to 1.0)
"f1 =~ NA*x1 + x2 + x3"

# Equality constraints
"f1 =~ x1 + a*x2 + a*x3"   # x2 and x3 loadings constrained equal
```

## Reliability

```python
fit.reliability()
# {'visual': {'omega': 0.601, 'alpha': 0.592},
#  'textual': {'omega': 0.887, 'alpha': 0.884}, ...}
```

McDonald's omega uses the model-implied factor structure. Cronbach's alpha is computed from the model-implied covariance matrix.

## Model Improvement with Modification Indices

```python
mi = fit.modindices(min_mi=5.0)
print(mi)
```

Large MI values suggest paths that, if freed, would significantly improve fit. Only add paths that are theoretically justified.

## Choosing an Estimator

```python
fit = cfa(model, data=df)                      # ML (default)
fit = cfa(model, data=df, estimator="MLR")     # robust (non-normal data)
fit = cfa(model, data=df, estimator="DWLS")    # ordinal data
fit = cfa(model, data=df, missing="fiml")      # missing data
fit = cfa(model, data=df, estimator="bayes")   # Bayesian MCMC
```

## Multivariate Normality Check

```python
from semla import mardia_test

result = mardia_test(df[["x1", "x2", "x3", "x4", "x5", "x6"]].values)
print(result["recommendation"])  # "ML" or "MLR"
```

If data is non-normal, consider `estimator="MLR"` for robust standard errors and a scaled chi-square test.

## Reporting CFA Results

A typical CFA results section reports:

1. **Model specification** -- how many factors, which indicators
2. **Fit indices** -- chi-square (with df and p), CFI, TLI, RMSEA (with 90% CI), SRMR
3. **Standardized factor loadings** -- from `standardized_estimates()`
4. **Factor correlations** -- from the `~~` parameters between latent variables
5. **Reliability** -- omega and/or alpha per factor
