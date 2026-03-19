# Structural Equation Modeling

SEM combines a measurement model (CFA) with structural paths (regressions between latent variables).

## Basic SEM

```python
from semla import sem

model = """
    # Measurement model
    ind60 =~ x1 + x2 + x3
    dem60 =~ y1 + y2 + y3 + y4
    dem65 =~ y5 + y6 + y7 + y8

    # Structural model (regressions)
    dem60 ~ ind60
    dem65 ~ ind60 + dem60
"""

fit = sem(model, data=df)
fit.summary()
```

## Difference Between `cfa()` and `sem()`

| | `cfa()` | `sem()` |
|---|---------|---------|
| Latent covariances | Auto-added (all factors correlate) | Auto-added between exogenous latent variables only |
| Use case | Testing measurement structure | Testing causal/structural paths |

If your model has regressions between latent variables (`~`), use `sem()`. If it's purely a factor model, use `cfa()`.

## Mediation Analysis

Test indirect effects using the `:=` operator with labeled paths:

```python
from semla import sem

model = """
    M ~ a*X          # X -> M (path a)
    Y ~ b*M + c*X    # M -> Y (path b), X -> Y (direct, path c)

    indirect := a*b   # indirect effect
    total := a*b + c  # total effect
"""

fit = sem(model, data=df)
fit.defined_estimates()  # indirect effect with delta method SE
```

You can also use bootstrap confidence intervals for the indirect effect:

```python
fit.bootstrap(nboot=1000)
```

## Correlated Residuals

Sometimes indicators share variance beyond their factor (e.g., similar wording):

```python
model = """
    f1 =~ x1 + x2 + x3 + x4
    x1 ~~ x2   # correlated residuals
"""
```

## Mean Structure

Estimate intercepts for observed variables:

```python
fit = sem(model, data=df, meanstructure=True)
```

Or use the `~1` operator in syntax:

```python
model = """
    f1 =~ x1 + x2 + x3
    x1 ~1
"""
```

## Equality Constraints

Constrain parameters to be equal using labels:

```python
model = """
    f1 =~ x1 + a*x2 + a*x3   # x2 and x3 loadings forced equal
"""
```

## Nonlinear Constraints

Go beyond simple equality with inequality and nonlinear constraints on labeled parameters:

```python
# Inequality constraints
model = """
    f1 =~ x1 + a*x2 + b*x3
    f2 =~ x4 + x5 + x6
    a > 0          # loading must be positive
    b < 1          # loading must be less than 1
    a < b          # ordering constraint
"""

# Nonlinear equality constraints
model = """
    f1 =~ x1 + a*x2 + b*x3
    f2 =~ x4 + c*x5 + x6
    a + b == c     # sum of two loadings equals a third
    a*b == 0.5     # product constraint
"""
```

Supported operators: `>`, `<`, `>=`, `<=`, `==`. Expressions can reference any labeled parameter and use arithmetic (`+`, `-`, `*`, `/`).

When constraints are present, the optimizer switches from BFGS to SLSQP (sequential least-squares programming). Models without constraints are unaffected.

## R-Squared and Factor Scores

```python
fit.r_squared()                          # variance explained per endogenous variable
fit.predict(method="regression")         # Thurstone regression scores
fit.predict(method="bartlett")           # Bartlett scores
```

## Bayesian SEM

Structural models can also be estimated with Bayesian MCMC:

```python
fit = sem(model, data=df, estimator="bayes", chains=4, draws=2000)
fit.results.estimates()     # posterior mean, CI, R-hat, ESS
```

See the [Bayesian Estimation](bayesian.md) guide for details.
