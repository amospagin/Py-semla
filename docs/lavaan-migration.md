# Coming from R?

If you know lavaan, blavaan, or mirt in R, you already know semla. The syntax is identical and the API is deliberately similar.

## Function Mapping

| lavaan / blavaan / mirt (R) | semla (Python) |
|------------------------------|----------------|
| `cfa(model, data)` | `cfa(model, data)` |
| `sem(model, data)` | `sem(model, data)` |
| `summary(fit)` | `fit.summary()` |
| `fitMeasures(fit)` | `fit.fit_indices()` |
| `parameterEstimates(fit)` | `fit.estimates()` |
| `standardizedSolution(fit)` | `fit.standardized_estimates()` |
| `modindices(fit)` | `fit.modindices()` |
| `cfa(model, data, group="x")` | `cfa(model, data, group="x")` |
| `cfa(model, data, ordered=TRUE)` | `cfa(model, data, estimator="DWLS")` |
| `cfa(model, data, estimator="MLR")` | `cfa(model, data, estimator="MLR")` |
| `cfa(model, data, missing="fiml")` | `cfa(model, data, missing="fiml")` |
| `blavaan::bcfa(model, data)` | `cfa(model, data, estimator="bayes")` |
| `mirt(data, 1, itemtype="2PL")` | `irt(items, data, model_type="2PL")` |
| `anova(fit1, fit2)` | `chi_square_diff_test(fit1, fit2)` |
| `lavPredict(fit)` | `fit.predict()` |
| `reliability(fit)` | `fit.reliability()` |
| `bootstrapLavaan(fit)` | `fit.bootstrap(nboot=1000)` |

## Syntax Comparison

The model syntax is identical:

```python
# Same syntax in both lavaan and semla
model = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
    speed ~ visual + textual
    x1 ~~ x2
    indirect := a*b
"""
```

## Key Differences

1. **Return values** -- lavaan uses R's S4 objects with `summary()`, `coef()`, etc. semla returns Python objects with methods like `.summary()`, `.estimates()`.

2. **DataFrames** -- semla returns pandas DataFrames instead of R data.frames. Filter, sort, and manipulate with standard pandas operations.

3. **Ordinal data** -- lavaan uses `ordered=TRUE`, semla uses `estimator="DWLS"`.

4. **Bayesian** -- lavaan requires the separate blavaan package. semla uses `estimator="bayes"` with NumPyro (includes GPU support).

5. **IRT** -- R uses the separate mirt package. semla uses `irt()` from the same package.

## Validation

semla is validated against lavaan 0.6-21 on the Holzinger-Swineford dataset. Parameter estimates match within 0.005 for ML, and within 0.02 for FIML. See the test suite for exact comparisons.
