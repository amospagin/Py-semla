# API Reference

## Main Functions

### `cfa(model, data, group=None, **kwargs)`

Fit a Confirmatory Factor Analysis model. Automatically adds covariances between latent variables.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Model syntax in lavaan format |
| `data` | `DataFrame` | Data with columns matching observed variables |
| `group` | `str`, optional | Column name for multi-group analysis |
| `estimator` | `str` | `"ML"` (default), `"MLR"`, `"DWLS"`, or `"bayes"` |
| `missing` | `str` | `"listwise"` (default) or `"fiml"` |
| `invariance` | `str` | `"configural"`, `"metric"`, `"scalar"`, `"strict"` (multi-group) |
| `meanstructure` | `bool` | Estimate intercepts (default: False) |

**Bayesian-specific kwargs:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chains` | `int` | 4 | Number of MCMC chains |
| `warmup` | `int` | 1000 | Warmup iterations per chain |
| `draws` | `int` | 1000 | Posterior draws per chain |
| `cores` | `int` | = chains | CPU cores for parallel chains |
| `seed` | `int` | 0 | Random seed |
| `priors` | `str/dict` | None | `"weak"`, or dict of Prior objects |
| `adapt_delta` | `float` | 0.8 | NUTS target acceptance rate |
| `adapt_convergence` | `bool` | True | Auto-retry on poor convergence |
| `positive_loadings` | `bool` | True | Constrain loadings > 0 |

**Returns:** `Model` (single-group) or `MultiGroupModel` (multi-group)

---

### `sem(model, data, group=None, **kwargs)`

Fit a Structural Equation Model. Same as `cfa()` but does NOT auto-add covariances between latent variables.

---

### `irt(items, data, model_type="2PL", **kwargs)`

Fit an Item Response Theory model.

| Parameter | Type | Description |
|-----------|------|-------------|
| `items` | `list[str]` | Column names for the items |
| `data` | `DataFrame` | Data with item columns |
| `model_type` | `str` | `"1PL"`, `"2PL"`, or `"GRM"` |

**Returns:** `IRTModel`

---

### `chi_square_diff_test(model_restricted, model_free)`

Chi-square difference test for nested model comparison.

**Returns:** `dict` with `chi_sq_diff`, `df_diff`, `p_value`

---

### `mardia_test(data)`

Mardia's test for multivariate normality.

**Returns:** `dict` with `skewness`, `skewness_p`, `kurtosis`, `kurtosis_p`, `recommendation`

---

### `set_host_devices(n)`

Set the number of CPU devices for parallel MCMC chains. Call before any Bayesian fitting.

---

## Model Object

Returned by `cfa()` and `sem()` for single-group frequentist models.

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Lavaan-style summary |
| `fit_indices()` | `dict` | Chi-square, CFI, TLI, RMSEA, SRMR, AIC, BIC |
| `estimates()` | `DataFrame` | Parameter estimates with SEs, z-values, p-values |
| `standardized_estimates(type)` | `DataFrame` | `"std.all"` or `"std.lv"` |
| `modindices(min_mi=0)` | `DataFrame` | Modification indices |
| `defined_estimates()` | `DataFrame` | Defined parameters (`:=` operator) |
| `residuals(type="raw")` | `ndarray` | Residual covariance matrix |
| `r_squared()` | `dict` | R-squared for endogenous variables |
| `reliability()` | `dict` | Omega and alpha per factor |
| `predict(method="regression")` | `DataFrame` | Factor scores |
| `bootstrap(nboot=1000)` | `DataFrame` | Bootstrap confidence intervals |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `converged` | `bool` | Whether the optimizer converged |

---

## BayesianResults Object

Returned as `fit.results` when `estimator="bayes"`.

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Bayesian summary with posterior stats |
| `estimates()` | `DataFrame` | Mean, median, SD, CI, R-hat, ESS |
| `draws()` | `DataFrame` | Raw posterior samples |
| `diagnostics()` | `dict` | Divergences, min ESS, max R-hat |
| `waic()` | `dict` | WAIC, p_waic, SE |
| `loo()` | `dict` | LOO, p_loo, SE, k_max |
| `fit_indices()` | `dict` | Same as `waic()` |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `converged` | `bool` | max R-hat < 1.05 |

---

## IRTModel Object

Returned by `irt()`.

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | IRT model summary |
| `irt_params()` | `DataFrame` | Discrimination, difficulty, thresholds |
| `icc()` | `DataFrame` | Item characteristic curves |
| `item_information()` | `DataFrame` | Item information functions |
| `test_information()` | `DataFrame` | Total test information and SE |
| `abilities(method="regression")` | `DataFrame` | Person ability estimates |

---

## MultiGroupModel Object

Returned by `cfa()` and `sem()` when `group=` is specified.

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Multi-group summary |
| `fit_indices()` | `dict` | Multi-group fit indices |
| `estimates()` | `DataFrame` | Estimates with `group` column |

---

## Prior Classes (`semla.priors`)

All priors are dataclasses with a `to_numpyro()` method.

**Unbounded:** `Normal(mu, sigma)`, `StudentT(df, loc, scale)`, `Cauchy(loc, scale)`, `Uniform(low, high)`, `Laplace(loc, scale)`

**Positive:** `HalfCauchy(scale)`, `HalfNormal(scale)`, `InverseGamma(concentration, rate)`, `Exponential(rate)`, `Gamma(concentration, rate)`, `LogNormal(loc, scale)`

**Correlations:** `LKJ(dim, concentration)`, `Beta(concentration1, concentration0)`

---

## Datasets

### `semla.datasets.HolzingerSwineford1939()`

Classic dataset: 301 students, 9 mental ability tests, 2 schools.

**Columns:** id, sex, ageyr, agemo, school, grade, x1-x9

**Factors:** visual (x1-x3), textual (x4-x6), speed (x7-x9)
