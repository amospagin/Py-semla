# Bayesian Estimation

semla supports full Bayesian SEM estimation via [NumPyro](https://num.pyro.ai/). Pass `estimator="bayes"` to use the NUTS (No-U-Turn Sampler) with automatic convergence monitoring.

## Installation

```bash
pip install semla[bayes]       # CPU
pip install semla[bayes-cuda]  # NVIDIA GPU (much faster for large models)
```

## Basic Usage

```python
from semla import cfa
from semla.datasets import HolzingerSwineford1939

df = HolzingerSwineford1939()

model = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""

fit = cfa(model, data=df, estimator="bayes")
fit.summary()
```

## Sampling Options

```python
fit = cfa(model, data=df, estimator="bayes",
          chains=4,              # number of MCMC chains (default: 4)
          warmup=1000,           # warmup iterations per chain (default: 1000)
          draws=2000,            # posterior draws per chain (default: 1000)
          cores=4,               # CPU cores for parallel chains (default: = chains)
          seed=42,               # random seed for reproducibility
          adapt_delta=0.8,       # NUTS target acceptance rate (default: 0.8)
          adapt_convergence=True,  # auto-retry if R-hat > 1.01 (default: True)
          progress_bar=True,     # show sampling progress (default: True)
          positive_loadings=True,  # constrain loadings > 0 (default: True)
)
```

## Priors

By default, semla uses **data-adaptive priors** scaled by observed standard deviations (brms-style):

- **Loadings**: Normal(0, 2.5 * SD(indicator))
- **Residual variances**: InverseGamma(2, SD(indicator)^2)
- **Factor variances**: InverseGamma(2, median_SD^2)
- **Covariances**: Normal(0, 2.5 * median_SD^2)

### Weak Informative Priors

```python
fit = cfa(model, data=df, estimator="bayes", priors="weak")
```

### Custom Priors

Override at the matrix level or per parameter:

```python
from semla.priors import Normal, InverseGamma, HalfCauchy

# Matrix-level: all loadings get the same prior
fit = cfa(model, data=df, estimator="bayes",
          priors={"loadings": Normal(0, 1)})

# Per-parameter: override specific parameters
fit = cfa(model, data=df, estimator="bayes",
          priors={
              "loadings": Normal(0, 1),           # default for all loadings
              "f1=~x2": Normal(0.7, 0.2),          # specific override
              "residual_variances": InverseGamma(2, 1),
          })
```

Matrix-level keys: `loadings`, `regressions`, `residual_variances`, `factor_variances`, `covariances`, `intercepts`.

### Available Prior Distributions

```python
from semla.priors import (
    Normal, StudentT, Cauchy, Uniform, Laplace,       # unbounded
    HalfCauchy, HalfNormal, InverseGamma,              # positive
    Exponential, Gamma, LogNormal,                      # positive
    LKJ, Beta,                                          # correlations
)
```

## Working with Results

```python
# Posterior summary (mean, median, SD, CI, R-hat, ESS)
fit.results.estimates()

# Raw posterior draws as DataFrame
draws = fit.results.draws()

# MCMC diagnostics
fit.results.diagnostics()
# {'divergences': 0, 'divergence_pct': 0.0, 'min_ess': 3200, 'max_rhat': 1.001, ...}

# Model comparison
fit.results.waic()    # {'waic': 7483.2, 'p_waic': 20.5, 'se': 42.1}
fit.results.loo()     # {'loo': 7484.1, 'p_loo': 21.0, 'se': 42.3, 'k_max': 0.45}
```

## Adaptive Convergence

When `adapt_convergence=True` (default), semla monitors convergence after sampling:

1. Checks R-hat across all parameters
2. If max R-hat > 1.01: extends draws by 2x
3. If still bad: restarts with longer warmup and higher adapt_delta
4. Checks divergence proportion and auto-retries if > 5%

This means most models "just work" without manual tuning.

## GPU Acceleration

With `pip install semla[bayes-cuda]`, all MCMC computation runs on NVIDIA GPUs automatically. No code changes needed -- JAX handles device placement transparently.

For multi-GPU setups, NumPyro distributes chains across available GPUs (one chain per GPU).

```python
# Same code, runs on GPU if jax[cuda] is installed
fit = cfa(model, data=df, estimator="bayes", chains=4)
```

For large models (many latent variables, large datasets), GPU acceleration can provide 10-50x speedups over CPU.

## Parallel Chains on CPU

By default, chains run in parallel (one core per chain). Control this with `cores`:

```python
fit = cfa(model, data=df, estimator="bayes", chains=4, cores=2)  # 2 batches of 2
```

Or set it once at the start of your script:

```python
import semla
semla.set_host_devices(8)  # before any Bayesian fitting
```

## Positive Loading Constraints

By default, free loadings are constrained to be positive. This prevents sign-flipping non-identifiability that can occur in structural models with latent regressions. Disable with:

```python
fit = cfa(model, data=df, estimator="bayes", positive_loadings=False)
```

## Comparison with ML

With sufficient data and diffuse priors, Bayesian posterior means converge to ML point estimates. On the Holzinger-Swineford dataset, all parameters match within 0.033 and 100% of ML estimates fall within the Bayesian 95% credible intervals.
