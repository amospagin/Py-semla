# Roadmap

## Shipped

- [x] ML estimation with lavaan-validated results
- [x] MLR estimator (robust Satorra-Bentler)
- [x] DWLS estimator for ordinal data (polychoric correlations)
- [x] FIML for missing data
- [x] Bayesian MCMC estimation via NumPyro (NUTS sampler)
- [x] GPU-accelerated Bayesian inference via JAX
- [x] Batch Bayesian estimation with CPU+GPU parallel scheduling
- [x] Data-adaptive and weak informative priors
- [x] Adaptive convergence monitoring
- [x] WAIC and PSIS-LOO model comparison
- [x] IRT models (1PL, 2PL, GRM) with ICC, information functions, ability estimation
- [x] Multi-group CFA (configural, metric, scalar, strict invariance)
- [x] Growth curve models (linear and nonlinear)
- [x] Higher-order (second-order) factor models
- [x] Cross-lagged panel models
- [x] Standardized solutions (std.all, std.lv)
- [x] Modification indices
- [x] Chi-square difference test
- [x] Multi-model comparison table (compare_models)
- [x] AIC / BIC / adjusted BIC
- [x] R-squared for endogenous variables
- [x] Mean structure and intercepts (~1)
- [x] Equality constraints via parameter labels
- [x] Indirect effects and mediation (:= operator with delta method SEs)
- [x] Reliability measures (McDonald's omega, Cronbach's alpha)
- [x] Factor score prediction (regression and Bartlett)
- [x] Bootstrap confidence intervals
- [x] Residual diagnostics and Mardia's multivariate normality test
- [x] Model-implied matrices (fitted)
- [x] Parameter variance-covariance matrix (vcov)
- [x] Parallel chain execution on CPU
- [x] Positive loading constraints for sign identification
- [x] Validated against lavaan 0.6-20

## Known Issues

- [ ] MLR robust SEs diverge from lavaan for some parameters ([#36](https://github.com/amospagin/semla/issues/36))
- [ ] Modification indices inflated for residual covariances ([#37](https://github.com/amospagin/semla/issues/37))

## Future Directions

- [ ] Path diagram visualization ([#38](https://github.com/amospagin/semla/issues/38))
- [ ] Auto-add exogenous observed variable covariances in sem() ([#42](https://github.com/amospagin/semla/issues/42))
- [ ] Nonlinear parameter constraints ([#45](https://github.com/amospagin/semla/issues/45))
- [ ] Publication-ready table export ([#46](https://github.com/amospagin/semla/issues/46))
- [ ] Complex survey weight support ([#47](https://github.com/amospagin/semla/issues/47))
- [ ] Bayesian model comparison (Bayes factors)
- [ ] Traceplots and posterior visualization
- [ ] Multi-group Bayesian estimation
- [ ] 3PL and multidimensional IRT models
- [ ] Latent class / mixture models
- [ ] Penalized likelihood (lasso/ridge SEM)
- [ ] Simulation / power analysis utilities
- [ ] PyPI release
