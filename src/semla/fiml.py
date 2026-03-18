"""Full Information Maximum Likelihood (FIML) for missing data.

Instead of computing a single sample covariance matrix, FIML evaluates
the log-likelihood for each observation using only its observed variables.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import optimize

from .estimation import EstimationResult, _model_implied_cov, _model_implied_mean
from .specification import ModelSpecification


def _group_by_pattern(data: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group observations by missingness pattern.

    Returns list of (obs_var_indices, case_data_matrix) tuples.
    """
    patterns: dict[tuple, list[int]] = {}
    n, p = data.shape
    for i in range(n):
        obs = tuple(np.where(~np.isnan(data[i]))[0])
        patterns.setdefault(obs, []).append(i)

    groups = []
    for obs_tuple, row_indices in patterns.items():
        obs_idx = np.array(obs_tuple, dtype=int)
        rows = data[np.ix_(row_indices, obs_idx)]
        groups.append((obs_idx, rows))
    return groups


def fiml_objective(
    theta: np.ndarray,
    spec: ModelSpecification,
    pattern_groups: list[tuple[np.ndarray, np.ndarray]],
    n_obs: int,
) -> float:
    """FIML fitting function: sum of casewise log-likelihoods.

    Returns F = -2/(N) * sum(loglik_i), scaled for chi-square compatibility.
    """
    A, S_mat = spec.unpack(theta)
    sigma_full = _model_implied_cov(A, S_mat, spec.F)
    if sigma_full is None:
        return 1e10

    m = spec.unpack_m(theta)
    mu_full = _model_implied_mean(A, m, spec.F)
    if mu_full is None:
        return 1e10

    total_loglik = 0.0

    for obs_idx, case_data in pattern_groups:
        p_k = len(obs_idx)
        n_k = case_data.shape[0]

        if p_k == 0:
            continue

        # Extract sub-matrices
        sigma_k = sigma_full[np.ix_(obs_idx, obs_idx)]
        mu_k = mu_full[obs_idx]

        try:
            sign, logdet = np.linalg.slogdet(sigma_k)
            if sign <= 0:
                return 1e10
            sigma_k_inv = np.linalg.inv(sigma_k)
        except np.linalg.LinAlgError:
            return 1e10

        # Log-likelihood for this group
        # -0.5 * sum_i [p_k*log(2pi) + log|Sigma_k| + (x_i - mu_k)' Sigma_k^{-1} (x_i - mu_k)]
        centered = case_data - mu_k
        # Quadratic form for all cases in this group
        quad = np.sum(centered @ sigma_k_inv * centered, axis=1)
        group_loglik = -0.5 * (n_k * p_k * np.log(2 * np.pi) + n_k * logdet + np.sum(quad))
        total_loglik += group_loglik

    # Return scaled objective: F = -2/N * loglik
    return -2.0 / n_obs * total_loglik


def fiml_gradient(
    theta: np.ndarray,
    spec: ModelSpecification,
    pattern_groups: list[tuple[np.ndarray, np.ndarray]],
    n_obs: int,
) -> np.ndarray:
    """Numerical gradient of FIML objective."""
    eps = 1e-7
    grad = np.zeros_like(theta)
    f0 = fiml_objective(theta, spec, pattern_groups, n_obs)
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        grad[i] = (fiml_objective(theta_plus, spec, pattern_groups, n_obs) - f0) / eps
    return grad


def _compute_se_fiml(
    theta: np.ndarray,
    spec: ModelSpecification,
    pattern_groups: list[tuple[np.ndarray, np.ndarray]],
    n_obs: int,
) -> np.ndarray:
    """Compute SEs via numerical Hessian of FIML objective."""
    eps = 1e-5
    k = len(theta)
    H = np.zeros((k, k))

    for i in range(k):
        for j in range(i, k):
            theta_pp = theta.copy()
            theta_pm = theta.copy()
            theta_mp = theta.copy()
            theta_mm = theta.copy()

            theta_pp[i] += eps; theta_pp[j] += eps
            theta_pm[i] += eps; theta_pm[j] -= eps
            theta_mp[i] -= eps; theta_mp[j] += eps
            theta_mm[i] -= eps; theta_mm[j] -= eps

            H[i, j] = (
                fiml_objective(theta_pp, spec, pattern_groups, n_obs)
                - fiml_objective(theta_pm, spec, pattern_groups, n_obs)
                - fiml_objective(theta_mp, spec, pattern_groups, n_obs)
                + fiml_objective(theta_mm, spec, pattern_groups, n_obs)
            ) / (4 * eps * eps)
            H[j, i] = H[i, j]

    # F_FIML = -2/N * loglik, so Hessian of F = -2/N * Hessian of loglik
    # Var(theta) = inv(-Hessian of loglik) = inv(N/2 * H) = 2/(N*H)
    # But since H already has the -2/N factor, Var = inv(N/2 * H_F)
    # Actually: d2(-2*loglik/N) = H, so d2(loglik) = -N/2 * H
    # Fisher info = -d2(loglik) = N/2 * H
    # Var = inv(Fisher) = 2/(N*H) ... but H should be the per-observation info
    # Since H = d2(F)/d(theta)^2 and F = -2/N*loglik:
    # Var(theta) = 2 / (N * diag(H)) ... this is approximate
    try:
        H_inv = np.linalg.inv(H)
        # Var(theta) = 2/N * H_F^{-1} where H_F is the Hessian of F_FIML
        var_theta = 2.0 / n_obs * np.diag(H_inv)
        se = np.where(var_theta > 0, np.sqrt(var_theta), np.nan)
        return se
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)


def _saturated_loglik(
    data: np.ndarray,
    pattern_groups: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Compute FIML log-likelihood for the saturated model.

    Uses overall (EM-like) sample means and pairwise covariance as the
    unrestricted model, evaluated via the same casewise FIML approach.
    """
    n, p = data.shape

    # Compute overall sample statistics from available data
    mu_sat = np.nanmean(data, axis=0)

    # Pairwise covariance
    sigma_sat = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1):
            valid = ~(np.isnan(data[:, i]) | np.isnan(data[:, j]))
            if valid.sum() > 1:
                xi = data[valid, i] - mu_sat[i]
                xj = data[valid, j] - mu_sat[j]
                sigma_sat[i, j] = np.sum(xi * xj) / (valid.sum() - 1)
                sigma_sat[j, i] = sigma_sat[i, j]

    # Ensure PD
    eigvals = np.linalg.eigvalsh(sigma_sat)
    if eigvals[0] <= 0:
        sigma_sat += (-eigvals[0] + 1e-6) * np.eye(p)

    # Evaluate FIML log-likelihood with saturated parameters
    total_loglik = 0.0
    for obs_idx, case_data in pattern_groups:
        p_k = len(obs_idx)
        n_k = case_data.shape[0]
        if p_k == 0:
            continue

        sigma_k = sigma_sat[np.ix_(obs_idx, obs_idx)]
        mu_k = mu_sat[obs_idx]

        try:
            sign, logdet = np.linalg.slogdet(sigma_k)
            if sign <= 0:
                continue
            sigma_k_inv = np.linalg.inv(sigma_k)
        except np.linalg.LinAlgError:
            continue

        centered = case_data - mu_k
        quad = np.sum(centered @ sigma_k_inv * centered, axis=1)
        group_loglik = -0.5 * (n_k * p_k * np.log(2 * np.pi) + n_k * logdet + np.sum(quad))
        total_loglik += group_loglik

    return total_loglik


def estimate_fiml(
    spec: ModelSpecification,
    data: pd.DataFrame,
) -> EstimationResult:
    """Estimate model via FIML for data with missing values.

    Parameters
    ----------
    spec : ModelSpecification
        Model specification (must have meanstructure=True).
    data : pd.DataFrame
        Data with possible NaN values.

    Returns
    -------
    EstimationResult
    """
    obs_data = data[spec.observed_vars].values
    n_obs = obs_data.shape[0]

    # Check for completely missing variables
    all_missing = np.all(np.isnan(obs_data), axis=0)
    if any(all_missing):
        bad_vars = [spec.observed_vars[i] for i in np.where(all_missing)[0]]
        raise ValueError(f"Variable(s) completely missing: {bad_vars}")

    # Group by missingness pattern
    pattern_groups = _group_by_pattern(obs_data)
    n_patterns = len(pattern_groups)

    # Set starting values for intercepts from available data
    for i, var in enumerate(spec.observed_vars):
        idx = spec._idx(var)
        if spec.m_free is not None and spec.m_free[idx]:
            col_data = obs_data[:, i]
            spec.m_values[idx] = np.nanmean(col_data)

    # Try listwise-deletion ML for starting values
    complete_mask = ~np.any(np.isnan(obs_data), axis=1)
    n_complete = complete_mask.sum()
    theta0 = spec.pack_start()

    if n_complete > spec.n_free + 10:
        from .estimation import ml_objective, ml_gradient
        complete_data = obs_data[complete_mask]
        sample_cov = np.cov(complete_data, rowvar=False, ddof=1)
        sample_mean = np.mean(complete_data, axis=0)
        result_lw = optimize.minimize(
            ml_objective, theta0,
            args=(spec, sample_cov, n_complete, sample_mean),
            method="BFGS",
            jac=ml_gradient,
            options={"maxiter": 2000, "gtol": 1e-5},
        )
        if result_lw.fun < 1e8:
            theta0 = result_lw.x

    # Optimize FIML
    result = optimize.minimize(
        fiml_objective,
        theta0,
        args=(spec, pattern_groups, n_obs),
        method="BFGS",
        jac=fiml_gradient,
        options={"maxiter": 10000, "gtol": 1e-6},
    )

    # Polish
    if result.success:
        result2 = optimize.minimize(
            fiml_objective,
            result.x,
            args=(spec, pattern_groups, n_obs),
            method="BFGS",
            jac=fiml_gradient,
            options={"maxiter": 10000, "gtol": 1e-9},
        )
        if result2.fun <= result.fun + 1e-10:
            result2.success = True
            result = result2

    if not result.success:
        warnings.warn(
            f"FIML optimization did not converge: {result.message}",
            RuntimeWarning,
            stacklevel=2,
        )

    # Compute sample cov from complete cases (for SRMR/fit indices)
    if n_complete > 1:
        complete_data = obs_data[complete_mask]
        sample_cov = np.cov(complete_data, rowvar=False, ddof=1)
        sample_mean = np.mean(complete_data, axis=0)
    else:
        # Pairwise covariance as fallback
        sample_cov = np.ma.cov(np.ma.array(obs_data, mask=np.isnan(obs_data)), rowvar=False).data
        sample_mean = np.nanmean(obs_data, axis=0)

    if sample_cov.ndim == 0:
        sample_cov = sample_cov.reshape(1, 1)

    # Compute FIML log-likelihood for AIC/BIC
    fiml_loglik = -0.5 * n_obs * result.fun  # F = -2/N * loglik => loglik = -N/2 * F

    # Compute FIML log-likelihood and saturated log-likelihood for chi-square
    fiml_loglik = -0.5 * n_obs * result.fun  # F = -2/N * loglik
    sat_loglik = _saturated_loglik(obs_data, pattern_groups)

    # Chi-square as likelihood ratio test
    fiml_chi_square = -2.0 * (fiml_loglik - sat_loglik)

    est = EstimationResult(
        converged=result.success,
        iterations=result.nit,
        fmin=result.fun,
        theta=result.x,
        hessian_inv=getattr(result, "hess_inv", None),
        sample_cov=sample_cov,
        n_obs=n_obs,
        spec=spec,
        sample_mean=sample_mean,
    )
    est._missing_method = "fiml"
    est._pattern_groups = pattern_groups
    est._fiml_loglik = fiml_loglik
    est._fiml_chi_square = fiml_chi_square
    return est
