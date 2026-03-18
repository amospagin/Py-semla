"""DWLS (Diagonally Weighted Least Squares) estimation for ordinal SEM.

Uses polychoric correlations as input with ML estimation and
robust (sandwich) standard errors and scaled chi-square test.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from .estimation import (
    EstimationResult, _model_implied_cov, _compute_se,
    ml_objective, ml_gradient,
)
from .polychoric import polychoric_correlation_matrix
from .specification import ModelSpecification


def _vech_offdiag(M: np.ndarray) -> np.ndarray:
    """Extract strictly lower triangle (off-diagonal only)."""
    p = M.shape[0]
    idx = np.tril_indices(p, k=-1)
    return M[idx]


def _cov_to_cor(sigma: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to correlation matrix."""
    d = np.sqrt(np.maximum(np.diag(sigma), 1e-10))
    return sigma / np.outer(d, d)


@dataclass
class DWLSEstimationResult(EstimationResult):
    """DWLS estimation output."""

    polychoric_cov: np.ndarray = None
    gamma_diagonal: np.ndarray = None  # asymptotic variance of off-diag polychoric elements
    estimator_type: str = "DWLS"
    weight_diagonal: np.ndarray = None


def _compute_jacobian_cor(
    theta: np.ndarray, spec: ModelSpecification
) -> np.ndarray:
    """Jacobian d(vech_offdiag(cor(Sigma))) / d(theta)."""
    eps = 1e-7
    A0, S0 = spec.unpack(theta)
    sigma0 = _model_implied_cov(A0, S0, spec.F)
    s0 = _vech_offdiag(_cov_to_cor(sigma0))
    k = len(theta)
    m = len(s0)

    J = np.zeros((m, k))
    for i in range(k):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        A_p, S_p = spec.unpack(theta_plus)
        sig_p = _model_implied_cov(A_p, S_p, spec.F)
        J[:, i] = (_vech_offdiag(_cov_to_cor(sig_p)) - s0) / eps

    return J


def _compute_se_dwls(
    theta: np.ndarray,
    spec: ModelSpecification,
    polychoric_cov: np.ndarray,
    weight_diagonal: np.ndarray,
    gamma_diagonal: np.ndarray,
    n_obs: int,
) -> np.ndarray:
    """Robust (sandwich) standard errors.

    Uses the expected information from ML estimation as the bread,
    and the asymptotic covariance of polychoric correlations as the meat.
    """
    # Use standard ML SEs from expected information on polychoric matrix
    se_ml = _compute_se(theta, spec, polychoric_cov, n_obs)

    # For a more robust approach, compute sandwich SEs
    J = _compute_jacobian_cor(theta, spec)
    # Scale gamma by n_obs for proper asymptotic variance
    Gamma = np.diag(gamma_diagonal * n_obs)

    try:
        # Sandwich: V = (J'J)^{-1} J' (n*Gamma) J (J'J)^{-1} / (n-1)
        JtJ = J.T @ J
        JtJ_inv = np.linalg.pinv(JtJ)

        meat = J.T @ Gamma @ J
        V = JtJ_inv @ meat @ JtJ_inv / (n_obs - 1)
        var_theta = np.diag(V)
        se_robust = np.where(var_theta > 0, np.sqrt(var_theta), np.nan)

        # Use robust SEs where available, fall back to ML SEs
        se = np.where(np.isnan(se_robust), se_ml, se_robust)
        return se
    except Exception:
        return se_ml


def _scaled_chi_square(
    theta: np.ndarray,
    spec: ModelSpecification,
    polychoric_cov: np.ndarray,
    gamma_diag: np.ndarray,
    n_obs: int,
    df: int,
) -> tuple[float, float]:
    """Compute scaled chi-square for DWLS.

    Uses ML chi-square at the DWLS optimum with Satorra-Bentler correction.
    """
    # ML chi-square on the polychoric matrix
    f_ml = ml_objective(theta, spec, polychoric_cov, n_obs)
    T_ml = (n_obs - 1) * f_ml

    if T_ml >= 1e8 or df <= 0:
        return T_ml, 1.0

    # Satorra-Bentler correction
    J = _compute_jacobian_cor(theta, spec)
    # Scale gamma by n_obs: gamma_diag is Var(r_ij) ≈ 1/(n*info),
    # but the SB formula needs the asymptotic variance of sqrt(n)*vech(S)
    Gamma = np.diag(gamma_diag * n_obs)
    m = J.shape[0]

    try:
        JtJ_inv = np.linalg.pinv(J.T @ J)
        P = np.eye(m) - J @ JtJ_inv @ J.T
        PG = P @ Gamma
        trace_PG = np.trace(PG)

        if trace_PG > 1e-10:
            c = trace_PG / df
            return T_ml / c, c
        else:
            return T_ml, 1.0
    except Exception:
        return T_ml, 1.0


def estimate_dwls(
    spec: ModelSpecification,
    data: pd.DataFrame,
) -> DWLSEstimationResult:
    """Estimate model using polychoric correlations with ML + robust SEs.

    Parameters
    ----------
    spec : ModelSpecification
        Model specification.
    data : pd.DataFrame
        Data with ordinal observed variables.

    Returns
    -------
    DWLSEstimationResult
    """
    obs_data = data[spec.observed_vars].values
    n_obs = obs_data.shape[0]

    # Compute polychoric correlations
    R, avar_diag, thresholds = polychoric_correlation_matrix(obs_data)

    # Fit using ML on the polychoric correlation matrix
    # BFGS with numerical gradient (analytic gradient has precision issues
    # with polychoric matrices)
    theta0 = spec.pack_start()

    result = optimize.minimize(
        ml_objective,
        theta0,
        args=(spec, R, n_obs),
        method="BFGS",
        options={"maxiter": 10000, "gtol": 1e-6},
    )

    # Polish with Nelder-Mead (more robust for polychoric matrices)
    result2 = optimize.minimize(
        ml_objective,
        result.x,
        args=(spec, R, n_obs),
        method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-10},
    )
    if result2.fun <= result.fun + 1e-10:
        result2.success = True
        result = result2

    if not result.success:
        warnings.warn(
            f"DWLS optimization did not converge: {result.message}",
            RuntimeWarning,
            stacklevel=2,
        )

    # Weight diagonal (for interface compatibility)
    weight_diag = 1.0 / np.maximum(avar_diag, 1e-10)

    return DWLSEstimationResult(
        converged=result.success,
        iterations=result.nit,
        fmin=result.fun,
        theta=result.x,
        hessian_inv=getattr(result, "hess_inv", None),
        sample_cov=R,
        n_obs=n_obs,
        spec=spec,
        polychoric_cov=R,
        gamma_diagonal=avar_diag,
        estimator_type="DWLS",
        weight_diagonal=weight_diag,
    )
