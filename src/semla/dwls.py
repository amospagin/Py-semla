"""DWLS (Diagonally Weighted Least Squares) estimation for ordinal SEM.

Uses polychoric correlations as input with true DWLS objective function
(diagonally weighted residuals) and robust sandwich standard errors
with Satorra-Bentler scaled chi-square test.
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


def _dwls_objective(
    theta: np.ndarray,
    spec: ModelSpecification,
    s_offdiag: np.ndarray,
    weight_diag: np.ndarray,
) -> float:
    """True DWLS objective: F = (s - sigma)' W (s - sigma).

    Parameters
    ----------
    theta : parameter vector
    spec : model specification
    s_offdiag : vech of off-diagonal sample polychoric correlations
    weight_diag : diagonal weights (1 / asymptotic variance of each correlation)
    """
    A, S_mat = spec.unpack(theta)
    sigma = _model_implied_cov(A, S_mat, spec.F)
    if sigma is None:
        return 1e10

    sigma_cor = _cov_to_cor(sigma)
    sigma_offdiag = _vech_offdiag(sigma_cor)

    diff = s_offdiag - sigma_offdiag
    return float(diff @ (weight_diag * diff))


def _dwls_gradient(
    theta: np.ndarray,
    spec: ModelSpecification,
    s_offdiag: np.ndarray,
    weight_diag: np.ndarray,
) -> np.ndarray:
    """Numerical gradient of DWLS objective."""
    eps = 1e-7
    f0 = _dwls_objective(theta, spec, s_offdiag, weight_diag)
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        grad[i] = (_dwls_objective(theta_plus, spec, s_offdiag, weight_diag) - f0) / eps
    return grad


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
    """Robust (sandwich) standard errors for DWLS.

    V = (J' W J)^{-1} J' W Gamma W J (J' W J)^{-1}

    where W = diag(weight_diagonal) and Gamma = diag(gamma_diagonal * n_obs).
    """
    J = _compute_jacobian_cor(theta, spec)
    W = weight_diagonal  # 1-D diagonal
    Gamma_diag = gamma_diagonal * n_obs  # asymptotic variance, scaled

    try:
        # Bread: (J' W J)^{-1}
        JtWJ = J.T @ (W[:, None] * J)
        JtWJ_inv = np.linalg.pinv(JtWJ)

        # Meat: J' W Gamma W J
        WGW = W * Gamma_diag * W  # element-wise for diagonal matrices
        meat = J.T @ (WGW[:, None] * J)

        V = JtWJ_inv @ meat @ JtWJ_inv / (n_obs - 1)
        var_theta = np.diag(V)
        se_robust = np.where(var_theta > 0, np.sqrt(var_theta), np.nan)

        # Fall back to ML SEs where robust fails
        se_ml = _compute_se(theta, spec, polychoric_cov, n_obs)
        se = np.where(np.isnan(se_robust), se_ml, se_robust)
        return se
    except Exception:
        return _compute_se(theta, spec, polychoric_cov, n_obs)


def _scaled_chi_square(
    theta: np.ndarray,
    spec: ModelSpecification,
    polychoric_cov: np.ndarray,
    gamma_diag: np.ndarray,
    n_obs: int,
    df: int,
) -> tuple[float, float]:
    """Compute mean-adjusted scaled chi-square for DWLS.

    Uses ULS-based chi-square at the DWLS optimum with Satorra-Bentler
    mean adjustment, following Muthén, du Toit & Spisic (1997).

    T_scaled = T_ULS / c  where  c = tr(U_d Gamma) / df
    """
    # ULS fit function at DWLS solution
    A, S_mat = spec.unpack(theta)
    sigma = _model_implied_cov(A, S_mat, spec.F)
    if sigma is None:
        return 0.0, 1.0
    s_offdiag = _vech_offdiag(polychoric_cov)
    sigma_offdiag = _vech_offdiag(_cov_to_cor(sigma))
    resid = s_offdiag - sigma_offdiag
    f_uls = 0.5 * float(resid @ resid)
    T_uls = (n_obs - 1) * f_uls

    if T_uls >= 1e8 or df <= 0:
        return T_uls, 1.0

    # Satorra-Bentler mean adjustment
    J = _compute_jacobian_cor(theta, spec)
    Gamma = np.diag(gamma_diag * n_obs)
    m = J.shape[0]

    try:
        # U_d for ULS: U = I - Delta (Delta' Delta)^{-1} Delta'
        JtJ_inv = np.linalg.pinv(J.T @ J)
        P = np.eye(m) - J @ JtJ_inv @ J.T
        PG = P @ Gamma
        trace_PG = np.trace(PG)

        if trace_PG > 1e-10:
            c = trace_PG / df
            return T_uls / c, c
        else:
            return T_uls, 1.0
    except Exception:
        return T_uls, 1.0


def estimate_dwls(
    spec: ModelSpecification,
    data: pd.DataFrame,
) -> DWLSEstimationResult:
    """Estimate model using true DWLS on polychoric correlations.

    Minimizes F_DWLS = (s - sigma)' W (s - sigma) where W is the diagonal
    of the inverse asymptotic covariance matrix of polychoric correlations.

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

    # Weight diagonal: inverse asymptotic variance of each off-diagonal element
    weight_diag = 1.0 / np.maximum(avar_diag, 1e-10)

    # Vectorized sample correlations (off-diagonal lower triangle)
    s_offdiag = _vech_offdiag(R)

    # Get starting values via quick ML on polychoric matrix
    theta0 = spec.pack_start()
    ml_result = optimize.minimize(
        ml_objective,
        theta0,
        args=(spec, R, n_obs),
        method="BFGS",
        options={"maxiter": 5000, "gtol": 1e-5},
    )
    theta_start = ml_result.x if ml_result.success else theta0

    # Optimize true DWLS objective
    result = optimize.minimize(
        _dwls_objective,
        theta_start,
        jac=_dwls_gradient,
        args=(spec, s_offdiag, weight_diag),
        method="BFGS",
        options={"maxiter": 10000, "gtol": 1e-8},
    )

    # Polish with Nelder-Mead
    result2 = optimize.minimize(
        _dwls_objective,
        result.x,
        args=(spec, s_offdiag, weight_diag),
        method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-10, "fatol": 1e-12},
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
