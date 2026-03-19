"""Robust ML (MLR) standard errors and scaled chi-square.

Implements the Huber-White sandwich estimator and Satorra-Bentler
scaled test statistic for non-normal continuous data.
"""

from __future__ import annotations

import numpy as np

from .estimation import _model_implied_cov, _compute_se
from .specification import ModelSpecification


def _vech_indices(p: int) -> list[tuple[int, int]]:
    """Return (i, j) pairs for vech ordering (lower triangle incl. diagonal)."""
    return [(i, j) for i in range(p) for j in range(i + 1)]


def compute_gamma(data: np.ndarray, sample_cov: np.ndarray) -> np.ndarray:
    """Compute the Gamma matrix (asymptotic covariance of vech(S)).

    Gamma_ab = (1/N) sum_k (x_ki*x_kj - s_ij)(x_km*x_kn - s_mn)

    where a=(i,j) and b=(m,n) are vech indices.

    Parameters
    ----------
    data : np.ndarray
        Centered data matrix (n x p).
    sample_cov : np.ndarray
        Sample covariance matrix (p x p).

    Returns
    -------
    np.ndarray
        Gamma matrix, shape (p*(p+1)/2, p*(p+1)/2).
    """
    n, p = data.shape
    indices = _vech_indices(p)
    m = len(indices)

    # Compute outer products minus expected for each observation
    # w_k[a] = x_ki * x_kj - s_ij for vech index a=(i,j)
    W = np.zeros((n, m))
    for a, (i, j) in enumerate(indices):
        W[:, a] = data[:, i] * data[:, j] - sample_cov[i, j]

    # Gamma = W'W / n — ML-consistent estimate
    Gamma = (W.T @ W) / n
    return Gamma


def compute_robust_se(
    theta: np.ndarray,
    spec: ModelSpecification,
    sample_cov: np.ndarray,
    n_obs: int,
    gamma: np.ndarray,
    raw_data: np.ndarray = None,
) -> np.ndarray:
    """Compute robust (sandwich) standard errors for ML.

    Uses the sandwich formula matching lavaan's MLR:
        V = I_obs^{-1} @ B @ I_obs^{-1} / N
    where I_obs = observed information (per obs) and
    B = outer product of casewise scores (per obs).
    """
    from .estimation import ml_objective

    eps = 1e-7
    k = len(theta)
    N = n_obs

    A_mat, S_mat = spec.unpack(theta)
    sigma = _model_implied_cov(A_mat, S_mat, spec.F)
    if sigma is None:
        return np.full(k, np.nan)

    try:
        sigma_inv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)

    p = sigma.shape[0]

    # Use ML sample covariance (/ N) for consistency with casewise scores
    sample_cov_ml = sample_cov * (N - 1) / N

    # Bread: observed information (numerical Hessian of -loglik per obs)
    # -loglik ∝ N/2 * F_ML, so Hessian of -loglik/N = 1/2 * Hessian of F_ML
    eps_h = 1e-5
    I_obs = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            tp = theta.copy(); tm = theta.copy()
            tpp = theta.copy(); tmm = theta.copy()
            tp[i] += eps_h; tp[j] += eps_h
            tm[i] += eps_h; tm[j] -= eps_h
            tpp[i] -= eps_h; tpp[j] += eps_h
            tmm[i] -= eps_h; tmm[j] -= eps_h
            H_ij = (
                ml_objective(tp, spec, sample_cov, N)
                - ml_objective(tm, spec, sample_cov, N)
                - ml_objective(tpp, spec, sample_cov, N)
                + ml_objective(tmm, spec, sample_cov, N)
            ) / (4 * eps_h ** 2)
            # Per-observation info: (N/2) * H / N = H/2
            I_obs[i, j] = 0.5 * H_ij
            I_obs[j, i] = I_obs[i, j]

    try:
        I_obs_inv = np.linalg.inv(I_obs)
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)

    if raw_data is None:
        # Fallback: return ML SEs
        var_theta = np.diag(I_obs_inv) / N
        return np.where(var_theta > 0, np.sqrt(var_theta), np.nan)

    # Meat: casewise scores (per observation)
    # score_ki = -1/2 * tr(Σ⁻¹ dΣ_i) + 1/2 * e_k' Σ⁻¹ dΣ_i Σ⁻¹ e_k
    dSigma = np.zeros((k, p, p))
    for i in range(k):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += eps
        theta_minus[i] -= eps
        A_p, S_p = spec.unpack(theta_plus)
        A_m, S_m = spec.unpack(theta_minus)
        sig_p = _model_implied_cov(A_p, S_p, spec.F)
        sig_m = _model_implied_cov(A_m, S_m, spec.F)
        if sig_p is None or sig_m is None:
            return np.full(k, np.nan)
        dSigma[i] = (sig_p - sig_m) / (2 * eps)

    SinvdSiSinv = np.zeros((k, p, p))
    trace_SinvdSi = np.zeros(k)
    for i in range(k):
        SinvdS_i = sigma_inv @ dSigma[i]
        SinvdSiSinv[i] = SinvdS_i @ sigma_inv
        trace_SinvdSi[i] = np.trace(SinvdS_i)

    scores = np.zeros((N, k))
    for i in range(k):
        Me = raw_data @ SinvdSiSinv[i].T  # (N, p)
        quad = np.sum(raw_data * Me, axis=1)  # (N,)
        scores[:, i] = -0.5 * trace_SinvdSi[i] + 0.5 * quad

    # Mean-center scores to ensure they sum to zero
    scores -= scores.mean(axis=0)

    # B = (1/N) * scores' @ scores (per-observation outer product)
    B = (scores.T @ scores) / N

    # Sandwich: V = I_obs^{-1} @ B @ I_obs^{-1} / N
    V = I_obs_inv @ B @ I_obs_inv / N

    var_theta = np.diag(V)
    se = np.where(var_theta > 0, np.sqrt(var_theta), np.nan)
    return se


def satorra_bentler_chi_square(
    fmin: float,
    n_obs: int,
    df: int,
    theta: np.ndarray,
    spec: ModelSpecification,
    sample_cov: np.ndarray,
    gamma: np.ndarray,
) -> tuple[float, float]:
    """Compute Satorra-Bentler scaled chi-square.

    T_scaled = T_ML / c where c = tr(UG) / df

    Returns (T_scaled, scaling_factor).
    """
    T_ml = (n_obs - 1) * fmin
    if df <= 0:
        return T_ml, 1.0

    eps = 1e-7
    k = len(theta)
    A, S_mat = spec.unpack(theta)
    sigma = _model_implied_cov(A, S_mat, spec.F)
    if sigma is None:
        return T_ml, 1.0

    try:
        sigma_inv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        return T_ml, 1.0

    p = sigma.shape[0]
    indices = _vech_indices(p)
    m = len(indices)

    # Jacobian Delta (m x k)
    Delta = np.zeros((m, k))
    for i in range(k):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += eps
        theta_minus[i] -= eps
        A_p, S_p = spec.unpack(theta_plus)
        A_m, S_m = spec.unpack(theta_minus)
        sig_p = _model_implied_cov(A_p, S_p, spec.F)
        sig_m = _model_implied_cov(A_m, S_m, spec.F)
        if sig_p is None or sig_m is None:
            return T_ml, 1.0
        dSig = (sig_p - sig_m) / (2 * eps)
        for a, (r, c) in enumerate(indices):
            Delta[a, i] = dSig[r, c]

    # W_ml (vech version of Sigma_inv kron Sigma_inv)
    W_ml = np.zeros((m, m))
    for a, (i, j) in enumerate(indices):
        for b, (r, c) in enumerate(indices):
            val = sigma_inv[i, r] * sigma_inv[j, c]
            if r != c:
                val += sigma_inv[i, c] * sigma_inv[j, r]
            W_ml[a, b] = val

    # UG = W_ml - W_ml Delta (Delta' W_ml Delta)^{-1} Delta' W_ml
    try:
        DtW = Delta.T @ W_ml
        DtWD_inv = np.linalg.inv(DtW @ Delta)
        UG = W_ml - W_ml @ Delta @ DtWD_inv @ DtW
    except np.linalg.LinAlgError:
        return T_ml, 1.0

    # Scaling factor
    trace_UG_Gamma = np.trace(UG @ gamma)
    if trace_UG_Gamma > 0:
        c = trace_UG_Gamma / df
        return T_ml / c, c
    else:
        return T_ml, 1.0
