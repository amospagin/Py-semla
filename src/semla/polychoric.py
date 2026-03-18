"""Polychoric correlation estimation for ordinal data.

Estimates the Pearson correlation between latent continuous variables
assumed to underlie observed ordinal variables, using maximum likelihood.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import optimize, stats


def _threshold_estimates(x: np.ndarray) -> np.ndarray:
    """Estimate thresholds for an ordinal variable from marginal proportions."""
    values, counts = np.unique(x[~np.isnan(x)], return_counts=True)
    proportions = counts / counts.sum()
    cum_props = np.cumsum(proportions)[:-1]  # exclude last (=1.0)
    # Clip to avoid inf thresholds
    cum_props = np.clip(cum_props, 1e-8, 1 - 1e-8)
    return stats.norm.ppf(cum_props)


def _bivariate_normal_prob(a1: float, a2: float, b1: float, b2: float,
                           rho: float) -> float:
    """P(a1 < X < b1, a2 < Y < b2) for bivariate normal with correlation rho."""
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, rho], [rho, 1.0]])
    # Use mvn.mvnun for rectangle probability
    lower = np.array([a1, a2])
    upper = np.array([b1, b2])
    try:
        from scipy.stats import mvnun
        prob, _ = mvnun(lower, upper, mean, cov)
    except (ImportError, AttributeError):
        # Fallback: use multivariate_normal CDF differences
        from scipy.stats import multivariate_normal
        rv = multivariate_normal(mean, cov)
        # P(X<b1, Y<b2) - P(X<a1, Y<b2) - P(X<b1, Y<a2) + P(X<a1, Y<a2)
        def cdf2(x, y):
            return rv.cdf([x, y])
        prob = cdf2(b1, b2) - cdf2(a1, b2) - cdf2(b1, a2) + cdf2(a1, a2)
    return max(prob, 1e-15)


def polychoric_corr_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Estimate polychoric correlation for a pair of ordinal variables.

    Returns
    -------
    tuple[float, float]
        (polychoric_correlation, fisher_information)
    """
    # Remove missing pairs
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    n = len(x)

    if n < 5:
        return 0.0, 0.0

    # Get thresholds
    tau_x = _threshold_estimates(x)
    tau_y = _threshold_estimates(y)

    # Categories
    cats_x = np.unique(x)
    cats_y = np.unique(y)
    n_cat_x = len(cats_x)
    n_cat_y = len(cats_y)

    # If either variable is continuous or has too many categories, use Pearson
    if n_cat_x > 10 or n_cat_y > 10:
        r = np.corrcoef(x, y)[0, 1]
        info = n / (1 - r**2)**2 if abs(r) < 0.999 else n
        return r, info

    # Contingency table
    x_idx = np.searchsorted(cats_x, x)
    y_idx = np.searchsorted(cats_y, y)
    table = np.zeros((n_cat_x, n_cat_y))
    for xi, yi in zip(x_idx, y_idx):
        table[xi, yi] += 1

    # Extended thresholds (with -inf and +inf)
    tau_x_ext = np.concatenate([[-10.0], tau_x, [10.0]])
    tau_y_ext = np.concatenate([[-10.0], tau_y, [10.0]])

    def neg_loglik(rho):
        ll = 0.0
        for i in range(n_cat_x):
            for j in range(n_cat_y):
                if table[i, j] > 0:
                    prob = _bivariate_normal_prob(
                        tau_x_ext[i], tau_y_ext[j],
                        tau_x_ext[i + 1], tau_y_ext[j + 1],
                        rho
                    )
                    ll += table[i, j] * np.log(prob)
        return -ll

    # Optimize
    result = optimize.minimize_scalar(
        neg_loglik, bounds=(-0.999, 0.999), method="bounded"
    )
    rho_hat = result.x

    # Fisher information (numerical second derivative)
    eps = 1e-4
    f0 = neg_loglik(rho_hat)
    fp = neg_loglik(min(rho_hat + eps, 0.999))
    fm = neg_loglik(max(rho_hat - eps, -0.999))
    info = (fp - 2 * f0 + fm) / (eps ** 2)
    info = max(info, 1e-10)

    return rho_hat, info


def polychoric_correlation_matrix(
    data: np.ndarray, columns: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Compute polychoric correlation matrix for ordinal data.

    Parameters
    ----------
    data : np.ndarray
        Data matrix (n_obs x p).
    columns : list[str], optional
        Column names (for warnings).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[np.ndarray]]
        (correlation_matrix, avar_diagonal, thresholds_per_variable)

        avar_diagonal: diagonal of asymptotic covariance matrix of
        the vectorized (vech) correlation elements.
    """
    n, p = data.shape
    R = np.eye(p)
    info_matrix = np.zeros((p, p))
    thresholds = [_threshold_estimates(data[:, j]) for j in range(p)]

    for i in range(p):
        for j in range(i):
            rho, info = polychoric_corr_pair(data[:, i], data[:, j])
            R[i, j] = rho
            R[j, i] = rho
            info_matrix[i, j] = info
            info_matrix[j, i] = info

    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(R)
    if eigvals[0] < 0:
        warnings.warn(
            "Polychoric correlation matrix is not positive definite. "
            "Applying nearest PD correction.",
            RuntimeWarning,
            stacklevel=2,
        )
        R = _nearest_pd(R)

    # Asymptotic variance diagonal for off-diagonal elements of R
    # For off-diagonal: Var(r_ij) ≈ 1/info_ij
    avar_diag = []
    for i in range(p):
        for j in range(i):  # strictly lower triangle (off-diagonal only)
            info = info_matrix[i, j]
            avar_diag.append(1.0 / info if info > 0 else 1.0)

    return R, np.array(avar_diag), thresholds


def _nearest_pd(A: np.ndarray) -> np.ndarray:
    """Find nearest positive definite matrix (Higham's method simplified)."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-6)
    B = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalize diagonal to 1.0 (correlation matrix)
    d = np.sqrt(np.diag(B))
    B = B / np.outer(d, d)
    return B
