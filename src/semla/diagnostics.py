"""Diagnostic tests for SEM data."""

from __future__ import annotations

import numpy as np
from scipy import stats


def mardia_test(data: np.ndarray) -> dict:
    """Mardia's test for multivariate normality.

    Tests multivariate skewness and kurtosis, which inform the choice
    between ML and MLR estimators.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Data matrix (n x p). NaN rows are dropped.

    Returns
    -------
    dict
        skewness: Mardia's multivariate skewness statistic
        skewness_p: p-value (chi-squared)
        kurtosis: Mardia's multivariate kurtosis statistic
        kurtosis_z: z-score for kurtosis
        kurtosis_p: p-value (normal approximation)
        recommendation: "ML" if data is approximately normal, "MLR" otherwise
    """
    import pandas as pd
    if isinstance(data, pd.DataFrame):
        data = data.select_dtypes(include=[np.number]).values

    # Drop rows with NaN
    data = data[~np.any(np.isnan(data), axis=1)]
    n, p = data.shape

    # Center and compute inverse covariance
    mean = np.mean(data, axis=0)
    centered = data - mean
    S = np.cov(centered, rowvar=False, ddof=0)

    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return {
            "skewness": np.nan, "skewness_p": np.nan,
            "kurtosis": np.nan, "kurtosis_z": np.nan, "kurtosis_p": np.nan,
            "recommendation": "MLR",
        }

    # Mahalanobis distances: D_ij = (x_i - mean)' S^{-1} (x_j - mean)
    D = centered @ S_inv @ centered.T

    # Mardia's skewness: b1p = (1/n^2) * sum_ij D_ij^3
    b1p = np.sum(D ** 3) / (n ** 2)
    # Test statistic: n*b1p/6 ~ chi2(p*(p+1)*(p+2)/6)
    skew_stat = n * b1p / 6
    skew_df = p * (p + 1) * (p + 2) / 6
    skew_p = 1.0 - stats.chi2.cdf(skew_stat, skew_df)

    # Mardia's kurtosis: b2p = (1/n) * sum_i D_ii^2
    b2p = np.sum(np.diag(D) ** 2) / n
    # Expected value under normality: p*(p+2)
    expected = p * (p + 2)
    # Variance under normality: 8*p*(p+2)/n
    var_kurt = 8 * p * (p + 2) / n
    kurt_z = (b2p - expected) / np.sqrt(var_kurt)
    kurt_p = 2 * (1 - stats.norm.cdf(abs(kurt_z)))

    # Recommendation
    non_normal = skew_p < 0.05 or kurt_p < 0.05
    recommendation = "MLR" if non_normal else "ML"

    return {
        "skewness": b1p,
        "skewness_stat": skew_stat,
        "skewness_p": skew_p,
        "kurtosis": b2p,
        "kurtosis_z": kurt_z,
        "kurtosis_p": kurt_p,
        "recommendation": recommendation,
    }
