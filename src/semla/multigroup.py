"""Multi-group SEM/CFA estimation.

Supports configural and metric invariance by fitting a single model
structure across multiple groups with a combined ML objective function.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .estimation import ml_objective, ml_gradient, _model_implied_cov, _compute_se
from .specification import ModelSpecification, build_specification
from .syntax import parse_syntax, FormulaToken


@dataclass
class MultiGroupSpec:
    """Specification for multi-group SEM."""

    group_names: list[str]
    group_specs: list[ModelSpecification]
    group_sample_covs: list[np.ndarray]
    group_sample_means: list[np.ndarray]
    group_n_obs: list[int]
    n_total: int
    invariance: str  # "configural" or "metric"

    # theta_mapping[g][i] = index into combined theta for group g's i-th free param
    theta_mapping: list[np.ndarray] = field(default_factory=list)
    n_free_combined: int = 0


def build_multigroup_spec(
    tokens: list[FormulaToken],
    data: pd.DataFrame,
    group_col: str,
    invariance: str = "configural",
    auto_cov_latent: bool = True,
    meanstructure: bool = False,
) -> MultiGroupSpec:
    """Build multi-group specification from parsed syntax and grouped data.

    Parameters
    ----------
    tokens : list[FormulaToken]
        Parsed model syntax.
    data : pd.DataFrame
        Full dataset with a grouping column.
    group_col : str
        Column name defining groups.
    invariance : str
        "configural" (same structure, all free) or "metric" (equal loadings).
    auto_cov_latent : bool
        Auto-add covariances between latent variables.
    """
    if group_col not in data.columns:
        raise ValueError(
            f"Group column '{group_col}' not found in data. "
            f"Available columns: {sorted(data.columns.tolist())}"
        )

    groups = data[group_col].unique()
    if len(groups) < 2:
        raise ValueError(
            f"Need at least 2 groups, found {len(groups)} in column '{group_col}'."
        )

    # Scalar and strict invariance require mean structure
    if invariance in ("scalar", "strict"):
        auto_cov_latent = True  # force for these levels

    group_names = sorted(str(g) for g in groups)
    group_specs = []
    group_sample_covs = []
    group_sample_means = []
    group_n_obs = []

    for gname in group_names:
        gdata = data[data[group_col].astype(str) == gname]
        n_g = len(gdata)

        if n_g < 10:
            warnings.warn(
                f"Group '{gname}' has only {n_g} observations.",
                RuntimeWarning,
                stacklevel=3,
            )

        spec = build_specification(
            tokens, gdata.columns.tolist(),
            auto_cov_latent=auto_cov_latent,
            meanstructure=meanstructure,
        )
        obs_data = gdata[spec.observed_vars].values
        # Set intercept starting values from group means
        if spec.meanstructure and spec.m_values is not None:
            for i_v, var in enumerate(spec.observed_vars):
                idx = spec._idx(var)
                if spec.m_free[idx]:
                    spec.m_values[idx] = np.mean(obs_data[:, i_v])
        sample_cov = np.cov(obs_data, rowvar=False, ddof=1)
        if sample_cov.ndim == 0:
            sample_cov = sample_cov.reshape(1, 1)

        group_specs.append(spec)
        group_sample_covs.append(sample_cov)
        group_sample_means.append(np.mean(obs_data, axis=0))
        group_n_obs.append(n_g)

    n_total = sum(group_n_obs)
    n_groups = len(group_names)
    k_per_group = group_specs[0].n_free  # all groups have same structure

    # Build theta_mapping
    if invariance == "configural":
        # Each group gets fully independent parameters
        theta_mapping = []
        for g in range(n_groups):
            theta_mapping.append(np.arange(g * k_per_group, (g + 1) * k_per_group))
        n_free_combined = n_groups * k_per_group

    elif invariance == "metric":
        # Loadings (A matrix) are shared; variances/covariances (S matrix) are per-group
        n_a = int(np.sum(group_specs[0].A_free))
        n_s = k_per_group - n_a  # S parameters per group

        theta_mapping = []
        # Combined theta layout: [shared_A_params | group0_S_params | group1_S_params | ...]
        for g in range(n_groups):
            mapping = np.zeros(k_per_group, dtype=int)
            # A params: all groups share indices 0..n_a-1
            mapping[:n_a] = np.arange(n_a)
            # S params: each group gets unique indices
            s_offset = n_a + g * n_s
            mapping[n_a:] = np.arange(s_offset, s_offset + n_s)
            theta_mapping.append(mapping)

        n_free_combined = n_a + n_groups * n_s

    elif invariance == "scalar":
        # Loadings (A) + intercepts (M) shared; variances/covariances (S) per-group
        # Requires meanstructure
        if not group_specs[0].meanstructure:
            raise ValueError(
                "Scalar invariance requires mean structure. "
                "It will be auto-enabled."
            )
        n_a = int(np.sum(group_specs[0].A_free))
        n_s = int(np.sum(group_specs[0]._S_free_lower))
        n_m = int(np.sum(group_specs[0].m_free)) if group_specs[0].m_free is not None else 0

        theta_mapping = []
        # Layout: [shared_A | shared_M | group0_S | group1_S | ...]
        for g in range(n_groups):
            mapping = np.zeros(k_per_group, dtype=int)
            mapping[:n_a] = np.arange(n_a)  # shared loadings
            mapping[n_a:n_a + n_s] = np.arange(n_a + n_m + g * n_s, n_a + n_m + g * n_s + n_s)
            if n_m > 0:
                mapping[n_a + n_s:] = np.arange(n_a, n_a + n_m)  # shared intercepts
            theta_mapping.append(mapping)

        n_free_combined = n_a + n_m + n_groups * n_s

    elif invariance == "strict":
        # Loadings (A) + intercepts (M) + residual variances shared;
        # only latent variances/covariances per-group
        if not group_specs[0].meanstructure:
            raise ValueError(
                "Strict invariance requires mean structure. "
                "It will be auto-enabled."
            )
        n_a = int(np.sum(group_specs[0].A_free))
        n_s = int(np.sum(group_specs[0]._S_free_lower))
        n_m = int(np.sum(group_specs[0].m_free)) if group_specs[0].m_free is not None else 0

        # Split S params into observed residual variances (shared) and
        # latent variances/covariances (per-group)
        s_lower = group_specs[0]._S_free_lower
        n_vars = group_specs[0].n_vars
        n_obs_vars = group_specs[0].n_obs

        # Count S params: observed residuals vs latent var/cov
        n_s_obs = 0  # residual variances of observed vars (shared in strict)
        n_s_lat = 0  # latent var/cov (per-group)
        s_is_obs = []  # True if this S param is an observed residual
        for r in range(n_vars):
            for c in range(r + 1):
                if s_lower[r, c]:
                    is_obs_resid = (r < n_obs_vars and c < n_obs_vars and r == c)
                    s_is_obs.append(is_obs_resid)
                    if is_obs_resid:
                        n_s_obs += 1
                    else:
                        n_s_lat += 1

        theta_mapping = []
        # Layout: [shared_A | shared_M | shared_S_obs | group0_S_lat | group1_S_lat | ...]
        shared_count = n_a + n_m + n_s_obs
        for g in range(n_groups):
            mapping = np.zeros(k_per_group, dtype=int)
            # A params: shared
            mapping[:n_a] = np.arange(n_a)
            # S params: split observed (shared) vs latent (per-group)
            s_shared_idx = n_a + n_m  # start of shared S_obs in combined theta
            s_lat_offset = shared_count + g * n_s_lat
            s_shared_count = 0
            s_lat_count = 0
            for s_idx, is_obs in enumerate(s_is_obs):
                if is_obs:
                    mapping[n_a + s_idx] = s_shared_idx + s_shared_count
                    s_shared_count += 1
                else:
                    mapping[n_a + s_idx] = s_lat_offset + s_lat_count
                    s_lat_count += 1
            # M params: shared
            if n_m > 0:
                mapping[n_a + n_s:] = np.arange(n_a, n_a + n_m)
            theta_mapping.append(mapping)

        n_free_combined = shared_count + n_groups * n_s_lat

    else:
        raise ValueError(
            f"invariance must be 'configural', 'metric', 'scalar', or 'strict', "
            f"got '{invariance}'"
        )

    return MultiGroupSpec(
        group_names=group_names,
        group_specs=group_specs,
        group_sample_covs=group_sample_covs,
        group_sample_means=group_sample_means,
        group_n_obs=group_n_obs,
        n_total=n_total,
        invariance=invariance,
        theta_mapping=theta_mapping,
        n_free_combined=n_free_combined,
    )


def _pack_multigroup_start(mg_spec: MultiGroupSpec) -> np.ndarray:
    """Pack starting values into the combined theta vector."""
    theta = np.zeros(mg_spec.n_free_combined)
    counts = np.zeros(mg_spec.n_free_combined)

    for g, spec in enumerate(mg_spec.group_specs):
        theta_g = spec.pack_start()
        mapping = mg_spec.theta_mapping[g]
        # For shared params, average the starting values
        np.add.at(theta, mapping, theta_g)
        np.add.at(counts, mapping, 1.0)

    theta /= np.maximum(counts, 1.0)
    return theta


def multigroup_ml_objective(
    theta_combined: np.ndarray, mg_spec: MultiGroupSpec
) -> float:
    """Combined ML objective: weighted sum of per-group F_ML."""
    f_total = 0.0
    for g in range(len(mg_spec.group_names)):
        theta_g = theta_combined[mg_spec.theta_mapping[g]]
        sample_mean_g = mg_spec.group_sample_means[g] if mg_spec.group_specs[g].meanstructure else None
        f_g = ml_objective(
            theta_g,
            mg_spec.group_specs[g],
            mg_spec.group_sample_covs[g],
            mg_spec.group_n_obs[g],
            sample_mean_g,
        )
        if f_g >= 1e9:
            return 1e10
        f_total += (mg_spec.group_n_obs[g] / mg_spec.n_total) * f_g
    return f_total


def multigroup_ml_gradient(
    theta_combined: np.ndarray, mg_spec: MultiGroupSpec
) -> np.ndarray:
    """Combined ML gradient using scatter-add for shared parameters."""
    grad = np.zeros(mg_spec.n_free_combined)
    for g in range(len(mg_spec.group_names)):
        theta_g = theta_combined[mg_spec.theta_mapping[g]]
        sample_mean_g = mg_spec.group_sample_means[g] if mg_spec.group_specs[g].meanstructure else None
        grad_g = ml_gradient(
            theta_g,
            mg_spec.group_specs[g],
            mg_spec.group_sample_covs[g],
            mg_spec.group_n_obs[g],
            sample_mean_g,
        )
        weight = mg_spec.group_n_obs[g] / mg_spec.n_total
        np.add.at(grad, mg_spec.theta_mapping[g], weight * grad_g)
    return grad


@dataclass
class MultiGroupEstimationResult:
    """Raw output from multi-group optimization."""

    converged: bool
    iterations: int
    fmin: float
    theta_combined: np.ndarray
    mg_spec: MultiGroupSpec


def estimate_multigroup(mg_spec: MultiGroupSpec) -> MultiGroupEstimationResult:
    """Estimate multi-group model via ML."""
    theta0 = _pack_multigroup_start(mg_spec)

    result = optimize.minimize(
        multigroup_ml_objective,
        theta0,
        args=(mg_spec,),
        method="BFGS",
        jac=multigroup_ml_gradient,
        options={"maxiter": 10000, "gtol": 1e-6},
    )

    # Polish
    if result.success:
        result2 = optimize.minimize(
            multigroup_ml_objective,
            result.x,
            args=(mg_spec,),
            method="BFGS",
            jac=multigroup_ml_gradient,
            options={"maxiter": 10000, "gtol": 1e-9},
        )
        if result2.fun <= result.fun + 1e-10:
            result2.success = True
            result = result2

    if not result.success:
        warnings.warn(
            f"Optimization did not converge: {result.message}",
            RuntimeWarning,
            stacklevel=2,
        )

    return MultiGroupEstimationResult(
        converged=result.success,
        iterations=result.nit,
        fmin=result.fun,
        theta_combined=result.x,
        mg_spec=mg_spec,
    )


def _multigroup_compute_se(
    theta_combined: np.ndarray, mg_spec: MultiGroupSpec
) -> np.ndarray:
    """Compute SEs for multi-group model using expected information."""
    k = mg_spec.n_free_combined
    eps = 1e-7

    # For each group, compute per-group implied covariance and its derivatives
    # Then sum information contributions across groups
    info = np.zeros((k, k))

    for g in range(len(mg_spec.group_names)):
        spec_g = mg_spec.group_specs[g]
        n_g = mg_spec.group_n_obs[g]
        mapping_g = mg_spec.theta_mapping[g]
        theta_g = theta_combined[mapping_g]

        A_g, S_g = spec_g.unpack(theta_g)
        sigma_g = _model_implied_cov(A_g, S_g, spec_g.F)
        if sigma_g is None:
            return np.full(k, np.nan)

        try:
            sigma_inv_g = np.linalg.inv(sigma_g)
        except np.linalg.LinAlgError:
            return np.full(k, np.nan)

        p = sigma_g.shape[0]
        k_g = len(theta_g)

        # Compute dSigma for each local parameter
        dSigma_g = []
        for i in range(k_g):
            tg_plus = theta_g.copy()
            tg_minus = theta_g.copy()
            tg_plus[i] += eps
            tg_minus[i] -= eps
            A_p, S_p = spec_g.unpack(tg_plus)
            A_m, S_m = spec_g.unpack(tg_minus)
            sig_p = _model_implied_cov(A_p, S_p, spec_g.F)
            sig_m = _model_implied_cov(A_m, S_m, spec_g.F)
            if sig_p is None or sig_m is None:
                return np.full(k, np.nan)
            dSigma_g.append((sig_p - sig_m) / (2 * eps))

        SinvdS_g = [sigma_inv_g @ ds for ds in dSigma_g]

        # Add this group's information contribution to the combined matrix
        for i in range(k_g):
            for j in range(i, k_g):
                val = 0.5 * (n_g - 1) * np.trace(SinvdS_g[i] @ SinvdS_g[j])
                ci = mapping_g[i]
                cj = mapping_g[j]
                info[ci, cj] += val
                if ci != cj:
                    info[cj, ci] += val

    try:
        info_inv = np.linalg.inv(info)
        var_theta = np.diag(info_inv)
        se = np.where(var_theta > 0, np.sqrt(var_theta), np.nan)
        return se
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)
