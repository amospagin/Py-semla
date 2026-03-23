"""Local Structural Equation Modeling (LSEM).

Fits SEM models locally across a continuous moderator using Gaussian
kernel weighting, producing smooth parameter trajectories.

Reference: Hildebrandt, Lüdtke, Robitzsch, Sommer & Wilhelm (2016).
Exploring factor model parameters across continuous variables with
local structural equation models. Multivariate Behavioral Research.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import optimize

from .estimation import (
    ml_objective, ml_gradient, _model_implied_cov, EstimationResult,
)
from .model import Model
from .results import ModelResults
from .specification import build_specification
from .syntax import parse_syntax


class LSEMResult:
    """Results of Local SEM analysis."""

    def __init__(
        self,
        focal_points: np.ndarray,
        moderator: str,
        param_names: list[str],
        estimates: np.ndarray,   # (n_focal, n_params)
        se: np.ndarray,          # (n_focal, n_params)
        effective_n: np.ndarray, # (n_focal,)
        converged: np.ndarray,   # (n_focal,) bool
        bandwidth: float,
    ):
        self._focal = focal_points
        self._moderator = moderator
        self._param_names = param_names
        self._estimates = estimates
        self._se = se
        self._effective_n = effective_n
        self._converged = converged
        self._bandwidth = bandwidth

    def trajectories(self) -> pd.DataFrame:
        """Parameter estimates at each focal point.

        Returns
        -------
        pd.DataFrame
            Columns: focal_point, effective_n, converged, then one column
            per parameter (estimate).
        """
        df = pd.DataFrame({
            "focal_point": self._focal,
            "effective_n": self._effective_n,
            "converged": self._converged,
        })
        for i, name in enumerate(self._param_names):
            df[name] = self._estimates[:, i]
        return df

    def trajectory(self, param: str) -> pd.DataFrame:
        """Get trajectory for a single parameter with CIs.

        Parameters
        ----------
        param : str
            Parameter name, e.g. ``"f1 =~ x2"`` or ``"x1 ~~ x1"``.
        """
        if param not in self._param_names:
            raise ValueError(
                f"Parameter '{param}' not found. "
                f"Available: {self._param_names}"
            )
        idx = self._param_names.index(param)
        est = self._estimates[:, idx]
        se = self._se[:, idx]
        return pd.DataFrame({
            "focal_point": self._focal,
            "est": est,
            "se": se,
            "ci_lower": est - 1.96 * se,
            "ci_upper": est + 1.96 * se,
        })

    @property
    def convergence_rate(self) -> float:
        return float(np.mean(self._converged))

    @property
    def param_names(self) -> list[str]:
        return list(self._param_names)

    def summary(self) -> str:
        lines = []
        lines.append("Local SEM Results")
        lines.append("=" * 65)
        lines.append(f"  Moderator: {self._moderator}")
        lines.append(f"  Bandwidth: {self._bandwidth:.3f}")
        lines.append(f"  Focal points: {len(self._focal)}")
        lines.append(
            f"  Moderator range: [{self._focal[0]:.2f}, "
            f"{self._focal[-1]:.2f}]"
        )
        lines.append(f"  Convergence rate: {self.convergence_rate:.1%}")
        lines.append(f"  Effective n range: "
                     f"[{self._effective_n.min():.0f}, "
                     f"{self._effective_n.max():.0f}]")
        lines.append("")

        # Parameter variation summary
        lines.append(f"  {'Parameter':<25s} {'Min':>8s} {'Max':>8s} "
                     f"{'Range':>8s}")
        lines.append("  " + "-" * 51)
        converged_mask = self._converged
        for i, name in enumerate(self._param_names):
            vals = self._estimates[converged_mask, i]
            if len(vals) > 0:
                lines.append(
                    f"  {name:<25s} {vals.min():>8.3f} {vals.max():>8.3f} "
                    f"{vals.max() - vals.min():>8.3f}"
                )
        lines.append("=" * 65)
        output = "\n".join(lines)
        print(output)
        return output

    def __repr__(self) -> str:
        return (
            f"LSEMResult(moderator='{self._moderator}', "
            f"focal_points={len(self._focal)}, "
            f"params={len(self._param_names)}, "
            f"convergence={self.convergence_rate:.0%})"
        )


def lsem(
    model: str,
    data: pd.DataFrame,
    moderator: str,
    focal_points: int | np.ndarray = 50,
    bandwidth: float | str = "auto",
    **kwargs,
) -> LSEMResult:
    """Fit a Local SEM across a continuous moderator.

    Parameters
    ----------
    model : str
        Model syntax in lavaan format.
    data : pd.DataFrame
        Data with observed variables and the moderator column.
    moderator : str
        Column name of the continuous moderator variable.
    focal_points : int or array
        Number of equally spaced focal points (int), or explicit
        array of focal values.
    bandwidth : float or "auto"
        Gaussian kernel bandwidth. ``"auto"`` uses Silverman's rule.
    **kwargs
        Additional arguments passed to the model (e.g., ``meanstructure``).

    Returns
    -------
    LSEMResult
    """
    if moderator not in data.columns:
        raise ValueError(
            f"Moderator '{moderator}' not found in data. "
            f"Available: {sorted(data.columns.tolist())}"
        )

    mod_values = data[moderator].values.astype(float)

    # Set up focal points
    if isinstance(focal_points, (int, np.integer)):
        focal = np.linspace(
            np.percentile(mod_values, 5),
            np.percentile(mod_values, 95),
            int(focal_points),
        )
    else:
        focal = np.asarray(focal_points, dtype=float)

    # Bandwidth
    if bandwidth == "auto":
        # Silverman's rule of thumb
        h = 1.06 * np.std(mod_values) * len(mod_values) ** (-1 / 5)
    else:
        h = float(bandwidth)

    # Parse model and build spec
    tokens = parse_syntax(model)
    _non_model_ops = {":=", ">", "<", ">=", "<=", "=="}
    model_tokens = [t for t in tokens if t.op not in _non_model_ops]

    auto_cov_latent = kwargs.pop("auto_cov_latent", True)
    meanstructure = kwargs.pop("meanstructure", False)

    spec = build_specification(
        model_tokens, data.columns.tolist(),
        auto_cov_latent=auto_cov_latent,
        meanstructure=meanstructure,
    )
    obs_vars = spec.observed_vars

    # Get parameter names from spec
    param_names = []
    for p in spec.params:
        if p.free:
            if p.op == "~1":
                param_names.append(f"{p.lhs} {p.op}")
            else:
                param_names.append(f"{p.lhs} {p.op} {p.rhs}")
    # Deduplicate (equality constraints)
    if spec._constraint_map is not None:
        seen = set()
        unique_names = []
        for i, name in enumerate(param_names):
            eff_idx = spec._constraint_map[i]
            if eff_idx not in seen:
                seen.add(eff_idx)
                unique_names.append(name)
        param_names = unique_names

    n_params = spec.n_free
    n_focal = len(focal)

    estimates = np.full((n_focal, n_params), np.nan)
    se_array = np.full((n_focal, n_params), np.nan)
    effective_n = np.zeros(n_focal)
    converged = np.zeros(n_focal, dtype=bool)

    obs_data = data[obs_vars].values
    n_obs = len(obs_data)

    # Global fit for starting values
    global_cov = np.cov(obs_data, rowvar=False, ddof=1)
    if global_cov.ndim == 0:
        global_cov = global_cov.reshape(1, 1)
    global_mean = np.mean(obs_data, axis=0) if meanstructure else None

    # Get global starting values
    theta_start = spec.pack_start()

    # Try global fit for better starting values
    try:
        global_result = optimize.minimize(
            ml_objective, theta_start,
            args=(spec, global_cov, n_obs, global_mean),
            method="BFGS", jac=ml_gradient,
            options={"maxiter": 5000, "gtol": 1e-6},
        )
        if global_result.success:
            theta_start = global_result.x
    except Exception:
        pass

    # Fit at each focal point (warm-starting from previous solution)
    current_start = theta_start.copy()

    for f_idx, fp in enumerate(focal):
        # Compute kernel weights
        weights = np.exp(-0.5 * ((mod_values - fp) / h) ** 2)
        weights /= weights.sum()

        # Effective sample size
        eff_n = 1.0 / np.sum(weights ** 2)
        effective_n[f_idx] = eff_n

        if eff_n < max(spec.n_free, 20):
            continue  # skip focal points with insufficient data

        # Weighted covariance matrix
        weighted_mean = np.average(obs_data, axis=0, weights=weights)
        centered = obs_data - weighted_mean
        weighted_cov = (centered * weights[:, None]).T @ centered
        # Bessel-like correction using effective n
        weighted_cov *= eff_n / (eff_n - 1)

        # Check PD
        try:
            eigvals = np.linalg.eigvalsh(weighted_cov)
            if eigvals.min() <= 0:
                continue
        except np.linalg.LinAlgError:
            continue

        w_mean = weighted_mean if meanstructure else None

        # Fit model (try current_start first, fall back to global start)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = optimize.minimize(
                    ml_objective, current_start,
                    args=(spec, weighted_cov, int(round(eff_n)), w_mean),
                    method="BFGS", jac=ml_gradient,
                    options={"maxiter": 5000, "gtol": 1e-6},
                )

                # Fall back to global start if warm start failed
                if not result.success or result.fun >= 1e8:
                    result = optimize.minimize(
                        ml_objective, theta_start,
                        args=(spec, weighted_cov, int(round(eff_n)), w_mean),
                        method="BFGS", jac=ml_gradient,
                        options={"maxiter": 5000, "gtol": 1e-6},
                    )

                if result.success or result.fun < 1e8:
                    estimates[f_idx] = result.x
                    converged[f_idx] = True
                    current_start = result.x.copy()  # warm start

                    # Compute SEs
                    from .estimation import _compute_se
                    se_vals = _compute_se(
                        result.x, spec, weighted_cov, int(round(eff_n)),
                    )
                    se_array[f_idx] = se_vals

            except Exception:
                continue

    return LSEMResult(
        focal_points=focal,
        moderator=moderator,
        param_names=param_names,
        estimates=estimates,
        se=se_array,
        effective_n=effective_n,
        converged=converged,
        bandwidth=h,
    )
