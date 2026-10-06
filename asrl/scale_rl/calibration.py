"""Rolling calibration error E_cal,t (DOCX line 14) and quantile-coverage metrics.

DOCX line 14 (mode "docx", default):
    E_cal,t = (1/W) sum_{j=max(0,t-W)}^{t-1} (1/N) sum_i | 1{r_j <= z_i(s_j,a_j)} - tau_i |,
    E_cal,0 = 1.
Uses data up to t-1 only. Divides by W even when fewer than W terms exist
(literal reading; irrelevant to the reward when B >= W because D_hat = 0 before B).

AMBIGUITY: for a perfectly calibrated critic the expectation of each summand is
2 tau_i (1 - tau_i), so this error never goes to 0 (~1/3 for midpoint quantiles).
Mode "coverage" (option) is the coverage error that does vanish under calibration:
    E_cal,t = (1/N) sum_i | (1/n_j) sum_j 1{r_j <= z_i} - tau_i |.
"""
from __future__ import annotations

import numpy as np


def calibration_error(r_hist: np.ndarray, z_hist: np.ndarray, taus: np.ndarray, t: int, W: int,
                      mode: str = "docx") -> np.ndarray:
    """Vectorised over envs. r_hist [n_env, >=t], z_hist [n_env, >=t, N]. Returns [n_env]."""
    n_env = r_hist.shape[0]
    if t == 0:
        return np.ones(n_env)                                     # E_cal,0 <- 1
    j0 = max(0, t - W)
    ind = (r_hist[:, j0:t, None] <= z_hist[:, j0:t, :]).astype(np.float64)   # [n_env, w, N]
    if mode == "docx":
        per_step = np.abs(ind - taus).mean(-1)                    # (1/N) sum_i |.|
        return per_step.sum(-1) / W
    if mode == "coverage":
        return np.abs(ind.mean(1) - taus).mean(-1)
    raise ValueError(mode)


def coverage_stats(r: np.ndarray, z: np.ndarray, taus: np.ndarray) -> dict:
    """Whole-period metrics. r [n], z [n, N].
    * per-quantile empirical coverage P(r <= z_i) vs tau_i -> mean abs coverage error
    * central-interval coverage of [z_1, z_N] vs nominal tau_N - tau_1."""
    ind = (r[:, None] <= z).astype(np.float64)
    cov = ind.mean(0)
    zs = np.sort(z, axis=1)
    inside = ((r >= zs[:, 0]) & (r <= zs[:, -1])).mean()
    return {
        "coverage_error": float(np.abs(cov - taus).mean()),
        "central_interval_coverage": float(inside),
        "central_interval_nominal": float(taus[-1] - taus[0]),
    }
