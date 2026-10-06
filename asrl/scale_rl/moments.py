"""Running Sharpe moments and the gated Differential Sharpe reward (DOCX lines 8-20).

All functions are vectorised over environments (numpy broadcasting).
"""
from __future__ import annotations

import numpy as np


def init_moments(n_env: int, b0: float, eps: float):
    """Line 8: A_{-1} <- 0 ; B_{-1} <- b0 (b0 >= eps)."""
    assert b0 >= eps
    return np.zeros(n_env), np.full(n_env, float(b0))


def floored_variance(A_prev, B_prev, eps):
    """Line 10: V_{t-1} <- max(B_{t-1} - A_{t-1}^2, eps)."""
    return np.maximum(B_prev - A_prev ** 2, eps)


def sharpe_fn(A, B):
    """S(A,B) = A / sqrt(B - A^2) (no floor; caller must ensure B - A^2 > 0)."""
    return A / np.sqrt(B - A ** 2)


def gated_innovations(r, mu_hat, var_hat, A_prev, B_prev, lam, alpha):
    """Lines 17-18:
    dA = alpha [ lam (r - A) + (1 - lam)(mu - A) ]
    dB = alpha [ lam (r^2 - B) + (1 - lam)(mu^2 + sigma^2 - B) ]"""
    dA = alpha * (lam * (r - A_prev) + (1.0 - lam) * (mu_hat - A_prev))
    dB = alpha * (lam * (r ** 2 - B_prev) + (1.0 - lam) * (mu_hat ** 2 + var_hat - B_prev))
    return dA, dB


def gated_dsr(dA, dB, A_prev, B_prev, V_prev_floored, t, burn_in):
    """Line 19: D = (B_{t-1} dA - 0.5 A_{t-1} dB) / V_{t-1}^{3/2} if t >= B else 0.
    V_prev_floored must be the line-10 floored variance."""
    D = (B_prev * dA - 0.5 * A_prev * dB) / V_prev_floored ** 1.5
    return np.where(np.asarray(t) >= burn_in, D, 0.0)


def update_moments(A_prev, B_prev, r, alpha):
    """Line 20 (raw returns only):
    A_t = (1-alpha) A_{t-1} + alpha r ; B_t = (1-alpha) B_{t-1} + alpha r^2 ; V_t = B_t - A_t^2 (unfloored)."""
    A = (1.0 - alpha) * A_prev + alpha * r
    B = (1.0 - alpha) * B_prev + alpha * r ** 2
    return A, B, B - A ** 2
