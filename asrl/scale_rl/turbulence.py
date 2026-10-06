"""Turbulence index, DOCX line 15:

    Turb_t = (y_{t-1} - ybar)^T  Sigma_LW^{-1}  (y_{t-1} - ybar)

with ybar and Sigma_LW the rolling mean and Ledoit-Wolf shrunk covariance over a
window of W observations. Ambiguity: the DOCX does not say whether the window
contains y_{t-1}. Default follows Kritzman & Li (2010): the window is the W
returns *before* y_{t-1}, i.e. y_{t-1-W}, ..., y_{t-2}. Set ``include_current``
to use y_{t-W}, ..., y_{t-1} instead.

Turb_t depends only on y_{<t}, so it is F_{t-1}-measurable and can be
precomputed once from data (it does not depend on the policy).
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


def ledoit_wolf_cov(window: np.ndarray) -> np.ndarray:
    return LedoitWolf(assume_centered=False).fit(window).covariance_


def turbulence_at(y: np.ndarray, t: int, W: int, include_current: bool = False) -> float:
    """Turb_t for a single data index t (needs t-1-W >= 0, or t-W >= 0 if include_current)."""
    x = y[t - 1]
    win = y[t - W:t] if include_current else y[t - 1 - W:t - 1]
    mu = win.mean(axis=0)
    cov = ledoit_wolf_cov(win)
    d = x - mu
    try:
        return float(d @ np.linalg.solve(cov, d))
    except np.linalg.LinAlgError:
        # degenerate window (e.g. constant returns): LW cannot regularise a zero matrix.
        # Numerical guard, not in the DOCX: use the pseudo-inverse.
        return float(d @ np.linalg.pinv(cov) @ d)


def turbulence_series(y: np.ndarray, W: int, include_current: bool = False):
    """Returns (turb[T], valid[T]). Invalid entries (not enough history) are 0."""
    T = y.shape[0]
    out = np.zeros(T)
    valid = np.zeros(T, dtype=bool)
    first = W if include_current else W + 1
    for t in range(first, T):
        out[t] = turbulence_at(y, t, W, include_current)
        valid[t] = True
    return out, valid
