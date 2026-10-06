"""Critic losses with analytic gradients w.r.t. the quantile outputs z [M, N].

Line 31 (pinball, one-step return target):
    L_pin = E_B[ (1/N) sum_i (tau_i - 1{r < z_i}) (r - z_i) ]
Line 32 / Eq. (5) (Sharpe calibration loss):
    L_cal = E_B[ (mu_phi - A_t)^2 + (sigma^2_phi - V_t)^2 ]
with mu, sigma^2 from Eqs. (1)-(2) and A_t, V_t the stored post-update running moments.
"""
from __future__ import annotations

import numpy as np


def pinball_loss(z: np.ndarray, r: np.ndarray, taus: np.ndarray):
    M, N = z.shape
    ind = (r[:, None] < z).astype(np.float64)
    w = taus[None, :] - ind
    loss = float((w * (r[:, None] - z)).mean())
    dz = -w / (M * N)
    return loss, dz


def calibration_loss(z: np.ndarray, A: np.ndarray, V: np.ndarray):
    M, N = z.shape
    mu = z.mean(1)
    var = (z ** 2).mean(1) - mu ** 2
    e1, e2 = mu - A, var - V
    loss = float((e1 ** 2 + e2 ** 2).mean())
    dmu = 2.0 * e1 / M                      # dL/dmu
    dvar = 2.0 * e2 / M                     # dL/dvar
    # dmu/dz_i = 1/N ; dvar/dz_i = 2 (z_i - mu) / N
    dz = dmu[:, None] / N + dvar[:, None] * 2.0 * (z - mu[:, None]) / N
    return loss, dz
