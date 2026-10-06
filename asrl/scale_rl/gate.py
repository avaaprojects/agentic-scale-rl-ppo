"""Reliability gate, DOCX line 16:
    lambda_t = sigmoid(w0 + w1 E_cal,t + w2 Turb_t + w3 ell_t).
(The SCALE-RL paper's Eq. 6 has a different form; the DOCX version is used.)"""
from __future__ import annotations

import numpy as np


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def reliability_gate(E_cal, turb, ell, w0, w1, w2, w3):
    return sigmoid(w0 + w1 * np.asarray(E_cal) + w2 * np.asarray(turb) + w3 * np.asarray(ell))
