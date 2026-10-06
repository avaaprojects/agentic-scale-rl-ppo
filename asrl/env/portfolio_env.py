"""Vectorised long-only portfolio environment.

Action a_t = portfolio weights over n assets + cash (softmax of the policy's
Gaussian sample, done in the trainer). Each env runs a contiguous window of T
data days starting at ``starts[e]``.

    r_t = w_t . [y_d, 0] - c * || w_t - h_{t-1} ||_1

where h_{t-1} are the holdings after the previous day's price drift. The market
state is s_t = (F[d], h_{t-1}): features of past returns plus current holdings
(DOCX: "prices, indicators, holdings").
"""
from __future__ import annotations

import numpy as np

from ..data.dataset import MarketDataset


class PortfolioEnv:
    def __init__(self, ds: MarketDataset, cost: float):
        self.ds = ds
        self.cost = float(cost)
        self.n = ds.n_assets
        self.action_dim = self.n + 1
        self.state_dim = ds.feat_dim + self.action_dim

    def reset(self, starts: np.ndarray) -> np.ndarray:
        self.starts = np.asarray(starts, dtype=int)
        self.t = 0
        self.hold = np.zeros((len(self.starts), self.action_dim))
        self.hold[:, -1] = 1.0                       # start fully in cash
        return self.obs()

    @property
    def d(self) -> np.ndarray:
        return self.starts + self.t

    def obs(self) -> np.ndarray:
        return np.concatenate([self.ds.features[self.d], self.hold], axis=1)

    def step(self, w: np.ndarray):
        d = self.d
        y = self.ds.y[d]                                       # [n_env, n]
        y_full = np.concatenate([y, np.zeros((len(d), 1))], axis=1)
        gross = (w * y_full).sum(1)
        turnover = np.abs(w - self.hold).sum(1)
        r = gross - self.cost * turnover
        grown = w * (1.0 + y_full)
        self.hold = grown / grown.sum(1, keepdims=True)        # drift holdings to end of day
        self.t += 1
        return r, y


def portfolio_returns_fixed_weights(y: np.ndarray, w: np.ndarray, cost: float) -> np.ndarray:
    """Reference strategy: rebalance daily to fixed weights w (n+1, incl. cash)."""
    n_days = y.shape[0]
    hold = np.zeros_like(w)
    hold[-1] = 1.0
    out = np.zeros(n_days)
    for d in range(n_days):
        yf = np.append(y[d], 0.0)
        out[d] = w @ yf - cost * np.abs(w - hold).sum()
        g = w * (1 + yf)
        hold = g / g.sum()
    return out
