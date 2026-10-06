"""Actor pi_theta, value head V_psi and quantile head Z_phi.

Design choice (see README, ambiguity 5): the three are SEPARATE networks, so
theta, psi and phi are disjoint parameter sets and each learning rate eta_theta,
eta_psi, eta_phi owns exactly its own loss terms:
    grad_theta L = grad_theta (L_clip - c_e H)
    grad_psi   L = c_v grad_psi L_V
    grad_phi   L = c_q grad_phi (L_pin + beta L_cal)
"""
from __future__ import annotations

import numpy as np

from .mlp import MLP

LOG_2PI = float(np.log(2.0 * np.pi))


class GaussianActor:
    """pi_theta(u | s~) = N(mu_theta(s~), diag(exp(log_std))^2); weights = softmax(u).
    Log-probabilities are taken on the pre-softmax sample u (the stored action)."""

    def __init__(self, in_dim, act_dim, hidden, n_hidden, rng, log_std_init=-0.5, log_std_bounds=(-5.0, 1.0)):
        self.net = MLP([in_dim] + [hidden] * n_hidden + [act_dim], rng, out_gain=0.01, prefix="pi_")
        self.params = self.net.params
        self.params["log_std"] = np.full(act_dim, float(log_std_init))
        self.act_dim = act_dim
        self.log_std_bounds = log_std_bounds

    def mean(self, s):
        return self.net.forward(s)

    def sample(self, s, rng):
        mu, _ = self.net.forward(s)
        std = np.exp(self.params["log_std"])
        u = mu + std * rng.standard_normal(mu.shape)
        return u, self.log_prob(u, mu), mu

    def log_prob(self, u, mu):
        ls = self.params["log_std"]
        z = (u - mu) / np.exp(ls)
        return -0.5 * (z ** 2).sum(-1) - ls.sum() - 0.5 * self.act_dim * LOG_2PI

    def entropy(self) -> float:
        return float(self.params["log_std"].sum() + 0.5 * self.act_dim * (1.0 + LOG_2PI))

    def grads(self, cache, u, mu, dlogp, d_entropy: float) -> dict:
        """dL/dparams given dL/dlogp per sample and dL/dH (scalar)."""
        ls = self.params["log_std"]
        var = np.exp(2 * ls)
        dmu = dlogp[:, None] * (u - mu) / var
        g = self.net.backward(cache, dmu)
        z2 = (u - mu) ** 2 / var
        g["log_std"] = (dlogp[:, None] * (z2 - 1.0)).sum(0) + d_entropy
        return g

    def clamp(self) -> None:
        lo, hi = self.log_std_bounds
        np.clip(self.params["log_std"], lo, hi, out=self.params["log_std"])


class ValueNet:
    def __init__(self, in_dim, hidden, n_hidden, rng):
        self.net = MLP([in_dim] + [hidden] * n_hidden + [1], rng, out_gain=1.0, prefix="v_")
        self.params = self.net.params

    def forward(self, s):
        out, cache = self.net.forward(s)
        return out[:, 0], cache

    def grads(self, cache, dv) -> dict:
        return self.net.backward(cache, dv[:, None])


class QuantileNet:
    """Z_phi(s, a) -> N quantiles of the one-step return r_t. Output parameterised as
    z = out_scale * net(x) because returns are O(1e-2)."""

    def __init__(self, in_dim, n_quantiles, hidden, n_hidden, rng, out_scale=0.01):
        self.net = MLP([in_dim] + [hidden] * n_hidden + [n_quantiles], rng, out_gain=0.1, prefix="q_")
        self.params = self.net.params
        self.N = n_quantiles
        self.out_scale = out_scale
        self.taus = (2.0 * np.arange(1, n_quantiles + 1) - 1.0) / (2.0 * n_quantiles)  # tau_i=(2i-1)/(2N)

    def forward(self, x):
        out, cache = self.net.forward(x)
        return self.out_scale * out, cache

    def grads(self, cache, dz) -> dict:
        return self.net.backward(cache, self.out_scale * dz)


def quantile_moments(z: np.ndarray):
    """Eqs. (1)-(2): mu = mean_i z_i ; sigma^2 = mean_i z_i^2 - mu^2."""
    mu = z.mean(-1)
    var = (z ** 2).mean(-1) - mu ** 2
    return mu, var


def softmax(u: np.ndarray) -> np.ndarray:
    m = u - u.max(-1, keepdims=True)
    e = np.exp(m)
    return e / e.sum(-1, keepdims=True)
