"""Minimal numpy MLP with manual backprop (PyTorch was unavailable in the build
environment; this keeps the prototype dependency-light). Gradients are verified
against finite differences in tests/test_models.py."""
from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, sizes, rng: np.random.Generator, out_gain: float = 0.01, prefix: str = ""):
        self.sizes = list(sizes)
        self.prefix = prefix
        self.params: dict[str, np.ndarray] = {}
        n_layers = len(sizes) - 1
        for i in range(n_layers):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            gain = out_gain if i == n_layers - 1 else np.sqrt(2.0)
            W = rng.standard_normal((fan_in, fan_out))
            # orthogonal-style init via QR for stable tanh nets
            q, _ = np.linalg.qr(W if fan_in >= fan_out else W.T)
            q = q if fan_in >= fan_out else q.T
            self.params[f"{prefix}W{i}"] = (gain * q[:fan_in, :fan_out]).astype(np.float64)
            self.params[f"{prefix}b{i}"] = np.zeros(fan_out)

    @property
    def n_layers(self) -> int:
        return len(self.sizes) - 1

    def forward(self, x: np.ndarray):
        cache = [x]
        h = x
        for i in range(self.n_layers):
            z = h @ self.params[f"{self.prefix}W{i}"] + self.params[f"{self.prefix}b{i}"]
            h = np.tanh(z) if i < self.n_layers - 1 else z
            cache.append(h)
        return h, cache

    def backward(self, cache, dout: np.ndarray) -> dict:
        grads = {}
        g = dout
        for i in reversed(range(self.n_layers)):
            h_in = cache[i]
            grads[f"{self.prefix}W{i}"] = h_in.T @ g
            grads[f"{self.prefix}b{i}"] = g.sum(0)
            if i > 0:
                g = g @ self.params[f"{self.prefix}W{i}"].T
                g = g * (1.0 - cache[i] ** 2)          # tanh'
        return grads
