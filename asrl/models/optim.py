"""Optimisers. The learning rate is passed at every step so the meta-agent can
change eta_theta / eta_psi / eta_phi between iterations.

* ``SGD``  - DOCX line 34 literally: p <- p - eta * grad.
* ``Adam`` - practical default (APPROXIMATION of line 34; see README).
"""
from __future__ import annotations

import copy

import numpy as np


class SGD:
    def __init__(self, params: dict):
        self.params = params

    def step(self, grads: dict, lr: float) -> None:
        for k, g in grads.items():
            self.params[k] -= lr * g

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, s: dict) -> None:
        pass


class Adam:
    def __init__(self, params: dict, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict, lr: float) -> None:
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            self.params[k] -= lr * (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + self.eps)

    def state_dict(self) -> dict:
        return {"m": copy.deepcopy(self.m), "v": copy.deepcopy(self.v), "t": self.t}

    def load_state_dict(self, s: dict) -> None:
        self.m = copy.deepcopy(s["m"])
        self.v = copy.deepcopy(s["v"])
        self.t = s["t"]


def make_optimizer(kind: str, params: dict):
    if kind == "adam":
        return Adam(params)
    if kind == "sgd":
        return SGD(params)
    raise ValueError(kind)
