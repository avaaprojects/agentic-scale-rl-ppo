"""GAE (DOCX lines 24-25) and the PPO clipped surrogate (line 29) with its gradient."""
from __future__ import annotations

import numpy as np


def compute_gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float):
    """rewards, values: [n_env, T]. Terminal bootstrap V(s~_T) = 0 (DOCX line 24).
    xi_t = D_t + gamma V(s~_{t+1}) - V(s~_t);  Adv_t = sum_l (gamma lam)^l xi_{t+l};  G_t = Adv_t + V(s~_t)."""
    n_env, T = rewards.shape
    next_v = np.concatenate([values[:, 1:], np.zeros((n_env, 1))], axis=1)
    xi = rewards + gamma * next_v - values
    adv = np.zeros_like(rewards)
    acc = np.zeros(n_env)
    for t in reversed(range(T)):
        acc = xi[:, t] + gamma * lam * acc
        adv[:, t] = acc
    return adv, adv + values


def normalise(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def ppo_clip_loss(logp_new, logp_old, adv, eps_clip):
    """L_clip = -mean(min(rho A, clip(rho, 1-eps, 1+eps) A)). Returns (loss, dL/dlogp_new, stats)."""
    M = len(adv)
    log_ratio = logp_new - logp_old
    rho = np.exp(log_ratio)
    unclipped = rho * adv
    clipped = np.clip(rho, 1.0 - eps_clip, 1.0 + eps_clip) * adv
    loss = -float(np.minimum(unclipped, clipped).mean())
    # gradient flows only where the unclipped term is the active minimum
    active = ~(((adv > 0) & (rho > 1.0 + eps_clip)) | ((adv < 0) & (rho < 1.0 - eps_clip)))
    dlogp = -(adv * rho * active) / M                     # d(rho)/d(logp) = rho
    stats = {
        "clip_frac": float((np.abs(rho - 1.0) > eps_clip).mean()),
        "approx_kl": float(((rho - 1.0) - log_ratio).mean()),   # k3 estimator, >= 0
    }
    return loss, dlogp, stats
