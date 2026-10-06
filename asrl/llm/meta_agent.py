"""LLM meta-agent A_LLM for the outer loop (DOCX lines 38-44).

* ``HyperparamSpace`` - the bounded discrete proposal set H: one hyper-parameter at a
  time, multiplied by a fixed step, inside box bounds, and only if the two-timescale
  ordering eta_theta <= min(eta_psi, eta_phi) still holds. Reward-defining parameters
  (alpha, B, eps, b0, w0..w3) are not in the space and are rejected by ``apply``.
* ``RuleBasedMetaAgent`` - offline, deterministic stand-in for the LLM (APPROXIMATION).
  It reads the same diagnostics and memory a real LLM would and maps them to one move
  with simple heuristics; it never re-proposes a move rejected at the same h.
* ``LLMMetaAgent`` - the real interface: builds a prompt from (S_k, M_k, h, H), calls any
  ``llm_fn(prompt) -> str`` (version-pinned, temperature 0), and parses a JSON choice of
  ONE enumerated move. Invalid output -> no proposal (safe fallback).
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional

import numpy as np

from ..config import HYPERPARAM_NAMES, REWARD_PARAM_NAMES, Hyperparams, MetaConfig


class HyperparamSpace:
    def __init__(self, meta_cfg: MetaConfig):
        self.bounds = dict(meta_cfg.bounds)
        self.steps = dict(meta_cfg.steps)
        for k in list(self.bounds) + list(self.steps):
            assert k in HYPERPARAM_NAMES, f"{k} is not an optimisation hyper-parameter"
            assert k not in REWARD_PARAM_NAMES

    @staticmethod
    def ordering_ok(h: Hyperparams) -> bool:
        return h.eta_theta <= min(h.eta_psi, h.eta_phi) + 1e-15

    def apply(self, h: Hyperparams, delta_h: dict) -> Hyperparams:
        name, factor = delta_h["name"], float(delta_h["factor"])
        if name in REWARD_PARAM_NAMES or name not in HYPERPARAM_NAMES:
            raise ValueError(f"meta-agent may not modify {name!r}")
        if factor not in self.steps[name]:
            raise ValueError(f"factor {factor} not in allowed steps {self.steps[name]} for {name}")
        h2 = h.copy()
        setattr(h2, name, getattr(h, name) * factor)
        lo, hi = self.bounds[name]
        if not (lo - 1e-15 <= getattr(h2, name) <= hi + 1e-15):
            raise ValueError(f"{name}={getattr(h2, name)} outside bounds {self.bounds[name]}")
        if not self.ordering_ok(h2):
            raise ValueError("proposal breaks the two-timescale learning-rate ordering")
        return h2

    def valid_moves(self, h: Hyperparams) -> list:
        moves = []
        for name, facs in self.steps.items():
            for f in facs:
                try:
                    self.apply(h, {"name": name, "factor": f})
                    moves.append({"name": name, "factor": f})
                except ValueError:
                    pass
        return moves


def _tabu(memory: list, h: Hyperparams) -> set:
    """Moves already rejected from (essentially) the same h."""
    out = set()
    hd = h.as_dict()
    for m in memory:
        if m.get("decision") == "reject_rollback" and m.get("proposal"):
            hb = m["h_before"]
            if all(np.isclose(hb[k], hd[k]) for k in hd):
                d = m["proposal"]["delta_h"]
                out.add((d["name"], float(d["factor"])))
    return out


class RuleBasedMetaAgent:
    name = "rule-based-mock"

    def __init__(self, seed: int = 0, kl_high: float = 0.02, clip_high: float = 0.25,
                 clip_low: float = 0.02, cov_err_high: float = 0.05):
        self.rng = np.random.default_rng(seed)
        self.kl_high, self.clip_high, self.clip_low, self.cov_err_high = kl_high, clip_high, clip_low, cov_err_high

    def propose(self, diag: dict, memory: list, h: Hyperparams, space: HyperparamSpace) -> Optional[dict]:
        valid = space.valid_moves(h)
        tabu = _tabu(memory, h)
        valid = [m for m in valid if (m["name"], float(m["factor"])) not in tabu]
        if not valid:
            return None
        prefs = []
        kl, cf = diag.get("approx_kl") or 0.0, diag.get("clip_frac") or 0.0
        cov = diag.get("coverage_error")
        if kl > self.kl_high or cf > self.clip_high:
            prefs += [("eta_theta", 0.5, "policy updates too large (KL/clip fraction high)"),
                      ("eps_clip", 0.75, "policy updates too large")]
        elif cf < self.clip_low and kl < self.kl_high / 4:
            prefs += [("eta_theta", 2.0, "policy updates very small; speed up the actor"),
                      ("eps_clip", 1.5, "clip rarely active")]
        if cov is not None and cov > self.cov_err_high:
            prefs += [("eta_phi", 2.0, "quantile head under-calibrated; speed up critic"),
                      ("c_q", 2.0, "quantile head under-calibrated")]
        if (diag.get("val_sharpe") or 0.0) < 0:
            prefs += [("c_e", 2.0, "negative validation Sharpe; encourage exploration")]
        for name, f, why in prefs:
            if any(m["name"] == name and float(m["factor"]) == f for m in valid):
                return {"delta_h": {"name": name, "factor": f}, "rationale": why}
        m = valid[int(self.rng.integers(len(valid)))]
        return {"delta_h": m, "rationale": "no rule fired; exploratory move from H"}


def build_meta_prompt(diag: dict, memory: list, h: Hyperparams, moves: list) -> str:
    recent = [{"iteration": m["iteration"], "proposal": m.get("proposal"), "decision": m.get("decision"),
               "S_V_baseline": m.get("S_V_baseline"), "S_V_probe": m.get("S_V_probe")} for m in memory[-8:]]
    menu = "\n".join(f"{i}: {m['name']} x {m['factor']}" for i, m in enumerate(moves))
    return (
        "You tune the optimisation hyper-parameters of a PPO trading agent trained with a "
        "Sharpe-calibrated reward. You may NOT change reward parameters. Propose exactly ONE "
        "move from the menu. A probe will be accepted only if validation Sharpe improves.\n\n"
        f"Current hyper-parameters h: {json.dumps(h.as_dict())}\n"
        f"Validation diagnostics: {json.dumps(diag, default=float)}\n"
        f"Recent decisions (memory): {json.dumps(recent, default=str)}\n\n"
        f"Allowed moves:\n{menu}\n\n"
        'Reply with JSON only: {"move_id": <int>, "rationale": "<one sentence>"}'
    )


class LLMMetaAgent:
    """Wraps any text-in/text-out LLM callable (e.g. an API client pinned to one model
    version at temperature 0)."""
    name = "llm"

    def __init__(self, llm_fn: Callable[[str], str]):
        self.llm_fn = llm_fn
        self.last_prompt = None
        self.last_response = None

    def propose(self, diag, memory, h, space):
        moves = space.valid_moves(h)
        if not moves:
            return None
        prompt = build_meta_prompt(diag, memory, h, moves)
        self.last_prompt = prompt
        resp = self.llm_fn(prompt)
        self.last_response = resp
        m = re.search(r"\{.*\}", resp, re.S)
        try:
            obj = json.loads(m.group(0)) if m else {}
            idx = int(obj["move_id"])
            move = moves[idx]
        except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError):
            return None
        return {"delta_h": move, "rationale": str(obj.get("rationale", ""))}


def make_meta_agent(kind: str, seed: int, llm_fn: Optional[Callable] = None):
    if kind == "rule":
        return RuleBasedMetaAgent(seed=seed)
    if kind == "llm":
        if llm_fn is None:
            raise ValueError("meta.agent='llm' needs an llm_fn backend (see asrl/llm/providers.py)")
        return LLMMetaAgent(llm_fn)
    raise ValueError(kind)
