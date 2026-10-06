"""Benchmarks and ablations.

Benchmarks (requested):
  ppo                 standard PPO, reward = raw portfolio return r_t, no critic head
  scale_rl            PPO + SCALE-RL gated DSR (gate uses E_cal, Turb; no LLM)
  scale_rl_llm_risk   + LLM risk score ell_t in state and gate (no embedding, no meta-agent)
  agentic_full        full Agentic SCALE-RL-PPO: ell_t + e_t + meta-agent

Ablations of the full system (requested):
  abl_no_llm_score    full minus ell_t (state and gate); embedding and meta-agent kept
  abl_gate_one        full with lambda_t == 1 (moments use realised returns only = classical DSR)
  abl_no_meta         full minus meta-agent (ell_t + e_t kept)
"""
from __future__ import annotations

from ..config import VariantConfig

VARIANTS = {
    "ppo": VariantConfig("ppo", reward="return", use_quantile=False, use_ab_state=False,
                         use_llm_risk=False, use_llm_embedding=False, gate_fixed_one=False, use_meta=False),
    "scale_rl": VariantConfig("scale_rl", use_llm_risk=False, use_llm_embedding=False, use_meta=False),
    "scale_rl_llm_risk": VariantConfig("scale_rl_llm_risk", use_llm_risk=True, use_llm_embedding=False,
                                       use_meta=False),
    "agentic_full": VariantConfig("agentic_full"),
    "abl_no_llm_score": VariantConfig("abl_no_llm_score", use_llm_risk=False),
    "abl_gate_one": VariantConfig("abl_gate_one", gate_fixed_one=True),
    "abl_no_meta": VariantConfig("abl_no_meta", use_meta=False),
}

BENCHMARKS = ["ppo", "scale_rl", "scale_rl_llm_risk", "agentic_full"]
ABLATIONS = ["abl_no_llm_score", "abl_gate_one", "abl_no_meta"]
