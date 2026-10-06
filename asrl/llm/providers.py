"""Optional real-LLM adapters. Not used by the default pipeline (offline mock).

To plug in a real model for the SIGNAL layer, implement the ``SignalProvider``
protocol in ``signals.py`` (``encode`` and ``generate``). ``CallableSignalProvider``
does this from two functions, e.g. a sentence-embedding model for ``encode_fn``
and a temperature-0 chat model for ``generate_fn``.

For the META-AGENT, any ``llm_fn(prompt: str) -> str`` works; ``anthropic_llm_fn``
is a thin example (requires ``pip install anthropic`` and ANTHROPIC_API_KEY).

Leakage rule for real LLMs: the test window must start after the model's knowledge
cutoff (ExperimentConfig.llm_knowledge_cutoff), or entities/dates must be masked.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


class CallableSignalProvider:
    def __init__(self, name: str, version: str, embedding_dim: int,
                 encode_fn: Callable[[list], np.ndarray], generate_fn: Callable[[str], str]):
        self.name, self.version, self.embedding_dim = name, version, embedding_dim
        self.encode_fn, self.generate_fn = encode_fn, generate_fn

    def encode(self, items: Sequence, t: int) -> np.ndarray:
        if not items:
            return np.zeros(self.embedding_dim)
        vecs = np.asarray(self.encode_fn([it.text for it in items]), dtype=float)
        return vecs.mean(0)                                  # Pool = mean pooling

    def generate(self, prompt: str, items: Sequence, t: int) -> str:
        body = "\n".join(f"[day {it.day}] {it.text}" for it in items) or "(no news)"
        return self.generate_fn(f"{prompt}\n\nNews:\n{body}")


def anthropic_llm_fn(model: str, max_tokens: int = 300) -> Callable[[str], str]:
    import anthropic  # optional dependency
    client = anthropic.Anthropic()

    def fn(prompt: str) -> str:
        msg = client.messages.create(model=model, max_tokens=max_tokens, temperature=0,
                                     messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return fn
