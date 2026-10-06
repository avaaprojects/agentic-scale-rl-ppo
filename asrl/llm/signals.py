"""Offline LLM signal layer (DOCX lines 2-4):

    e_t   <- Pool(M_enc(T_<t))
    ell_t <- g(M_gen(P_risk, T_<t))      text timestamped strictly before t
    cache {e_t, ell_t}; frozen thereafter

``SignalProvider`` is the replaceable interface (a real LLM provider implements
``encode`` and ``generate``). ``MockLLMProvider`` is a deterministic, offline
stand-in (APPROXIMATION): a hashed bag-of-words encoder and a keyword/announcement
reader that writes a verbalised probability. It is deterministic like a frozen,
temperature-0 model, but it is NOT an LLM.

APPROXIMATION: the provider sees only items with t - lookback <= day < t
(a bounded context window) instead of the full history T_<t.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from ..data.synthetic import TextItem
from ..utils import sha256_of, stable_hash_int

P_RISK = (
    "You are a financial risk analyst. Read the dated news items below, all published "
    "before the current trading day. Estimate the probability (0 to 1) that a major "
    "market-moving event or regime shift occurs in the next 5 trading days. "
    "Answer exactly as: 'probability: <number>'."
)


class SignalProvider(Protocol):
    name: str
    version: str
    embedding_dim: int

    def encode(self, items: Sequence[TextItem], t: int) -> np.ndarray:
        """Pooled embedding of the items (all with day < t). Shape [embedding_dim]."""

    def generate(self, prompt: str, items: Sequence[TextItem], t: int) -> str:
        """Raw generative output for the risk prompt."""


def g_verbalised_probability(text: str) -> float:
    """g(.): map model output to [0, 1]. Parses the first number after 'probability'
    (or the first number); percentages are divided by 100. Unparseable -> 0.5."""
    m = re.search(r"probability\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*(%?)", text, re.I) or \
        re.search(r"([0-9]*\.?[0-9]+)\s*(%?)", text)
    if not m:
        return 0.5
    v = float(m.group(1))
    if m.group(2) == "%" or v > 1.0:
        v = v / 100.0
    return float(np.clip(v, 0.0, 1.0))


_STRESS = {"volatility", "selloff", "fear", "fears", "panic", "slide", "concerns", "risk",
           "recession", "widening", "uncertainty", "selling", "aversion", "spikes", "jumps"}
_CALM = {"steady", "calm", "stable", "healthy", "light", "higher", "quiet", "tight"}
_WS = re.compile(r"[a-z]+|\d+")


class MockLLMProvider:
    name = "mock-keyword-hash"
    version = "1.0"

    def __init__(self, embedding_dim: int = 16, horizon: int = 5):
        self.embedding_dim = embedding_dim
        self.horizon = horizon

    def _tokens(self, text: str):
        return _WS.findall(text.lower())

    def encode(self, items, t):
        out = np.zeros(self.embedding_dim)
        if not items:
            return out
        for it in items:
            v = np.zeros(self.embedding_dim)
            for tok in self._tokens(it.text):
                h = stable_hash_int(tok)
                v[h % self.embedding_dim] += 1.0 if (h >> 32) & 1 else -1.0
            n = np.linalg.norm(v)
            out += v / n if n > 0 else v
        out /= len(items)                                    # mean pooling
        return out

    def generate(self, prompt, items, t):
        score = -2.0
        for it in items:
            toks = self._tokens(it.text)
            age = t - it.day                                  # >= 1 by construction
            m = re.search(r"in (\d+) days", it.text)
            if m:                                             # scheduled announcement
                event_day = it.day + int(m.group(1))
                if 0 <= event_day - t <= self.horizon:
                    score += 2.5
                continue
            w = 0.6 ** (age - 1)                             # recency weighting
            score += w * (0.8 * sum(tok in _STRESS for tok in toks) - 0.5 * sum(tok in _CALM for tok in toks))
        p = 1.0 / (1.0 + np.exp(-score))
        return f"probability: {p:.3f}"


def items_before(corpus: Sequence[TextItem], t: int, lookback: int) -> list:
    """T_<t restricted to [t - lookback, t): strictly before step t."""
    return [it for it in corpus if t - lookback <= it.day < t]


def build_signals(corpus: Sequence[TextItem], n_steps: int, provider, lookback: int,
                  prompt: str = P_RISK):
    """Run once, offline. Returns (E [n_steps, d_e], ell [n_steps], raw_outputs)."""
    days = np.array([it.day for it in corpus]) if corpus else np.zeros(0, dtype=int)
    order = np.argsort(days, kind="stable")
    corpus = [corpus[i] for i in order]
    days = days[order]
    E = np.zeros((n_steps, provider.embedding_dim))
    ell = np.zeros(n_steps)
    raw = []
    for t in range(n_steps):
        lo = np.searchsorted(days, t - lookback, side="left")
        hi = np.searchsorted(days, t, side="left")           # day < t (strict)
        items = corpus[lo:hi]
        E[t] = provider.encode(items, t)
        out = provider.generate(prompt, items, t)
        raw.append(out)
        ell[t] = g_verbalised_probability(out)
    return E, ell, raw


def load_or_build_signals(cache_dir: str, corpus, n_steps: int, provider, lookback: int,
                          prompt: str = P_RISK):
    """Content-addressed cache: any change to corpus, provider, prompt or window -> new key.
    Signals are frozen once written; training only ever reads them."""
    key = sha256_of([(it.day, it.text) for it in corpus], provider.name, provider.version,
                    provider.embedding_dim, prompt, lookback, n_steps)[:16]
    path = Path(cache_dir) / f"llm_signals_{key}.npz"
    if path.exists():
        z = np.load(path)
        return z["E"], z["ell"], {"cache_path": str(path), "cache_key": key, "from_cache": True}
    E, ell, _ = build_signals(corpus, n_steps, provider, lookback, prompt)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, E=E, ell=ell)
    return E, ell, {"cache_path": str(path), "cache_key": key, "from_cache": False}
