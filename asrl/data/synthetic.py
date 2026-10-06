"""Synthetic multi-asset market with a time-stamped text corpus.

APPROXIMATION: used because real market data and news could not be downloaded
in the build environment. Replace via ``asrl.data.loaders`` (CSV) for real data.

Design (all causal; text is stamped with the day it is published):
* Two-regime Markov market factor (calm / stressed) with different drift and vol.
* Per-asset beta, idiosyncratic vol, and a slowly varying AR(1) alpha so that
  momentum-type features carry a weak, learnable signal.
* Scheduled "events" (e.g. central-bank meetings). Most are ANNOUNCED k days in
  advance by a text item; some are surprises; some announcements are benign.
  A real event produces a market shock and raises the chance of a stressed regime.
* Daily routine text whose tone reflects that day's realised conditions; it is
  published at the end of the day, so it is only usable from the next step on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TextItem:
    day: int        # integer data index of publication (end of day)
    text: str


CALM_PHRASES = [
    "markets steady as earnings come in line with expectations",
    "investors calm and volumes light in a quiet session",
    "stocks edge higher on stable growth outlook",
    "credit spreads tight and liquidity healthy",
]
STRESS_PHRASES = [
    "volatility jumps as investors fear a broad selloff",
    "risk aversion rises amid liquidity concerns and heavy selling",
    "markets slide on recession fears and widening credit spreads",
    "panic selling hits equities as uncertainty spikes",
]
ANNOUNCE_TEMPLATES = [
    "central bank policy meeting scheduled in {k} days, markets brace for a rate decision",
    "major regulatory ruling expected in {k} days, analysts warn of uncertainty",
    "key inflation report due in {k} days could trigger repricing",
]


def generate_synthetic_market(n_assets: int = 8, n_days: int = 2520, seed: int = 1234,
                              start_date: str = "2012-01-02"):
    rng = np.random.default_rng(seed)
    # regime process
    p_stay = {0: 0.985, 1: 0.95}
    mkt_mu = {0: 0.0008, 1: -0.0008}
    mkt_sd = {0: 0.008, 1: 0.022}
    beta = rng.uniform(0.5, 1.5, n_assets)
    idio_sd = rng.uniform(0.006, 0.015, n_assets)
    base_alpha = rng.normal(0.0002, 0.0002, n_assets)
    ar_phi, ar_sd = 0.99, 0.00007          # stationary sd ~ 0.0005
    alpha_t = np.zeros(n_assets)

    # events: roughly one every 40 days
    event_days = np.sort(rng.choice(np.arange(30, n_days - 1), size=max(1, n_days // 40), replace=False))
    events = {}
    corpus: list[TextItem] = []
    for d in event_days:
        real = rng.random() < 0.75                 # 25% of announcements turn out benign
        announced = rng.random() < 0.8             # 20% of real events are surprises
        if not real:
            announced = True
        events[int(d)] = real
        if announced:
            k = int(rng.integers(2, 11))
            a = int(d) - k
            if a >= 0:
                tmpl = ANNOUNCE_TEMPLATES[int(rng.integers(len(ANNOUNCE_TEMPLATES)))]
                corpus.append(TextItem(a, tmpl.format(k=k)))

    regime = 0
    regimes = np.zeros(n_days, dtype=int)
    y = np.zeros((n_days, n_assets))
    for t in range(n_days):
        if rng.random() > p_stay[regime]:
            regime = 1 - regime
        shock = 0.0
        if events.get(t, False):
            shock = rng.normal(-0.006, 0.025)
            if rng.random() < 0.5:
                regime = 1
        regimes[t] = regime
        alpha_t = ar_phi * alpha_t + rng.normal(0.0, ar_sd, n_assets)
        f = rng.normal(mkt_mu[regime], mkt_sd[regime]) + shock
        idio_scale = 1.0 if regime == 0 else 1.8
        eps = rng.standard_t(df=5, size=n_assets) * idio_sd * idio_scale / np.sqrt(5 / 3)
        y[t] = base_alpha + alpha_t + beta * f + eps
        # routine end-of-day text reflecting today's realised tone
        if rng.random() < 0.7:
            stressed_today = regime == 1 or abs(f) > 0.025
            noisy = rng.random() < 0.15            # 15% of items have the "wrong" tone
            pool = STRESS_PHRASES if (stressed_today ^ noisy) else CALM_PHRASES
            corpus.append(TextItem(t, pool[int(rng.integers(len(pool)))]))

    y = np.clip(y, -0.5, 0.5)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    names = [f"SYN{i:02d}" for i in range(n_assets)]
    corpus.sort(key=lambda it: it.day)
    meta = {"regimes": regimes, "event_days": event_days, "beta": beta}
    return dates, names, y, corpus, meta
