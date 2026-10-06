"""MarketDataset: returns, causal features, chronological split, turbulence.

Timing convention (used everywhere in the code):
* ``y[d]`` is the vector of asset returns realised over data day d.
* At step d the agent acts using information up to the END of day d-1:
  market features ``F[d]`` use y[:d]; ``turb[d]`` uses y[:d]; LLM signals for
  step d use text published on days < d (DOCX: "text timestamped strictly before t").
* The portfolio return r_d = w_d . y[d] - costs is observed after acting.

Split is strictly chronological: train = [0, n_train), val = [n_train, n_train+n_val),
test = [n_train+n_val, n_days). Feature normalisation statistics are computed on
the training range only. Validation/test evaluation may *warm up* on earlier
data (past information), never on later data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import DataConfig
from ..scale_rl.turbulence import turbulence_series
from .loaders import load_csv_prices
from .synthetic import TextItem, generate_synthetic_market


def market_features(y: np.ndarray, lookback: int, vol_window: int) -> np.ndarray:
    """F[d] built from y[:d] only. Rows with insufficient history are zero.
    Layout per row: [y[d-lookback..d-1] flattened (lookback*n), rolling mean (n), rolling std (n)]."""
    T, n = y.shape
    F = np.zeros((T, lookback * n + 2 * n))
    start = max(lookback, vol_window)
    for d in range(start, T):
        lag = y[d - lookback:d][::-1].reshape(-1)          # most recent first
        win = y[d - vol_window:d]
        F[d] = np.concatenate([lag, win.mean(0), win.std(0)])
    return F


@dataclass
class Split:
    train: tuple
    val: tuple
    test: tuple


@dataclass
class MarketDataset:
    dates: object
    names: list
    y: np.ndarray                    # [n_days, n_assets]
    corpus: list                     # list[TextItem]
    features: np.ndarray             # normalised F[d]
    turb: np.ndarray                 # Turb[d]
    turb_valid: np.ndarray
    split: Split
    first_valid: int                 # first index with full features and turbulence
    meta: dict = field(default_factory=dict)

    @property
    def n_assets(self) -> int:
        return self.y.shape[1]

    @property
    def n_days(self) -> int:
        return self.y.shape[0]

    @property
    def feat_dim(self) -> int:
        return self.features.shape[1]

    def describe(self) -> dict:
        s = self.split
        return {
            "n_assets": self.n_assets, "n_days": self.n_days, "assets": self.names,
            "train": [str(self.dates[s.train[0]].date()), str(self.dates[s.train[1] - 1].date()), s.train],
            "val": [str(self.dates[s.val[0]].date()), str(self.dates[s.val[1] - 1].date()), s.val],
            "test": [str(self.dates[s.test[0]].date()), str(self.dates[s.test[1] - 1].date()), s.test],
            "n_text_items": len(self.corpus), "first_valid_index": self.first_valid,
        }


def chronological_split(n_days: int, train_frac: float, val_frac: float) -> Split:
    n_train = int(round(n_days * train_frac))
    n_val = int(round(n_days * val_frac))
    assert n_train > 0 and n_val > 0 and n_train + n_val < n_days
    return Split((0, n_train), (n_train, n_train + n_val), (n_train + n_val, n_days))


def build_dataset(cfg: DataConfig, W: int, turb_include_current: bool = False) -> MarketDataset:
    if cfg.source == "synthetic":
        dates, names, y, corpus, meta = generate_synthetic_market(
            cfg.n_assets, cfg.n_days, cfg.data_seed, cfg.start_date)
    elif cfg.source == "csv":
        assert cfg.csv_path, "DataConfig.csv_path required for source='csv'"
        dates, names, y, corpus, meta = load_csv_prices(cfg.csv_path)
    else:
        raise ValueError(cfg.source)

    split = chronological_split(len(y), cfg.train_frac, cfg.val_frac)
    raw = market_features(y, cfg.lookback, cfg.vol_window)
    first_feat = max(cfg.lookback, cfg.vol_window)
    tr0, tr1 = split.train
    ref = raw[max(first_feat, tr0):tr1]                     # TRAIN-ONLY statistics
    mu, sd = ref.mean(0), ref.std(0) + 1e-8
    feats = (raw - mu) / sd
    feats[:first_feat] = 0.0
    turb, valid = turbulence_series(y, W, turb_include_current)
    first_valid = max(first_feat, int(np.argmax(valid)))
    return MarketDataset(dates, names, y, corpus, feats.astype(np.float64), turb, valid,
                         split, first_valid, meta)
