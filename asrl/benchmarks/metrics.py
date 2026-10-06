"""Backtest performance metrics (daily returns, 252 trading days per year, rf = 0)."""
from __future__ import annotations

import numpy as np

ANN = 252


def sharpe_ratio(r: np.ndarray) -> float:
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else 0.0


def max_drawdown(r: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return float((wealth / peak - 1.0).min()) if len(r) else 0.0


def performance_metrics(r: np.ndarray) -> dict:
    r = np.asarray(r, dtype=float)
    cum = float(np.prod(1.0 + r) - 1.0)
    n_years = len(r) / ANN
    ann_ret = float((1.0 + cum) ** (1.0 / n_years) - 1.0) if n_years > 0 and cum > -1 else float("nan")
    vol = float(r.std(ddof=1) * np.sqrt(ANN)) if len(r) > 1 else 0.0
    downside = r[r < 0]
    dsd = float(np.sqrt((downside ** 2).sum() / len(r)) * np.sqrt(ANN)) if len(r) else 0.0
    mdd = max_drawdown(r)
    return {
        "sharpe": sharpe_ratio(r),
        "cumulative_return": cum,
        "annual_return": ann_ret,
        "annual_volatility": vol,
        "max_drawdown": mdd,
        "sortino": float(r.mean() * ANN / dsd) if dsd > 0 else 0.0,
        "calmar": float(ann_ret / abs(mdd)) if mdd < 0 else 0.0,
        "n_days": int(len(r)),
    }
