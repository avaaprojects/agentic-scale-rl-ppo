"""Replaceable data loaders.

``load_csv_prices`` expects a wide CSV: a date column followed by one adjusted
close column per asset (e.g. exported from yfinance/CRSP). An optional text CSV
(columns: date, text) can be supplied as the corpus; each item is stamped with
the index of the trading day on or after its date, and is therefore usable only
from the NEXT step on (see dataset.MarketDataset).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .synthetic import TextItem


def load_csv_prices(path: str, text_csv: Optional[str] = None):
    df = pd.read_csv(path)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col).dropna(how="any")
    prices = df.to_numpy(dtype=float)
    y = prices[1:] / prices[:-1] - 1.0      # y[t] = simple return over day t
    dates = df.index[1:]
    names = list(df.columns)
    corpus: list[TextItem] = []
    if text_csv:
        tdf = pd.read_csv(text_csv)
        tdf["date"] = pd.to_datetime(tdf["date"])
        idx = np.searchsorted(dates.values, tdf["date"].values, side="left")
        for i, txt in zip(idx, tdf["text"].astype(str)):
            if i < len(dates):
                corpus.append(TextItem(int(i), txt))
        corpus.sort(key=lambda it: it.day)
    return dates, names, y, corpus, {}
