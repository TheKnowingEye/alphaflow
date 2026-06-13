"""Price loading: live yfinance, or a deterministic synthetic fallback (offline-safe).

Synthetic prices have NO injected lead-lag (stocks just co-move with a sector factor),
so the honest backtest verdict on synthetic is ~0 edge. Real lead-lag, if it exists at
all, only shows up on `--live` data. That's the point.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd

_TRADING = 252


def synthetic_prices(tickers: list[str], days: int = 750, seed: int = 42) -> pd.DataFrame:
    """Sector-factor GBM, no lead-lag. Deterministic. Returns a close-price DataFrame."""
    rng = random.Random(seed)
    idx = pd.bdate_range(end="2026-06-12", periods=days)
    factor = [0.011 * rng.gauss(0, 1) for _ in range(days)]
    cols = {}
    for t in tickers:
        r = random.Random(f"{seed}:{t}")
        beta = r.uniform(0.8, 1.3)
        idio = r.uniform(0.006, 0.016)
        drift = r.uniform(0.02, 0.14)
        px, price = [], r.uniform(40, 400)
        for f in factor:
            price *= math.exp(drift / _TRADING + beta * f + idio * r.gauss(0, 1))
            px.append(round(price, 4))
        cols[t] = px
    return pd.DataFrame(cols, index=idx)


def load_prices(tickers: list[str], period: str = "2y", live: bool = False) -> pd.DataFrame:
    """Close prices, columns=tickers, aligned on common dates. Live -> yfinance."""
    if not live:
        return synthetic_prices(tickers)
    import yfinance as yf

    df = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False)
    close = df["Close"] if "Close" in df.columns.get_level_values(0) else df
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close.dropna(how="all")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))
