"""Multi-window robustness test — the honest re-test.

Takes ONE pre-chosen, parameter-free config (10-day formation, weekly hold) and runs it
on several non-overlapping time windows. Each window is an independent out-of-sample test
of the *same fixed rule* (nothing is fitted), so a real effect should show up repeatedly,
not just in the one window we happened to look at first.

Scoring is Sharpe-vs-ZERO with a standard error (Lo 2002), not vs buy-and-hold:
    per-period sharpe sr = mean/std over T obs
    SE(sr) = sqrt((1 + 0.5*sr^2) / T)
    t-stat = sr / SE(sr)             # |t| > ~2 ≈ significant
Annualized Sharpe = sr * sqrt(252).

    python -m simple_leadlag.robustness [--live] [--windows 5]
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from .config import SECTORS, all_tickers
from .data import load_prices
from .leadlag import sector_net_series

LOOKBACK, HOLD = 10, 5  # the config that looked best in the sweep
_PERIODS = 252


def portfolio_returns(prices: pd.DataFrame) -> pd.Series:
    """Equal-weight the 6 sector reversal books into one market-neutral daily series."""
    books = []
    for cfg in SECTORS.values():
        bench = str(cfg["benchmark"])
        stocks = [s for s in cfg["stocks"] if s in prices.columns]  # type: ignore[union-attr]
        if bench in prices.columns and len(stocks) >= 2:
            s = sector_net_series(prices, bench, stocks, LOOKBACK, HOLD)
            if not s.empty:
                books.append(s.rename(bench))
    if not books:
        return pd.Series(dtype=float)
    return pd.concat(books, axis=1).mean(axis=1).dropna()


def sharpe_stats(ret: pd.Series) -> tuple[float, float, float, int]:
    """Annualized Sharpe, its SE, t-stat vs 0, and obs count."""
    ret = ret[ret != 0].dropna()
    T = len(ret)
    if T < 30 or ret.std() == 0:
        return 0.0, 0.0, 0.0, T
    sr = ret.mean() / ret.std()
    se = math.sqrt((1 + 0.5 * sr**2) / T)
    return sr * math.sqrt(_PERIODS), se * math.sqrt(_PERIODS), sr / se, T


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-window robustness of the lead-lag book")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--period", default="10y")
    ap.add_argument("--windows", type=int, default=5)
    args = ap.parse_args()

    print(f"Loading prices ({'LIVE' if args.live else 'synthetic'})...")
    prices = load_prices(all_tickers(), period=args.period, live=args.live)
    print(f"config under test: {LOOKBACK}d formation, {HOLD}d hold (parameter-free)\n")

    chunks = np.array_split(np.arange(len(prices)), args.windows)
    print(f"{'window':<22}{'ann Sharpe':>12}{'+/- SE':>10}{'t-stat':>9}{'days':>7}")
    print("-" * 60)
    pooled = []
    pos_windows = 0
    for c in chunks:
        sub = prices.iloc[c[0] : c[-1] + 1]
        port = portfolio_returns(sub)
        ann, se, t, T = sharpe_stats(port)
        if T >= 30:
            pooled.append(port)
            pos_windows += ann > 0
            lo, hi = sub.index[0].date(), sub.index[-1].date()
            print(f"{str(lo)+'..'+str(hi):<22}{ann:>+12.2f}{se:>10.2f}{t:>+9.2f}{T:>7}")

    if pooled:
        allret = pd.concat(pooled)
        ann, se, t, T = sharpe_stats(allret)
        print("-" * 60)
        print(f"{'POOLED':<22}{ann:>+12.2f}{se:>10.2f}{t:>+9.2f}{T:>7}")
        print(f"\nwindows with positive Sharpe: {pos_windows}/{len(pooled)}")
        if abs(t) < 2 or pos_windows <= len(pooled) // 2:
            print("Not robust: inconsistent across windows and/or pooled |t| < 2.")
            print("This is noise, not edge. (The honest, expected outcome.)")
        else:
            print("Survives multiple windows with |t| > 2 — worth a serious, careful look")
            print("(check other costs, capacity, and a truly fresh period before believing).")


if __name__ == "__main__":
    main()
