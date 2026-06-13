"""Lead-lag diagnostics + a simple, leak-free relative-strength reversal backtest.

Diagnostic: cross-correlation of a stock's return with the benchmark's return at
shifted lags. corr(stock_t, bench_{t-1}) > corr(stock_t, bench_{t+1}) => the stock
tends to LAG the benchmark (benchmark leads).

Strategy (parameter-free, nothing fitted -> nothing to overfit):
  spread_t       = cumulative (stock - benchmark) return over LOOKBACK days
  signal_t       = -zscore(spread)         # a laggard (negative spread) -> go long
  position uses ONLY data through t-1 (everything .shift(1)) -> no look-ahead
  pnl_t          = position_{t-1} * stock_return_t  - turnover * cost

Reported metrics are the SECOND half (out-of-sample); the first half is never used to
tune anything (there is nothing to tune) — the split just keeps reporting honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COST_PER_SIDE, LOOKBACK, TRAIN_FRAC, Z_WINDOW
from .data import log_returns


def lead_lag_corr(stock_ret: pd.Series, bench_ret: pd.Series, max_lag: int = 2) -> dict[int, float]:
    """corr(stock_t, bench_{t-lag}). lag>0 => benchmark leads (stock lags)."""
    return {
        lag: float(stock_ret.corr(bench_ret.shift(lag))) for lag in range(-max_lag, max_lag + 1)
    }


def verdict(corrs: dict[int, float], tol: float = 0.02) -> str:
    leads = corrs.get(-1, 0.0)  # stock leads benchmark
    lags = corrs.get(1, 0.0)  # stock lags benchmark
    if lags - leads > tol:
        return "LAGS"
    if leads - lags > tol:
        return "LEADS"
    return "~same"


def _metrics(ret: pd.Series, periods: int = 252) -> dict[str, float]:
    ret = ret.dropna()
    if len(ret) < 2 or ret.std() == 0:
        return {"sharpe": 0.0, "ann_return": 0.0, "hit_rate": 0.0}
    traded = ret[ret != 0]
    return {
        "sharpe": float(ret.mean() / ret.std() * np.sqrt(periods)),
        "ann_return": float(np.expm1(ret.mean() * periods)),
        "hit_rate": float((traded > 0).mean()) if len(traded) else 0.0,
    }


def backtest_sector(prices: pd.DataFrame, benchmark: str, stocks: list[str]) -> dict:
    """Relative-strength reversal across a sector's stocks. Returns OOS metrics +
    a buy-and-hold baseline for the same names/period."""
    cols = [c for c in stocks if c in prices.columns] + [benchmark]
    px = prices[cols].dropna()
    r = log_returns(px).dropna()
    names = [c for c in stocks if c in px.columns]
    bench = r[benchmark]

    rel = r[names].sub(bench, axis=0)  # daily relative return
    spread = rel.rolling(LOOKBACK).sum()  # under/out-performance window
    z = (spread - spread.rolling(Z_WINDOW).mean()) / spread.rolling(Z_WINDOW).std()
    pos = (-np.sign(z)).shift(1).fillna(0.0)  # long laggards; decided at t-1

    strat = (pos * r[names]).mean(axis=1)  # equal-weight book
    turn = pos.diff().abs().mean(axis=1).fillna(0.0)
    net = strat - turn * COST_PER_SIDE
    baseline = r[names].mean(axis=1)  # equal-weight buy & hold

    cut = int(len(net) * TRAIN_FRAC)
    return {
        "n_oos": len(net) - cut,
        "strategy": _metrics(net.iloc[cut:]),
        "baseline": _metrics(baseline.iloc[cut:]),
        "avg_daily_turnover": float(turn.iloc[cut:].mean()),
    }
