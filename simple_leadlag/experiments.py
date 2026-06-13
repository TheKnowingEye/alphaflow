"""Pre-specified hypothesis sweep (small + honest).

Tests FOUR configs only. A bigger sweep would be data-dredging: try enough knobs and
one will look good by luck. Reported = mean OOS Sharpe across the 6 sectors, net of
costs, vs the buy-and-hold baseline.

    python -m simple_leadlag.experiments [--live]
"""

from __future__ import annotations

import argparse

from .config import SECTORS, all_tickers
from .data import load_prices
from .leadlag import backtest_sector

# (label, kwargs) — fixed in advance, not chosen after seeing results.
CONFIGS = [
    ("daily reversal (baseline)", dict(lookback=5, hold=1)),
    ("weekly hold", dict(lookback=5, hold=5)),
    ("10d formation, weekly hold", dict(lookback=10, hold=5)),
    ("weekly + lag-filter", dict(lookback=5, hold=5, direction_filter=True)),
]


def run_config(prices, kwargs) -> tuple[float, float, float]:
    strat, base, turn = [], [], []
    for cfg in SECTORS.values():
        bench = str(cfg["benchmark"])
        stocks = [s for s in cfg["stocks"] if s in prices.columns]  # type: ignore[union-attr]
        if bench not in prices.columns or len(stocks) < 2:
            continue
        bt = backtest_sector(prices, bench, stocks, **kwargs)
        if bt["n_oos"] == 0:
            continue
        strat.append(bt["strategy"]["sharpe"])
        base.append(bt["baseline"]["sharpe"])
        turn.append(bt["avg_daily_turnover"])
    n = max(len(strat), 1)
    return sum(strat) / n, sum(base) / n, sum(turn) / max(len(turn), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Lead-lag hypothesis sweep")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--period", default="2y")
    args = ap.parse_args()

    print(f"Loading prices ({'LIVE' if args.live else 'synthetic'})...")
    prices = load_prices(all_tickers(), period=args.period, live=args.live)

    print(f"\n{'config':<30}{'strat Sharpe':>14}{'buy&hold':>12}{'turnover/d':>12}")
    print("-" * 68)
    best = None
    for label, kwargs in CONFIGS:
        s, b, t = run_config(prices, kwargs)
        print(f"{label:<30}{s:>+14.2f}{b:>+12.2f}{t:>12.2f}")
        if best is None or s > best[1]:
            best = (label, s, b)

    print("-" * 68)
    label, s, b = best
    print(f"best config: '{label}'  (strat {s:+.2f} vs buy&hold {b:+.2f})")
    if s <= b + 0.2 or s < 0.5:
        print("Still no edge that beats just holding. Honest negative.")
    else:
        print("Looks interesting — but it's the best of 4 tries. Re-test on FRESH/held-out")
        print("data before believing it; one winner out of several is usually luck.")


if __name__ == "__main__":
    main()
