"""Run the lead-lag study across all sectors and print an honest report.

python -m simple_leadlag.run            # synthetic (offline, deterministic)
python -m simple_leadlag.run --live     # real data via yfinance
"""

from __future__ import annotations

import argparse

from .config import SECTORS, all_tickers
from .data import load_prices, log_returns
from .leadlag import backtest_sector, lead_lag_corr, verdict


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple stock-vs-sector lead-lag study")
    ap.add_argument("--live", action="store_true", help="real data via yfinance")
    ap.add_argument("--period", default="2y", help="yfinance history window (live only)")
    args = ap.parse_args()

    print(f"Loading prices ({'LIVE yfinance' if args.live else 'synthetic'})...")
    prices = load_prices(all_tickers(), period=args.period, live=args.live)
    rets = log_returns(prices)

    agg_strat, agg_base = [], []
    for sector, cfg in SECTORS.items():
        bench = str(cfg["benchmark"])
        stocks = [s for s in cfg["stocks"] if s in prices.columns]  # type: ignore[union-attr]
        if bench not in prices.columns or len(stocks) < 2:
            print(f"\n[{sector}] skipped (missing data)")
            continue

        print(f"\n=== {sector}  (benchmark {bench}) ===")
        print("  lead-lag diagnosis (corr stock_t vs bench_t-1 / t+1):")
        for s in stocks:
            c = lead_lag_corr(rets[s], rets[bench])
            print(
                f"    {s:6s} lag+1={c.get(1, 0):+.3f}  lag-1={c.get(-1, 0):+.3f}  -> {verdict(c)}"
            )

        bt = backtest_sector(prices, bench, stocks)
        st, ba = bt["strategy"], bt["baseline"]
        agg_strat.append(st["sharpe"])
        agg_base.append(ba["sharpe"])
        print(f"  OOS backtest ({bt['n_oos']} days, net of costs):")
        print(
            f"    strategy  sharpe={st['sharpe']:+.2f}  ann_ret={st['ann_return']:+.1%}  "
            f"hit={st['hit_rate']:.1%}  turnover/day={bt['avg_daily_turnover']:.2f}"
        )
        print(f"    buy&hold  sharpe={ba['sharpe']:+.2f}  ann_ret={ba['ann_return']:+.1%}")

    if agg_strat:
        ms = sum(agg_strat) / len(agg_strat)
        mb = sum(agg_base) / len(agg_base)
        print("\n--- VERDICT ---")
        print(f"mean OOS Sharpe: strategy {ms:+.2f}  vs  buy&hold {mb:+.2f}")
        if ms <= mb + 0.2 or ms < 0.5:
            print("No convincing lead-lag edge after costs. (Expected. The market is efficient.)")
        else:
            print("Strategy beats buy&hold OOS here — be suspicious (likely a leak, not edge).")


if __name__ == "__main__":
    main()
