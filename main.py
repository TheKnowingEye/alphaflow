"""AlphaFlow Engine: global multi-factor pipeline.

ingest (prices + JSON fundamentals) -> features (dual-spread) -> train -> signals.
`--backtest` runs walk-forward OOS; `--live` pulls yfinance via the auto-router.
"""

import argparse
import asyncio

from alphaflow.conformal import ConformalCalibrator
from alphaflow.config import SETTINGS, Holding
from alphaflow.data_source import fetch_prices
from alphaflow.features import compute_features
from alphaflow.fundamentals import fetch_fundamentals
from alphaflow.ingestion import (
    bar_count,
    ingest_bars,
    ingest_fundamentals,
    load_fundamentals,
    open_db,
)
from alphaflow.model import predict_latest, train
from alphaflow.signals import build_signals, write_signals
from execution.backtester import walk_forward


def _universe_tickers(holdings: tuple[Holding, ...]) -> tuple[tuple[str, ...], frozenset[str]]:
    assets = [h.ticker for h in holdings]
    benches: set[str] = set()
    for h in holdings:
        benches.add(h.macro_benchmark)
        benches.add(h.sector_benchmark)
    tickers = tuple(dict.fromkeys(assets + sorted(benches)))  # assets first, distinct
    return tickers, frozenset(benches)


async def _load_features(conn, s, live: bool):
    holdings = s.holdings()
    for h in holdings:
        print(
            f"[route] {h.ticker:11s} -> {h.region}/{h.sector} "
            f"macro={h.macro_benchmark} sector={h.sector_benchmark}"
        )
    tickers, benches = _universe_tickers(holdings)

    prices = await fetch_prices(
        tickers, s.history_days, s.synthetic_seed, live=live, low_idio=benches
    )
    for ticker, bars in prices.items():
        await ingest_bars(conn, bars)
        print(f"[ingest] {ticker}: {await bar_count(conn, ticker)} bars")

    # Fundamentals -> JSON column -> reload (exercises DB fluidity).
    funds = await fetch_fundamentals(holdings, s.history_days, s.synthetic_seed, live=live)
    n_fund = 0
    for recs in funds.values():
        n_fund += await ingest_fundamentals(conn, recs)
    print(f"[fundamentals] {n_fund} PIT records (JSON)")
    loaded = {h.ticker: await load_fundamentals(conn, h.ticker) for h in holdings}

    rows = await compute_features(conn, s, holdings=holdings, fundamentals=loaded)
    labelled = sum(1 for r in rows if r.fwd_alpha_3d is not None)
    print(f"[features] {len(rows)} rows ({labelled} labelled)")
    return rows


async def run(live: bool, backtest: bool) -> None:
    s = SETTINGS
    conn = await open_db(s.db_path)
    try:
        rows = await _load_features(conn, s, live)

        if backtest:
            metrics = walk_forward(rows, s)
            s.metrics_path.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
            print(f"[backtest] -> {s.metrics_path}")
            print(
                f"  rebalances={metrics.n_rebalances} trades={metrics.n_trades} "
                f"sharpe={metrics.sharpe:+.2f} hit_rate={metrics.hit_rate:.3f}"
            )
            print(
                f"  max_dd={metrics.max_drawdown:.3f} ann_turnover={metrics.annual_turnover:.2f} "
                f"total_return={metrics.total_return:+.4f}"
            )
            return

        result = train(rows, s)
        print(
            f"[model] CatBoost trained: {result.n_train} train / {result.n_val} val "
            f"({result.n_purged} purged), val RMSE = {result.val_rmse:.6f}"
        )
        calibrator = ConformalCalibrator(list(result.val_residuals), window=s.conformal_window)
        preds = predict_latest(result.model, rows)
        batch = build_signals(preds, s, calibrator)
        write_signals(batch, s.signals_path)
        print(f"[signals] -> {s.signals_path}")
        for sig in batch.signals:
            print(
                f"  {sig.asset_ticker:11s} {sig.target_action.value:4s} "
                f"alpha={sig.predicted_alpha:+.5f} weight={sig.allocation_weight:.4f} "
                f"conf={sig.model_confidence_score:.3f}"
            )
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaFlow Engine")
    parser.add_argument("--live", action="store_true", help="fetch real data via yfinance")
    parser.add_argument("--backtest", action="store_true", help="run walk-forward OOS backtest")
    args = parser.parse_args()
    asyncio.run(run(live=args.live, backtest=args.backtest))


if __name__ == "__main__":
    main()
