# AlphaFlow Engine — SPEC

## Core Thesis
**Global Multi-Factor** residual-alpha engine across **US + India**, 6 sectors (Tech,
Finance, Energy, Materials, FMCG, Pharma). Each holding's idiosyncratic move is measured
vs its regional **macro** benchmark (market factor) and **sector** benchmark (relative
strength). Edge = predicted 3-day forward residual alpha vs the macro factor.

## Dual-Region Sector Registry (`config.py`)
`GLOBAL_REGISTRY[REGION][SECTOR] = {macro_benchmark, sector_benchmark, fundamental_keys}`.
- US macro `SPY`; sectors -> SPDR ETFs (XLK/XLF/XLE/XLB/XLP/XLV).
- India macro `^NSEI`; sectors -> Nifty indices (^CNXIT/^NSEBANK/^CNXENERGY/^CNXMETAL/
  ^CNXFMCG/^CNXPHARMA).
- `fundamental_keys` per sector (e.g. Tech=trailingPE+revenueGrowth, Finance=priceToBook+
  trailingPE). **Auto-router**: ticker `.NS`/`.BO` -> IN, else US (`resolve_holding`).

## Goal
Predict **3-day forward residual alpha** (vs macro benchmark) per holding with CatBoost.
Emit a Pydantic v2-validated JSON signal payload (target_action + allocation_weight +
model_confidence_score).

## Architecture

```
alphaflow/
  config.py        Settings + GLOBAL_REGISTRY + auto-router (Holding, resolve_holding)
  models.py        Pydantic v2 strict models: PriceBar, FundamentalRecord, FeatureRow, Signal
  data_source.py   Prices: yfinance (live, per-ticker) or synthetic GBM w/ per-region factor
  fundamentals.py  Heterogeneous fundamentals + Point-In-Time forward-fill alignment
  ingestion.py     Async DB: price_bars + fundamentals JSON column (schema fluidity)
  features.py      Per-holding dual-spread SQL + EWMA beta / FFD / GARCH (Python)
  model.py         CatBoost regressor + purged/embargo split: 3d forward residual alpha
  conformal.py     Split-conformal calibrator: distribution-free confidence + intervals
  signals.py       Signal generation: action + softmax allocation + conformal confidence
execution/
  backtester.py    Walk-forward OOS backtest + friction engine: BacktestMetrics
main.py            asyncio orchestrator (--backtest walk-forward, --live yfinance)
tests/             pytest suite
```

## Data Flow
1. **Ingest** (auto-routed per holding): fetch OHLCV for the asset + its macro & sector
   benchmarks -> `PriceBar` -> upsert `price_bars`. Fundamentals -> JSON `fundamentals`
   table (heterogeneous keys per sector, no column churn). Idempotent.
   - **Benchmark lag**: SQL `LEAD(bar_date)` maps each benchmark close to the next session
     so the asset on day t joins the benchmark from t-1 (no same-session lookahead margin).
2. **Features** (per-holding SQL window pass + Python layers):
   - **Dual spreads** — `macro_spread` = log(asset) - log(macro_bench); `sector_spread` =
     log(asset) - log(sector_bench); `spread_z` = 20d z-score of sector_spread.
   - **Fractional Differentiation** — `frac_diff_close`: FFD of log price (d≈0.5, width 50);
     stationary w/ long memory. Weights `w[k]=-w[k-1]*(d-k+1)/k`. Python (not windowable).
   - **GARCH(1,1) Volatility** — `garch_vol`: 1d-ahead conditional vol (zero-mean GARCH via
     `arch`). `garch_vol_ratio`: asset / macro-benchmark garch_vol. Replaces rolling stddev.
   - **Momentum** — `mom_5d / mom_10d / mom_20d`.
   - **Region/Sector** — `region`/`sector` strings + `region_id`/`sector_id` numeric codes
     (so one pooled model generalises across the grid).
   - **Fundamentals (PIT)** — `fund_0`/`fund_1` = the sector's two `fundamental_keys`,
     forward-filled from announce date (no lookahead). Heterogeneous semantics, fixed slots.
   - `log_ret_1d`; `beta_60d`: **EWMA** beta vs macro benchmark (recurrence, span=beta_window).
   - **Target** `fwd_alpha_3d` = (asset 3d fwd log ret) - beta_60d * (macro 3d fwd log ret).
3. **Train**: CatBoostRegressor (RMSE) on `fwd_alpha_3d`. **Purged + embargo**
   chronological split (`purged_embargo_split`): val = chronological tail; purge the
   `forward_horizon` train rows whose forward labels overlap val; embargo `forward_horizon`
   rows after val (no-op for tail). Guarantees >= horizon gap -> zero label leakage.
   Report val RMSE + purged count.
4. **Signals**: latest-date predictions ->
   - `target_action`: BUY if pred_alpha > +threshold; SELL if < -threshold; else HOLD
   - `allocation_weight`: softmax over positive predicted alphas; non-BUY weight 0;
     sum(weights) <= 1.0 enforced by Pydantic validator
   - `model_confidence_score`: **conformal** empirical CDF of |alpha| over a sliding
     window of OOS residuals (distribution-free, replaces the `erf` heuristic)
   - `alpha_ci_low/high`: conformal prediction interval = predicted_alpha ± empirical
     `conformal_level` quantile of |residual|
   - output: `signals.json` validated through `SignalBatch`
5. **Backtest** (`execution/backtester.py`, `python main.py --backtest`): rolling
   walk-forward. Per rebalance (stride `bt_step`): purge train rows whose label isn't
   realised before the decision date, retrain CatBoost every `bt_retrain_every` folds,
   size a long/short book from predicted alpha. **Friction engine**: a name must clear a
   round-trip cost (2*`txn_cost_per_side`, 5 bps default) to trade; returns net of
   turnover*cost. OOS metrics -> strict `BacktestMetrics`: Sharpe, hit-rate, max drawdown,
   annualised turnover, total return. Output: `backtest_metrics.json`.

## Models (Pydantic v2, strict)
- `PriceBar`: ticker, date, open/high/low/close (>0), volume (>=0); high >= low invariant
- `FeatureRow`: ticker, date, technical + fundamental feature floats (NaN-free post-filter),
  target optional
- `FundamentalRecord`: ticker, announce_date (PIT boundary), trailing_pe (>0),
  rev_growth_quarterly
- `Signal` (output contract): `timestamp`, `asset_ticker`, `target_action` (enum
  BUY/HOLD/SELL), `allocation_weight` [0,1], `model_confidence_score` [0,1]; plus
  `as_of_date`, `predicted_alpha`, conformal `alpha_ci_low/high` for traceability.
  Validator: non-BUY -> zero weight
- `SignalBatch`: model_version, generated_at, benchmark, signals list; validator:
  sum(weights) <= 1.0 + tolerance
- `BacktestMetrics` (strict): n_rebalances, n_trades, sharpe, hit_rate, max_drawdown,
  annual_turnover, total_return, txn_cost_per_side

## Conventions
- Python 3.11+ (venv 3.12.10), black + flake8 (max-line-length 100)
- Synthetic data default: seeded GBM with common sector factor -> reproducible runs/tests
- `python main.py` = full pipeline; `--live` = yfinance data; `--backtest` = walk-forward OOS

## Validation Checkpoints
1. Deps import smoke test
2. `pytest tests/test_models.py` — strict validation rejects bad rows
3. Ingestion idempotency — re-ingest same rows -> no dupes
4. SQL feature calcs cross-checked vs pandas in `tests/test_features.py`
5. CatBoost trains, val RMSE finite
6. `signals.json` round-trips through `SignalBatch`; weight sum <= 1
7. `python main.py` end-to-end + `pytest` + `black . && flake8` green
