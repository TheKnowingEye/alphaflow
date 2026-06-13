# AlphaFlow Engine — SPEC

## Core Thesis
Cross-Asset Lead-Lag and Relative Strength Spread analysis. US hardware leaders
(NVDA, AMD) are the **leading** indicators; the global tech benchmark — Mirae Asset
Hang Seng TECH ETF (`MAHKTECH.NS`, sandbox-wrapped synthetic by default) — is the
**lagging** vector. Edge = leader move not yet priced into the lagging benchmark.

## Goal
Async FinTech pipeline. Predict **3-day forward residual alpha** of each leader vs the
benchmark with CatBoost. Emit a Pydantic v2-validated JSON signal payload
(target_action + allocation_weight + model_confidence_score).

## Architecture

```
alphaflow/
  config.py        Pydantic settings (tickers, benchmark, DB path, model params)
  models.py        Pydantic v2 strict models: PriceBar, FeatureRow, Signal, SignalBatch
  data_source.py   Market data: yfinance (live) or seeded synthetic GBM (default, offline-safe)
  fundamentals.py  Quarterly fundamentals + Point-In-Time forward-fill alignment
  ingestion.py     Async DB ingestion: aiosqlite, validated batch upsert
  features.py      SQL window-function features + EWMA beta (Python)
  model.py         CatBoost regressor + purged/embargo split: 3d forward residual alpha
  conformal.py     Split-conformal calibrator: distribution-free confidence + intervals
  signals.py       Signal generation: action + softmax allocation + conformal confidence
execution/
  backtester.py    Walk-forward OOS backtest + friction engine: BacktestMetrics
main.py            asyncio orchestrator (--backtest for walk-forward eval)
tests/             pytest suite
```

## Data Flow
1. **Ingest**: fetch OHLCV bars per ticker -> validate via `PriceBar` -> async upsert into `price_bars` (PK: ticker, date). Idempotent.
   - **Lead-lag alignment**: US leaders (NVDA/AMD) lead; HK benchmark lags one session.
     SQL `LEAD(bar_date)` maps each HK close to the next session, so leader day t joins
     the HK close from t-1. No same-calendar-day US/HK simultaneity (no lookahead); HK
     reaction to the leader move lands inside the forward target window.
2. **Features** (SQL `LAG`/`LEAD`/`AVG OVER` window functions over `price_bars`; EWMA beta in Python):
   - **Asset-to-ETF Price Spread** — `spread`: log(asset_close) - log(etf_close), plus
     `spread_z`: 20d z-score of spread (relative-strength dislocation)
   - **Fractional Differentiation** — `frac_diff_close`: FFD of log price (d≈`frac_diff_d`=0.5,
     fixed width `frac_diff_width`=50). Stationary while retaining long-memory support/
     resistance. Weights `w[k]=-w[k-1]*(d-k+1)/k` (power-law decay). Python (not windowable).
   - **GARCH(1,1) Volatility** — `garch_vol`: 1-day-ahead conditional vol forecast (zero-mean
     GARCH(1,1) via `arch`, `conditional_volatility[t]` = forecast from info ≤ t-1).
     `garch_vol_ratio`: asset garch_vol / benchmark garch_vol. **Replaces rolling stddev.**
   - **Momentum Vector** — `mom_5d / mom_10d / mom_20d`: cumulative log returns
   - **Fundamentals (PIT-aligned)** — `trailing_pe`, `rev_growth_quarterly`: quarterly
     metrics keyed by announce date, forward-filled onto daily rows from the announcement
     forward (no lookahead). Synthetic default; `--live` -> yfinance `.info`.
   - `log_ret_1d`: 1-day log return
   - `beta_60d`: **EWMA** beta = EWMA cov(asset, etf) / EWMA var(etf), span = `beta_window`,
     recurrence `S_t=(1-a)S_{t-1}+a*x_t`, a=2/(span+1). Computed in Python over the full
     per-ticker series (EWMA is recursive) -> dampens noise vs raw rolling cov/var.
   - **Target** `fwd_alpha_3d` = (asset 3d fwd log ret) - beta_60d * (etf 3d fwd log ret)
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
