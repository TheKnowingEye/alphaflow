# AlphaFlow Engine

A global, multi-factor **quantitative research pipeline** — data ingestion → feature
engineering → ML model → trade signals → walk-forward backtest — across US and Indian
equities, 6 sectors, in async Python.

> **Honest disclaimer up front:** this is a *research scaffold*, not a profitable
> strategy. The model shows **no demonstrated trading edge**, and the backtest Sharpe is
> inflated by a known look-ahead leak (see [Known Limitations](#known-limitations)). It
> was built to learn how a real quant pipeline is structured end-to-end — and that part
> works. The "no edge" result is the honest, expected outcome, not a bug.

---

## What it does

| Stage | Module | What happens |
|---|---|---|
| **Route** | `config.py` | `GLOBAL_REGISTRY[region][sector]` + auto-router: a ticker's suffix (`.NS`/`.BO` → India, else US) picks its macro + sector benchmarks and fundamental keys. |
| **Ingest** | `data_source.py`, `ingestion.py` | OHLCV via yfinance (or seeded synthetic GBM) → SQLite. Fundamentals stored in a **JSON column** so heterogeneous keys (Bank `priceToBook` vs Tech `revenueGrowth`) coexist without schema churn. |
| **Features** | `features.py`, `fundamentals.py` | Per-holding SQL window pass + Python layers: dual spreads (vs macro & sector), EWMA beta, **fractional differentiation**, **GARCH(1,1)** volatility, momentum, and **Point-In-Time** fundamentals (forward-filled from announce date, no lookahead). |
| **Train** | `model.py` | CatBoost regressor predicts 3-day forward residual alpha. **Purged + embargo** chronological split to prevent label leakage. One pooled model; region/sector enter as numeric features. |
| **Signals** | `signals.py`, `conformal.py` | BUY/SELL/HOLD + softmax allocation + **conformal** (distribution-free) confidence & prediction intervals. Validated JSON output. |
| **Backtest** | `execution/backtester.py` | Rolling walk-forward, rolling retrain, **friction engine** (5 bps/side), long/short book. Reports Sharpe, hit-rate, max drawdown, turnover. |

## Stack
Python 3.11+ · SQLite (aiosqlite) · Pydantic v2 (strict) · CatBoost · `arch` (GARCH) ·
pandas/numpy · pytest. Synthetic data is the deterministic default → fully offline & reproducible.

## Run

```bash
pip install -r requirements.txt

python main.py                       # 2-ticker demo (synthetic), emits signals.json
python main.py --backtest            # walk-forward backtest
python main.py --grid                # full 60-ticker dual-region sector grid
python main.py --grid --live         # real data via yfinance (slow; some IN tickers may be sparse)
python main.py --grid --live --backtest

pytest                               # 47 tests
black . && flake8                    # format + lint
```

The `--grid` universe is **5 liquid large-caps × 6 sectors × 2 regions = 60 holdings**
(curated in `config.py`).

## What actually works (the engineering)

- **Leak-aware *validation*** — purged + embargo split, benchmark session-lag, PIT
  fundamentals. The *design* prevents the obvious leaks.
- **Schema fluidity** — JSON fundamentals column handles per-sector heterogeneous metrics.
- **Auto-routing** — add `("TICKER", "Sector")` to the universe; region + benchmarks resolve automatically.
- **Reproducibility** — synthetic mode is fully deterministic; 47 tests including
  independent pandas/`arch` cross-checks of every engineered feature.
- **Depth** — the grid produces ~2,300+ labelled rows per region×sector cell.

## Known Limitations

This is where the honesty lives:

1. **Look-ahead leak in the volatility feature.** GARCH(1,1) parameters are fit **once on
   the full sample**, so `garch_vol` at an early date peeks slightly at the future. The
   backtester retrains the model per fold but reuses these precomputed features → the
   reported Sharpe (~5) is **inflated and not real**. A correct version refits GARCH inside
   each walk-forward fold. *Deferred, documented, not fixed.*
2. **No demonstrated edge.** Once the leak is removed, the expected honest result is
   roughly break-even (~50% hit-rate, Sharpe ≈ 0). Predicting daily equity moves is hard;
   most ideas have no edge. This one hasn't shown any.
3. **Degraded live fundamentals.** yfinance exposes only a current snapshot (no historical
   series), so in `--live` mode each fundamental is a single backdated value ≈ constant per
   ticker — not genuine point-in-time history. Real PIT fundamentals need a proper vendor.
4. **Signal thresholds** (`±0.004`) sit well below the live model's error (~0.03) → it
   over-trades. Would need tuning to error scale.

## Why publish a "no edge" project?

Because the lesson *is* the deliverable. In quant research, **every too-good-to-be-true
backtest is a leak, and honest backtests usually say "no edge."** Building a clean pipeline
that lets you *detect* that — rather than fooling yourself — is the actual skill. The code
is sound; the market is efficient.

## Layout

```
alphaflow/      config, models, data_source, fundamentals, ingestion,
                features, model, conformal, signals
execution/      backtester
main.py         orchestrator (--live / --grid / --backtest)
tests/          pytest suite
SPEC.md         design spec   ·   lessons.md   build log & edge cases
```
