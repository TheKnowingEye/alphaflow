# Lessons / Edge Cases

Running log of architecture changes + debugging edge cases.

## 2026-06-13 — Initial build
- SQLite lacks `STDDEV` aggregate -> rolling stddev via window identity `AVG(x*x) - AVG(x)*AVG(x)` (clamp negative epsilon from float error before sqrt).
- Forward-looking target (`fwd_alpha_3d`) uses `LEAD()` -> last 3 rows per ticker have NULL target -> excluded from training, kept for prediction.
- Chronological train/val split mandatory: random shuffle leaks future info through overlapping forward windows.
- Synthetic data must share a common sector factor between assets and ETF, else spread/beta features are pure noise and model degenerates.

## 2026-06-13 — Pivot to AlphaFlow lead-lag strategy
- Benchmark switched `3032.HK` -> `MAHKTECH.NS` (Mirae Asset Hang Seng TECH ETF) as the lagging vector vs NVDA/AMD leaders.
- Added `vol_ratio_5_20` (asset 5d std / 20d std): short-vs-long vol regime. Needs its own `wvs` window (`ROWS BETWEEN 4 PRECEDING`). 5d window fully populated after the 60-row beta warmup, so no extra NULL handling vs pandas `rolling(5)`.
- Reused existing 20d `AVG(x*x)-AVG(x)^2` identity for the 5d branch — no STDDEV needed.
- Signal model reshaped to the public output contract (`timestamp`/`asset_ticker`/`target_action`/`allocation_weight`/`model_confidence_score`). Field rename rippled into signals.py + every test referencing `.ticker`/`.action`/`.weight`.
- `model_confidence_score` derived, not placeholder: `erf(|alpha|/(val_rmse*sqrt2))` = P(|N(0,rmse)| < |alpha|). HOLD signals score near 0; strong alpha vs model error -> near 1. Requires threading `val_rmse` from TrainResult into `build_signals`.

## 2026-06-13 — P0 correctness (zero-leakage)
- **Lead-lag timezone fix**: old `JOIN USING(bar_date)` paired US(t) with HK(t) — same calendar day, but HK/US sessions don't overlap -> ambiguous causality. Now `etf` CTE uses `LEAD(bar_date) OVER (ORDER BY bar_date)` to map each HK close to the NEXT session; leader day t joins HK close from t-1. Removes same-day simultaneity; HK reaction lands in forward window. Costs first asset session (no prior HK pair) -> joined rows = N-1; warmup test updated `-1`.
- Pandas cross-check must mirror exactly: `df["e"] = e.shift(1)` before any return/spread calc, else SQL vs pandas diverge.
- **Purged + embargo split** (`purged_embargo_split`, Lopez de Prado): `fwd_alpha_3d` spans `forward_horizon` rows -> train rows near the val boundary leak future. Purge the `horizon` rows immediately before val; embargo `horizon` after (no-op for tail val). Enforces >= horizon gap. val RMSE rose 0.034 -> 0.042 after fix = previous number was leakage-inflated optimism, not real skill.
- `purged_embargo_split` reads only `.bar_date` -> unit-testable with `SimpleNamespace`, no CatBoost/DB needed (fast pure tests).
- `/cavekit build` not a registered skill -> ran suite directly (`pytest` + `black`/`flake8` + `python main.py`).

## 2026-06-13 — P1 validation & trust
- **EWMA beta**: EWMA is a recurrence (`S_t=(1-a)S_{t-1}+a*x_t`), NOT expressible in plain SQL window funcs. Moved beta + `fwd_alpha_3d` target out of SQL into Python (`features._ewma_beta`); SQL now emits raw `a_ret/e_ret/fwd_a/fwd_e/cnt_b`. EWMA runs over FULL per-ticker series (warmup incl) then rows filtered -> converged by emission.
- Cross-check parity: pandas must use `ewm(alpha=2/(span+1), adjust=False)` (adjust=False = the recurrence; adjust=True is normalized weights, won't match). Seed both at first non-null return (drop leading LAG-NULL row).
- **Conformal** replaces `erf` confidence: `confidence(alpha)=ECDF(|alpha|)` over sliding window of OOS residuals; interval = empirical quantile. Distribution-free, no Gaussian assumption. Threads `TrainResult.val_residuals` -> `ConformalCalibrator` -> `build_signals`. Added optional `alpha_ci_low/high` to Signal (defaulted None -> back-compat, no contract break).
- **Walk-forward backtester** (`execution/`): per-rebalance purge (train labels realised before decision date) = same leakage guard as P0 split but rolling. Friction throttle: skip names whose |pred alpha| <= 2*cost (can't clear round-trip); net returns by turnover*cost. Non-overlap via `bt_step=horizon`; annualize by 252/step.
- Backtest CatBoost cost: retrain every fold = slow. `bt_retrain_every` + small `bt_iterations` for test config keeps suite <1s. Synthetic GBM has no real alpha -> Sharpe ~0.19, hit ~0.46 expected; framework correctness is what's verified, not PnL.
- `import execution` works in tests via root `conftest.py` on sys.path (no install).

## 2026-06-13 — Phase 1 advanced feature layer (arch/GARCH + FFD)
- Added `arch>=7.0` dep (GARCH). Installed in venv; pinned in requirements.txt.
- **Fractional differentiation** (`ffd_weights`/`_frac_diff`): fixed-width FFD on log price, d=0.5. Weights `w[k]=-w[k-1]*(d-k+1)/k` decay power-law (~k^-1.5) -> long memory. Chose `frac_diff_width=50 < beta_window=60` deliberately so FFD warmup never extends past the beta warmup -> emitted-row count unchanged, warmup test stays `n_joined-1-beta_window`.
- **GARCH(1,1)** (`_garch_vol`): zero-mean, fit once per series; `res.conditional_volatility` IS the 1-step-ahead forecast (info ≤ t-1) -> use directly as per-row vector, no per-row refit. Fit on percent-scale returns (`*100`) — arch warns/ill-conditions on ~0.02 log-rets; divide back. try/except -> degrade to constant sample vol, never crash the pipeline.
- Replaced BOTH rolling-std `vol_ratio_*` features with GARCH (`garch_vol`, `garch_vol_ratio`). Dropped SQL `wv`/`wvs` windows + vol aggregates + config `vol_window`/`vol_short_window`. `FEATURE_COLS` + `FeatureRow` + Pydantic validator updated in lockstep.
- Param-leakage caveat: GARCH params + FFD use full-sample batch (features computed once). conditional_volatility VALUES are causal; only PARAMS are full-sample (same as EWMA). For strict walk-forward purity would refit per fold — deferred, noted.
- GARCH is an optimizer -> can't cross-check vs a closed form. Test instead refits `arch` independently on the same return series and asserts equality (deterministic lib -> identical). FFD IS closed-form -> cross-checked via independent weight sum.
- `/cavekit build` still not a real skill -> ran `pytest`+`black`/`flake8`+`main.py`(+`--backtest`).

## 2026-06-13 — Phase 2 fundamental ingestion (PIT)
- New `alphaflow/fundamentals.py`: `FundamentalRecord` (ticker, announce_date, trailing_pe, rev_growth_quarterly), synthetic default + live yfinance `.info`. `data/db_client.py` is `ingestion.py` (prices only) -> fundamentals kept in dedicated module.
- **PIT rule** (`pit_lookup`): bisect_right(announce_dates, day)-1 = latest record announced ON/BEFORE day, forward-filled. day before first announce -> None. NO lookahead: future quarter never bleeds into earlier rows (test asserts 04-19 still carries old quarter, switches exactly 04-20).
- Coverage constraint to protect P0/P1 boundaries: synthetic first announce seeded ONE quarter BEFORE the window start (`start - 91d`) so every emitted daily row has a PIT fundamental -> no extra row drop -> warmup count stays `n_joined-1-beta_window`, backtest stays 70 rebalances. Verified: 924 rows unchanged before/after Phase 2.
- Determinism: `random.Random(f"{seed}:{t}")` — str seed is reproducible (sha-based), unlike `hash((seed,t))` which PYTHONHASHSEED-randomizes. Don't use `hash()` for seeding.
- yfinance `.info` is current-snapshot only (no historical fundamental series) -> live PIT is degraded (single record at latest earnings date). Synthetic default gives real quarterly PIT history. Noted limitation.
- Live yfinance rev-growth key varies: try `revenueQuarterlyGrowth` then `revenueGrowth`.
- `compute_features(conn, settings, fundamentals=None)` — default None -> synthetic, so existing test/backtester callers unchanged; main passes fetched (live/synthetic).
- FEATURE_COLS now 12 (10 technical + 2 fundamental); FeatureRow + validator synced. CatBoost trains on combined matrix, val RMSE ~0.0419, 70 backtest steps intact.
