"""Feature engineering: SQL window pass + advanced statistical layers in Python.

SQLite portability: ln/sqrt registered as UDFs (af_ln, af_sqrt) — no reliance on
SQLITE_ENABLE_MATH_FUNCTIONS. Spread z-score stddev via the AVG(x*x) - AVG(x)*AVG(x)
window identity, clamped at 0 against float error before sqrt.

Three layers are inherently non-windowable and computed in Python over the full
per-ticker series (then warmup rows are dropped):
  - EWMA beta: recurrence S_t = (1-a)S_{t-1} + a*x_t -> stable residual-alpha target.
  - Fractional differentiation (FFD, d~0.5): stationary log-price retaining long memory
    (support/resistance), via fixed-width weighted sum (Lopez de Prado).
  - GARCH(1,1): conditional 1-day-ahead volatility forecast — replaces rolling stddev.
"""

import math
from datetime import date

import aiosqlite
import numpy as np
from arch import arch_model

from .config import Settings
from .fundamentals import pit_lookup, synthetic_fundamentals
from .models import FeatureRow, FundamentalRecord

# Required (non-target) feature columns — a NULL/None here drops the row.
_REQUIRED = (
    "log_ret_1d",
    "spread",
    "spread_z",
    "frac_diff_close",
    "garch_vol",
    "garch_vol_ratio",
    "mom_5d",
    "mom_10d",
    "mom_20d",
    "trailing_pe",
    "rev_growth_quarterly",
)
_GARCH_SCALE = 100.0  # arch fits better on percent-scale returns
_GARCH_MIN_OBS = 50


def _query(s: Settings) -> str:
    z = s.spread_z_window - 1
    b = s.beta_window - 1
    m5, m10, m20 = (w - 1 for w in s.momentum_windows)
    h = s.forward_horizon
    return f"""
WITH etf AS (
    -- Lead-lag alignment: US leaders lead, HK benchmark lags by one session.
    -- Map each HK close to the NEXT session so a leader bar on day t joins the HK
    -- close from t-1. Removes same-calendar-day US/HK simultaneity (no lookahead);
    -- the HK reaction to the leader move falls inside the forward target window.
    SELECT
        LEAD(bar_date) OVER (ORDER BY bar_date) AS map_date,
        close AS e_close
    FROM price_bars WHERE ticker = :benchmark
),
joined AS (
    SELECT p.ticker, p.bar_date, p.close AS a_close, e.e_close
    FROM price_bars p JOIN etf e ON p.bar_date = e.map_date
    WHERE p.ticker != :benchmark
),
rets AS (
    SELECT ticker, bar_date, a_close, e_close,
        af_ln(a_close / LAG(a_close) OVER w) AS a_ret,
        af_ln(e_close / LAG(e_close) OVER w) AS e_ret,
        af_ln(a_close) - af_ln(e_close) AS spread,
        af_ln(LEAD(a_close, {h}) OVER w / a_close) AS fwd_a,
        af_ln(LEAD(e_close, {h}) OVER w / e_close) AS fwd_e
    FROM joined
    WINDOW w AS (PARTITION BY ticker ORDER BY bar_date)
),
wins AS (
    SELECT ticker, bar_date, a_close, a_ret, e_ret, spread, fwd_a, fwd_e,
        COUNT(a_ret)       OVER wb  AS cnt_b,
        AVG(spread)        OVER wz  AS sp_mu,
        AVG(spread*spread) OVER wz  AS sp_sq,
        SUM(a_ret)         OVER w5  AS mom_5d,
        SUM(a_ret)         OVER w10 AS mom_10d,
        SUM(a_ret)         OVER w20 AS mom_20d
    FROM rets
    WINDOW
        wz  AS (PARTITION BY ticker ORDER BY bar_date ROWS BETWEEN {z} PRECEDING AND CURRENT ROW),
        wb  AS (PARTITION BY ticker ORDER BY bar_date ROWS BETWEEN {b} PRECEDING AND CURRENT ROW),
        w5  AS (PARTITION BY ticker ORDER BY bar_date ROWS BETWEEN {m5} PRECEDING AND CURRENT ROW),
        w10 AS (PARTITION BY ticker ORDER BY bar_date ROWS BETWEEN {m10} PRECEDING AND CURRENT ROW),
        w20 AS (PARTITION BY ticker ORDER BY bar_date ROWS BETWEEN {m20} PRECEDING AND CURRENT ROW)
)
SELECT ticker, bar_date, a_close, a_ret AS log_ret_1d, e_ret, spread,
    (spread - sp_mu) / NULLIF(af_sqrt(MAX(sp_sq - sp_mu*sp_mu, 0)), 0) AS spread_z,
    mom_5d, mom_10d, mom_20d,
    cnt_b, fwd_a, fwd_e
FROM wins
ORDER BY ticker, bar_date
"""


def _af_ln(x: float | None) -> float | None:
    # LAG/LEAD on partition edges emit SQL NULL -> guard so NULL propagates.
    return None if x is None else math.log(x)


def _af_sqrt(x: float | None) -> float | None:
    return None if x is None else math.sqrt(x)


async def register_udfs(conn: aiosqlite.Connection) -> None:
    await conn.create_function("af_ln", 1, _af_ln, deterministic=True)
    await conn.create_function("af_sqrt", 1, _af_sqrt, deterministic=True)


def _ewma_beta(recs: list[dict], alpha: float) -> list[float | None]:
    """EWMA beta per row: cov(a,e)/var(e) from EWMAs of a_ret, e_ret, a*e, e*e.

    Recurrence S_t = (1-alpha)*S_{t-1} + alpha*x_t (pandas ewm adjust=False),
    seeded at the first row with non-NULL returns. NULL-return rows -> beta None.
    """
    ea = ee = eae = eee = None
    out: list[float | None] = []
    for r in recs:
        a, e = r["log_ret_1d"], r["e_ret"]
        if a is None or e is None:
            out.append(None)
            continue
        if ea is None:
            ea, ee, eae, eee = a, e, a * e, e * e
        else:
            ea = (1 - alpha) * ea + alpha * a
            ee = (1 - alpha) * ee + alpha * e
            eae = (1 - alpha) * eae + alpha * (a * e)
            eee = (1 - alpha) * eee + alpha * (e * e)
        var = eee - ee * ee
        out.append((eae - ea * ee) / var if var > 0 else None)
    return out


def ffd_weights(d: float, width: int) -> list[float]:
    """Fixed-width fractional-differentiation weights. w[k] multiplies x[t-k];
    w[0]=1, w[k] = -w[k-1]*(d-k+1)/k. Slow (power-law) decay -> long memory retained."""
    w = [1.0]
    for k in range(1, width):
        w.append(-w[-1] * (d - k + 1) / k)
    return w


def _frac_diff(prices: list[float], d: float, width: int) -> list[float | None]:
    """Apply FFD to a log-price series; first (width-1) rows have no full window -> None."""
    w = ffd_weights(d, width)
    x = [math.log(p) for p in prices]
    out: list[float | None] = [None] * len(x)
    for i in range(width - 1, len(x)):
        out[i] = sum(w[k] * x[i - k] for k in range(width))
    return out


def _garch_vol(returns: list[float | None]) -> list[float | None]:
    """GARCH(1,1) zero-mean conditional volatility = 1d-ahead forecast given the past.

    Fit once on the clean return series; `conditional_volatility[t]` is the variance
    forecast for t formed from information through t-1. Leading None-return -> None.
    """
    out: list[float | None] = [None] * len(returns)
    idx = [i for i, r in enumerate(returns) if r is not None]
    clean = np.array([returns[i] for i in idx], dtype=np.float64)
    if len(clean) < _GARCH_MIN_OBS:
        return out
    try:
        res = arch_model(
            clean * _GARCH_SCALE, mean="Zero", vol="Garch", p=1, q=1, dist="normal"
        ).fit(disp="off", show_warning=False)
        cond = np.asarray(res.conditional_volatility, dtype=np.float64) / _GARCH_SCALE
    except Exception:  # optimiser failure -> degrade to constant sample vol, never crash
        sd = float(np.std(clean))
        cond = np.full(len(clean), sd if sd > 0 else 1e-8)
    for j, i in enumerate(idx):
        v = float(cond[j])
        out[i] = v if v > 0 else None
    return out


async def compute_features(
    conn: aiosqlite.Connection,
    settings: Settings,
    fundamentals: dict[str, list[FundamentalRecord]] | None = None,
) -> list[FeatureRow]:
    """SQL window features + EWMA beta + FFD + GARCH vol + PIT fundamentals, in Python.

    The recursive/optimised layers run over the FULL per-ticker series (warmup included)
    so they converge; rows are emitted only past the beta-window warmup and free of
    degenerate NULLs. FFD width < beta_window so it adds no warmup beyond beta.
    Fundamentals are PIT-aligned: each row gets the latest record announced on/before it.
    """
    if fundamentals is None:
        fundamentals = synthetic_fundamentals(
            settings.asset_tickers, settings.history_days, settings.synthetic_seed
        )
    await register_udfs(conn)
    cur = await conn.execute(_query(settings), {"benchmark": settings.benchmark_ticker})
    cols = [c[0] for c in cur.description]
    raw = [dict(zip(cols, r)) async for r in cur]

    by_ticker: dict[str, list[dict]] = {}
    for rec in raw:
        by_ticker.setdefault(rec["ticker"], []).append(rec)
    for recs in by_ticker.values():
        recs.sort(key=lambda x: x["bar_date"])

    # Benchmark GARCH vol (lagged e_ret is identical across tickers) -> date -> vol map.
    bench_vol: dict[str, float] = {}
    if by_ticker:
        ref = next(iter(by_ticker.values()))
        for rec, v in zip(ref, _garch_vol([r["e_ret"] for r in ref])):
            if v is not None:
                bench_vol[rec["bar_date"]] = v

    alpha = 2.0 / (settings.beta_window + 1)  # EWMA span = beta_window
    rows: list[FeatureRow] = []
    for ticker, recs in by_ticker.items():
        betas = _ewma_beta(recs, alpha)
        fdiff = _frac_diff(
            [r["a_close"] for r in recs], settings.frac_diff_d, settings.frac_diff_width
        )
        gvol = _garch_vol([r["log_ret_1d"] for r in recs])
        frecs = sorted(fundamentals.get(ticker, []), key=lambda r: r.announce_date)
        for rec, beta, fdv, gv in zip(recs, betas, fdiff, gvol):
            if rec["cnt_b"] != settings.beta_window:
                continue  # warmup not complete
            bv = bench_vol.get(rec["bar_date"])
            if beta is None or fdv is None or gv is None or bv is None:
                continue
            d = date.fromisoformat(rec["bar_date"])
            fund = pit_lookup(frecs, d)  # PIT: latest announce <= d, forward-filled
            if fund is None:
                continue  # no fundamental announced yet -> no lookahead fabrication
            rec["frac_diff_close"] = fdv
            rec["garch_vol"] = gv
            rec["garch_vol_ratio"] = gv / bv
            rec["trailing_pe"] = fund.trailing_pe
            rec["rev_growth_quarterly"] = fund.rev_growth_quarterly
            if any(rec[k] is None for k in _REQUIRED):
                continue
            fa, fe = rec["fwd_a"], rec["fwd_e"]
            fwd_alpha = None if (fa is None or fe is None) else fa - beta * fe
            rows.append(
                FeatureRow(
                    ticker=ticker,
                    bar_date=d,
                    log_ret_1d=rec["log_ret_1d"],
                    spread=rec["spread"],
                    spread_z=rec["spread_z"],
                    frac_diff_close=fdv,
                    garch_vol=gv,
                    garch_vol_ratio=rec["garch_vol_ratio"],
                    mom_5d=rec["mom_5d"],
                    mom_10d=rec["mom_10d"],
                    mom_20d=rec["mom_20d"],
                    beta_60d=beta,
                    trailing_pe=fund.trailing_pe,
                    rev_growth_quarterly=fund.rev_growth_quarterly,
                    fwd_alpha_3d=fwd_alpha,
                )
            )
    rows.sort(key=lambda r: (r.ticker, r.bar_date))
    return rows
