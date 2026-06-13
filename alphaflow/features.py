"""Feature engineering: per-holding SQL window pass + advanced Python layers.

Each holding joins its asset to TWO same-region benchmarks (macro + sector), both
lagged one session (no same-session lookahead margin, P0 carry-over). Outputs a
dual-spread, multi-factor row:
  - macro_spread / sector_spread vs the two benchmarks; spread_z on sector_spread.
  - beta_60d (EWMA) + residual-alpha target measured vs the MACRO benchmark.
  - frac_diff (FFD log price), GARCH(1,1) vol + asset/macro vol ratio.
  - region/sector carried as strings + integer matrix codes; fund_0/fund_1 are the
    sector's two PIT fundamentals (heterogeneous semantics, fixed slots).
"""

import math
from datetime import date

import aiosqlite
import numpy as np
from arch import arch_model

from .config import REGION_ID, SECTOR_ID, Holding, Settings
from .fundamentals import pit_lookup
from .models import FeatureRow, FundamentalRecord

_REQUIRED = (
    "log_ret_1d",
    "macro_spread",
    "sector_spread",
    "spread_z",
    "mom_5d",
    "mom_10d",
    "mom_20d",
)
_GARCH_SCALE = 100.0  # arch fits better on percent-scale returns
_GARCH_MIN_OBS = 50


def _query(s: Settings) -> str:
    z = s.spread_z_window - 1
    b = s.beta_window - 1
    m5, m10, m20 = (w - 1 for w in s.momentum_windows)
    h = s.forward_horizon
    return f"""
WITH mac AS (
    SELECT LEAD(bar_date) OVER (ORDER BY bar_date) AS jd, close AS m_close
    FROM price_bars WHERE ticker = :macro
),
sec AS (
    SELECT LEAD(bar_date) OVER (ORDER BY bar_date) AS jd, close AS s_close
    FROM price_bars WHERE ticker = :sector
),
joined AS (
    -- benchmarks lag the asset by one session -> no same-session lookahead.
    SELECT p.bar_date, p.close AS a_close, mac.m_close, sec.s_close
    FROM price_bars p
    JOIN mac ON p.bar_date = mac.jd
    JOIN sec ON p.bar_date = sec.jd
    WHERE p.ticker = :asset
),
rets AS (
    SELECT bar_date, a_close,
        af_ln(a_close / LAG(a_close) OVER w) AS a_ret,
        af_ln(m_close / LAG(m_close) OVER w) AS m_ret,
        af_ln(a_close) - af_ln(m_close) AS macro_spread,
        af_ln(a_close) - af_ln(s_close) AS sector_spread,
        af_ln(LEAD(a_close, {h}) OVER w / a_close) AS fwd_a,
        af_ln(LEAD(m_close, {h}) OVER w / m_close) AS fwd_m
    FROM joined
    WINDOW w AS (ORDER BY bar_date)
),
wins AS (
    SELECT bar_date, a_close, a_ret, m_ret, macro_spread, sector_spread, fwd_a, fwd_m,
        COUNT(a_ret)              OVER wb AS cnt_b,
        AVG(sector_spread)        OVER wz AS sp_mu,
        AVG(sector_spread*sector_spread) OVER wz AS sp_sq,
        SUM(a_ret)                OVER w5  AS mom_5d,
        SUM(a_ret)                OVER w10 AS mom_10d,
        SUM(a_ret)                OVER w20 AS mom_20d
    FROM rets
    WINDOW
        wz  AS (ORDER BY bar_date ROWS BETWEEN {z} PRECEDING AND CURRENT ROW),
        wb  AS (ORDER BY bar_date ROWS BETWEEN {b} PRECEDING AND CURRENT ROW),
        w5  AS (ORDER BY bar_date ROWS BETWEEN {m5} PRECEDING AND CURRENT ROW),
        w10 AS (ORDER BY bar_date ROWS BETWEEN {m10} PRECEDING AND CURRENT ROW),
        w20 AS (ORDER BY bar_date ROWS BETWEEN {m20} PRECEDING AND CURRENT ROW)
)
SELECT bar_date, a_close, a_ret AS log_ret_1d, m_ret, macro_spread, sector_spread,
    (sector_spread - sp_mu) / NULLIF(af_sqrt(MAX(sp_sq - sp_mu*sp_mu, 0)), 0) AS spread_z,
    mom_5d, mom_10d, mom_20d,
    cnt_b, fwd_a, fwd_m
FROM wins
ORDER BY bar_date
"""


def _af_ln(x: float | None) -> float | None:
    return None if x is None else math.log(x)


def _af_sqrt(x: float | None) -> float | None:
    return None if x is None else math.sqrt(x)


async def register_udfs(conn: aiosqlite.Connection) -> None:
    await conn.create_function("af_ln", 1, _af_ln, deterministic=True)
    await conn.create_function("af_sqrt", 1, _af_sqrt, deterministic=True)


def _ewma_beta(a: list, e: list, alpha: float) -> list[float | None]:
    """EWMA beta cov(a,e)/var(e) per row (pandas ewm adjust=False recurrence)."""
    ea = ee = eae = eee = None
    out: list[float | None] = []
    for av, ev in zip(a, e):
        if av is None or ev is None:
            out.append(None)
            continue
        if ea is None:
            ea, ee, eae, eee = av, ev, av * ev, ev * ev
        else:
            ea = (1 - alpha) * ea + alpha * av
            ee = (1 - alpha) * ee + alpha * ev
            eae = (1 - alpha) * eae + alpha * (av * ev)
            eee = (1 - alpha) * eee + alpha * (ev * ev)
        var = eee - ee * ee
        out.append((eae - ea * ee) / var if var > 0 else None)
    return out


def ffd_weights(d: float, width: int) -> list[float]:
    """Fixed-width fractional-differentiation weights. w[k]=-w[k-1]*(d-k+1)/k."""
    w = [1.0]
    for k in range(1, width):
        w.append(-w[-1] * (d - k + 1) / k)
    return w


def _frac_diff(prices: list[float], d: float, width: int) -> list[float | None]:
    w = ffd_weights(d, width)
    x = [math.log(p) for p in prices]
    out: list[float | None] = [None] * len(x)
    for i in range(width - 1, len(x)):
        out[i] = sum(w[k] * x[i - k] for k in range(width))
    return out


def _garch_vol(returns: list[float | None]) -> list[float | None]:
    """GARCH(1,1) zero-mean conditional vol = 1d-ahead forecast given the past."""
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
    except Exception:
        sd = float(np.std(clean))
        cond = np.full(len(clean), sd if sd > 0 else 1e-8)
    for j, i in enumerate(idx):
        v = float(cond[j])
        out[i] = v if v > 0 else None
    return out


async def _holding_rows(
    conn: aiosqlite.Connection,
    settings: Settings,
    holding: Holding,
    funds: list[FundamentalRecord],
) -> list[FeatureRow]:
    cur = await conn.execute(
        _query(settings),
        {
            "asset": holding.ticker,
            "macro": holding.macro_benchmark,
            "sector": holding.sector_benchmark,
        },
    )
    cols = [c[0] for c in cur.description]
    recs = [dict(zip(cols, r)) async for r in cur]
    recs.sort(key=lambda x: x["bar_date"])

    alpha = 2.0 / (settings.beta_window + 1)
    betas = _ewma_beta([r["log_ret_1d"] for r in recs], [r["m_ret"] for r in recs], alpha)
    fdiff = _frac_diff([r["a_close"] for r in recs], settings.frac_diff_d, settings.frac_diff_width)
    a_vol = _garch_vol([r["log_ret_1d"] for r in recs])
    m_vol = _garch_vol([r["m_ret"] for r in recs])
    frecs = sorted(funds, key=lambda r: r.announce_date)
    k0, k1 = holding.fundamental_keys
    rid, sid = REGION_ID[holding.region], SECTOR_ID[holding.sector]

    rows: list[FeatureRow] = []
    for rec, beta, fdv, av, mv in zip(recs, betas, fdiff, a_vol, m_vol):
        if rec["cnt_b"] != settings.beta_window:
            continue
        if beta is None or fdv is None or av is None or mv is None or mv <= 0:
            continue
        if any(rec[k] is None for k in _REQUIRED):
            continue
        d = date.fromisoformat(rec["bar_date"])
        fund = pit_lookup(frecs, d)
        if fund is None or k0 not in fund.metrics or k1 not in fund.metrics:
            continue  # no PIT fundamental yet -> no fabrication
        # Robustness: clamp beta (var(macro) can momentarily collapse -> blowup) and the
        # residual target (else outlier labels train fantasy alphas + detonate the backtest).
        bc = settings.beta_clip
        beta = max(-bc, min(bc, beta))
        fa, fm = rec["fwd_a"], rec["fwd_m"]
        if fa is None or fm is None:
            fwd_alpha = None
        else:
            tc = settings.target_clip
            fwd_alpha = max(-tc, min(tc, fa - beta * fm))
        rows.append(
            FeatureRow(
                ticker=holding.ticker,
                bar_date=d,
                region=holding.region,
                sector=holding.sector,
                log_ret_1d=rec["log_ret_1d"],
                macro_spread=rec["macro_spread"],
                sector_spread=rec["sector_spread"],
                spread_z=rec["spread_z"],
                frac_diff_close=fdv,
                garch_vol=av,
                garch_vol_ratio=av / mv,
                mom_5d=rec["mom_5d"],
                mom_10d=rec["mom_10d"],
                mom_20d=rec["mom_20d"],
                beta_60d=beta,
                region_id=rid,
                sector_id=sid,
                fund_0=float(fund.metrics[k0]),
                fund_1=float(fund.metrics[k1]),
                fwd_alpha_3d=fwd_alpha,
            )
        )
    return rows


async def compute_features(
    conn: aiosqlite.Connection,
    settings: Settings,
    holdings: tuple[Holding, ...] | None = None,
    fundamentals: dict[str, list[FundamentalRecord]] | None = None,
) -> list[FeatureRow]:
    """Multi-factor features across all holdings. Defaults: settings.holdings() and
    synthetic fundamentals (so existing callers need no extra args)."""
    if holdings is None:
        holdings = settings.holdings()
    if fundamentals is None:
        from .fundamentals import synthetic_fundamentals

        fundamentals = synthetic_fundamentals(
            holdings, settings.history_days, settings.synthetic_seed
        )
    await register_udfs(conn)
    rows: list[FeatureRow] = []
    for h in holdings:
        rows.extend(await _holding_rows(conn, settings, h, fundamentals.get(h.ticker, [])))
    rows.sort(key=lambda r: (r.ticker, r.bar_date))
    return rows
