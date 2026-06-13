"""Cross-check multi-factor SQL/Python features against an independent pandas calc."""

import numpy as np
import pandas as pd
import pytest
from arch import arch_model

from alphaflow.config import Settings, resolve_holding
from alphaflow.data_source import synthetic_prices
from alphaflow.features import _GARCH_SCALE, compute_features, ffd_weights
from alphaflow.fundamentals import pit_lookup, synthetic_fundamentals
from alphaflow.ingestion import ingest_bars, open_db

SETTINGS = Settings(db_path=":memory:")
H = resolve_holding("NVDA", "Tech")  # US -> macro SPY, sector XLK
TICKERS = ("NVDA", "SPY", "XLK")


@pytest.fixture
async def built():
    conn = await open_db(":memory:")
    prices = synthetic_prices(TICKERS, days=400, seed=11, low_idio=frozenset({"SPY", "XLK"}))
    for bars in prices.values():
        await ingest_bars(conn, bars)
    funds = synthetic_fundamentals((H,), days=SETTINGS.history_days, seed=SETTINGS.synthetic_seed)
    rows = await compute_features(conn, SETTINGS, holdings=(H,), fundamentals=funds)
    await conn.close()
    return prices, funds, rows


def _pandas(prices):
    cl = {
        t: pd.Series({b.bar_date: b.close for b in prices[t]}, dtype=float).sort_index()
        for t in TICKERS
    }
    df = pd.DataFrame({"a": cl["NVDA"], "m_raw": cl["SPY"], "s_raw": cl["XLK"]})
    df["m"] = df["m_raw"].shift(1)  # benchmarks lag one session
    df["s"] = df["s_raw"].shift(1)
    df = df.dropna(subset=["a", "m", "s"])
    df["a_ret"] = np.log(df.a / df.a.shift(1))
    df["m_ret"] = np.log(df.m / df.m.shift(1))
    df["macro_spread"] = np.log(df.a) - np.log(df.m)
    df["sector_spread"] = np.log(df.a) - np.log(df.s)
    mu = df.sector_spread.rolling(20).mean()
    sd = df.sector_spread.rolling(20).std(ddof=0)
    df["spread_z"] = (df.sector_spread - mu) / sd
    wts = ffd_weights(SETTINGS.frac_diff_d, SETTINGS.frac_diff_width)
    logp = np.log(df.a.to_numpy())
    w = SETTINGS.frac_diff_width
    fd = [np.nan] * len(logp)
    for i in range(w - 1, len(logp)):
        fd[i] = sum(wts[k] * logp[i - k] for k in range(w))
    df["frac_diff_close"] = fd
    for win in (5, 10, 20):
        df[f"mom_{win}d"] = df.a_ret.rolling(win).sum()
    al = 2.0 / (SETTINGS.beta_window + 1)
    ea = df.a_ret.ewm(alpha=al, adjust=False).mean()
    em = df.m_ret.ewm(alpha=al, adjust=False).mean()
    eam = (df.a_ret * df.m_ret).ewm(alpha=al, adjust=False).mean()
    emm = (df.m_ret**2).ewm(alpha=al, adjust=False).mean()
    df["beta_60d"] = (eam - ea * em) / (emm - em * em)
    return df


async def test_sql_matches_pandas(built):
    prices, _, rows = built
    df = _pandas(prices)
    nvda = {r.bar_date: r for r in rows if r.ticker == "NVDA"}
    assert len(nvda) > 100
    for d, r in nvda.items():
        exp = df.loc[d]
        assert r.region == "US" and r.sector == "Tech"
        assert r.region_id == 0 and r.sector_id == 0
        for col in (
            "macro_spread",
            "sector_spread",
            "spread_z",
            "frac_diff_close",
            "mom_5d",
            "mom_10d",
            "mom_20d",
            "beta_60d",
        ):
            assert getattr(r, col) == pytest.approx(exp[col], rel=1e-7, abs=1e-10), f"{col}@{d}"
        assert r.log_ret_1d == pytest.approx(exp["a_ret"], rel=1e-7, abs=1e-10)


async def test_garch_vol_matches_refit(built):
    prices, _, rows = built
    df = _pandas(prices)
    rets = df.a_ret.dropna().to_numpy() * _GARCH_SCALE
    res = arch_model(rets, mean="Zero", vol="Garch", p=1, q=1, dist="normal").fit(
        disp="off", show_warning=False
    )
    cond = dict(zip(df.a_ret.dropna().index, np.asarray(res.conditional_volatility) / _GARCH_SCALE))
    for r in (r for r in rows if r.ticker == "NVDA"):
        assert r.garch_vol > 0 and r.garch_vol_ratio > 0
        assert r.garch_vol == pytest.approx(cond[r.bar_date], rel=1e-6, abs=1e-12)


async def test_fundamentals_pit_attached(built):
    _, funds, rows = built
    frecs = funds["NVDA"]
    for r in (r for r in rows if r.ticker == "NVDA"):
        fund = pit_lookup(frecs, r.bar_date)
        assert r.fund_0 == pytest.approx(fund.metrics["trailingPE"])
        assert r.fund_1 == pytest.approx(fund.metrics["revenueGrowth"])


async def test_warmup_and_forward_edges(built):
    prices, _, rows = built
    nvda = sorted((r for r in rows if r.ticker == "NVDA"), key=lambda r: r.bar_date)
    n_joined = len(prices["NVDA"]) - 1  # benchmark-lag drops first asset session
    assert len(nvda) == n_joined - SETTINGS.beta_window
    assert all(r.fwd_alpha_3d is None for r in nvda[-3:])
    assert all(r.fwd_alpha_3d is not None for r in nvda[:-3])
