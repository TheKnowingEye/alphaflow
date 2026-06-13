"""Cross-check SQL window-function features against independent pandas calc."""

import numpy as np
import pandas as pd
import pytest
from arch import arch_model

from alphaflow.config import Settings
from alphaflow.data_source import synthetic_history
from alphaflow.features import _GARCH_SCALE, compute_features, ffd_weights
from alphaflow.ingestion import ingest_bars, open_db

SETTINGS = Settings(db_path=":memory:")


@pytest.fixture
async def feature_rows():
    conn = await open_db(":memory:")
    history = synthetic_history(
        SETTINGS.asset_tickers, SETTINGS.benchmark_ticker, days=400, seed=11
    )
    for bars in history.values():
        await ingest_bars(conn, bars)
    rows = await compute_features(conn, SETTINGS)
    await conn.close()
    return history, rows


def _pandas_features(history, ticker):
    a = pd.Series({b.bar_date: b.close for b in history[ticker]}, dtype=float).sort_index()
    e = pd.Series(
        {b.bar_date: b.close for b in history[SETTINGS.benchmark_ticker]}, dtype=float
    ).sort_index()
    # Lead-lag: HK benchmark lags US by one session -> leader day t pairs with the
    # HK close from t-1. Mirrors the SQL LEAD(bar_date) map join.
    df = pd.DataFrame({"a": a, "e_raw": e})
    df["e"] = df["e_raw"].shift(1)
    df = df.dropna(subset=["a", "e"])
    df["a_ret"] = np.log(df.a / df.a.shift(1))
    df["e_ret"] = np.log(df.e / df.e.shift(1))
    df["spread"] = np.log(df.a) - np.log(df.e)
    # population std (ddof=0) — matches SQL AVG(x*x) - AVG(x)^2 identity
    mu = df.spread.rolling(20).mean()
    sd = df.spread.rolling(20).std(ddof=0)
    df["spread_z"] = (df.spread - mu) / sd
    # Fractional differentiation of log price (mirrors features._frac_diff).
    wts = ffd_weights(SETTINGS.frac_diff_d, SETTINGS.frac_diff_width)
    logp = np.log(df.a.to_numpy())
    width = SETTINGS.frac_diff_width
    fd = [np.nan] * len(logp)
    for i in range(width - 1, len(logp)):
        fd[i] = sum(wts[k] * logp[i - k] for k in range(width))
    df["frac_diff_close"] = fd
    for w in (5, 10, 20):
        df[f"mom_{w}d"] = df.a_ret.rolling(w).sum()
    # EWMA beta (adjust=False recurrence) — mirrors features._ewma_beta. span = beta_window.
    al = 2.0 / (SETTINGS.beta_window + 1)
    ea = df.a_ret.ewm(alpha=al, adjust=False).mean()
    ee = df.e_ret.ewm(alpha=al, adjust=False).mean()
    eae = (df.a_ret * df.e_ret).ewm(alpha=al, adjust=False).mean()
    eee = (df.e_ret**2).ewm(alpha=al, adjust=False).mean()
    df["beta_60d"] = (eae - ea * ee) / (eee - ee * ee)
    fwd_a = np.log(df.a.shift(-3) / df.a)
    fwd_e = np.log(df.e.shift(-3) / df.e)
    df["fwd_alpha_3d"] = fwd_a - df.beta_60d * fwd_e
    return df


async def test_sql_matches_pandas(feature_rows):
    history, rows = feature_rows
    df = _pandas_features(history, "NVDA")
    nvda = {r.bar_date: r for r in rows if r.ticker == "NVDA"}
    assert len(nvda) > 100

    checked = 0
    for d, r in nvda.items():
        exp = df.loc[d]
        for col in (
            "log_ret_1d",
            "spread",
            "spread_z",
            "frac_diff_close",
            "mom_5d",
            "mom_10d",
            "mom_20d",
            "beta_60d",
        ):
            pd_col = "a_ret" if col == "log_ret_1d" else col
            assert getattr(r, col) == pytest.approx(
                exp[pd_col], rel=1e-7, abs=1e-10
            ), f"{col} mismatch on {d}"
        if r.fwd_alpha_3d is not None:
            assert r.fwd_alpha_3d == pytest.approx(exp["fwd_alpha_3d"], rel=1e-7, abs=1e-10)
        checked += 1
    assert checked == len(nvda)


async def test_garch_vol_matches_independent_refit(feature_rows):
    history, rows = feature_rows
    df = _pandas_features(history, "NVDA")
    rets = df.a_ret.dropna().to_numpy() * _GARCH_SCALE
    res = arch_model(rets, mean="Zero", vol="Garch", p=1, q=1, dist="normal").fit(
        disp="off", show_warning=False
    )
    cond = np.asarray(res.conditional_volatility) / _GARCH_SCALE
    # align: conditional_volatility aligns to the non-NaN return dates
    ret_dates = df.a_ret.dropna().index
    expected = dict(zip(ret_dates, cond))

    nvda = {r.bar_date: r for r in rows if r.ticker == "NVDA"}
    checked = 0
    for d, r in nvda.items():
        assert r.garch_vol > 0
        assert r.garch_vol == pytest.approx(expected[d], rel=1e-6, abs=1e-12)
        assert r.garch_vol_ratio > 0  # asset GARCH vol / benchmark GARCH vol
        checked += 1
    assert checked == len(nvda)


async def test_warmup_and_forward_edges(feature_rows):
    history, rows = feature_rows
    nvda = sorted((r for r in rows if r.ticker == "NVDA"), key=lambda r: r.bar_date)
    # warmup: first beta_window joined dates excluded. Benchmark-lag join drops the
    # first asset session (no prior HK close to pair) -> one fewer joined row.
    n_joined = len(history["NVDA"]) - 1
    assert len(nvda) == n_joined - SETTINGS.beta_window
    # last forward_horizon rows have NULL target (LEAD runs off the end)
    assert all(r.fwd_alpha_3d is None for r in nvda[-3:])
    assert all(r.fwd_alpha_3d is not None for r in nvda[:-3])
