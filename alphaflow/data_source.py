"""Market data acquisition: live (yfinance) or deterministic synthetic GBM.

Global model: every ticker belongs to a region (US/IN). Synthetic prices share a
per-region macro factor so assets co-move with their regional benchmarks (non-degenerate
spreads/betas). Benchmark tickers get lower idiosyncratic vol than single names.
"""

import asyncio
import math
import random
from datetime import date, timedelta

from .config import detect_region
from .models import PriceBar

_TRADING_DAYS = 252
_REGION_VOL = 0.011  # daily macro-factor vol per region


def _trading_dates(days: int, end: date) -> list[date]:
    out: list[date] = []
    d = end - timedelta(days=days)
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _ticker_params(ticker: str, seed: int, low_idio: bool) -> tuple[float, float, float, float]:
    """Deterministic (s0, beta, idio_vol, drift) from a string seed."""
    rng = random.Random(f"{seed}:params:{ticker}")
    s0 = rng.uniform(40.0, 500.0)
    beta = rng.uniform(0.8, 1.4)
    idio = rng.uniform(0.004, 0.008) if low_idio else rng.uniform(0.012, 0.022)
    drift = rng.uniform(0.02, 0.18)
    return s0, beta, idio, drift


def _series(
    ticker: str, dates: list[date], shocks: list[float], seed: int, low_idio: bool
) -> list[PriceBar]:
    s0, beta, idio, drift = _ticker_params(ticker, seed, low_idio)
    rng = random.Random(f"{seed}:path:{ticker}")
    bars: list[PriceBar] = []
    price = s0
    for d, shock in zip(dates, shocks):
        ret = drift / _TRADING_DAYS + beta * shock + idio * rng.gauss(0, 1)
        new_price = price * math.exp(ret)
        intraday = abs(rng.gauss(0, 0.01))
        high = max(price, new_price) * (1 + intraday)
        low = min(price, new_price) * (1 - intraday)
        bars.append(
            PriceBar(
                ticker=ticker,
                bar_date=d,
                open=round(price, 4),
                high=round(high, 4),
                low=round(low, 4),
                close=round(new_price, 4),
                volume=float(rng.randint(1_000_000, 50_000_000)),
            )
        )
        price = new_price
    return bars


def synthetic_prices(
    tickers: tuple[str, ...],
    days: int,
    seed: int,
    low_idio: frozenset[str] = frozenset(),
    end: date | None = None,
) -> dict[str, list[PriceBar]]:
    """Deterministic seeded history. Tickers sharing a region share a macro factor."""
    end = end or date(2026, 6, 12)
    dates = _trading_dates(days, end)
    region_shocks: dict[str, list[float]] = {}
    for region in {detect_region(t) for t in tickers}:
        r = random.Random(f"{seed}:macro:{region}")
        region_shocks[region] = [_REGION_VOL * r.gauss(0, 1) for _ in dates]
    return {
        t: _series(t, dates, region_shocks[detect_region(t)], seed, t in low_idio) for t in tickers
    }


async def fetch_prices(
    tickers: tuple[str, ...],
    days: int,
    seed: int,
    live: bool = False,
    low_idio: frozenset[str] = frozenset(),
) -> dict[str, list[PriceBar]]:
    """Async entry point. Live -> yfinance (per-ticker, robust to gaps); else synthetic."""
    if not live:
        return synthetic_prices(tickers, days, seed, low_idio)
    return await asyncio.to_thread(_fetch_yfinance, tickers, days)


def _fetch_yfinance(tickers: tuple[str, ...], days: int) -> dict[str, list[PriceBar]]:
    import yfinance as yf

    out: dict[str, list[PriceBar]] = {}
    for t in tickers:
        try:
            df = yf.download(t, period=f"{days}d", interval="1d", auto_adjust=True, progress=False)
        except Exception:
            out[t] = []
            continue
        if df is None or df.empty:
            out[t] = []
            continue
        if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        bars: list[PriceBar] = []
        for idx, row in df.dropna().iterrows():
            o, h, low_, c = (
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
            )
            if min(o, h, low_, c) <= 0:
                continue
            bars.append(
                PriceBar(
                    ticker=t,
                    bar_date=idx.date(),
                    open=o,
                    high=max(h, o, c),
                    low=min(low_, o, c),
                    close=c,
                    volume=float(row.get("Volume", 0.0) or 0.0),
                )
            )
        out[t] = bars
    return out
