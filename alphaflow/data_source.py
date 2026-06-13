"""Market data acquisition: live (yfinance) or deterministic synthetic GBM."""

import asyncio
import math
import random
from datetime import date, timedelta

from .models import PriceBar

_TRADING_DAYS = 252


def _synthetic_ticker_bars(
    ticker: str,
    dates: list[date],
    rng: random.Random,
    sector_shocks: list[float],
    s0: float,
    beta: float,
    idio_vol: float,
    drift: float,
) -> list[PriceBar]:
    """GBM with shared sector factor -> realistic asset/ETF co-movement."""
    bars: list[PriceBar] = []
    price = s0
    for d, shock in zip(dates, sector_shocks):
        ret = drift / _TRADING_DAYS + beta * shock + idio_vol * rng.gauss(0, 1)
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


def synthetic_history(
    asset_tickers: tuple[str, ...],
    benchmark_ticker: str,
    days: int,
    seed: int,
    end: date | None = None,
) -> dict[str, list[PriceBar]]:
    """Deterministic seeded history for all tickers (weekdays only)."""
    end = end or date(2026, 6, 12)
    dates: list[date] = []
    d = end - timedelta(days=days)
    while d <= end:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)

    rng = random.Random(seed)
    sector_vol = 0.012
    sector_shocks = [sector_vol * rng.gauss(0, 1) for _ in dates]

    out: dict[str, list[PriceBar]] = {
        benchmark_ticker: _synthetic_ticker_bars(
            benchmark_ticker, dates, rng, sector_shocks, 5.0, 1.0, 0.004, 0.04
        )
    }
    params = {"NVDA": (450.0, 1.5, 0.018, 0.30), "AMD": (140.0, 1.3, 0.020, 0.15)}
    for t in asset_tickers:
        s0, beta, ivol, drift = params.get(t, (100.0, 1.2, 0.015, 0.10))
        out[t] = _synthetic_ticker_bars(t, dates, rng, sector_shocks, s0, beta, ivol, drift)
    return out


async def fetch_history(
    asset_tickers: tuple[str, ...],
    benchmark_ticker: str,
    days: int,
    seed: int,
    live: bool = False,
) -> dict[str, list[PriceBar]]:
    """Async entry point. Live mode hits yfinance in a thread; default is synthetic."""
    if not live:
        return synthetic_history(asset_tickers, benchmark_ticker, days, seed)
    return await asyncio.to_thread(_fetch_yfinance, asset_tickers + (benchmark_ticker,), days)


def _fetch_yfinance(tickers: tuple[str, ...], days: int) -> dict[str, list[PriceBar]]:
    import yfinance as yf

    out: dict[str, list[PriceBar]] = {}
    df = yf.download(
        list(tickers), period=f"{days}d", interval="1d", group_by="ticker", auto_adjust=True
    )
    for t in tickers:
        sub = df[t].dropna() if len(tickers) > 1 else df.dropna()
        out[t] = [
            PriceBar(
                ticker=t,
                bar_date=idx.date(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
            )
            for idx, row in sub.iterrows()
        ]
    return out
