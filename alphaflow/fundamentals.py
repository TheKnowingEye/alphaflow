"""Fundamental ingestion + Point-In-Time alignment.

Quarterly metrics (`trailing_pe`, `rev_growth_quarterly`) are keyed by official
announcement date. PIT rule: a metric applies ONLY to daily rows on/after its
announce date, forward-filled until the next release -> no lookahead.

Default source is deterministic synthetic (offline-safe, reproducible). `live=True`
pulls yfinance `.info` scalars stamped at the latest earnings date (degraded PIT:
vendor exposes no historical fundamental series).
"""

import asyncio
import random
from bisect import bisect_right
from datetime import date, timedelta

from .models import FundamentalRecord

_QUARTER_DAYS = 91
# (trailing_pe seed, rev_growth seed) per asset.
_BASE = {"NVDA": (55.0, 0.40), "AMD": (45.0, 0.22)}


def synthetic_fundamentals(
    tickers: tuple[str, ...], days: int, seed: int, end: date | None = None
) -> dict[str, list[FundamentalRecord]]:
    """Deterministic quarterly records. First announce precedes the window so PIT
    align covers every in-window daily row (no row dropped for want of a fundamental)."""
    end = end or date(2026, 6, 12)
    start = end - timedelta(days=days)
    out: dict[str, list[FundamentalRecord]] = {}
    for t in tickers:
        rng = random.Random(f"{seed}:{t}")  # str seed -> reproducible per ticker
        pe, g0 = _BASE.get(t, (30.0, 0.10))
        recs: list[FundamentalRecord] = []
        d = start - timedelta(days=_QUARTER_DAYS)  # one release before the window
        while d <= end:
            pe = max(5.0, pe * (1 + rng.gauss(0, 0.12)))
            g = g0 + rng.gauss(0, 0.08)
            recs.append(
                FundamentalRecord(
                    ticker=t,
                    announce_date=d,
                    trailing_pe=round(pe, 2),
                    rev_growth_quarterly=round(g, 4),
                )
            )
            d += timedelta(days=_QUARTER_DAYS)
        out[t] = recs
    return out


async def fetch_fundamentals(
    tickers: tuple[str, ...], days: int, seed: int, live: bool = False
) -> dict[str, list[FundamentalRecord]]:
    if not live:
        return synthetic_fundamentals(tickers, days, seed)
    return await asyncio.to_thread(_fetch_yf_fundamentals, tickers)


def _fetch_yf_fundamentals(tickers: tuple[str, ...]) -> dict[str, list[FundamentalRecord]]:
    import yfinance as yf

    out: dict[str, list[FundamentalRecord]] = {}
    for t in tickers:
        info = yf.Ticker(t).info
        pe = info.get("trailingPE")
        g = info.get("revenueQuarterlyGrowth", info.get("revenueGrowth"))
        if pe is None or pe <= 0 or g is None:
            out[t] = []
            continue
        try:
            ann = yf.Ticker(t).get_earnings_dates(limit=1).index[0].date()
        except Exception:
            ann = date.today()
        out[t] = [
            FundamentalRecord(
                ticker=t, announce_date=ann, trailing_pe=float(pe), rev_growth_quarterly=float(g)
            )
        ]
    return out


def pit_lookup(records: list[FundamentalRecord], day: date) -> FundamentalRecord | None:
    """Latest record with announce_date <= day (forward-fill). None if day precedes all."""
    anns = [r.announce_date for r in records]  # records assumed sorted by announce_date
    i = bisect_right(anns, day) - 1
    return records[i] if i >= 0 else None
