"""Fundamental ingestion + Point-In-Time alignment (heterogeneous keys).

Each holding's sector defines its `fundamental_keys`. Metrics are stored as a JSON
map keyed by announce date. PIT rule: a record applies only to daily rows on/after
its announce date, forward-filled until the next release -> no lookahead.

Default source is deterministic synthetic. `live=True` pulls yfinance `.info` for the
sector's keys, stamped at the latest earnings date (degraded PIT: the vendor exposes
no historical fundamental series).
"""

import asyncio
import random
from bisect import bisect_right
from datetime import date, timedelta

from .config import Holding
from .models import FundamentalRecord

_QUARTER_DAYS = 91

# Per-key synthetic generator: (low, high, quarterly_relative_vol, additive?).
_KEY_RANGE: dict[str, tuple[float, float, float, bool]] = {
    "trailingPE": (15.0, 60.0, 0.12, False),
    "revenueGrowth": (-0.10, 0.50, 0.08, True),
    "priceToBook": (1.0, 6.0, 0.10, False),
    "enterpriseToEbitda": (6.0, 18.0, 0.10, False),
    "profitMargins": (0.05, 0.35, 0.06, True),
}
_DEFAULT_RANGE = (10.0, 40.0, 0.10, False)


def _synthetic_key_series(ticker: str, key: str, seed: int, n: int) -> list[float]:
    rng = random.Random(f"{seed}:fund:{ticker}:{key}")
    low, high, vol, additive = _KEY_RANGE.get(key, _DEFAULT_RANGE)
    val = rng.uniform(low, high)
    out: list[float] = []
    for _ in range(n):
        if additive:
            val = val + rng.gauss(0, vol)
        else:
            val = max(0.5, val * (1 + rng.gauss(0, vol)))
        out.append(round(val, 4))
    return out


def synthetic_fundamentals(
    holdings: tuple[Holding, ...], days: int, seed: int, end: date | None = None
) -> dict[str, list[FundamentalRecord]]:
    """Deterministic quarterly records per holding. First announce precedes the window
    so PIT align covers every in-window daily row."""
    end = end or date(2026, 6, 12)
    start = end - timedelta(days=days)
    dates: list[date] = []
    d = start - timedelta(days=_QUARTER_DAYS)  # one release before the window
    while d <= end:
        dates.append(d)
        d += timedelta(days=_QUARTER_DAYS)

    out: dict[str, list[FundamentalRecord]] = {}
    for h in holdings:
        series = {
            k: _synthetic_key_series(h.ticker, k, seed, len(dates)) for k in h.fundamental_keys
        }
        out[h.ticker] = [
            FundamentalRecord(
                ticker=h.ticker,
                announce_date=ad,
                metrics={k: series[k][i] for k in h.fundamental_keys},
            )
            for i, ad in enumerate(dates)
        ]
    return out


async def fetch_fundamentals(
    holdings: tuple[Holding, ...], days: int, seed: int, live: bool = False
) -> dict[str, list[FundamentalRecord]]:
    if not live:
        return synthetic_fundamentals(holdings, days, seed)
    return await asyncio.to_thread(_fetch_yf_fundamentals, holdings, days)


def _fetch_yf_fundamentals(
    holdings: tuple[Holding, ...], days: int
) -> dict[str, list[FundamentalRecord]]:
    import yfinance as yf

    # Degraded PIT: yfinance .info exposes only a current snapshot (no historical
    # fundamental series). Backdate the single record to before the price window so it
    # forward-fills across all rows. NOT true PIT — flagged. Use synthetic for real PIT.
    ann = date.today() - timedelta(days=days + _QUARTER_DAYS)
    out: dict[str, list[FundamentalRecord]] = {}
    for h in holdings:
        try:
            info = yf.Ticker(h.ticker).info
        except Exception:
            out[h.ticker] = []
            continue
        metrics = {
            k: (float(v) if isinstance((v := info.get(k)), (int, float)) else 0.0)
            for k in h.fundamental_keys
        }
        out[h.ticker] = [FundamentalRecord(ticker=h.ticker, announce_date=ann, metrics=metrics)]
    return out


def pit_lookup(records: list[FundamentalRecord], day: date) -> FundamentalRecord | None:
    """Latest record with announce_date <= day (forward-fill). None if day precedes all."""
    anns = [r.announce_date for r in records]  # records assumed sorted by announce_date
    i = bisect_right(anns, day) - 1
    return records[i] if i >= 0 else None
