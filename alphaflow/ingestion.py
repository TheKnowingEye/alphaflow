"""Async ingestion layer: validated PriceBars + JSON fundamentals -> SQLite."""

import json
from datetime import date
from pathlib import Path

import aiosqlite

from .models import FundamentalRecord, PriceBar

_SCHEMA = """
CREATE TABLE IF NOT EXISTS price_bars (
    ticker   TEXT NOT NULL,
    bar_date TEXT NOT NULL,
    open     REAL NOT NULL,
    high     REAL NOT NULL,
    low      REAL NOT NULL,
    close    REAL NOT NULL,
    volume   REAL NOT NULL,
    PRIMARY KEY (ticker, bar_date)
);
CREATE INDEX IF NOT EXISTS idx_bars_date ON price_bars (bar_date);

-- Heterogeneous fundamentals: a JSON payload column keeps the schema fluid so a
-- Bank (priceToBook) and a Tech name (revenueGrowth) coexist without column churn.
CREATE TABLE IF NOT EXISTS fundamentals (
    ticker        TEXT NOT NULL,
    announce_date TEXT NOT NULL,
    payload       TEXT NOT NULL,   -- JSON: {metric_key: value}
    PRIMARY KEY (ticker, announce_date)
);
"""

_UPSERT = """
INSERT INTO price_bars (ticker, bar_date, open, high, low, close, volume)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (ticker, bar_date) DO UPDATE SET
    open = excluded.open, high = excluded.high, low = excluded.low,
    close = excluded.close, volume = excluded.volume
"""

_UPSERT_FUND = """
INSERT INTO fundamentals (ticker, announce_date, payload)
VALUES (?, ?, ?)
ON CONFLICT (ticker, announce_date) DO UPDATE SET payload = excluded.payload
"""


async def open_db(db_path: Path | str) -> aiosqlite.Connection:
    conn = await aiosqlite.connect(db_path)
    await conn.executescript(_SCHEMA)
    await conn.commit()
    return conn


async def ingest_bars(conn: aiosqlite.Connection, bars: list[PriceBar]) -> int:
    """Idempotent batch upsert of validated bars. Returns row count written."""
    rows = [
        (b.ticker, b.bar_date.isoformat(), b.open, b.high, b.low, b.close, b.volume) for b in bars
    ]
    await conn.executemany(_UPSERT, rows)
    await conn.commit()
    return len(rows)


async def ingest_fundamentals(conn: aiosqlite.Connection, records: list[FundamentalRecord]) -> int:
    """Idempotent upsert of fundamentals as JSON payloads. Returns rows written."""
    rows = [
        (r.ticker, r.announce_date.isoformat(), json.dumps(r.metrics, sort_keys=True))
        for r in records
    ]
    await conn.executemany(_UPSERT_FUND, rows)
    await conn.commit()
    return len(rows)


async def load_fundamentals(conn: aiosqlite.Connection, ticker: str) -> list[FundamentalRecord]:
    """Read PIT records for a ticker from the JSON column, sorted by announce_date."""
    cur = await conn.execute(
        "SELECT announce_date, payload FROM fundamentals WHERE ticker = ? ORDER BY announce_date",
        (ticker,),
    )
    return [
        FundamentalRecord(
            ticker=ticker,
            announce_date=date.fromisoformat(ad),
            metrics=json.loads(payload),
        )
        async for ad, payload in cur
    ]


async def bar_count(conn: aiosqlite.Connection, ticker: str | None = None) -> int:
    if ticker:
        cur = await conn.execute("SELECT COUNT(*) FROM price_bars WHERE ticker = ?", (ticker,))
    else:
        cur = await conn.execute("SELECT COUNT(*) FROM price_bars")
    (n,) = await cur.fetchone()
    return int(n)
