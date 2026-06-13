"""Async ingestion layer: validated PriceBars -> SQLite via aiosqlite."""

from pathlib import Path

import aiosqlite

from .models import PriceBar

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
"""

_UPSERT = """
INSERT INTO price_bars (ticker, bar_date, open, high, low, close, volume)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (ticker, bar_date) DO UPDATE SET
    open = excluded.open, high = excluded.high, low = excluded.low,
    close = excluded.close, volume = excluded.volume
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


async def bar_count(conn: aiosqlite.Connection, ticker: str | None = None) -> int:
    if ticker:
        cur = await conn.execute("SELECT COUNT(*) FROM price_bars WHERE ticker = ?", (ticker,))
    else:
        cur = await conn.execute("SELECT COUNT(*) FROM price_bars")
    (n,) = await cur.fetchone()
    return int(n)
