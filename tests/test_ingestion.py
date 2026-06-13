import pytest

from alphaflow.data_source import synthetic_prices
from alphaflow.fundamentals import synthetic_fundamentals
from alphaflow.ingestion import (
    bar_count,
    ingest_bars,
    ingest_fundamentals,
    load_fundamentals,
    open_db,
)
from alphaflow.config import resolve_holding


@pytest.fixture
async def conn():
    c = await open_db(":memory:")
    yield c
    await c.close()


@pytest.fixture
def prices():
    return synthetic_prices(
        ("NVDA", "SPY", "XLK"), days=120, seed=7, low_idio=frozenset({"SPY", "XLK"})
    )


async def test_ingest_counts(conn, prices):
    for bars in prices.values():
        await ingest_bars(conn, bars)
    n_expected = sum(len(b) for b in prices.values())
    assert await bar_count(conn) == n_expected
    assert await bar_count(conn, "NVDA") == len(prices["NVDA"])


async def test_ingest_idempotent(conn, prices):
    await ingest_bars(conn, prices["NVDA"])
    first = await bar_count(conn)
    await ingest_bars(conn, prices["NVDA"])  # re-ingest same rows
    assert await bar_count(conn) == first


async def test_fundamentals_json_roundtrip(conn):
    h = resolve_holding("HDFCBANK.NS", "Finance")  # heterogeneous keys: priceToBook, trailingPE
    funds = synthetic_fundamentals((h,), days=400, seed=3)
    n = await ingest_fundamentals(conn, funds[h.ticker])
    assert n == len(funds[h.ticker])
    loaded = await load_fundamentals(conn, h.ticker)
    assert [r.model_dump() for r in loaded] == [r.model_dump() for r in funds[h.ticker]]
    # heterogeneous keys survived the JSON column
    assert set(loaded[0].metrics) == {"priceToBook", "trailingPE"}


async def test_fundamentals_upsert(conn):
    h = resolve_holding("NVDA", "Tech")
    funds = synthetic_fundamentals((h,), days=200, seed=1)[h.ticker]
    await ingest_fundamentals(conn, funds)
    await ingest_fundamentals(conn, funds)  # idempotent
    loaded = await load_fundamentals(conn, "NVDA")
    assert len(loaded) == len(funds)
