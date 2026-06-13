import pytest

from alphaflow.data_source import synthetic_history
from alphaflow.ingestion import bar_count, ingest_bars, open_db


@pytest.fixture
async def conn():
    c = await open_db(":memory:")
    yield c
    await c.close()


@pytest.fixture
def history():
    return synthetic_history(("NVDA", "AMD"), "MAHKTECH.NS", days=120, seed=7)


async def test_ingest_counts(conn, history):
    for bars in history.values():
        await ingest_bars(conn, bars)
    n_expected = sum(len(b) for b in history.values())
    assert await bar_count(conn) == n_expected
    assert await bar_count(conn, "NVDA") == len(history["NVDA"])


async def test_ingest_idempotent(conn, history):
    await ingest_bars(conn, history["NVDA"])
    first = await bar_count(conn)
    await ingest_bars(conn, history["NVDA"])  # re-ingest same rows
    assert await bar_count(conn) == first


async def test_upsert_overwrites(conn, history):
    bars = history["NVDA"]
    await ingest_bars(conn, bars)
    patched = bars[0].model_copy(update={"volume": 999.0})
    await ingest_bars(conn, [patched])
    cur = await conn.execute(
        "SELECT volume FROM price_bars WHERE ticker = ? AND bar_date = ?",
        (patched.ticker, patched.bar_date.isoformat()),
    )
    (vol,) = await cur.fetchone()
    assert vol == 999.0
