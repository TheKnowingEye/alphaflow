"""Point-In-Time alignment + heterogeneous metric dicts."""

from datetime import date

from alphaflow.config import resolve_holding
from alphaflow.fundamentals import pit_lookup, synthetic_fundamentals
from alphaflow.models import FundamentalRecord

RECS = [
    FundamentalRecord(
        ticker="NVDA",
        announce_date=date(2025, 1, 15),
        metrics={"trailingPE": 50.0, "revenueGrowth": 0.30},
    ),
    FundamentalRecord(
        ticker="NVDA",
        announce_date=date(2025, 4, 20),
        metrics={"trailingPE": 60.0, "revenueGrowth": 0.40},
    ),
]


def test_no_lookahead_before_first_announce():
    assert pit_lookup(RECS, date(2025, 1, 14)) is None


def test_applies_on_announce_day_forward():
    assert pit_lookup(RECS, date(2025, 1, 15)).metrics["trailingPE"] == 50.0
    assert pit_lookup(RECS, date(2025, 3, 1)).metrics["trailingPE"] == 50.0  # forward-filled


def test_switches_only_at_next_release():
    assert pit_lookup(RECS, date(2025, 4, 19)).metrics["trailingPE"] == 50.0  # no future leak
    assert pit_lookup(RECS, date(2025, 4, 20)).metrics["trailingPE"] == 60.0
    assert pit_lookup(RECS, date(2025, 5, 1)).metrics["revenueGrowth"] == 0.40


def test_heterogeneous_keys_per_sector():
    tech = synthetic_fundamentals((resolve_holding("NVDA", "Tech"),), days=400, seed=11)
    fin = synthetic_fundamentals((resolve_holding("JPM", "Finance"),), days=400, seed=11)
    assert set(tech["NVDA"][0].metrics) == {"trailingPE", "revenueGrowth"}
    assert set(fin["JPM"][0].metrics) == {"priceToBook", "trailingPE"}


def test_synthetic_deterministic_and_covers_window():
    h = resolve_holding("NVDA", "Tech")
    a = synthetic_fundamentals((h,), days=400, seed=11)
    b = synthetic_fundamentals((h,), days=400, seed=11)
    assert [r.model_dump() for r in a["NVDA"]] == [r.model_dump() for r in b["NVDA"]]
    assert a["NVDA"][0].announce_date < date(2026, 6, 12)
