"""Point-In-Time alignment: a metric applies only on/after its announce date."""

from datetime import date

from alphaflow.fundamentals import pit_lookup, synthetic_fundamentals
from alphaflow.models import FundamentalRecord

RECS = [
    FundamentalRecord(
        ticker="NVDA", announce_date=date(2025, 1, 15), trailing_pe=50.0, rev_growth_quarterly=0.30
    ),
    FundamentalRecord(
        ticker="NVDA", announce_date=date(2025, 4, 20), trailing_pe=60.0, rev_growth_quarterly=0.40
    ),
]


def test_no_lookahead_before_first_announce():
    assert pit_lookup(RECS, date(2025, 1, 14)) is None


def test_applies_on_announce_day_forward():
    assert pit_lookup(RECS, date(2025, 1, 15)).trailing_pe == 50.0  # on the day
    assert pit_lookup(RECS, date(2025, 3, 1)).trailing_pe == 50.0  # forward-filled


def test_switches_only_at_next_release():
    # day before next release still carries the old quarter -> no future leak
    assert pit_lookup(RECS, date(2025, 4, 19)).trailing_pe == 50.0
    assert pit_lookup(RECS, date(2025, 4, 20)).trailing_pe == 60.0
    assert pit_lookup(RECS, date(2025, 5, 1)).rev_growth_quarterly == 0.40


def test_synthetic_deterministic_and_covers_window():
    a = synthetic_fundamentals(("NVDA", "AMD"), days=400, seed=11)
    b = synthetic_fundamentals(("NVDA", "AMD"), days=400, seed=11)
    assert [r.model_dump() for r in a["NVDA"]] == [r.model_dump() for r in b["NVDA"]]
    # first release precedes the price window start -> PIT covers every in-window row
    start = date(2026, 6, 12)
    assert a["NVDA"][0].announce_date < start
    assert all(r.trailing_pe > 0 for r in a["AMD"])
