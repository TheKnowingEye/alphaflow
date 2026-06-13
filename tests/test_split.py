"""Purged + embargo split: prove zero label overlap between train and validation."""

from datetime import date, timedelta
from types import SimpleNamespace

from alphaflow.model import purged_embargo_split

HORIZON = 3


def _rows(n):
    # purged_embargo_split only reads .bar_date — SimpleNamespace stands in for FeatureRow.
    return [SimpleNamespace(bar_date=date(2026, 1, 1) + timedelta(days=i)) for i in range(n)]


def test_val_is_chronological_tail():
    rows = _rows(200)
    train, val = purged_embargo_split(rows, val_fraction=0.2, horizon=HORIZON)
    assert val == rows[160:]
    assert all(t.bar_date < v.bar_date for t in train for v in val[:1])


def test_purge_drops_horizon_rows_before_val():
    rows = _rows(200)
    train, val = purged_embargo_split(rows, val_fraction=0.2, horizon=HORIZON)
    # cut = 160; purge removes rows [157, 160) -> 157 train rows
    assert len(train) == 160 - HORIZON
    assert train[-1].bar_date == rows[156].bar_date


def test_gap_at_least_horizon_no_overlap():
    rows = _rows(200)
    train, val = purged_embargo_split(rows, val_fraction=0.2, horizon=HORIZON)
    last_train = max(r.bar_date for r in train)
    first_val = min(r.bar_date for r in val)
    # a train row's H-forward label must end strictly before the val block begins
    assert (first_val - last_train).days >= HORIZON
