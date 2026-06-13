from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from alphaflow.models import Action, PriceBar, Signal, SignalBatch


def _bar(**kw):
    base = dict(
        ticker="NVDA",
        bar_date=date(2026, 6, 1),
        open=100.0,
        high=105.0,
        low=99.0,
        close=104.0,
        volume=1e6,
    )
    base.update(kw)
    return PriceBar(**base)


def test_valid_bar():
    assert _bar().close == 104.0


def test_high_below_low_rejected():
    with pytest.raises(ValidationError, match="high"):
        _bar(high=98.0, low=99.0, open=98.5, close=98.5)


def test_close_outside_range_rejected():
    with pytest.raises(ValidationError):
        _bar(close=200.0)


def test_strict_type_rejected():
    with pytest.raises(ValidationError):
        _bar(close="104.0")


def test_negative_price_rejected():
    with pytest.raises(ValidationError):
        _bar(open=-1.0)


def test_non_buy_with_weight_rejected():
    with pytest.raises(ValidationError, match="zero allocation_weight"):
        Signal(
            timestamp=datetime.now(timezone.utc),
            asset_ticker="AMD",
            target_action=Action.SELL,
            allocation_weight=0.5,
            model_confidence_score=0.8,
            as_of_date=date(2026, 6, 12),
            predicted_alpha=-0.01,
        )


def test_batch_weight_sum_capped():
    def sig(t, w):
        return Signal(
            timestamp=datetime.now(timezone.utc),
            asset_ticker=t,
            target_action=Action.BUY,
            allocation_weight=w,
            model_confidence_score=0.8,
            as_of_date=date(2026, 6, 12),
            predicted_alpha=0.01,
        )

    with pytest.raises(ValidationError, match="weights sum"):
        SignalBatch(
            model_version="0.1.0",
            generated_at=datetime.now(timezone.utc),
            benchmark="GLOBAL",
            signals=[sig("NVDA", 0.7), sig("AMD", 0.7)],
        )
