from datetime import date

from alphaflow.conformal import ConformalCalibrator
from alphaflow.config import Settings
from alphaflow.models import Action, SignalBatch
from alphaflow.signals import build_signals, write_signals

SETTINGS = Settings()
D = date(2026, 6, 12)
# calibration residuals ~ |0.005..0.04| -> empirical CDF for confidence
CAL = ConformalCalibrator([0.005, 0.01, 0.02, 0.03, 0.04, -0.015, -0.025])


def test_actions_from_thresholds():
    preds = {"NVDA": (D, 0.02), "AMD": (D, -0.02), "INTC": (D, 0.001)}
    batch = build_signals(preds, SETTINGS, CAL)
    by = {s.asset_ticker: s for s in batch.signals}
    assert by["NVDA"].target_action is Action.BUY
    assert by["AMD"].target_action is Action.SELL
    assert by["INTC"].target_action is Action.HOLD
    assert by["AMD"].allocation_weight == 0.0
    assert by["INTC"].allocation_weight == 0.0


def test_buy_weights_softmax_sum_one():
    preds = {"NVDA": (D, 0.02), "AMD": (D, 0.01)}
    batch = build_signals(preds, SETTINGS, CAL)
    by = {s.asset_ticker: s for s in batch.signals}
    assert by["NVDA"].allocation_weight > by["AMD"].allocation_weight > 0
    assert sum(s.allocation_weight for s in batch.signals) <= 1.0


def test_conformal_confidence_and_interval():
    preds = {"NVDA": (D, 0.05), "AMD": (D, 0.005)}
    by = {s.asset_ticker: s for s in build_signals(preds, SETTINGS, CAL).signals}
    # larger |alpha| -> exceeds more past residuals -> higher empirical-CDF confidence
    assert 0.0 <= by["AMD"].model_confidence_score < by["NVDA"].model_confidence_score <= 1.0
    # distribution-free interval brackets the point prediction
    n = by["NVDA"]
    assert n.alpha_ci_low < n.predicted_alpha < n.alpha_ci_high


def test_no_buys_all_zero_weight():
    preds = {"NVDA": (D, -0.02), "AMD": (D, 0.0)}
    batch = build_signals(preds, SETTINGS, CAL)
    assert all(s.allocation_weight == 0.0 for s in batch.signals)


def test_json_roundtrip(tmp_path):
    preds = {"NVDA": (D, 0.02), "AMD": (D, -0.02)}
    batch = build_signals(preds, SETTINGS, CAL)
    path = tmp_path / "signals.json"
    write_signals(batch, path)
    # strict models: JSON-mode validation (str dates OK in JSON mode, not Python mode)
    reparsed = SignalBatch.model_validate_json(path.read_text(encoding="utf-8"))
    assert reparsed.signals == batch.signals
    assert reparsed.benchmark == "GLOBAL"
