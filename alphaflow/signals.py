"""Signal generation: predicted residual alpha -> validated JSON payload."""

import math
from datetime import date, datetime, timezone
from pathlib import Path

from . import __version__
from .conformal import ConformalCalibrator
from .config import Settings
from .models import Action, Signal, SignalBatch


def build_signals(
    predictions: dict[str, tuple[date, float]],
    settings: Settings,
    calibrator: ConformalCalibrator,
) -> SignalBatch:
    """BUY/SELL on alpha thresholds; softmax allocation over BUY alphas only.

    Confidence + interval are distribution-free conformal estimates from the
    calibrator's sliding window of out-of-sample residuals.
    """
    now = datetime.now(timezone.utc)
    half_width = calibrator.interval(settings.conformal_level)
    actions: dict[str, Action] = {}
    for t, (_, alpha) in predictions.items():
        if alpha > settings.buy_threshold:
            actions[t] = Action.BUY
        elif alpha < settings.sell_threshold:
            actions[t] = Action.SELL
        else:
            actions[t] = Action.HOLD

    buys = {t: predictions[t][1] for t, a in actions.items() if a is Action.BUY}
    weights: dict[str, float] = {t: 0.0 for t in predictions}
    if buys:
        # scaled softmax: alphas are small (log-return units) -> scale for contrast
        exps = {t: math.exp(alpha * 100) for t, alpha in buys.items()}
        total = sum(exps.values())
        for t, e in exps.items():
            # floor at 6 dp: rounding up could push the batch sum past 1.0
            weights[t] = math.floor(e / total * 1e6) / 1e6

    signals = [
        Signal(
            timestamp=now,
            asset_ticker=t,
            target_action=actions[t],
            allocation_weight=weights[t],
            model_confidence_score=calibrator.confidence(predictions[t][1]),
            as_of_date=predictions[t][0],
            predicted_alpha=predictions[t][1],
            alpha_ci_low=predictions[t][1] - half_width,
            alpha_ci_high=predictions[t][1] + half_width,
        )
        for t in sorted(predictions)
    ]
    return SignalBatch(
        model_version=__version__,
        generated_at=now,
        benchmark=settings.benchmark_ticker,
        signals=signals,
    )


def write_signals(batch: SignalBatch, path: Path) -> None:
    path.write_text(batch.model_dump_json(indent=2), encoding="utf-8")
