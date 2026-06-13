"""CatBoost pipeline: predict 3-day forward residual alpha."""

import math
from dataclasses import dataclass

import numpy as np
from catboost import CatBoostRegressor

from .config import Settings
from .models import FeatureRow

FEATURE_COLS = (
    "log_ret_1d",
    "spread",
    "spread_z",
    "frac_diff_close",
    "garch_vol",
    "garch_vol_ratio",
    "mom_5d",
    "mom_10d",
    "mom_20d",
    "beta_60d",
    "trailing_pe",
    "rev_growth_quarterly",
)


@dataclass(frozen=True)
class TrainResult:
    model: CatBoostRegressor
    val_rmse: float
    n_train: int
    n_val: int
    n_purged: int
    val_residuals: tuple[float, ...]  # OOS errors for conformal calibration


def _matrix(rows: list[FeatureRow]) -> np.ndarray:
    return np.array([[getattr(r, c) for c in FEATURE_COLS] for r in rows], dtype=np.float64)


def purged_embargo_split(
    rows: list[FeatureRow], val_fraction: float, horizon: int
) -> tuple[list[FeatureRow], list[FeatureRow]]:
    """Chronological split with purge + embargo (Lopez de Prado) -> zero label leakage.

    The forward target spans `horizon` rows, so a train row's label window can reach
    into the validation block. Validation is the chronological tail (val_fraction).
      - Purge: drop the `horizon` train rows immediately before the val block; their
        forward labels overlap the val period.
      - Embargo: also drop `horizon` rows immediately AFTER the val block (no-op when
        val is the tail) so val labels never contaminate later train autocorrelation.
    Result: a strict >= `horizon` gap between any train row and the val block.
    """
    s = sorted(rows, key=lambda r: r.bar_date)
    n = len(s)
    cut = int(n * (1 - val_fraction))
    val = s[cut:]
    train_head = s[: max(0, cut - horizon)]  # purge overlapping labels before val
    train_tail = s[min(n, cut + len(val) + horizon) :]  # embargo after val (empty for tail)
    return train_head + train_tail, val


def train(rows: list[FeatureRow], settings: Settings) -> TrainResult:
    """Purged+embargo chronological split (no shuffle — forward-target leakage guard)."""
    labelled = sorted((r for r in rows if r.fwd_alpha_3d is not None), key=lambda r: r.bar_date)
    if len(labelled) < 100:
        raise ValueError(f"need >=100 labelled rows, got {len(labelled)}")
    cut = int(len(labelled) * (1 - settings.val_fraction))
    tr, va = purged_embargo_split(labelled, settings.val_fraction, settings.forward_horizon)
    n_purged = cut - len(tr)

    model = CatBoostRegressor(
        iterations=settings.iterations,
        learning_rate=settings.learning_rate,
        depth=settings.depth,
        loss_function="RMSE",
        random_seed=settings.synthetic_seed,
        verbose=False,
    )
    y_tr = np.array([r.fwd_alpha_3d for r in tr])
    y_va = np.array([r.fwd_alpha_3d for r in va])
    model.fit(_matrix(tr), y_tr, eval_set=(_matrix(va), y_va), use_best_model=True)

    pred_va = model.predict(_matrix(va))
    resid = pred_va - y_va
    rmse = float(math.sqrt(np.mean(resid**2)))
    if not math.isfinite(rmse):
        raise ValueError("validation RMSE not finite — degenerate training data")
    return TrainResult(
        model=model,
        val_rmse=rmse,
        n_train=len(tr),
        n_val=len(va),
        n_purged=n_purged,
        val_residuals=tuple(float(x) for x in resid),
    )


def predict_latest(model: CatBoostRegressor, rows: list[FeatureRow]) -> dict[str, tuple]:
    """Predict on each ticker's most recent feature row. Returns ticker -> (date, alpha)."""
    latest: dict[str, FeatureRow] = {}
    for r in rows:
        if r.ticker not in latest or r.bar_date > latest[r.ticker].bar_date:
            latest[r.ticker] = r
    preds = model.predict(_matrix(list(latest.values())))
    return {r.ticker: (r.bar_date, float(p)) for r, p in zip(latest.values(), preds)}
