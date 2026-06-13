"""Rolling walk-forward backtester with a friction engine.

Out-of-sample protocol per rebalance date d:
  - Purge: train only on rows whose 3d-forward label is fully realised before d
    (bar_date index <= d_index - horizon) -> no look-ahead into the decision.
  - Retrain CatBoost every `bt_retrain_every` rebalances (rolling).
  - Size a long/short book from predicted residual alpha, gross-normalised to 1.
  - Friction engine: a name must clear a round-trip cost (2 * txn_cost_per_side) on
    predicted alpha to be traded; realised return is netted of turnover * cost.

Reports OOS Sharpe, hit-rate (directional alpha accuracy), max drawdown, and
annualised turnover via a strict Pydantic `BacktestMetrics` payload.
"""

import math

import numpy as np
from catboost import CatBoostRegressor
from pydantic import BaseModel, ConfigDict, Field

from alphaflow.config import Settings
from alphaflow.model import _matrix
from alphaflow.models import FeatureRow

_TRADING_DAYS = 252


class BacktestMetrics(BaseModel):
    """Validated out-of-sample backtest report."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    n_rebalances: int = Field(ge=0)
    n_trades: int = Field(ge=0)
    sharpe: float
    hit_rate: float = Field(ge=0.0, le=1.0)
    max_drawdown: float = Field(ge=0.0)
    annual_turnover: float = Field(ge=0.0)
    total_return: float
    txn_cost_per_side: float = Field(ge=0.0)


def _fit(train_rows: list[FeatureRow], settings: Settings) -> CatBoostRegressor:
    y = np.array([r.fwd_alpha_3d for r in train_rows], dtype=np.float64)
    model = CatBoostRegressor(
        iterations=settings.bt_iterations,
        learning_rate=settings.learning_rate,
        depth=settings.depth,
        loss_function="RMSE",
        random_seed=settings.synthetic_seed,
        verbose=False,
    )
    model.fit(_matrix(train_rows), y)
    return model


def _positions(preds: dict[str, float], cost_per_side: float) -> dict[str, float]:
    """Long/short book, gross-normalised to 1. Friction throttle: drop names whose
    predicted alpha cannot clear a round-trip cost."""
    tradable = {t: p for t, p in preds.items() if abs(p) > 2 * cost_per_side}
    gross = sum(abs(p) for p in tradable.values())
    if gross <= 0:
        return {}
    return {t: p / gross for t, p in tradable.items()}  # signed weights, sum|w| = 1


def walk_forward(rows: list[FeatureRow], settings: Settings) -> BacktestMetrics:
    labelled = [r for r in rows if r.fwd_alpha_3d is not None]
    dates = sorted({r.bar_date for r in labelled})
    by_date: dict[object, list[FeatureRow]] = {}
    for r in labelled:
        by_date.setdefault(r.bar_date, []).append(r)

    h = settings.forward_horizon
    cost = settings.txn_cost_per_side
    idx_of = {d: i for i, d in enumerate(dates)}

    model: CatBoostRegressor | None = None
    prev_pos: dict[str, float] = {}
    rets: list[float] = []
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    turnover_sum = 0.0
    n_trades = 0
    hits = 0
    fold = 0

    for d in dates[settings.bt_min_train :: settings.bt_step]:
        cut_i = idx_of[d] - h  # purge: labels must be realised strictly before d
        if cut_i <= 0:
            continue
        cutoff = dates[cut_i]
        train_rows = [r for r in labelled if r.bar_date <= cutoff]
        if len(train_rows) < settings.bt_min_train:
            continue
        if model is None or fold % settings.bt_retrain_every == 0:
            model = _fit(train_rows, settings)
        fold += 1

        today = by_date[d]
        preds = {r.ticker: float(p) for r, p in zip(today, model.predict(_matrix(today)))}
        realized = {r.ticker: r.fwd_alpha_3d for r in today}
        pos = _positions(preds, cost)

        names = set(pos) | set(prev_pos)
        turn = sum(abs(pos.get(t, 0.0) - prev_pos.get(t, 0.0)) for t in names)
        turnover_sum += turn

        gross_ret = sum(w * realized[t] for t, w in pos.items())
        net = gross_ret - turn * cost
        rets.append(net)
        equity *= 1.0 + net
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)

        for t in pos:
            n_trades += 1
            if math.copysign(1.0, preds[t]) == math.copysign(1.0, realized[t]):
                hits += 1
        prev_pos = pos

    n_rebal = len(rets)
    periods_per_year = _TRADING_DAYS / settings.bt_step
    if n_rebal > 1 and (sd := float(np.std(rets, ddof=1))) > 0:
        sharpe = float(np.mean(rets)) / sd * math.sqrt(periods_per_year)
    else:
        sharpe = 0.0
    annual_turnover = (turnover_sum / n_rebal * periods_per_year) if n_rebal else 0.0

    return BacktestMetrics(
        n_rebalances=n_rebal,
        n_trades=n_trades,
        sharpe=sharpe,
        hit_rate=(hits / n_trades) if n_trades else 0.0,
        max_drawdown=max_dd,
        annual_turnover=annual_turnover,
        total_return=equity - 1.0,
        txn_cost_per_side=cost,
    )
