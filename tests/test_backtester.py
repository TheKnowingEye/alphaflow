"""Walk-forward backtester: OOS metrics are well-formed and friction is enforced."""

import pytest

from alphaflow.config import Settings
from alphaflow.data_source import synthetic_history
from alphaflow.features import compute_features
from alphaflow.ingestion import ingest_bars, open_db
from execution.backtester import BacktestMetrics, _positions, walk_forward

# Small/fast backtest config: few rebalances, single retrain, tiny CatBoost.
SETTINGS = Settings(
    db_path=":memory:",
    bt_min_train=120,
    bt_step=20,
    bt_retrain_every=100,
    bt_iterations=20,
)


@pytest.fixture
async def rows():
    conn = await open_db(":memory:")
    history = synthetic_history(SETTINGS.asset_tickers, SETTINGS.benchmark_ticker, days=420, seed=5)
    for bars in history.values():
        await ingest_bars(conn, bars)
    out = await compute_features(conn, SETTINGS)
    await conn.close()
    return out


async def test_walk_forward_metrics_well_formed(rows):
    m = walk_forward(rows, SETTINGS)
    assert isinstance(m, BacktestMetrics)
    assert m.n_rebalances > 0
    assert 0.0 <= m.hit_rate <= 1.0
    assert m.max_drawdown >= 0.0
    assert m.annual_turnover >= 0.0
    assert m.txn_cost_per_side == SETTINGS.txn_cost_per_side
    # JSON payload is strict-validatable
    assert BacktestMetrics.model_validate_json(m.model_dump_json()) == m


def test_friction_throttles_subthreshold_signals():
    cost = 0.0005  # round-trip threshold = 0.001
    preds = {"NVDA": 0.05, "AMD": 0.0003, "INTC": -0.04}
    pos = _positions(preds, cost)
    # AMD alpha cannot clear round-trip cost -> throttled out
    assert "AMD" not in pos
    assert set(pos) == {"NVDA", "INTC"}
    # gross-normalised long/short book
    assert sum(abs(w) for w in pos.values()) == pytest.approx(1.0)
    assert pos["INTC"] < 0 < pos["NVDA"]


def test_zero_positions_when_all_subthreshold():
    assert _positions({"NVDA": 0.0001, "AMD": -0.0002}, 0.0005) == {}
