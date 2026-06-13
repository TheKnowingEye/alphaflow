"""Pipeline configuration."""

from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Static pipeline settings."""

    model_config = {"frozen": True}

    asset_tickers: tuple[str, ...] = ("NVDA", "AMD")  # leading indicators
    benchmark_ticker: str = "MAHKTECH.NS"  # Mirae Asset Hang Seng TECH ETF (lagging vector)
    db_path: Path = Path("alphaflow.db")
    history_days: int = 730
    synthetic_seed: int = 42

    # Feature windows
    spread_z_window: int = 20
    beta_window: int = 60
    momentum_windows: tuple[int, ...] = (5, 10, 20)
    forward_horizon: int = 3

    # Fractional differentiation (stationary price w/ long memory). Width < beta_window
    # so its warmup never extends past the beta warmup.
    frac_diff_d: float = 0.5
    frac_diff_width: int = 50

    # CatBoost
    iterations: int = 400
    learning_rate: float = 0.05
    depth: int = 6
    val_fraction: float = Field(0.2, gt=0.0, lt=1.0)

    # Signal thresholds (predicted 3d residual alpha, log-return units)
    buy_threshold: float = 0.004
    sell_threshold: float = -0.004
    signals_path: Path = Path("signals.json")

    # Friction engine (per trade side, fraction) — 5 bps default
    txn_cost_per_side: float = Field(0.0005, ge=0.0)

    # Conformal prediction (distribution-free confidence)
    conformal_window: int = 252
    conformal_level: float = Field(0.90, gt=0.0, lt=1.0)

    # Walk-forward backtest
    bt_min_train: int = 250
    bt_step: int = 3  # rebalance stride; = forward_horizon -> non-overlapping returns
    bt_retrain_every: int = 5  # retrain CatBoost every N rebalances
    bt_iterations: int = 150
    metrics_path: Path = Path("backtest_metrics.json")


SETTINGS = Settings()
