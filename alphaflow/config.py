"""Pipeline configuration + dual-region multi-factor sector registry."""

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

# --- Global Multi-Factor Registry ------------------------------------------------
# REGION -> SECTOR -> {macro_benchmark, sector_benchmark, fundamental_keys}.
# Macro is denormalised onto every sector entry (per the registry contract).

SECTORS: tuple[str, ...] = ("Tech", "Finance", "Energy", "Materials", "FMCG", "Pharma")

# Heterogeneous fundamental keys per sector (yfinance .info keys). Two slots each ->
# fixed-width matrix even though semantics differ across sectors.
_FUNDAMENTAL_KEYS: dict[str, tuple[str, str]] = {
    "Tech": ("trailingPE", "revenueGrowth"),
    "Finance": ("priceToBook", "trailingPE"),
    "Energy": ("enterpriseToEbitda", "trailingPE"),
    "Materials": ("priceToBook", "trailingPE"),
    "FMCG": ("trailingPE", "profitMargins"),
    "Pharma": ("trailingPE", "revenueGrowth"),
}

_MACRO: dict[str, str] = {"US": "SPY", "IN": "^NSEI"}

_SECTOR_BENCH: dict[str, dict[str, str]] = {
    "US": {
        "Tech": "XLK",
        "Finance": "XLF",
        "Energy": "XLE",
        "Materials": "XLB",
        "FMCG": "XLP",
        "Pharma": "XLV",
    },
    "IN": {
        "Tech": "^CNXIT",
        "Finance": "^NSEBANK",
        "Energy": "^CNXENERGY",
        "Materials": "^CNXMETAL",
        "FMCG": "^CNXFMCG",
        "Pharma": "^CNXPHARMA",
    },
}

GLOBAL_REGISTRY: dict[str, dict[str, dict[str, object]]] = {
    region: {
        sector: {
            "macro_benchmark": _MACRO[region],
            "sector_benchmark": _SECTOR_BENCH[region][sector],
            "fundamental_keys": _FUNDAMENTAL_KEYS[sector],
        }
        for sector in SECTORS
    }
    for region in ("US", "IN")
}

# Integer codes for the CatBoost matrix (region/sector enter as numeric features).
REGION_ID: dict[str, int] = {"US": 0, "IN": 1}
SECTOR_ID: dict[str, int] = {s: i for i, s in enumerate(SECTORS)}


@dataclass(frozen=True)
class Holding:
    """A resolved (ticker, sector) mapped to its region + benchmarks + fund keys."""

    ticker: str
    sector: str
    region: str
    macro_benchmark: str
    sector_benchmark: str
    fundamental_keys: tuple[str, str]


def detect_region(ticker: str) -> str:
    """Auto-router: .NS/.BO suffix -> India, else US."""
    return "IN" if ticker.upper().endswith((".NS", ".BO")) else "US"


def resolve_holding(ticker: str, sector: str) -> Holding:
    """Map (ticker, sector) -> Holding via the registry; region auto-detected."""
    region = detect_region(ticker)
    if sector not in GLOBAL_REGISTRY[region]:
        raise ValueError(f"unknown sector {sector!r} for region {region!r}")
    entry = GLOBAL_REGISTRY[region][sector]
    return Holding(
        ticker=ticker,
        sector=sector,
        region=region,
        macro_benchmark=str(entry["macro_benchmark"]),
        sector_benchmark=str(entry["sector_benchmark"]),
        fundamental_keys=tuple(entry["fundamental_keys"]),  # type: ignore[arg-type]
    )


# --- Pipeline settings -----------------------------------------------------------


class Settings(BaseModel):
    """Static pipeline settings."""

    model_config = {"frozen": True}

    # Universe: list of (ticker, sector). Region auto-detected per ticker.
    universe: tuple[tuple[str, str], ...] = (("NVDA", "Tech"), ("AMD", "Tech"))
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

    def holdings(self) -> tuple[Holding, ...]:
        return tuple(resolve_holding(t, s) for t, s in self.universe)


SETTINGS = Settings()
