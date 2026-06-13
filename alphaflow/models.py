"""Strict Pydantic v2 data models for every pipeline boundary."""

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

STRICT = ConfigDict(strict=True, frozen=True, extra="forbid")


class PriceBar(BaseModel):
    """One validated OHLCV bar."""

    model_config = STRICT

    ticker: str = Field(min_length=1, max_length=12)
    bar_date: date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @model_validator(mode="after")
    def check_range(self) -> "PriceBar":
        if self.high < self.low:
            raise ValueError(f"high {self.high} < low {self.low}")
        if not (self.low <= self.open <= self.high):
            raise ValueError("open outside [low, high]")
        if not (self.low <= self.close <= self.high):
            raise ValueError("close outside [low, high]")
        return self


class FundamentalRecord(BaseModel):
    """One quarterly fundamental snapshot, keyed by its official announcement date.

    `announce_date` is the Point-In-Time boundary: the metric only applies to daily
    rows on/after this date (no lookahead).
    """

    model_config = STRICT

    ticker: str = Field(min_length=1, max_length=12)
    announce_date: date
    trailing_pe: float = Field(gt=0)
    rev_growth_quarterly: float

    @field_validator("trailing_pe", "rev_growth_quarterly")
    @classmethod
    def finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("fundamental must be finite")
        return v


class FeatureRow(BaseModel):
    """One engineered feature row (target optional: NULL on last horizon rows)."""

    model_config = STRICT

    ticker: str
    bar_date: date
    log_ret_1d: float
    spread: float
    spread_z: float
    frac_diff_close: float  # fractionally differentiated log price (stationary, long-memory)
    garch_vol: float = Field(gt=0)  # GARCH(1,1) 1d-ahead conditional vol forecast
    garch_vol_ratio: float = Field(gt=0)  # asset GARCH vol / benchmark GARCH vol
    mom_5d: float
    mom_10d: float
    mom_20d: float
    beta_60d: float
    trailing_pe: float = Field(gt=0)  # PIT-aligned fundamental
    rev_growth_quarterly: float  # PIT-aligned YoY quarterly revenue growth
    fwd_alpha_3d: float | None = None

    @field_validator(
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
    @classmethod
    def finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("feature must be finite")
        return v


class Action(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class Signal(BaseModel):
    """Per-asset trade signal — the unit of the public JSON payload.

    Output contract fields: timestamp, asset_ticker, target_action,
    allocation_weight, model_confidence_score. as_of_date / predicted_alpha
    are kept for traceability.
    """

    model_config = STRICT

    timestamp: datetime
    asset_ticker: str = Field(min_length=1, max_length=12)
    target_action: Action
    allocation_weight: float = Field(ge=0.0, le=1.0)
    model_confidence_score: float = Field(ge=0.0, le=1.0)  # conformal empirical CDF
    as_of_date: date
    predicted_alpha: float
    # Distribution-free conformal interval around predicted_alpha (None if uncalibrated)
    alpha_ci_low: float | None = None
    alpha_ci_high: float | None = None

    @model_validator(mode="after")
    def weight_consistent(self) -> "Signal":
        if self.target_action is not Action.BUY and self.allocation_weight != 0.0:
            raise ValueError("non-BUY signal must carry zero allocation_weight")
        return self


class SignalBatch(BaseModel):
    """Validated JSON signal payload — the pipeline's public output schema."""

    model_config = STRICT

    model_version: str
    generated_at: datetime
    benchmark: str
    signals: list[Signal]

    @model_validator(mode="after")
    def weights_sum(self) -> "SignalBatch":
        total = sum(s.allocation_weight for s in self.signals)
        if total > 1.0 + 1e-9:
            raise ValueError(f"allocation weights sum {total:.6f} > 1.0")
        return self
