"""Split-conformal calibration: distribution-free confidence from past residuals.

Replaces the parametric `erf` heuristic. Given out-of-sample residuals
|y_true - y_pred| over a sliding window, the empirical CDF turns a predicted alpha
into a calibrated confidence (fraction of past errors it exceeds), and an empirical
quantile gives a distribution-free prediction interval — no Gaussian assumption.
"""

import math
from bisect import bisect_right


class ConformalCalibrator:
    """Holds recent absolute residuals; serves confidence + interval half-widths."""

    def __init__(self, residuals: list[float], window: int | None = None) -> None:
        vals = [abs(r) for r in residuals if isinstance(r, (int, float)) and math.isfinite(r)]
        if window is not None and window > 0:
            vals = vals[-window:]  # sliding window of most-recent residuals (pre-sort)
        self._sorted = sorted(vals)

    def __len__(self) -> int:
        return len(self._sorted)

    def confidence(self, value: float) -> float:
        """Empirical CDF of |value|: fraction of past residuals it meets or exceeds. [0,1]."""
        n = len(self._sorted)
        if n == 0:
            return 0.0
        return bisect_right(self._sorted, abs(value)) / n

    def interval(self, level: float) -> float:
        """Half-width of the `level` prediction interval = empirical quantile of |residual|."""
        n = len(self._sorted)
        if n == 0:
            return 0.0
        idx = min(n - 1, max(0, math.ceil(level * n) - 1))
        return self._sorted[idx]
