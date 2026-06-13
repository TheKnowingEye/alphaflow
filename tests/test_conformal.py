"""Split-conformal calibrator: empirical CDF confidence + distribution-free interval."""

import math

from alphaflow.conformal import ConformalCalibrator

RESID = [0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, -0.08, 0.09, -0.10]


def test_confidence_is_empirical_cdf_monotonic():
    c = ConformalCalibrator(RESID)
    assert c.confidence(0.0) == 0.0
    assert c.confidence(0.055) == 0.5  # 5 of 10 |resid| <= 0.055
    assert c.confidence(1.0) == 1.0
    assert c.confidence(0.02) < c.confidence(0.08)


def test_interval_is_empirical_quantile():
    c = ConformalCalibrator(RESID)
    # 90th percentile of |resid| (0.01..0.10) -> 0.09
    assert c.interval(0.90) == 0.09
    assert c.interval(0.50) <= c.interval(0.90)


def test_sliding_window_keeps_recent():
    # window keeps the last 3 residuals (|0.08|,|0.09|,|0.10|) before sorting
    c = ConformalCalibrator(RESID, window=3)
    assert len(c) == 3
    assert c.interval(1.0) == 0.10
    assert c.confidence(0.085) == 1 / 3


def test_empty_calibrator_safe():
    c = ConformalCalibrator([])
    assert c.confidence(0.5) == 0.0
    assert c.interval(0.9) == 0.0


def test_non_finite_residuals_dropped():
    c = ConformalCalibrator([0.01, float("nan"), float("inf"), -0.02])
    assert len(c) == 2
    assert c.confidence(0.05) == 1.0
    assert all(math.isfinite(x) for x in (c.interval(0.9),))
