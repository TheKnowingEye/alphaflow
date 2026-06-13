"""Dual-region registry + auto-router."""

import pytest

from alphaflow.config import GLOBAL_REGISTRY, SECTORS, detect_region, resolve_holding


def test_registry_shape_region_sector_keys():
    assert set(GLOBAL_REGISTRY) == {"US", "IN"}
    for region in ("US", "IN"):
        assert set(GLOBAL_REGISTRY[region]) == set(SECTORS)
        for sector in SECTORS:
            entry = GLOBAL_REGISTRY[region][sector]
            assert set(entry) == {"macro_benchmark", "sector_benchmark", "fundamental_keys"}


def test_macro_and_sector_benchmarks():
    assert GLOBAL_REGISTRY["US"]["Tech"]["macro_benchmark"] == "SPY"
    assert GLOBAL_REGISTRY["IN"]["Tech"]["macro_benchmark"] == "^NSEI"
    assert GLOBAL_REGISTRY["US"]["Tech"]["sector_benchmark"] == "XLK"
    assert GLOBAL_REGISTRY["US"]["Finance"]["sector_benchmark"] == "XLF"
    assert GLOBAL_REGISTRY["IN"]["Finance"]["sector_benchmark"] == "^NSEBANK"
    assert GLOBAL_REGISTRY["IN"]["Pharma"]["sector_benchmark"] == "^CNXPHARMA"


def test_fundamental_keys_per_sector():
    assert GLOBAL_REGISTRY["US"]["Tech"]["fundamental_keys"] == ("trailingPE", "revenueGrowth")
    assert GLOBAL_REGISTRY["US"]["Finance"]["fundamental_keys"] == ("priceToBook", "trailingPE")
    # consistent across regions
    assert (
        GLOBAL_REGISTRY["US"]["Finance"]["fundamental_keys"]
        == GLOBAL_REGISTRY["IN"]["Finance"]["fundamental_keys"]
    )


@pytest.mark.parametrize(
    "ticker,region",
    [("NVDA", "US"), ("JPM", "US"), ("TCS.NS", "IN"), ("HDFCBANK.NS", "IN"), ("RELIANCE.BO", "IN")],
)
def test_detect_region(ticker, region):
    assert detect_region(ticker) == region


def test_resolve_holding_routes_to_correct_benchmarks():
    us = resolve_holding("NVDA", "Tech")
    assert (us.region, us.macro_benchmark, us.sector_benchmark) == ("US", "SPY", "XLK")
    ind = resolve_holding("HDFCBANK.NS", "Finance")
    assert (ind.region, ind.macro_benchmark, ind.sector_benchmark) == ("IN", "^NSEI", "^NSEBANK")
    assert ind.fundamental_keys == ("priceToBook", "trailingPE")


def test_unknown_sector_rejected():
    with pytest.raises(ValueError, match="unknown sector"):
        resolve_holding("NVDA", "Crypto")
