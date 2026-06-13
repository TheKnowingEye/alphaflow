"""Universe: 6 US sectors, each a sector-ETF benchmark + a basket of large-cap stocks.

Deliberately simple and US-only (one trading calendar). The hypothesis under test:
do individual names LAG their sector index — and if so, does trading that lag survive
costs out-of-sample? (Usually: no. The point is to find that out honestly.)
"""

SECTORS: dict[str, dict[str, object]] = {
    "Tech": {"benchmark": "XLK", "stocks": ["NVDA", "AAPL", "MSFT", "AVGO", "AMD", "QCOM"]},
    "Finance": {"benchmark": "XLF", "stocks": ["JPM", "BAC", "WFC", "GS", "MS", "C"]},
    "Energy": {"benchmark": "XLE", "stocks": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC"]},
    "Materials": {"benchmark": "XLB", "stocks": ["LIN", "SHW", "FCX", "NEM", "APD", "ECL"]},
    "FMCG": {"benchmark": "XLP", "stocks": ["PG", "KO", "PEP", "COST", "WMT", "MDLZ"]},
    "Pharma": {"benchmark": "XLV", "stocks": ["JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY"]},
}

# Strategy / eval knobs (all simple, no fitting).
LOOKBACK = 5  # days of relative under/out-performance the signal reads
Z_WINDOW = 60  # window to standardise the spread for sizing
COST_PER_SIDE = 0.0005  # 5 bps per trade side
TRAIN_FRAC = 0.5  # first half = in-sample peek; reported metrics are SECOND half (OOS)


def all_tickers() -> list[str]:
    out: list[str] = []
    for cfg in SECTORS.values():
        out.append(str(cfg["benchmark"]))
        out.extend(cfg["stocks"])  # type: ignore[arg-type]
    return list(dict.fromkeys(out))
