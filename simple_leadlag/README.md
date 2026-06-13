# simple_leadlag — a small, honest lead-lag study

A deliberately **simple** counterpart to the main engine: no ML, no GARCH, no leakage
tricks to get wrong. Just plain pandas testing one falsifiable idea.

## The hypothesis
*Do individual stocks **lag** their sector index — and if a name has under-performed its
sector recently, does it **catch up** enough to trade profitably after costs?*

That's it. One claim, testable, killable.

## How it tests it (and why it's leak-free)
1. **Diagnose direction** — cross-correlate each stock's return with the benchmark's
   return at ±1 day. `corr(stock_t, bench_{t-1}) > corr(stock_t, bench_{t+1})` ⇒ the stock
   lags the benchmark.
2. **Strategy** — relative-strength reversal: go long recent laggards, short recent
   leaders within each sector. Parameter-free (nothing is fitted ⇒ nothing to overfit).
   Every position is decided from data through *yesterday* (`.shift(1)`) ⇒ no look-ahead.
3. **Honest scoring** — metrics are the **second half** (out-of-sample), net of 5 bps/side
   costs, always compared against a plain **buy-and-hold** baseline of the same names.

## Run
```bash
python -m simple_leadlag.run          # synthetic, offline, deterministic
python -m simple_leadlag.run --live   # real data via yfinance
```

## What to expect
On synthetic data there's no lead-lag by construction → ~0 edge. On live data the honest
answer is **almost certainly also ~0 after costs** — daily cross-sectional reversal is
well known and largely arbitraged away. The value here isn't the result; it's a rig small
enough that you can trust what it tells you.

**Reading the verdict:** if the strategy quietly matches or loses to buy-and-hold, that's
the truth — no edge. If it suddenly *beats* buy-and-hold by a lot, be suspicious first:
in a test this simple, a big edge usually means a mistake, not a discovery.
