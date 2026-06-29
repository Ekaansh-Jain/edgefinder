# Findings — an honest account

This documents what we actually tested with `edgefinder` and what we concluded.
It is deliberately blunt: most of what we tried did **not** work, and saying so
clearly is the whole point.

## Objective

Find a small, real, *backtestable* edge in Indian equities (NSE) using only free
data and free/open-source AI — and judge it honestly, not against a soft
benchmark.

## Method (why the results are trustworthy)

- **Walk-forward, point-in-time**: features at date T use only data up to T;
  models train only on past periods. No look-ahead.
- **Realistic costs**: post-2026 STT, exchange/SEBI fees, stamp duty, GST,
  slippage — charged on turnover.
- **The honest bar is EQUAL-WEIGHT, not NIFTY.** Beating the cap-weighted NIFTY
  is easy (size/breadth premium); beating an equal-weight basket of the same
  universe on a *risk-adjusted* basis is what would show real skill.
- **Significance**: a t-stat / information ratio on the excess return over
  equal-weight (|t|>2 ~ unlikely to be chance), plus a **news placebo** test.

## Results (vs the equal-weight bar)

| Strategy | vs EqualWeight Sharpe | Significant? | Notes |
|---|---|---|---|
| LightGBM momentum factors | ~ -0.15 | no | momentum-dominated; loses |
| + demeaned (cross-sectional) labels | ~ -0.14 | no | no improvement |
| + inverse-vol weighting + turnover buffer | ~ -0.07 | no | better risk, still loses |
| + regime/trend filter (full cash) | ~ -0.06 | no | cuts drawdown, cash drag |
| + regime/trend filter (half exposure) | ~ -0.01 | no | ~tie; risk tool, not alpha |
| "News" overlay (GDELT) on midcap | +0.11 | **no (t≈1.3)** | **GDELT returned 0 data — overlay was empty; the gain was the midcap universe, i.e. noise** |
| **Low-volatility anomaly (nifty200)** | **+0.03** | no (t≈-1.3) | **~tie on Sharpe, but vol -23%, drawdown -36%, turnover 7.7%** |
| Low-vol + regime filter | -0.23 | **yes, WORSE (t≈-2.5)** | double de-risking hurts; don't combine |

### Market-neutral pairs trading (judged on absolute Sharpe + market correlation)

| Variant | Sharpe | Corr to NIFTY | Notes |
|---|---|---|---|
| Distance method | -4.27 | +0.10 | market-neutral ✓ but loses; spreads trended, not reverted |
| + cointegration (ADF) filter | -4.55 | +0.11 | filtering didn't rescue it; OOS breakdown + decay |

(Caveat: the starkly negative pairs Sharpe is partly implementation — entry/exit
bands, no forced close at window edges, idealised shorting costs — and was not
fully debuggable in an offline environment. Read as "this free-data
implementation lost", not "pairs trading is impossible".)

## Conclusions

1. **No statistically significant *return* edge over equal-weight** was found from
   price-factor ML, demeaning, weighting tweaks, regime filters, or news. This is
   consistent with the academic literature: liquid Indian equities are largely
   efficient for a retail player using free price data.

2. **The "news edge" was a mirage.** GDELT's free DOC 2.0 API only covers ~the
   last 1–1.5 years, so the 2015+ query returned nothing and the overlay was
   silently all-zeros. The apparent +0.11 Sharpe was just the midcap universe,
   and it was statistically insignificant (t≈1.3). A genuine historical news test
   would require the GDELT GKG via BigQuery (not built here).

3. **Low-volatility is the one defensible, literature-backed result.** It ties
   equal-weight on Sharpe (~0.90 vs ~0.87) while delivering dramatically lower
   volatility and drawdown and near-zero turnover. Crucially, **survivorship bias
   in this backtest works *against* low-vol** (the universe over-rewards the
   high-risk names that happened to survive), so in a survivorship-free, real-
   world setting low-vol's relative risk-adjusted performance would likely be
   *better* than shown here.

4. **Equal-weight midcap** captures the size/breadth premium and beats NIFTY by a
   wide margin — but that gap is heavily inflated by survivorship bias.

## What this means in practice (not investment advice)

- The realistic, honest "edge" for an individual is **not** an AI alpha machine.
  It is: own a **low-cost, broad/equal-weight basket** for the size premium, and
  use **low-volatility selection** as a risk-efficient core (similar Sharpe, much
  smaller drawdowns, negligible trading costs).
- Do **not** stack a regime filter on top of low-vol — it over-de-risks.
- Treat all absolute returns here as **overstated** (survivorship bias + backtest
  optimism). The *relative* (vs equal-weight) comparisons are the trustworthy part.

## Honest limitations

- **Survivorship bias**: universe = current constituents; delisted losers absent.
- **No fundamentals**: value/quality factors need point-in-time fundamentals we
  don't have for free (yfinance snapshots would leak look-ahead).
- **News untested historically**: see conclusion #2.
- **Backtest ≠ live**: slippage, liquidity, and impact are modelled crudely.

## Reproduce

```bash
pip install -r requirements.txt
python run.py --strategy lowvol --universe nifty200     # the defensible result
python run.py                                           # the momentum ML baseline
```



## Final word (after an exhaustive sweep)

We tested essentially every major strategy class available with free data:
cross-sectional factor ML, factor refinements, regime filtering, news sentiment,
the low-volatility anomaly, and market-neutral pairs/stat-arb (with cointegration
filtering). **None produced a statistically significant, cost-surviving return
edge over a simple equal-weight basket.** The single positive, defensible result
is the **low-volatility anomaly**: similar Sharpe to equal-weight but materially
lower volatility, drawdown, and turnover — and likely *better* once survivorship
bias (which penalises low-vol here) is removed.

This is the honest answer to "can AI find an edge in the Indian market on free
data": for a retail participant, **not a return edge** — the markets are largely
efficient at this resolution. The realistic, evidence-based takeaways are
risk-management, not alpha: own a broad/equal-weight basket for the size premium,
use low-volatility selection for a smoother, cheaper ride. Anything beyond that
needs data or infrastructure (point-in-time fundamentals, historical news via
BigQuery GKG, intraday/order-book, faster execution) that free tools don't give.
