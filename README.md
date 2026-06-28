# edgefinder

An **honest** search for a small statistical edge in Indian equities (NSE), using
only **free data** and **free / open-source AI**. It runs a walk-forward,
point-in-time backtest of a cross-sectional ranking strategy with **realistic
post-Budget-2026 transaction costs**, and tells you plainly whether the "AI"
actually beats a dumb benchmark.

> This is research/engineering tooling, **not investment advice**. Backtests
> routinely overstate live performance. Past performance does not predict future
> returns. Trade real money at your own risk.

---

## The thesis (why this design)

The research is clear on a few things:

- **"Predict tomorrow's price" deep-learning models are mostly fake** — they leak
  the future or just lag the price. We avoid that entirely.
- **Gradient-boosted trees beat deep nets** on tabular factor data and are the
  real workhorse at quant funds.
- The durable, documented edges in India are **slow factors** (momentum,
  low-volatility, quality), and they **decay** as they get crowded.
- High-turnover **intraday / F&O is where retail loses** (SEBI: ~91% of
  individual derivatives traders lost money in FY25). So we trade **delivery
  equity at low frequency** where the cost math can work.
- The genuinely new frontier is **LLMs reading text** (news/filings), strongest
  in **smaller, less-covered stocks** — hence the optional sentiment layer and
  midcap tilt.

So: a **monthly cross-sectional ranking** of NSE stocks by classic factors,
combined by a **LightGBM** model, validated walk-forward, with an **optional
free-LLM news overlay**. We expect (at best) a *small, fragile* edge — not a
money printer. The point is to measure it honestly.

---

## What it does

1. Downloads free adjusted-close prices via `yfinance` (cached locally).
2. At each month-end, builds **point-in-time** factor features (momentum,
   reversal, volatility, trend, drawdown).
3. Trains a ranking model on **past periods only**, predicts next-month relative
   returns, holds the **top-N equal-weighted**.
4. Charges **realistic costs** (STT, exchange/SEBI fees, stamp duty, GST,
   slippage) on turnover.
5. Compares the **AI strategy vs Equal-Weight vs NIFTY** and prints a verdict.

---

## Quick start

### Option A — GitHub Actions (zero local setup)

1. Push this repo to GitHub.
2. Go to the **Actions** tab → **backtest** → **Run workflow**.
3. Read the real CAGR / Sharpe / drawdown in the logs; download the
   `backtest-results` artifact for the equity-curve plot.

GitHub's runners have internet, so they fetch data and run everything for you.

### Option B — Run locally

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run.py                          # NIFTY200, monthly, top-25 (defaults)
python run.py --universe nifty100 --top-n 20
python run.py --rebalance W-FRI --annualization 52    # weekly variant
```

First run downloads data (a few minutes); later runs use the cache and are fast.

---

## The optional free-LLM news layer

The baseline needs **no API key**. To test the news-sentiment overlay, pick a
free provider and set an env var — the code auto-detects it:

| Provider | Cost | Setup |
|----------|------|-------|
| **Groq** | Free tier, very fast | `export GROQ_API_KEY=...`, `pip install groq` |
| **Gemini** | Free tier | `export GEMINI_API_KEY=...`, `pip install google-generativeai` |
| **Ollama** | Fully free / local | run `ollama serve`, `export USE_OLLAMA=1` |

Provide news as a CSV with columns `date,ticker,text` (ticker in `.NS` form),
then:

```bash
python run.py --news news.csv
```

Each headline is scored for short-term bullish/bearish impact; scores are
aggregated **point-in-time** (only news on/before each rebalance date, 30-day
window) and added to the model's ranking.

> **Look-ahead warning:** if the LLM was trained on data overlapping your
> backtest window, it may "remember" the future and inflate results. For an
> honest test, restrict the news layer to dates **after** the model's training
> cutoff. Treat any LLM backtest alpha as an upper bound, not deployment proof.

---

## Configuration

Tune everything in `edgefinder/config.py` or via CLI flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--universe` | `nifty200` | `nifty50` / `nifty100` / `nifty200` / `midcap` |
| `--rebalance` | `ME` | `ME` monthly, `W-FRI` weekly |
| `--top-n` | `25` | stocks held (equal-weighted) |
| `--train-min-periods` | `24` | warm-up periods before trading |
| `--slippage-bps` | `5.0` | modelled slippage per side |
| `--no-lightgbm` | off | force sklearn/z-score model |
| `--news` | none | CSV path to enable LLM overlay |
| `--refresh` | off | re-download price data |

Cost assumptions (post-Budget-2026 delivery equity) live in `CostModel` —
adjust to your actual broker.

---

## How to read the results

```
RESULTS (after realistic costs)
Strategy          | CAGR  | Sharpe | Vol   | MaxDD  | Hit%  | TotRet | Turnover | N
------------------+-------+--------+-------+--------+-------+--------+----------+----
AI Strategy       | ...   | ...    | ...   | ...    | ...   | ...    | ...      | ...
EqualWeight (ref) | ...   | ...    | ...   | ...    | ...   | ...    | -        | ...
NIFTY (Buy&Hold)  | ...   | ...    | ...   | ...    | ...   | ...    | -        | ...
```

- An edge means **AI Strategy beats both EqualWeight and NIFTY** on
  risk-adjusted terms (Sharpe), not just a positive return.
- If the AI strategy can't beat the `EqualWeight` reference, the ML is adding
  nothing — that's a real and useful (negative) result.
- Watch `Turnover`: high turnover gets eaten by costs.

---

## Honest limitations (read these)

- **Survivorship bias:** the ticker lists are current constituents, so delisted
  losers are missing. This *flatters* results. For rigour, rebuild the universe
  from historical index membership.
- **Free data is imperfect:** `yfinance` adjusted prices can have gaps/errors.
- **Backtest ≠ live:** slippage, liquidity, and impact are modelled crudely.
- **Edges decay:** an edge visible in history may already be arbitraged away.
- **Small universe:** ~150 names; broaden it before drawing strong conclusions.

---

## Project layout

```
edgefinder/
  config.py        # BacktestConfig + CostModel (post-2026 costs)
  universe.py      # NSE ticker lists
  data.py          # free yfinance loader + cache
  features.py      # point-in-time factor features (leakage-safe)
  model.py         # LightGBM -> sklearn -> z-score ranking model
  backtest.py      # walk-forward engine + cost accounting
  metrics.py       # CAGR / Sharpe / drawdown / turnover
  llm_sentiment.py # optional free-LLM news overlay
run.py             # CLI entrypoint
.github/workflows/backtest.yml  # run in CI, get real numbers
```
