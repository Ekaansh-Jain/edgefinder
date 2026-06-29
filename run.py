#!/usr/bin/env python3
"""edgefinder CLI — run the walk-forward NSE backtest and print real metrics.

Examples
--------
  python run.py                                   # defaults (NIFTY200, monthly)
  python run.py --universe nifty100 --top-n 20
  python run.py --rebalance W-FRI --annualization 52   # weekly
  python run.py --news news.csv                   # enable free-LLM overlay
  python run.py --refresh                         # re-download price data

Output: a summary table comparing the AI strategy vs an equal-weight reference
vs NIFTY, plus results/ artifacts (equity curve CSV + PNG).
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from edgefinder.backtest import benchmark_returns, run_backtest
from edgefinder.config import BacktestConfig, CostModel
from edgefinder.data import load_benchmark, load_prices
from edgefinder.metrics import equity_curve, format_summary_table, summarize
from edgefinder.universe import get_universe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI-driven NSE cross-sectional backtest")
    p.add_argument("--universe", default="nifty200",
                   help="nifty50 | nifty100 | nifty200 | midcap")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--rebalance", default="ME", help="pandas offset, e.g. ME or W-FRI")
    p.add_argument("--top-n", type=int, default=25)
    p.add_argument("--strategy", default="ml", choices=["ml", "lowvol"],
                   help="'ml' learned ranking, or 'lowvol' (low-volatility "
                        "anomaly: rank by lowest risk, no training)")
    p.add_argument("--weighting", default="inv_vol", choices=["inv_vol", "equal"],
                   help="how to weight selected names (inv_vol = risk-parity tilt)")
    p.add_argument("--turnover-buffer", type=float, default=0.5,
                   help="keep a holding while ranked within (1+buffer)*top_n")
    p.add_argument("--annualization", type=int, default=12,
                   help="12 for monthly, 52 for weekly")
    p.add_argument("--train-min-periods", type=int, default=24)
    p.add_argument("--no-lightgbm", action="store_true",
                   help="force sklearn/zscore instead of LightGBM")
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--news", default=None,
                   help="'gdelt' to auto-build a free news overlay, OR a path to "
                        "a CSV with columns date,ticker,text for the LLM overlay")
    p.add_argument("--news-window", type=int, default=30,
                   help="trailing days of news aggregated per rebalance date")
    p.add_argument("--news-cache-dir", default="news_cache")
    p.add_argument("--shuffle-news", action="store_true",
                   help="PLACEBO TEST: randomly reassign news scores across "
                        "tickers. If the edge survives, it was never the news.")
    p.add_argument("--regime-filter", action="store_true",
                   help="de-risk toward cash when the benchmark is below its MA")
    p.add_argument("--regime-ma-days", type=int, default=200)
    p.add_argument("--risk-off-exposure", type=float, default=0.0,
                   help="exposure when risk-off (0.0 = full cash, 0.5 = half)")
    p.add_argument("--refresh", action="store_true", help="re-download price data")
    p.add_argument("--cache-dir", default="data_cache")
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BacktestConfig(
        universe=args.universe,
        start=args.start,
        end=args.end,
        rebalance=args.rebalance,
        top_n=args.top_n,
        strategy=args.strategy,
        weighting=args.weighting,
        turnover_buffer=args.turnover_buffer,
        annualization=args.annualization,
        train_min_periods=args.train_min_periods,
        use_lightgbm=not args.no_lightgbm,
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        costs=CostModel(slippage_bps=args.slippage_bps),
        regime_filter=args.regime_filter,
        regime_ma_days=args.regime_ma_days,
        risk_off_exposure=args.risk_off_exposure,
    )

    tickers = get_universe(cfg.universe)
    end = cfg.resolved_end()
    print(f"\n=== edgefinder | universe={cfg.universe} ({len(tickers)} names) "
          f"| {cfg.start}..{end} | rebalance={cfg.rebalance} | top_n={cfg.top_n} ===\n")

    prices = load_prices(
        tickers, cfg.start, end, cache_dir=cfg.cache_dir, force_refresh=args.refresh
    )

    # Benchmark prices (also used by the regime/trend filter, point-in-time).
    bench_px = load_benchmark(cfg.benchmark, cfg.start, end, cache_dir=cfg.cache_dir)

    # Optional news overlay: 'gdelt' (free, no key) or a CSV for the LLM layer.
    score_overlay = None
    if args.news:
        from edgefinder.backtest import _rebalance_dates

        rb = _rebalance_dates(prices, cfg.rebalance)
        if str(args.news).lower() == "gdelt":
            from edgefinder.gdelt_news import gdelt_overlay

            print("Building free GDELT news-sentiment overlay ...")
            score_overlay = gdelt_overlay(
                list(prices.columns), cfg.start, end, rb,
                cache_dir=args.news_cache_dir, window_days=args.news_window,
                force_refresh=args.refresh,
            )
        else:
            from edgefinder.llm_sentiment import build_sentiment_overlay

            news = pd.read_csv(args.news)
            print(f"Building LLM sentiment overlay from {len(news)} news rows ...")
            score_overlay = build_sentiment_overlay(news, rb, list(prices.columns))

    # PLACEBO: scramble which ticker each news score attaches to (per date),
    # keeping the value distribution. If the edge persists, it isn't the news.
    if score_overlay is not None and args.shuffle_news:
        import numpy as _np

        rng = _np.random.default_rng(cfg.random_state)
        shuffled = score_overlay.copy()
        for idx in shuffled.index:
            vals = shuffled.loc[idx].to_numpy()
            rng.shuffle(vals)
            shuffled.loc[idx] = vals
        score_overlay = shuffled
        print("PLACEBO MODE: news scores randomly reassigned across tickers.")

    regime_series = bench_px if cfg.regime_filter else None
    result = run_backtest(
        prices, cfg, score_overlay=score_overlay, regime_series=regime_series
    )
    strat = result["strategy_returns"]
    ew = result["equalweight_returns"]

    # Benchmark returns aligned to the strategy's realised dates.
    bench = benchmark_returns(bench_px, list(strat.index)) if len(strat) else pd.Series(dtype=float)
    bench = bench.reindex(strat.index).dropna()

    regime_note = ""
    if cfg.regime_filter:
        regime_note = (f"  |  regime filter: ON (risk-off {result['risk_off_periods']} "
                       f"periods, exposure {cfg.risk_off_exposure})")
    print(f"\nModel used: {result['model_kind']}  |  weighting: {cfg.weighting}  |  "
          f"turnover buffer: {cfg.turnover_buffer}  |  avg one-way turnover/period: "
          f"{result['avg_turnover'] * 100:.1f}%{regime_note}")
    if score_overlay is not None:
        # Guard against the silent-zeros trap: an all-zero overlay adds nothing,
        # so any "edge" must NOT be attributed to news.
        nonzero = float(score_overlay.abs().to_numpy().sum()) > 0
        if nonzero:
            print("News overlay: ACTIVE (added to model ranking score)")
        else:
            print("News overlay: INACTIVE — overlay is empty/all-zero, so results "
                  "reflect the PRICE MODEL ONLY. Do NOT attribute any edge to news.")
    if result["feature_importance"] is not None:
        print("\nTop features (LightGBM importance):")
        print(result["feature_importance"].head(6).to_string())

    rows = [
        summarize("AI Strategy", strat, cfg.annualization, result["avg_turnover"]),
        summarize("EqualWeight (ref)", ew, cfg.annualization),
        summarize("NIFTY (Buy&Hold)", bench, cfg.annualization),
    ]
    print("\n" + "=" * 70)
    print("RESULTS (after realistic costs)")
    print("=" * 70)
    print(format_summary_table(rows))
    print("=" * 70)

    # Honest verdict: the real bar is the EQUAL-WEIGHT reference, not just NIFTY.
    # Beating NIFTY can be pure size/equal-weight beta; beating EqualWeight on a
    # risk-adjusted basis is what would indicate genuine selection skill.
    if len(strat) and len(ew):
        from edgefinder.metrics import cagr, sharpe

        s_cagr = cagr(strat, cfg.annualization)
        s_shp = sharpe(strat, cfg.annualization)
        ew_cagr = cagr(ew, cfg.annualization)
        ew_shp = sharpe(ew, cfg.annualization)
        print("\n--- Verdict ---")
        if len(bench):
            print(f"vs NIFTY      : CAGR {(s_cagr - cagr(bench, cfg.annualization)) * 100:+.2f}%"
                  f" (beating the index can be pure size/equal-weight beta)")
        beats_ew = s_shp > ew_shp and s_cagr >= ew_cagr * 0.99
        print(f"vs EqualWeight: CAGR {(s_cagr - ew_cagr) * 100:+.2f}%, "
              f"Sharpe {s_shp - ew_shp:+.2f}  <-- THE REAL BAR")

        # Significance of the excess return over EqualWeight (is it noise?).
        from edgefinder.metrics import excess_stats

        es = excess_stats(strat, ew, cfg.annualization)
        if es:
            sig = ("likely SIGNIFICANT" if abs(es["t_stat"]) > 2
                   else "NOT significant (could be noise)")
            print(f"Significance : t-stat {es['t_stat']:+.2f} over {es['n']} periods, "
                  f"info ratio {es['information_ratio']:+.2f}  -> {sig}")
            print("               (|t|>2 ~ unlikely to be chance; mind multiple-testing)")
        if beats_ew:
            print("=> The model shows genuine selection skill (beats equal-weight "
                  "risk-adjusted). Treat as a candidate edge, still subject to "
                  "survivorship bias and backtest overstatement.")
        else:
            print("=> NO real edge yet: the model does NOT beat a naive equal-weight "
                  "basket on a risk-adjusted basis. The 'AI' is not adding alpha.")
        print("Reminder: survivorship bias inflates ALL rows; backtests overstate "
              "live performance. Not investment advice.\n")

    _save_outputs(cfg, strat, ew, bench)


def _save_outputs(cfg, strat, ew, bench) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    curves = pd.DataFrame({
        "AI_Strategy": equity_curve(strat),
        "EqualWeight": equity_curve(ew),
        "NIFTY": equity_curve(bench),
    })
    csv_path = os.path.join(cfg.out_dir, "equity_curves.csv")
    curves.to_csv(csv_path)
    print(f"Saved equity curves -> {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax = curves.plot(figsize=(10, 6), logy=True)
        ax.set_title("edgefinder: growth of 1 (log scale, after costs)")
        ax.set_ylabel("Equity (x)")
        ax.grid(True, alpha=0.3)
        png_path = os.path.join(cfg.out_dir, "equity_curves.png")
        plt.tight_layout()
        plt.savefig(png_path, dpi=120)
        print(f"Saved plot -> {png_path}")
    except Exception as exc:
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":
    main()
