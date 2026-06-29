#!/usr/bin/env python3
"""Run the market-neutral pairs-trading backtest and report honest metrics.

Examples
--------
  python run_pairs.py
  python run_pairs.py --universe nifty100 --top-k 25 --entry-z 2.5
  python run_pairs.py --cost-bps 20

Judge this on: positive Sharpe AND low correlation to NIFTY (market-neutral).
A daily strategy, so metrics are annualised with 252.
"""

from __future__ import annotations

import argparse

import pandas as pd

from edgefinder.data import load_benchmark, load_prices
from edgefinder.metrics import cagr, max_drawdown, sharpe, volatility
from edgefinder.pairs import backtest_pairs
from edgefinder.universe import get_universe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Market-neutral pairs trading backtest")
    p.add_argument("--universe", default="nifty100",
                   help="nifty50 | nifty100 | nifty200 | midcap")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--formation-days", type=int, default=252)
    p.add_argument("--trading-days", type=int, default=126)
    p.add_argument("--top-k", type=int, default=20, help="number of pairs traded")
    p.add_argument("--entry-z", type=float, default=2.0)
    p.add_argument("--exit-z", type=float, default=0.5)
    p.add_argument("--stop-z", type=float, default=4.0)
    p.add_argument("--cost-bps", type=float, default=15.0,
                   help="per-leg round-trip cost in basis points")
    p.add_argument("--cointegration", action="store_true",
                   help="only trade pairs whose spread passes an ADF stationarity "
                        "test (needs statsmodels: pip install statsmodels)")
    p.add_argument("--adf-pvalue", type=float, default=0.05)
    p.add_argument("--cache-dir", default="data_cache")
    p.add_argument("--refresh", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tickers = get_universe(args.universe)
    end = args.end or pd.Timestamp.today().date().isoformat()
    print(f"\n=== edgefinder PAIRS | universe={args.universe} ({len(tickers)} names) "
          f"| {args.start}..{end} | top_k={args.top_k} | entry_z={args.entry_z} ===\n")

    prices = load_prices(tickers, args.start, end, cache_dir=args.cache_dir,
                         force_refresh=args.refresh)

    result = backtest_pairs(
        prices,
        formation_days=args.formation_days,
        trading_days=args.trading_days,
        top_k=args.top_k,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stop_z=args.stop_z,
        cost_per_leg=args.cost_bps / 10_000.0,
        require_stationary=args.cointegration,
        adf_pvalue=args.adf_pvalue,
    )
    daily = result["daily_returns"]

    # Benchmark daily returns aligned to the strategy, for the market-neutrality check.
    bench_px = load_benchmark("^NSEI", args.start, end, cache_dir=args.cache_dir)
    bench_daily = bench_px.reindex(daily.index).pct_change()
    aligned = pd.concat([daily.rename("strat"), bench_daily.rename("nifty")], axis=1).dropna()
    corr = float(aligned["strat"].corr(aligned["nifty"])) if len(aligned) > 2 else float("nan")

    PPY = 252
    print(f"Windows: {result['n_windows']}  |  avg pairs held: {result['avg_pairs']:.1f}  "
          f"|  trading days: {len(daily)}")
    print("\n" + "=" * 60)
    print("PAIRS TRADING RESULTS (market-neutral, after costs)")
    print("=" * 60)
    print(f"  CAGR (ann.)         : {cagr(daily, PPY) * 100:6.2f}%")
    print(f"  Volatility (ann.)   : {volatility(daily, PPY) * 100:6.2f}%")
    print(f"  Sharpe (ann.)       : {sharpe(daily, PPY):6.2f}")
    print(f"  Max drawdown        : {max_drawdown(daily) * 100:6.2f}%")
    print(f"  Correlation to NIFTY: {corr:+6.2f}   <-- want this near 0")
    print("=" * 60)

    if sharpe(daily, PPY) > 0.5 and abs(corr) < 0.3:
        print("=> Positive risk-adjusted return with low market correlation: a "
              "genuine market-neutral candidate. Validate out-of-sample before trusting.")
    else:
        print("=> Not a clear market-neutral edge after costs (Sharpe too low or "
              "correlation too high). Consistent with pairs-edge decay/crowding.")
    print("Reminder: idealised shorting; real overnight shorts need futures. "
          "Backtests overstate live performance. Not investment advice.\n")


if __name__ == "__main__":
    main()
