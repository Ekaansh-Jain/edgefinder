#!/usr/bin/env python3
"""Run the defined-risk Volatility Risk Premium (VRP) backtest on Nifty options.

You MUST supply historical options data (this sandbox/most setups have none by
default). Easiest free path: NSE F&O bhavcopy -> a tidy CSV. See
edgefinder/options_data.py for the schema and sources.

Examples
--------
  # 1) From a tidy/bhavcopy CSV you already have:
  python run_vrp.py --data nifty_options.csv

  # 2) Best-effort auto-fetch via jugaad-data (fragile), then run:
  python run_vrp.py --fetch --start 2019-01-01 --end 2024-12-31

  # tune the structure / risk:
  python run_vrp.py --data opts.csv --short-pct 0.02 --wing-points 300 --sizing-pct 0.02

Judge it on: positive Sharpe AND a tolerable WORST trade / drawdown / skew.
The whole point is to SEE the tail, not just the steady wins.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from edgefinder.options_data import (
    derive_spot,
    fetch_bhavcopy_jugaad,
    load_options_csv,
)
from edgefinder.vrp import backtest_iron_condor, tail_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Defined-risk VRP (iron condor) backtest")
    p.add_argument("--data", default="nifty_options.csv",
                   help="path to tidy/bhavcopy options CSV")
    p.add_argument("--spot", default=None,
                   help="optional CSV with columns date,close for NIFTY spot")
    p.add_argument("--fetch", action="store_true",
                   help="best-effort fetch bhavcopy via jugaad-data into --data")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--short-pct", type=float, default=0.02,
                   help="short strikes ~this fraction OTM (0.02 = 2%)")
    p.add_argument("--wing-points", type=int, default=300, help="wing width in points")
    p.add_argument("--strike-step", type=int, default=50)
    p.add_argument("--entry-offset-days", type=int, default=4)
    p.add_argument("--sizing-pct", type=float, default=0.02,
                   help="max loss per trade as fraction of capital (KEEP SMALL)")
    p.add_argument("--cost-points", type=float, default=4.0,
                   help="round-trip cost per trade in index points (STT+brokerage+slippage)")
    p.add_argument("--capital", type=float, default=1_000_000.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.fetch and not os.path.exists(args.data):
        print("Fetching bhavcopy via jugaad-data (best effort)...")
        fetch_bhavcopy_jugaad(args.start, args.end, args.data)

    options_df = load_options_csv(args.data)
    options_df = options_df[
        (options_df["date"] >= pd.Timestamp(args.start))
        & (options_df["date"] <= pd.Timestamp(args.end))
    ]
    if options_df.empty:
        raise SystemExit("No option rows in the chosen date range.")

    if args.spot and os.path.exists(args.spot):
        sp = pd.read_csv(args.spot)
        sp.columns = [c.lower() for c in sp.columns]
        spot = pd.Series(
            pd.to_numeric(sp["close"], errors="coerce").values,
            index=pd.to_datetime(sp["date"]),
        ).dropna().sort_index()
    else:
        print("No --spot file; deriving spot from option chain (put-call parity).")
        spot = derive_spot(options_df)

    print(f"\n=== VRP defined-risk iron condor | {args.start}..{args.end} | "
          f"short {args.short_pct*100:.1f}% OTM | wings {args.wing_points}pt | "
          f"size {args.sizing_pct*100:.1f}%/trade ===\n")

    result = backtest_iron_condor(
        options_df, spot,
        short_pct=args.short_pct, wing_points=args.wing_points,
        strike_step=args.strike_step, entry_offset_days=args.entry_offset_days,
        sizing_pct=args.sizing_pct, cost_points=args.cost_points,
    )
    rep = tail_report(result, periods_per_year=52, starting_capital=args.capital)

    def pct(x):
        return f"{x*100:6.2f}%"

    print("=" * 64)
    print("VRP RESULTS (defined-risk, after costs)")
    print("=" * 64)
    print(f"  Trades              : {rep['n_trades']}")
    print(f"  CAGR (ann.)         : {pct(rep['CAGR'])}")
    print(f"  Sharpe (ann.)       : {rep['Sharpe']:6.2f}")
    print(f"  Volatility (ann.)   : {pct(rep['Volatility'])}")
    print(f"  Win rate            : {pct(rep['WinRate'])}")
    print(f"  Final equity        : Rs {rep['FinalEquity']:,.0f}  (from {args.capital:,.0f})")
    print("-" * 64)
    print("  TAIL RISK (the part that ruins people) ")
    print(f"  Max drawdown        : {pct(rep['MaxDrawdown'])}")
    print(f"  WORST single trade  : {pct(rep['WorstTrade'])}")
    print(f"  Avg win / Avg loss  : {pct(rep['AvgWin'])} / {pct(rep['AvgLoss'])}")
    print(f"  Win/Loss size ratio : {rep['Win/LossRatio']:.2f}")
    print(f"  Longest losing run  : {rep['LongestLosingStreak']} trades")
    print(f"  Return skew         : {rep['Skew']:+.2f}   (negative = fat left tail)")
    print("=" * 64)

    if rep["Sharpe"] > 1.0 and rep["MaxDrawdown"] > -0.25:
        print("=> Attractive risk-adjusted profile AND survivable tail. Validate "
              "out-of-sample (esp. across a crash) before trusting.")
    else:
        print("=> Either weak risk-adjusted return or a tail too deep to trust. "
              "Premium selling is unforgiving — respect the skew.")
    print("Reminder: EOD payoff model; real fills/gaps/early-assignment differ. "
          "Backtests overstate live results. Not investment advice.\n")


if __name__ == "__main__":
    main()
