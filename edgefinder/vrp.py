"""Defined-risk Volatility Risk Premium (VRP) backtester — short iron condors.

The edge: Indian index options are systematically overpriced (implied vol >
realized vol), so selling premium has positive expectancy. The DANGER: it's
negatively skewed — you win small repeatedly, then a big move hands back a lot.
So we (a) only sell DEFINED-RISK iron condors (max loss capped by long wings)
and (b) size every trade so the capped max loss is a small % of capital. This
file's job is to report the SKEW honestly, not hide it.

Strategy (one trade per weekly expiry):
  * Enter ~`entry_offset_days` before expiry.
  * Sell a call and a put each ~`short_pct` out-of-the-money (strikes rounded to
    `strike_step`); buy wings `wing_points` further out -> an iron condor.
  * Net credit = shorts' premium - wings' premium (collected up front).
  * Hold to expiry; payoff is deterministic from the expiry spot.
  * Size so max loss (= wing_points - credit) equals `sizing_pct` of capital.
    => worst case on a trade is about -sizing_pct. That's the whole discipline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _round_to(x: float, step: int) -> float:
    return round(x / step) * step


def _premium(chain: pd.DataFrame, strike: float, opt: str) -> "float | None":
    row = chain[(chain["strike"] == strike) & (chain["option_type"] == opt)]
    if row.empty:
        return None
    return float(row["close"].iloc[-1])


def backtest_iron_condor(
    options_df: pd.DataFrame,
    spot: pd.Series,
    short_pct: float = 0.02,
    wing_points: int = 300,
    strike_step: int = 50,
    entry_offset_days: int = 4,
    sizing_pct: float = 0.02,
    cost_points: float = 4.0,
    profit_target: float = 0.5,
) -> dict:
    """Run the walk-forward iron-condor backtest. Returns trades + per-trade returns.

    profit_target: close early if credit decays to (1-target) of entry credit;
                   here applied approximately at expiry only (EOD data), so it
                   mainly documents intent. Set 1.0 to always hold to expiry.
    """
    spot = spot.sort_index()
    trades = []

    for expiry, grp in options_df.groupby("expiry"):
        expiry = pd.Timestamp(expiry)
        # Entry date: latest available quote on/before (expiry - offset).
        target_entry = expiry - pd.Timedelta(days=entry_offset_days)
        avail = grp[grp["date"] <= target_entry]["date"]
        if avail.empty:
            continue
        entry_date = avail.max()
        chain = grp[grp["date"] == entry_date]
        if chain.empty:
            continue

        # Spot at entry and expiry.
        s_entry = spot.loc[:entry_date]
        s_exp = spot.loc[:expiry]
        if s_entry.empty or s_exp.empty:
            continue
        S0 = float(s_entry.iloc[-1])
        S1 = float(s_exp.iloc[-1])

        call_short = _round_to(S0 * (1 + short_pct), strike_step)
        put_short = _round_to(S0 * (1 - short_pct), strike_step)
        call_wing = call_short + wing_points
        put_wing = put_short - wing_points

        cs = _premium(chain, call_short, "CE")
        ps = _premium(chain, put_short, "PE")
        cw = _premium(chain, call_wing, "CE")
        pw = _premium(chain, put_wing, "PE")
        if None in (cs, ps, cw, pw):
            continue  # incomplete chain for this expiry

        credit = (cs + ps) - (cw + pw)
        if credit <= 0:
            continue  # no premium to harvest (bad strikes/data)

        # Deterministic expiry payoff of the iron condor (in index points).
        call_loss = max(0.0, min(S1 - call_short, call_wing - call_short))
        put_loss = max(0.0, min(put_short - S1, put_short - put_wing))
        pnl_points = credit - (call_loss + put_loss) - cost_points

        max_loss_points = wing_points - credit
        if max_loss_points <= 0:
            continue
        # Size so worst-case loss = sizing_pct of capital -> return on capital.
        trade_return = (pnl_points / max_loss_points) * sizing_pct

        trades.append({
            "entry_date": entry_date,
            "expiry": expiry,
            "spot_entry": S0,
            "spot_expiry": S1,
            "credit": credit,
            "max_loss_pts": max_loss_points,
            "pnl_pts": pnl_points,
            "return": trade_return,
            "win": pnl_points > 0,
        })

    if not trades:
        raise RuntimeError(
            "No tradable expiries. Check that the options data + spot cover the "
            "same dates and that strikes near +/-short_pct exist in the chain."
        )

    tdf = pd.DataFrame(trades).sort_values("expiry").reset_index(drop=True)
    returns = pd.Series(tdf["return"].values, index=pd.DatetimeIndex(tdf["expiry"]))
    return {"trades": tdf, "returns": returns}


# --------------------------------------------------------------------------- #
# Honest tail-risk reporting                                                   #
# --------------------------------------------------------------------------- #
def _longest_losing_streak(wins: pd.Series) -> int:
    streak = worst = 0
    for w in wins:
        streak = 0 if w else streak + 1
        worst = max(worst, streak)
    return worst


def tail_report(result: dict, periods_per_year: int = 52,
                starting_capital: float = 1_000_000.0) -> dict:
    """Metrics that EXPOSE the negative skew of premium selling."""
    from .metrics import cagr, max_drawdown, sharpe, volatility

    r = result["returns"].dropna()
    tdf = result["trades"]
    wins = tdf["win"]
    win_ret = r[r > 0]
    loss_ret = r[r <= 0]

    equity = (1 + r).cumprod() * starting_capital
    return {
        "n_trades": int(len(r)),
        "CAGR": cagr(r, periods_per_year),
        "Sharpe": sharpe(r, periods_per_year),
        "Volatility": volatility(r, periods_per_year),
        "MaxDrawdown": max_drawdown(r),
        "WinRate": float(wins.mean()),
        "AvgWin": float(win_ret.mean()) if len(win_ret) else 0.0,
        "AvgLoss": float(loss_ret.mean()) if len(loss_ret) else 0.0,
        "WorstTrade": float(r.min()),
        "BestTrade": float(r.max()),
        "Skew": float(r.skew()) if len(r) > 2 else float("nan"),
        "LongestLosingStreak": _longest_losing_streak(wins),
        "Win/LossRatio": (abs(win_ret.mean() / loss_ret.mean())
                          if len(loss_ret) and loss_ret.mean() != 0 else float("nan")),
        "FinalEquity": float(equity.iloc[-1]) if len(equity) else starting_capital,
        "returns": r,
    }
