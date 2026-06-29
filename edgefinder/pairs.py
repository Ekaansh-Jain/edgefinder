"""Market-neutral pairs trading (statistical arbitrage) backtester.

A completely different class from the long-only cross-sectional model: we trade
the *spread* between two historically-similar stocks, betting it mean-reverts.
Being market-neutral, success is judged on absolute Sharpe and LOW correlation
to the index — not on beating an equal-weight basket.

Approach (the classic Gatev et al. "distance" method — needs only price data,
no extra dependencies):
  1. FORMATION window: normalise each stock to a return index (start = 1) and
     pick the pairs with the smallest sum-of-squared-distance (most similar).
     Record each pair's spread mean and std.
  2. TRADING window: z-score the spread using the formation mean/std. Go long
     the spread (long the laggard, short the leader) when z <= -entry; short the
     spread when z >= +entry; close when it reverts through `exit`; stop out at
     `stop`. Dollar-neutral, 1:1 legs.
  3. Roll formation/trading windows forward (walk-forward, no look-ahead).

HONEST CAVEATS (see README/FINDINGS):
  * Pairs edges decay and get crowded; published 30%+ figures are optimistic.
  * Cointegration/similarity can break down permanently (the main risk).
  * Overnight short legs in India require FUTURES (cash shorts are intraday
    only). We model an idealised market-neutral spread; real costs are higher.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def _normalize(window: pd.DataFrame) -> pd.DataFrame:
    """Return index starting at 1.0 for each column over the window."""
    first = window.iloc[0]
    return window.divide(first).where(first != 0)


def select_pairs_distance(
    formation_prices: pd.DataFrame, top_k: int, min_obs: int
) -> list[tuple[str, str, float, float]]:
    """Pick the top_k most-similar pairs by sum-of-squared distance.

    Returns list of (ticker_a, ticker_b, spread_mean, spread_std).
    """
    valid = formation_prices.dropna(axis=1, thresh=min_obs)
    if valid.shape[1] < 2:
        return []
    norm = _normalize(valid).dropna(axis=1, how="any")
    cols = list(norm.columns)
    results = []
    for a, b in itertools.combinations(cols, 2):
        spread = norm[a] - norm[b]
        ssd = float((spread ** 2).sum())
        results.append((a, b, ssd, spread.mean(), spread.std()))
    results.sort(key=lambda r: r[2])  # smallest distance first
    out = []
    for a, b, _ssd, mu, sd in results[:top_k]:
        if sd and sd > 0:
            out.append((a, b, float(mu), float(sd)))
    return out


def _trade_one_pair(
    prices: pd.DataFrame,
    a: str,
    b: str,
    base: pd.Series,
    mu: float,
    sd: float,
    trade_idx: pd.DatetimeIndex,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    cost_per_leg: float,
) -> pd.Series:
    """Daily net returns for one pair over the trading window.

    base : the formation-start prices (to keep the normalised index consistent).
    """
    # Normalised levels during trading, anchored to formation start.
    na = prices[a].reindex(trade_idx) / base[a]
    nb = prices[b].reindex(trade_idx) / base[b]
    spread = na - nb
    z = (spread - mu) / sd

    ret_a = prices[a].reindex(trade_idx).pct_change()
    ret_b = prices[b].reindex(trade_idx).pct_change()

    pos = 0  # +1 long spread (long a / short b), -1 short spread, 0 flat
    daily = pd.Series(0.0, index=trade_idx)

    for i in range(1, len(trade_idx)):
        t = trade_idx[i]
        # P&L accrues on the position held coming into day t.
        if pos != 0 and np.isfinite(ret_a.iloc[i]) and np.isfinite(ret_b.iloc[i]):
            daily.iloc[i] = pos * (ret_a.iloc[i] - ret_b.iloc[i])

        zt = z.iloc[i]
        if not np.isfinite(zt):
            continue

        prev_pos = pos
        if pos == 0:
            if zt <= -entry_z:
                pos = 1          # spread too low -> expect rise
            elif zt >= entry_z:
                pos = -1         # spread too high -> expect fall
        elif pos == 1:
            if zt >= -exit_z or zt <= -stop_z:
                pos = 0
        elif pos == -1:
            if zt <= exit_z or zt >= stop_z:
                pos = 0

        # Cost when the position changes (open or close): both legs trade.
        if pos != prev_pos:
            daily.iloc[i] -= 2 * cost_per_leg

    return daily


def backtest_pairs(
    prices: pd.DataFrame,
    formation_days: int = 252,
    trading_days: int = 126,
    top_k: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    cost_per_leg: float = 0.0015,
) -> dict:
    """Walk-forward market-neutral pairs backtest. Returns daily-return diagnostics."""
    prices = prices.sort_index()
    idx = prices.index
    all_daily = []
    pair_counts = []
    start = 0

    while start + formation_days + trading_days <= len(idx):
        form_idx = idx[start: start + formation_days]
        trade_idx = idx[start + formation_days: start + formation_days + trading_days]
        formation = prices.loc[form_idx]
        base = formation.iloc[0]

        pairs = select_pairs_distance(formation, top_k, min_obs=int(formation_days * 0.8))
        if pairs:
            pair_daily = []
            for a, b, mu, sd in pairs:
                if base.get(a, 0) and base.get(b, 0):
                    r = _trade_one_pair(
                        prices, a, b, base, mu, sd, trade_idx,
                        entry_z, exit_z, stop_z, cost_per_leg,
                    )
                    pair_daily.append(r)
            if pair_daily:
                # Equal capital across pairs -> average daily return.
                window_ret = pd.concat(pair_daily, axis=1).mean(axis=1)
                all_daily.append(window_ret)
                pair_counts.append(len(pair_daily))

        start += trading_days  # non-overlapping trading windows

    if not all_daily:
        raise RuntimeError("No tradable pairs/windows. Try a longer history.")

    daily = pd.concat(all_daily).sort_index()
    daily = daily[~daily.index.duplicated(keep="first")]
    return {
        "daily_returns": daily,
        "avg_pairs": float(np.mean(pair_counts)) if pair_counts else 0.0,
        "n_windows": len(all_daily),
    }
