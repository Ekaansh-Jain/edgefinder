"""Leakage-safe, point-in-time feature engineering.

The golden rule: a feature computed for rebalance date T may only use data
available *up to and including* T. The label (what we predict) is the forward
return from T to T+1, which is of course unknown at T and used ONLY for training
on past periods. This is what kills the look-ahead bias that makes most online
"99% accuracy" stock models fake.

All features are classic, well-documented equity factors. We let the model
combine them non-linearly rather than hand-tuning weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Feature columns produced by build_features(); kept in one place so the model
# and the backtest agree on exactly what is used.
FEATURE_COLUMNS = [
    "mom_12_1",   # 12-month momentum, skipping the most recent month
    "mom_6_1",    # 6-month momentum, skipping the most recent month
    "mom_3_1",    # 3-month momentum, skipping the most recent month
    "rev_1m",     # short-term reversal (last-month return, expect negative sign)
    "vol_6m",     # realised volatility (lower is better -> low-vol factor)
    "dist_200d",  # distance of price from its 200-day moving average (trend)
    "downside_vol",  # downside semivolatility over ~6m (risk feature)
    "max_dd_6m",  # max drawdown over last ~6m (risk feature)
]


def _trading_days(months: float) -> int:
    return int(round(months * 21))


def build_features(
    prices: pd.DataFrame, as_of: pd.Timestamp
) -> pd.DataFrame:
    """Compute cross-sectional features for every ticker as of ``as_of``.

    Parameters
    ----------
    prices : wide DataFrame (index=date, cols=ticker) of ADJUSTED close.
    as_of  : the rebalance date. Only rows with index <= as_of are used.

    Returns
    -------
    DataFrame indexed by ticker with FEATURE_COLUMNS. Tickers without enough
    history are dropped.
    """
    hist = prices.loc[:as_of]
    if hist.shape[0] < _trading_days(12):
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    # Daily simple returns up to as_of.
    rets = hist.pct_change()

    def cum_return(lookback_days: int, skip_days: int = 0) -> pd.Series:
        # return over [t-lookback, t-skip]; skip avoids the most-recent month
        end_idx = -1 - skip_days
        start_idx = -1 - lookback_days
        if hist.shape[0] < lookback_days + 1:
            return pd.Series(np.nan, index=hist.columns)
        p_end = hist.iloc[end_idx]
        p_start = hist.iloc[start_idx]
        return (p_end / p_start) - 1.0

    m1 = _trading_days(1)

    feats = pd.DataFrame(index=hist.columns)
    feats["mom_12_1"] = cum_return(_trading_days(12), skip_days=m1)
    feats["mom_6_1"] = cum_return(_trading_days(6), skip_days=m1)
    feats["mom_3_1"] = cum_return(_trading_days(3), skip_days=m1)
    feats["rev_1m"] = cum_return(m1, skip_days=0)

    # 6-month realised volatility (annualised), lower => 'safer'.
    win6 = _trading_days(6)
    recent = rets.iloc[-win6:]
    feats["vol_6m"] = recent.std() * np.sqrt(252)

    # Distance from 200-day MA: (price / MA200) - 1. Positive => uptrend.
    ma200 = hist.tail(200).mean()
    feats["dist_200d"] = (hist.iloc[-1] / ma200) - 1.0

    # Downside semivolatility (std of negative returns only) over ~6m.
    downside = recent.where(recent < 0)
    feats["downside_vol"] = downside.std() * np.sqrt(252)

    # Max drawdown over last ~6m, computed on the cumulative price path.
    window_prices = hist.iloc[-win6:]
    running_max = window_prices.cummax()
    drawdown = window_prices / running_max - 1.0
    feats["max_dd_6m"] = drawdown.min()

    feats = feats.replace([np.inf, -np.inf], np.nan)
    # Require the core momentum signal to exist; drop names that are too new.
    feats = feats.dropna(subset=["mom_12_1", "vol_6m"])
    return feats[FEATURE_COLUMNS]


def forward_return(
    prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    """Realised simple return for each ticker between two rebalance dates.

    Used to build training labels (on past periods) and to score the live
    portfolio. Tickers missing a price at either end get NaN.
    """
    try:
        p0 = prices.loc[:start].iloc[-1]
        p1 = prices.loc[:end].iloc[-1]
    except IndexError:
        return pd.Series(dtype=float)
    return (p1 / p0) - 1.0


def cross_sectional_zscore(feats: pd.DataFrame) -> pd.DataFrame:
    """Standardise features across the cross-section at a single date.

    This is point-in-time safe: it only uses values from the same date. It helps
    tree models and is essential for the simple z-score fallback ranker.
    """
    mu = feats.mean()
    sd = feats.std().replace(0, np.nan)
    z = (feats - mu) / sd
    return z.fillna(0.0)
