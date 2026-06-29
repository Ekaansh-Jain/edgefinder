"""Walk-forward cross-sectional backtest engine.

Flow at every rebalance date T:
  1. Build point-in-time features for all eligible names (data <= T only).
  2. Train the ranking model on ALL past (features@Ti -> return Ti..Ti+1) pairs.
     The current period's label is NOT known and never used for training.
  3. Predict scores at T, pick the top-N, equal-weight them.
  4. Realise the return over T..T+1 and charge costs based on turnover.

This expanding-window design is the standard defence against look-ahead bias.
We also run two honest reference strategies on the SAME dates:
  * 'EqualWeight'  : hold the whole universe equal-weighted (beta benchmark).
  * 'Buy&HoldNIFTY': the index itself.
so any "edge" must be judged as outperformance, not just a positive number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .features import build_features, forward_return
from .model import RankingModel


def _rebalance_dates(prices: pd.DataFrame, freq: str) -> list[pd.Timestamp]:
    """Last available trading day on or before each period boundary."""
    idx = prices.index
    period_ends = pd.date_range(idx.min(), idx.max(), freq=freq)
    dates: list[pd.Timestamp] = []
    for pe in period_ends:
        prior = idx[idx <= pe]
        if len(prior) > 0:
            d = prior[-1]
            if not dates or d != dates[-1]:
                dates.append(d)
    return dates


def _one_way_costs(
    prev_w: pd.Series, new_w: pd.Series, buy_cost: float, sell_cost: float
) -> tuple[float, float]:
    """Return (cost_fraction, one_way_turnover) for moving prev_w -> new_w."""
    all_idx = prev_w.index.union(new_w.index)
    pw = prev_w.reindex(all_idx).fillna(0.0)
    nw = new_w.reindex(all_idx).fillna(0.0)
    delta = nw - pw
    buys = delta.clip(lower=0).sum()      # fraction of book bought
    sells = (-delta.clip(upper=0)).sum()  # fraction of book sold
    cost = buys * buy_cost + sells * sell_cost
    one_way_turnover = 0.5 * delta.abs().sum()
    return cost, one_way_turnover


def _select_and_weight(
    scores: pd.Series, feats: pd.DataFrame, prev_w: pd.Series, cfg
) -> pd.Series:
    """Pick holdings (with a turnover buffer) and assign weights.

    Turnover buffer (hysteresis): a current holding is kept as long as it stays
    ranked within (1+buffer)*top_n, instead of being dumped the moment it leaves
    the top_n. This sharply cuts churn (and cost) versus rebuilding every period.

    Weighting: 'inv_vol' gives lower-volatility names more weight (a risk-parity
    tilt that usually improves Sharpe and tames drawdowns); 'equal' is the
    classic equal-weight.
    """
    ranked = scores.sort_values(ascending=False)
    n = cfg.top_n
    extended = set(ranked.head(int(round(n * (1 + cfg.turnover_buffer)))).index)

    # Keep current holdings still ranked inside the extended band (preserve rank order).
    kept = [t for t in ranked.index if t in prev_w.index and t in extended][:n]
    need = max(0, n - len(kept))
    additions = [t for t in ranked.index if t not in kept][:need]
    picks = [p for p in (kept + additions) if p in feats.index]
    if not picks:
        return pd.Series(dtype=float)

    if cfg.weighting == "inv_vol":
        vol = feats.loc[picks, "vol_6m"].astype(float)
        vol = vol.replace([np.inf, -np.inf], np.nan).fillna(vol.median())
        vol = vol.clip(lower=0.05)        # floor at 5% ann. vol to avoid blow-ups
        inv = 1.0 / vol
        w = inv / inv.sum()
    else:  # equal
        w = pd.Series(1.0 / len(picks), index=pd.Index(picks))
    return w


def _regime_exposure(
    regime_series: "pd.Series | None", as_of: pd.Timestamp, cfg
) -> float:
    """Exposure multiplier at a rebalance date (point-in-time).

    Full exposure (1.0) when the regime filter is off, or when the benchmark
    closes at/above its trailing moving average; otherwise ``risk_off_exposure``.
    Only uses benchmark data up to ``as_of`` (no look-ahead).
    """
    if not cfg.regime_filter or regime_series is None:
        return 1.0
    hist = regime_series.loc[:as_of].dropna()
    if len(hist) < cfg.regime_ma_days:
        return 1.0  # not enough history -> stay invested
    ma = hist.tail(cfg.regime_ma_days).mean()
    return 1.0 if hist.iloc[-1] >= ma else float(cfg.risk_off_exposure)


def run_backtest(
    prices: pd.DataFrame,
    cfg: BacktestConfig,
    score_overlay: "pd.DataFrame | None" = None,
    regime_series: "pd.Series | None" = None,
):
    """Execute the walk-forward backtest.

    Parameters
    ----------
    prices : adjusted close, wide (date x ticker). Benchmark excluded.
    cfg    : BacktestConfig.
    score_overlay : optional DataFrame (index=date, cols=ticker) of extra scores
        (e.g. GDELT/LLM news sentiment) added to the model score before ranking.
        Must be point-in-time (value at date T known at T). Missing -> 0.
    regime_series : optional benchmark price Series for the trend filter. When
        ``cfg.regime_filter`` is on and the benchmark is below its MA, exposure
        is scaled toward cash. Strictly point-in-time.

    Returns
    -------
    dict with per-strategy period-return Series and diagnostics.
    """
    dates = _rebalance_dates(prices, cfg.rebalance)
    if len(dates) < cfg.train_min_periods + 3:
        raise RuntimeError(
            f"Not enough rebalance periods ({len(dates)}). Use a longer history "
            f"or a higher-frequency rebalance."
        )

    buy_c = cfg.costs.buy_cost()
    sell_c = cfg.costs.sell_cost()

    # Accumulated training data across the walk.
    feat_history: list[pd.DataFrame] = []
    label_history: list[pd.Series] = []

    strat_rets: list[float] = []
    ew_rets: list[float] = []
    strat_dates: list[pd.Timestamp] = []
    turnovers: list[float] = []

    prev_w = pd.Series(dtype=float)        # strategy weights
    prev_ew = pd.Series(dtype=float)       # equal-weight weights
    last_importance = None
    risk_off_periods = 0

    for i in range(len(dates) - 1):
        t0, t1 = dates[i], dates[i + 1]
        feats = build_features(prices, t0)
        if feats.empty:
            continue

        fwd = forward_return(prices, t0, t1).reindex(feats.index)

        # ---- train on PAST periods only, then predict for t0 ----
        traded = False
        if len(feat_history) >= cfg.train_min_periods:
            X_train = pd.concat(feat_history)
            y_train = pd.concat(label_history)
            model = RankingModel(
                use_lightgbm=cfg.use_lightgbm, random_state=cfg.random_state
            )
            model.fit(X_train, y_train)
            scores = model.predict(feats)
            last_importance = model.feature_importance()

            if score_overlay is not None and t0 in score_overlay.index:
                ov = score_overlay.loc[t0].reindex(scores.index).fillna(0.0)
                # standardise overlay so it doesn't dominate the model score
                if ov.std() > 0:
                    ov = (ov - ov.mean()) / ov.std()
                scores = scores + ov

            new_w = _select_and_weight(scores, feats, prev_w, cfg)
            # Apply the regime/trend filter: scale exposure toward cash in
            # risk-off periods. Remaining weight is implicitly cash (0 return).
            exposure = _regime_exposure(regime_series, t0, cfg)
            if exposure < 1.0:
                new_w = new_w * exposure
                risk_off_periods += 1
            traded = len(new_w) > 0
        else:
            new_w = pd.Series(dtype=float)  # not trading yet (warm-up)

        # ---- realise strategy return over t0..t1, net of costs ----
        if traded:
            gross = float((new_w * fwd.reindex(new_w.index).fillna(0.0)).sum())
            cost, to = _one_way_costs(prev_w, new_w, buy_c, sell_c)
            strat_rets.append(gross - cost)
            turnovers.append(to)
            strat_dates.append(t1)
            prev_w = new_w

        # ---- equal-weight reference on the same eligible universe ----
        ew_w = pd.Series(1.0 / len(feats), index=feats.index)
        ew_gross = float((ew_w * fwd.fillna(0.0)).sum())
        ew_cost, _ = _one_way_costs(prev_ew, ew_w, buy_c, sell_c)
        if traded:
            ew_rets.append(ew_gross - ew_cost)
        prev_ew = ew_w

        # ---- append this period to training history for the NEXT step ----
        # Labels are DEMEANED cross-sectionally: the model learns what makes a
        # stock beat its PEERS this period, not what makes the whole market rise.
        # This isolates selection skill and avoids just loading up on high-beta
        # momentum in a bull market. Demeaning uses only this period's cross-
        # section, so it stays point-in-time safe.
        feat_history.append(feats)
        demeaned = fwd - fwd.mean(skipna=True)
        label_history.append(demeaned)

    strat = pd.Series(strat_rets, index=pd.DatetimeIndex(strat_dates), name="Strategy")
    ew = pd.Series(ew_rets, index=pd.DatetimeIndex(strat_dates[: len(ew_rets)]), name="EqualWeight")

    return {
        "strategy_returns": strat,
        "equalweight_returns": ew,
        "avg_turnover": float(np.mean(turnovers)) if turnovers else float("nan"),
        "risk_off_periods": risk_off_periods,
        "rebalance_dates": strat_dates,
        "feature_importance": last_importance,
        "model_kind": RankingModel(use_lightgbm=cfg.use_lightgbm).kind,
    }


def benchmark_returns(
    benchmark_prices: pd.Series, on_dates: list[pd.Timestamp]
) -> pd.Series:
    """Period returns of the benchmark aligned to the strategy's rebalance dates."""
    vals = []
    idx = []
    bp = benchmark_prices.dropna()
    for i in range(1, len(on_dates)):
        t0, t1 = on_dates[i - 1], on_dates[i]
        try:
            p0 = bp.loc[:t0].iloc[-1]
            p1 = bp.loc[:t1].iloc[-1]
            vals.append(p1 / p0 - 1.0)
            idx.append(t1)
        except (IndexError, KeyError):
            continue
    return pd.Series(vals, index=pd.DatetimeIndex(idx), name="NIFTY")
