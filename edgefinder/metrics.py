"""Performance metrics computed from a period-return series.

Everything here operates on the realised per-period (e.g. monthly) returns of
the strategy AFTER costs, so the numbers are the ones that actually matter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(period_returns: pd.Series) -> pd.Series:
    return (1.0 + period_returns.fillna(0.0)).cumprod()


def cagr(period_returns: pd.Series, periods_per_year: int) -> float:
    eq = equity_curve(period_returns)
    if len(eq) == 0:
        return float("nan")
    total = eq.iloc[-1]
    years = len(period_returns) / periods_per_year
    if years <= 0 or total <= 0:
        return float("nan")
    return total ** (1.0 / years) - 1.0


def sharpe(period_returns: pd.Series, periods_per_year: int, rf_annual: float = 0.06) -> float:
    """Annualised Sharpe using a risk-free rate (default 6%, typical Indian repo-ish)."""
    rf_period = (1 + rf_annual) ** (1 / periods_per_year) - 1
    excess = period_returns.dropna() - rf_period
    if excess.std() == 0 or len(excess) < 2:
        return float("nan")
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()


def max_drawdown(period_returns: pd.Series) -> float:
    eq = equity_curve(period_returns)
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return dd.min()


def hit_rate(period_returns: pd.Series) -> float:
    r = period_returns.dropna()
    if len(r) == 0:
        return float("nan")
    return (r > 0).mean()


def volatility(period_returns: pd.Series, periods_per_year: int) -> float:
    return period_returns.dropna().std() * np.sqrt(periods_per_year)


def summarize(
    name: str, period_returns: pd.Series, periods_per_year: int, avg_turnover: float | None = None
) -> dict:
    return {
        "strategy": name,
        "CAGR": cagr(period_returns, periods_per_year),
        "Sharpe": sharpe(period_returns, periods_per_year),
        "Volatility": volatility(period_returns, periods_per_year),
        "MaxDrawdown": max_drawdown(period_returns),
        "HitRate": hit_rate(period_returns),
        "Periods": int(period_returns.notna().sum()),
        "AvgTurnover": avg_turnover,
        "TotalReturn": equity_curve(period_returns).iloc[-1] - 1.0 if len(period_returns) else float("nan"),
    }


def format_summary_table(rows: list[dict]) -> str:
    """Pretty fixed-width table for the console / CI logs."""
    cols = ["strategy", "CAGR", "Sharpe", "Volatility", "MaxDrawdown",
            "HitRate", "TotalReturn", "AvgTurnover", "Periods"]
    headers = {
        "strategy": "Strategy", "CAGR": "CAGR", "Sharpe": "Sharpe",
        "Volatility": "Vol", "MaxDrawdown": "MaxDD", "HitRate": "Hit%",
        "TotalReturn": "TotRet", "AvgTurnover": "Turnover", "Periods": "N",
    }

    def fmt(key, val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "-"
        if key in {"CAGR", "Volatility", "MaxDrawdown", "HitRate", "TotalReturn", "AvgTurnover"}:
            return f"{val * 100:.1f}%"
        if key == "Sharpe":
            return f"{val:.2f}"
        if key == "strategy":
            return str(val)
        return str(val)

    widths = {c: max(len(headers[c]), *(len(fmt(c, r.get(c))) for r in rows)) for c in cols}
    line = " | ".join(headers[c].ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    body = "\n".join(
        " | ".join(fmt(c, r.get(c)).ljust(widths[c]) for c in cols) for r in rows
    )
    return f"{line}\n{sep}\n{body}"



def excess_stats(
    strat: pd.Series, ref: pd.Series, periods_per_year: int
) -> dict:
    """Statistical significance of the strategy's excess return over a reference.

    Aligns the two return series on common dates, takes the per-period
    difference, and reports:
      * t_stat            : a paired t-stat on the mean excess return. As a rough
                            rule, |t| > ~2 suggests the edge is unlikely to be
                            pure chance (with the usual multiple-testing caveats).
      * information_ratio : annualised mean excess / tracking error.
      * mean_excess_ann   : annualised mean excess return.
      * n                 : number of overlapping periods.
    """
    df = pd.concat([strat.rename("s"), ref.rename("r")], axis=1).dropna()
    if len(df) < 3:
        return {}
    d = df["s"] - df["r"]
    sd = d.std()
    if sd == 0:
        return {}
    n = len(d)
    return {
        "n": int(n),
        "t_stat": float(d.mean() / (sd / np.sqrt(n))),
        "information_ratio": float(np.sqrt(periods_per_year) * d.mean() / sd),
        "mean_excess_ann": float(d.mean() * periods_per_year),
    }
