"""Free price-data loading via yfinance, with local CSV caching.

Caching matters for two reasons:
  1. Reruns are instant and don't hammer Yahoo's free endpoint.
  2. Once cached, the backtest runs fully OFFLINE (useful in restricted envs).

We always use ADJUSTED close prices so corporate actions (splits/dividends)
don't create fake jumps that a naive backtest would mistake for returns.
"""

from __future__ import annotations

import os
import time

import pandas as pd


def _cache_path(cache_dir: str, ticker: str) -> str:
    safe = ticker.replace("/", "_")
    return os.path.join(cache_dir, f"{safe}.csv")


def _download_one(ticker: str, start: str, end: str) -> pd.Series | None:
    """Download adjusted close for one ticker. Returns None on failure."""
    import yfinance as yf

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,      # 'Close' becomes split/dividend adjusted
            progress=False,
            threads=False,
        )
    except Exception as exc:  # network/ticker errors should not kill the run
        print(f"  ! download failed for {ticker}: {exc}")
        return None

    if df is None or df.empty or "Close" not in df:
        return None
    s = df["Close"].copy()
    if isinstance(s, pd.DataFrame):  # yfinance sometimes returns a 1-col frame
        s = s.iloc[:, 0]
    s.name = ticker
    return s


def load_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str = "data_cache",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return a wide DataFrame of adjusted close prices: index=date, cols=ticker.

    Cached per-ticker CSVs are reused unless ``force_refresh`` is set.
    """
    os.makedirs(cache_dir, exist_ok=True)
    series: dict[str, pd.Series] = {}

    for i, ticker in enumerate(tickers, 1):
        path = _cache_path(cache_dir, ticker)
        s: pd.Series | None = None

        if os.path.exists(path) and not force_refresh:
            try:
                cached = pd.read_csv(path, index_col=0, parse_dates=True)
                if not cached.empty:
                    s = cached.iloc[:, 0]
                    s.name = ticker
            except Exception:
                s = None

        if s is None:
            print(f"[{i}/{len(tickers)}] downloading {ticker} ...")
            s = _download_one(ticker, start, end)
            if s is not None and not s.empty:
                s.to_frame(name=ticker).to_csv(path)
                time.sleep(0.2)  # be polite to the free endpoint

        if s is not None and not s.empty:
            series[ticker] = s

    if not series:
        raise RuntimeError(
            "No price data could be loaded. If you are offline, populate the "
            "cache first from a machine with internet access."
        )

    prices = pd.DataFrame(series).sort_index()
    # Drop fully-empty columns and forward-fill small gaps (holidays etc.)
    prices = prices.dropna(axis=1, how="all").ffill(limit=5)
    print(f"Loaded prices: {prices.shape[1]} tickers x {prices.shape[0]} days")
    return prices


def load_benchmark(
    symbol: str, start: str, end: str, cache_dir: str = "data_cache"
) -> pd.Series:
    """Load a single benchmark series (e.g. ^NSEI for NIFTY 50)."""
    df = load_prices([symbol], start, end, cache_dir=cache_dir)
    return df[symbol]
