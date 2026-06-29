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


def _extract_close(df: "pd.DataFrame | None", ticker: str) -> "pd.Series | None":
    if df is None or df.empty or "Close" not in df:
        return None
    s = df["Close"].copy()
    if isinstance(s, pd.DataFrame):  # yfinance sometimes returns a 1-col frame
        s = s.iloc[:, 0]
    s = s.dropna()
    if s.empty:
        return None
    s = _normalize_index(s)
    s.name = ticker
    return s


def _normalize_index(s: pd.Series) -> pd.Series:
    """Force a tz-naive, date-only DatetimeIndex.

    download() returns tz-naive dates, but the history() fallback returns
    tz-AWARE timestamps. Mixing the two breaks DataFrame alignment with
    'Cannot join tz-naive with tz-aware DatetimeIndex'. Normalising everything
    here keeps all sources (download / history / CSV cache) compatible.
    """
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")]
    return s


def _download_one(
    ticker: str, start: str, end: str, retries: int = 3, base_delay: float = 1.5
) -> pd.Series | None:
    """Download adjusted close for one ticker, with retry + backoff.

    Yahoo's free endpoint frequently returns transient 404 / "possibly delisted"
    errors under load — these are RATE LIMITING, not real delistings. We retry
    with exponential backoff and fall back to the per-ticker history() API,
    which often succeeds when the bulk download() endpoint is throttled.
    """
    import yfinance as yf

    last_err = None
    for attempt in range(retries):
        # --- primary: bulk download endpoint ---
        try:
            df = yf.download(
                ticker, start=start, end=end, auto_adjust=True,
                progress=False, threads=False,
            )
            s = _extract_close(df, ticker)
            if s is not None:
                return s
        except Exception as exc:
            last_err = exc

        # --- fallback: per-ticker history endpoint ---
        try:
            df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            s = _extract_close(df, ticker)
            if s is not None:
                return s
        except Exception as exc:
            last_err = exc

        if attempt < retries - 1:
            time.sleep(base_delay * (2 ** attempt))  # 1.5s, 3s, 6s ...

    if last_err is not None:
        print(f"  ! {ticker}: giving up after {retries} tries ({last_err})")
    else:
        print(f"  ! {ticker}: no data returned (likely transient rate-limit)")
    return None


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
    failed: list[str] = []

    for i, ticker in enumerate(tickers, 1):
        path = _cache_path(cache_dir, ticker)
        s: pd.Series | None = None

        if os.path.exists(path) and not force_refresh:
            try:
                cached = pd.read_csv(path, index_col=0, parse_dates=True)
                if not cached.empty:
                    s = cached.iloc[:, 0]
                    s = _normalize_index(s)  # fix any tz-aware cached files
                    s.name = ticker
            except Exception:
                s = None

        if s is None:
            print(f"[{i}/{len(tickers)}] downloading {ticker} ...")
            s = _download_one(ticker, start, end)
            if s is not None and not s.empty:
                s.to_frame(name=ticker).to_csv(path)
            time.sleep(0.4)  # be polite to the free endpoint (reduces throttling)

        if s is not None and not s.empty:
            series[ticker] = s
        else:
            failed.append(ticker)

    if not series:
        raise RuntimeError(
            "No price data could be loaded. If you are offline, populate the "
            "cache first from a machine with internet access."
        )

    prices = pd.DataFrame(series).sort_index()
    # Drop fully-empty columns and forward-fill small gaps (holidays etc.)
    prices = prices.dropna(axis=1, how="all").ffill(limit=5)
    print(f"Loaded prices: {prices.shape[1]} tickers x {prices.shape[0]} days")
    if failed:
        print(
            f"NOTE: {len(failed)} ticker(s) failed (usually transient Yahoo "
            f"rate-limiting, NOT delisting): {', '.join(failed)}\n"
            f"      Successful tickers are cached — just RE-RUN the same command "
            f"to retry only the failed ones."
        )
    return prices


def load_benchmark(
    symbol: str, start: str, end: str, cache_dir: str = "data_cache"
) -> pd.Series:
    """Load a single benchmark series (e.g. ^NSEI for NIFTY 50)."""
    df = load_prices([symbol], start, end, cache_dir=cache_dir)
    return df[symbol]
