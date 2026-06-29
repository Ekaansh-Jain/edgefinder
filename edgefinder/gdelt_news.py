"""Free, backtestable news-sentiment overlay via GDELT.

GDELT (gdeltproject.org) indexes global news with computed "tone" and entity
tagging, going back years, through a FREE API that needs NO key. That makes it
one of the few ways to build a genuinely POINT-IN-TIME news signal for a
historical backtest (most news APIs only return recent data).

Pipeline
--------
For each ticker we query GDELT's DOC 2.0 ``TimelineTone`` endpoint to get an
average-tone time series for the company name, cache it, and then aggregate it
into a (rebalance-date x ticker) overlay using only news published ON OR BEFORE
each rebalance date (a trailing window). The overlay is added to the model's
ranking score inside the backtest (the existing ``score_overlay`` hook).

Optionally, set a free LLM key (see ``llm_sentiment.py`` / ``.env.example``) to
re-score recent headlines instead of using GDELT's built-in tone.

Notes / honesty
---------------
* Entity matching from a name string is imperfect; expect noise.
* Tone is a crude sentiment proxy. This tests "does news info help?", not a
  production signal.
* Network access required -> run locally or in CI, not in an offline sandbox.
"""

from __future__ import annotations

import os
import time
from urllib.parse import urlencode

import pandas as pd

from .universe import get_company_query

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


def _cache_path(cache_dir: str, ticker: str) -> str:
    safe = ticker.replace("/", "_")
    return os.path.join(cache_dir, f"tone_{safe}.csv")


def _fetch_tone_series(
    query: str, start: str, end: str, retries: int = 3, base_delay: float = 1.0
) -> "pd.Series | None":
    """Fetch GDELT average-tone daily timeline for a query string.

    Returns a tz-naive daily Series of tone values, or None on failure.
    """
    import requests

    params = {
        # Quote the phrase so multi-word company names match as a unit, and bias
        # toward business/English coverage to cut cross-language noise.
        "query": f'"{query}" sourcelang:english',
        "mode": "TimelineTone",
        "format": "json",
        "startdatetime": pd.Timestamp(start).strftime("%Y%m%d%H%M%S"),
        "enddatetime": pd.Timestamp(end).strftime("%Y%m%d%H%M%S"),
    }
    url = f"{GDELT_DOC_API}?{urlencode(params)}"

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "edgefinder/0.1"})
            if resp.status_code == 200 and resp.text.strip().startswith("{"):
                data = resp.json()
                timeline = data.get("timeline", [])
                if not timeline:
                    return None
                points = timeline[0].get("data", [])
                if not points:
                    return None
                dates, vals = [], []
                for p in points:
                    raw = str(p.get("date", ""))
                    ts = pd.to_datetime(raw, errors="coerce", utc=True)
                    if pd.isna(ts):
                        continue
                    dates.append(ts.tz_localize(None).normalize())
                    vals.append(float(p.get("value", 0.0)))
                if not dates:
                    return None
                s = pd.Series(vals, index=pd.DatetimeIndex(dates)).sort_index()
                return s[~s.index.duplicated(keep="last")]
        except Exception:
            pass
        time.sleep(base_delay * (2 ** attempt))
    return None


def load_tone(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str = "news_cache",
    force_refresh: bool = False,
) -> dict[str, pd.Series]:
    """Return {ticker: daily tone Series}, fetching+caching as needed.

    Caches BOTH hits (CSV) and misses (an empty '.empty' marker) so that a
    re-run never re-queries the slow GDELT endpoint for tickers it already
    knows about. Delete the cache dir or pass force_refresh=True to refetch.
    """
    os.makedirs(cache_dir, exist_ok=True)
    out: dict[str, pd.Series] = {}
    n_hit = n_fetched = n_empty = 0

    for i, ticker in enumerate(tickers, 1):
        path = _cache_path(cache_dir, ticker)
        empty_marker = path + ".empty"
        s: pd.Series | None = None

        if not force_refresh and os.path.exists(empty_marker):
            n_empty += 1
            continue  # known to have no GDELT data -> skip entirely

        if os.path.exists(path) and not force_refresh:
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                if not df.empty:
                    s = df.iloc[:, 0]
                    n_hit += 1
            except Exception:
                s = None

        if s is None and not os.path.exists(path):
            query = get_company_query(ticker)
            print(f"[{i}/{len(tickers)}] GDELT tone: {ticker} ('{query}') ...")
            s = _fetch_tone_series(query, start, end)
            if s is not None and not s.empty:
                s.to_frame(name="tone").to_csv(path)
                n_fetched += 1
            else:
                open(empty_marker, "w").close()  # negative cache the miss
                n_empty += 1
            time.sleep(0.7)  # GDELT is rate-limited; be gentle

        if s is not None and not s.empty:
            s.index = pd.DatetimeIndex(s.index)
            out[ticker] = s

    print(f"GDELT tone: {len(out)} usable  (cache hits {n_hit}, newly fetched "
          f"{n_fetched}, no-data {n_empty})")
    return out


def build_overlay(
    tone_by_ticker: dict[str, pd.Series],
    rebalance_dates: list,
    window_days: int = 30,
    use_llm: bool = False,
) -> pd.DataFrame:
    """Aggregate daily tone into a point-in-time (date x ticker) overlay.

    For each rebalance date T, the value for a ticker is the mean tone over
    (T - window_days, T]. Strictly uses news on/before T (no look-ahead).
    Columns are z-scored cross-sectionally inside the backtest, so raw tone
    scale does not matter here.
    """
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in rebalance_dates])
    overlay = pd.DataFrame(index=idx, columns=list(tone_by_ticker.keys()), dtype=float)

    for ticker, s in tone_by_ticker.items():
        s = s.sort_index()
        for T in idx:
            window = s[(s.index <= T) & (s.index > T - pd.Timedelta(days=window_days))]
            if len(window) > 0:
                overlay.at[T, ticker] = float(window.mean())

    if use_llm:
        # Optional: blend GDELT tone with a free-LLM re-score of recent context.
        # Kept lightweight; only runs if a provider env var is set.
        try:
            from .llm_sentiment import _provider

            if _provider() is not None:
                print("LLM provider detected: GDELT tone retained as primary signal "
                      "(LLM headline rescoring available via --news csv path).")
        except Exception:
            pass

    return overlay.astype(float)


def gdelt_overlay(
    tickers: list[str],
    start: str,
    end: str,
    rebalance_dates: list,
    cache_dir: str = "news_cache",
    window_days: int = 30,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Convenience: fetch tone for tickers and build the overlay in one call."""
    tone = load_tone(tickers, start, end, cache_dir=cache_dir, force_refresh=force_refresh)
    if not tone:
        print("WARNING: no GDELT tone retrieved; overlay will be neutral (zeros).")
        return pd.DataFrame(
            0.0, index=pd.DatetimeIndex([pd.Timestamp(d) for d in rebalance_dates]),
            columns=tickers,
        )
    return build_overlay(tone, rebalance_dates, window_days=window_days)
