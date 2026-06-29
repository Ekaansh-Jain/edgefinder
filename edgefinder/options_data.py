"""Source-agnostic historical options data loader for the VRP backtester.

The backtester works off a TIDY dataset with these columns:
    date         : trade date (the day the quote/settlement is for)
    expiry       : contract expiry date
    strike       : strike price (float)
    option_type  : 'CE' (call) or 'PE' (put)
    close        : settlement/close price of that option contract

Plus an underlying NIFTY spot series (date -> close) for expiry settlement and
strike selection.

Why source-agnostic? NSE scrapers (nsepy / jugaad-data) break often because NSE
changes its site/format. So the RELIABLE path is: get a CSV from any source and
load it here. We also provide a best-effort NSE bhavcopy fetcher, but treat it
as a convenience, not a guarantee.

Where to get the data (all viable):
  * FREE  : NSE F&O bhavcopy archives (one EOD file/day, all contracts) via
            `jugaad-data` or manual download. 15+ years available.
  * FREE  : `nsepy` (per-contract history).
  * PAID  : TrueData / GDFL (cleaner, intraday).
  * EXPORT: AlgoTest / Opstra let you export option data too.
Just shape whatever you get into the 5 tidy columns above.
"""

from __future__ import annotations

import os

import pandas as pd

TIDY_COLUMNS = ["date", "expiry", "strike", "option_type", "close"]

# Common column-name variants across NSE bhavcopy (old + new UDiFF) and vendors.
_COLUMN_ALIASES = {
    "date": ["date", "timestamp", "traddt", "trade_date", "TIMESTAMP", "TradDt"],
    "expiry": ["expiry", "expiry_dt", "EXPIRY_DT", "XpryDt", "expiry_date"],
    "strike": ["strike", "strike_pr", "STRIKE_PR", "StrkPric", "strike_price"],
    "option_type": ["option_type", "option_typ", "OPTION_TYP", "OptnTp", "optiontype"],
    "close": ["close", "close_pric", "CLOSE_PRIC", "ClsPric", "settle_pr",
              "SETTLE_PR", "close_price", "ClsgPric"],
    "symbol": ["symbol", "SYMBOL", "TckrSymb", "instrument"],
    "instrument": ["instrument", "INSTRUMENT", "FinInstrmTp", "instrument_type"],
}


def _find_col(df: pd.DataFrame, key: str) -> "str | None":
    lower = {c.lower(): c for c in df.columns}
    for cand in _COLUMN_ALIASES.get(key, []):
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def normalize_bhavcopy(raw: pd.DataFrame, underlying: str = "NIFTY") -> pd.DataFrame:
    """Map a raw NSE F&O bhavcopy (old or new format) to the tidy schema.

    Filters to index options on ``underlying`` (CE/PE only).
    """
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    sym_col = _find_col(df, "symbol")
    inst_col = _find_col(df, "instrument")
    if sym_col:
        df = df[df[sym_col].astype(str).str.upper() == underlying.upper()]
    if inst_col:
        # keep index options (OPTIDX / 'STO'/'OPT' variants)
        mask = df[inst_col].astype(str).str.upper().str.contains("OPT")
        df = df[mask]

    cols = {k: _find_col(df, k) for k in ["date", "expiry", "strike", "option_type", "close"]}
    missing = [k for k, v in cols.items() if v is None]
    if missing:
        raise ValueError(
            f"Bhavcopy missing columns for {missing}. Found: {list(raw.columns)}. "
            f"Rename/produce the tidy columns {TIDY_COLUMNS} manually."
        )
    out = pd.DataFrame({
        "date": pd.to_datetime(df[cols["date"]], errors="coerce", dayfirst=True),
        "expiry": pd.to_datetime(df[cols["expiry"]], errors="coerce", dayfirst=True),
        "strike": pd.to_numeric(df[cols["strike"]], errors="coerce"),
        "option_type": df[cols["option_type"]].astype(str).str.upper().str.strip(),
        "close": pd.to_numeric(df[cols["close"]], errors="coerce"),
    })
    out = out[out["option_type"].isin(["CE", "PE"])]
    return out.dropna(subset=["date", "expiry", "strike", "close"])


def load_options_csv(path: str) -> pd.DataFrame:
    """Load a tidy options CSV (or a raw bhavcopy, auto-normalised)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Options data not found at '{path}'. See edgefinder/options_data.py "
            f"docstring for free/paid sources; shape it into columns {TIDY_COLUMNS}."
        )
    raw = pd.read_csv(path)
    cols_lower = {c.lower() for c in raw.columns}
    if set(TIDY_COLUMNS).issubset(cols_lower):
        # already tidy (case-insensitive)
        ren = {c: c.lower() for c in raw.columns}
        df = raw.rename(columns=ren)[TIDY_COLUMNS].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["option_type"] = df["option_type"].astype(str).str.upper().str.strip()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "expiry", "strike", "close"])
    else:
        df = normalize_bhavcopy(raw)
    df = df[df["option_type"].isin(["CE", "PE"])].sort_values(["expiry", "date", "strike"])
    return df.reset_index(drop=True)


def derive_spot(options_df: pd.DataFrame) -> pd.Series:
    """Best-effort underlying spot per date when a separate spot file is absent.

    Approximates spot on each date as the strike where |CE_close - PE_close| is
    smallest for the nearest expiry (put-call parity: C-P=0 near the forward).
    A real NIFTY spot series is better; pass one if you have it.
    """
    spots = {}
    for d, day in options_df.groupby("date"):
        near_expiry = day["expiry"].min()
        chain = day[day["expiry"] == near_expiry]
        ce = chain[chain["option_type"] == "CE"].set_index("strike")["close"]
        pe = chain[chain["option_type"] == "PE"].set_index("strike")["close"]
        common = ce.index.intersection(pe.index)
        if len(common) == 0:
            continue
        diff = (ce.reindex(common) - pe.reindex(common)).abs()
        spots[d] = float(diff.idxmin())
    return pd.Series(spots).sort_index()


def fetch_bhavcopy_jugaad(start: str, end: str, out_csv: str,
                          underlying: str = "NIFTY") -> str:
    """BEST-EFFORT: download NSE F&O bhavcopy day-by-day via jugaad-data.

    Fragile (NSE changes formats / rate-limits). If it fails, download the
    bhavcopy files manually and point load_options_csv at a concatenated CSV.
    """
    try:
        from jugaad_data.nse import bhavcopy_fo_raw  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "jugaad-data not installed or API changed. `pip install jugaad-data`, "
            f"or download bhavcopy manually. ({exc})"
        )
    import datetime as dt

    days = pd.bdate_range(start, end)
    frames = []
    for d in days:
        try:
            raw = pd.read_csv(pd.io.common.StringIO(
                bhavcopy_fo_raw(dt.date(d.year, d.month, d.day))))
            frames.append(normalize_bhavcopy(raw, underlying=underlying))
        except Exception:
            continue  # holiday / missing / format hiccup
    if not frames:
        raise RuntimeError("No bhavcopy data fetched; use manual download instead.")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} option rows -> {out_csv}")
    return out_csv
