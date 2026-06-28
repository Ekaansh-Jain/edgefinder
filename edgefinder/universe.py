"""Investable universes of NSE tickers (yfinance format, '.NS' suffix).

We focus on liquid large/mid-caps. Per the research, the documented LLM-news
edge is strongest in smaller, less-covered names, so a 'midcap' tilt is provided
too. These lists are static snapshots for reproducibility; refresh periodically
from the NSE indices if you want survivorship-bias control (see README).
"""

from __future__ import annotations

# NIFTY 50 (large-cap core)
NIFTY50 = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
    "BAJFINANCE", "HCLTECH", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO",
    "NESTLEIND", "ONGC", "NTPC", "POWERGRID", "TATAMOTORS", "TATASTEEL",
    "ADANIENT", "ADANIPORTS", "JSWSTEEL", "COALINDIA", "BAJAJFINSV", "GRASIM",
    "HINDALCO", "INDUSINDBK", "DRREDDY", "CIPLA", "EICHERMOT", "BRITANNIA",
    "HEROMOTOCO", "DIVISLAB", "BPCL", "TECHM", "APOLLOHOSP", "BAJAJ-AUTO",
    "TATACONSUM", "SBILIFE", "HDFCLIFE", "LTIM", "SHRIRAMFIN", "M&M",
]

# NIFTY Next 50 (large/mid-cap) — extends coverage beyond the top 50.
NIFTY_NEXT50 = [
    "ADANIGREEN", "ADANIPOWER", "AMBUJACEM", "DLF", "BANKBARODA", "BEL",
    "BOSCHLTD", "CHOLAFIN", "COLPAL", "DABUR", "GAIL", "GODREJCP", "HAVELLS",
    "ICICIGI", "ICICIPRULI", "IOC", "INDIGO", "NAUKRI", "PIDILITIND", "PNB",
    "SIEMENS", "SRF", "TORNTPHARM", "TVSMOTOR", "VEDL", "ZOMATO", "ZYDUSLIFE",
    "MARICO", "BERGEPAINT", "PAGEIND", "MUTHOOTFIN", "JINDALSTEL", "TATAPOWER",
    "UNIONBANK", "IRCTC", "LICI", "MAXHEALTH", "POLYCAB", "INDUSTOWER",
    "CANBK", "CGPOWER", "HINDPETRO", "JIOFIN", "MOTHERSON", "NHPC", "OFSS",
    "PFC", "RECLTD", "TRENT", "VBL",
]

# A midcap tilt list (less analyst coverage => where the text/news edge lives).
MIDCAP_EXTRA = [
    "AUBANK", "ASHOKLEY", "ASTRAL", "BALKRISIND", "BANDHANBNK", "BHARATFORG",
    "COFORGE", "CONCOR", "CUMMINSIND", "ESCORTS", "FEDERALBNK", "GMRINFRA",
    "GUJGASLTD", "IDFCFIRSTB", "INDHOTEL", "LUPIN", "MFSL", "MPHASIS",
    "MRF", "PERSISTENT", "PETRONET", "PIIND", "SAIL", "SUNDARMFIN",
    "SUPREMEIND", "TATACOMM", "TIINDIA", "UBL", "VOLTAS", "ABCAPITAL",
    "ALKEM", "APLAPOLLO", "AUROPHARMA", "BANKINDIA", "DEEPAKNTR", "DIXON",
    "GODREJPROP", "HAL", "IDEA", "IGL", "INDUSTOWER", "JUBLFOOD", "LTTS",
    "NMDC", "OBEROIRLTY", "PRESTIGE", "SHREECEM", "SOLARINDS", "TATAELXSI",
    "UPL",
]


def _suffix(tickers: list[str]) -> list[str]:
    """Append the '.NS' suffix yfinance uses for NSE, de-duplicating order."""
    seen: dict[str, None] = {}
    for t in tickers:
        seen.setdefault(f"{t}.NS", None)
    return list(seen.keys())


UNIVERSES: dict[str, list[str]] = {
    "nifty50": _suffix(NIFTY50),
    "nifty100": _suffix(NIFTY50 + NIFTY_NEXT50),
    "nifty200": _suffix(NIFTY50 + NIFTY_NEXT50 + MIDCAP_EXTRA),
    "midcap": _suffix(MIDCAP_EXTRA + NIFTY_NEXT50),
}


def get_universe(name: str) -> list[str]:
    name = name.lower()
    if name not in UNIVERSES:
        raise ValueError(
            f"Unknown universe '{name}'. Choose from: {sorted(UNIVERSES)}"
        )
    return UNIVERSES[name]
