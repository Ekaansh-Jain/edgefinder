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



# ---------------------------------------------------------------------------
# Company-name search terms for news lookups (GDELT etc.).
# Precise names improve news-entity matching. Anything not listed falls back to
# a cleaned-up version of the ticker root, which is usually good enough.
# ---------------------------------------------------------------------------
COMPANY_NAMES: dict[str, str] = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "INFY.NS": "Infosys",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC Limited",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen Toubro",
    "AXISBANK.NS": "Axis Bank",
    "ASIANPAINT.NS": "Asian Paints",
    "MARUTI.NS": "Maruti Suzuki",
    "BAJFINANCE.NS": "Bajaj Finance",
    "HCLTECH.NS": "HCL Technologies",
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "TITAN.NS": "Titan Company",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "WIPRO.NS": "Wipro",
    "NESTLEIND.NS": "Nestle India",
    "ONGC.NS": "Oil and Natural Gas Corporation",
    "NTPC.NS": "NTPC Limited",
    "POWERGRID.NS": "Power Grid Corporation of India",
    "TATAMOTORS.NS": "Tata Motors",
    "TATASTEEL.NS": "Tata Steel",
    "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports",
    "JSWSTEEL.NS": "JSW Steel",
    "COALINDIA.NS": "Coal India",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "GRASIM.NS": "Grasim Industries",
    "HINDALCO.NS": "Hindalco Industries",
    "INDUSINDBK.NS": "IndusInd Bank",
    "DRREDDY.NS": "Dr Reddys Laboratories",
    "CIPLA.NS": "Cipla",
    "EICHERMOT.NS": "Eicher Motors",
    "BRITANNIA.NS": "Britannia Industries",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "DIVISLAB.NS": "Divis Laboratories",
    "BPCL.NS": "Bharat Petroleum",
    "TECHM.NS": "Tech Mahindra",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "TATACONSUM.NS": "Tata Consumer Products",
    "SBILIFE.NS": "SBI Life Insurance",
    "HDFCLIFE.NS": "HDFC Life Insurance",
    "LTIM.NS": "LTIMindtree",
    "SHRIRAMFIN.NS": "Shriram Finance",
    "M&M.NS": "Mahindra Mahindra",
    "ZOMATO.NS": "Zomato",
    "JIOFIN.NS": "Jio Financial Services",
    "DLF.NS": "DLF Limited",
    "VEDL.NS": "Vedanta Limited",
    "TATAPOWER.NS": "Tata Power",
    "PNB.NS": "Punjab National Bank",
    "BANKBARODA.NS": "Bank of Baroda",
    "INDIGO.NS": "InterGlobe Aviation IndiGo",
    "SIEMENS.NS": "Siemens India",
    "TRENT.NS": "Trent Limited",
    "LICI.NS": "Life Insurance Corporation of India",
    "IRCTC.NS": "Indian Railway Catering IRCTC",
    "HAL.NS": "Hindustan Aeronautics",
    "BEL.NS": "Bharat Electronics",
}


def get_company_query(ticker: str) -> str:
    """Return a news search term for a ticker.

    Uses the curated name if known, else derives a readable fallback from the
    ticker root (strip '.NS', replace separators). Adds nothing fancy; the news
    fetcher can append context like 'India' if desired.
    """
    if ticker in COMPANY_NAMES:
        return COMPANY_NAMES[ticker]
    root = ticker.replace(".NS", "").replace("-", " ").replace("&", " ")
    return root.strip()
