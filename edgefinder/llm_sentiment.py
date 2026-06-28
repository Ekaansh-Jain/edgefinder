"""Optional free-LLM news-sentiment overlay.

This is the *cutting-edge* layer. The research (Lopez-Lira & Tang, 2024-25)
finds LLM news-sentiment scores predict next-period returns, with the effect
strongest in smaller, less-covered stocks and after negative news — which is
exactly the midcap tilt this project supports.

IMPORTANT GUARDRAILS (so the result isn't fake):
  * POINT-IN-TIME: only score news published on/before the rebalance date.
  * LOOK-AHEAD: ideally backtest the LLM layer ONLY on news dated AFTER the
    model's training cutoff, or the LLM may "remember" the future. See README.

Providers (all have a free path). None is required for the baseline backtest:
  * Groq      — free tier, very fast.  env GROQ_API_KEY
  * Gemini    — free tier.             env GEMINI_API_KEY
  * Ollama    — fully local/offline.   env OLLAMA_HOST (default localhost:11434)

This module is intentionally provider-agnostic and dependency-light. If no
provider is configured it returns neutral (zero) scores so the pipeline still
runs end-to-end.
"""

from __future__ import annotations

import json
import os

import pandas as pd

_SYSTEM_PROMPT = (
    "You are a financial news analyst. Given a headline/snippet about an Indian "
    "listed company, rate its likely short-term (1-4 week) impact on the stock "
    "price. Respond ONLY with a JSON object: {\"score\": <float -1..1>, "
    "\"confidence\": <float 0..1>}. Positive = bullish, negative = bearish. "
    "Be conservative; routine news should be near 0."
)


def _provider() -> str | None:
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("OLLAMA_HOST") or os.getenv("USE_OLLAMA"):
        return "ollama"
    return None


def _parse_score(text: str) -> float:
    """Extract the signed sentiment*confidence score from an LLM response."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        obj = json.loads(text[start:end])
        score = float(obj.get("score", 0.0))
        conf = float(obj.get("confidence", 0.5))
        return max(-1.0, min(1.0, score)) * max(0.0, min(1.0, conf))
    except Exception:
        return 0.0


def _score_groq(text: str, model: str = "llama-3.1-8b-instant") -> float:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return _parse_score(resp.choices[0].message.content)


def _score_gemini(text: str, model: str = "gemini-1.5-flash") -> float:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    m = genai.GenerativeModel(model, system_instruction=_SYSTEM_PROMPT)
    resp = m.generate_content(text)
    return _parse_score(resp.text)


def _score_ollama(text: str, model: str = "llama3.1") -> float:
    import requests

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    resp = requests.post(
        f"{host}/api/generate",
        json={
            "model": os.getenv("OLLAMA_MODEL", model),
            "prompt": f"{_SYSTEM_PROMPT}\n\nNEWS: {text}\n\nJSON:",
            "stream": False,
            "options": {"temperature": 0.0},
        },
        timeout=60,
    )
    return _parse_score(resp.json().get("response", ""))


def score_headline(text: str) -> float:
    """Return a signed score in [-1, 1] for one news item, 0 if no provider."""
    provider = _provider()
    if not provider or not text:
        return 0.0
    try:
        if provider == "groq":
            return _score_groq(text)
        if provider == "gemini":
            return _score_gemini(text)
        if provider == "ollama":
            return _score_ollama(text)
    except Exception as exc:
        print(f"  ! LLM scoring failed ({provider}): {exc}")
    return 0.0


def build_sentiment_overlay(
    news: pd.DataFrame, rebalance_dates: list, tickers: list[str]
) -> pd.DataFrame:
    """Aggregate per-headline LLM scores into a (date x ticker) overlay matrix.

    Parameters
    ----------
    news : DataFrame with columns ['date', 'ticker', 'text']. 'date' is the
        PUBLICATION date (must be <= the rebalance date it influences).
    rebalance_dates : list of rebalance Timestamps from the backtest.
    tickers : universe tickers (yfinance '.NS' format).

    For each rebalance date T, we average the LLM scores of all news for a
    ticker published in the lookback window (T-30d, T]. Strictly point-in-time.
    """
    if news is None or news.empty or _provider() is None:
        # neutral overlay -> backtest behaves like the baseline
        return pd.DataFrame(0.0, index=pd.DatetimeIndex(rebalance_dates), columns=tickers)

    news = news.copy()
    news["date"] = pd.to_datetime(news["date"])
    # Cache LLM calls per unique text to save quota.
    unique_texts = news["text"].dropna().unique().tolist()
    score_cache = {t: score_headline(t) for t in unique_texts}
    news["score"] = news["text"].map(score_cache).fillna(0.0)

    rows = []
    for T in rebalance_dates:
        T = pd.Timestamp(T)
        window = news[(news["date"] <= T) & (news["date"] > T - pd.Timedelta(days=30))]
        agg = window.groupby("ticker")["score"].mean()
        rows.append(agg.reindex(tickers))
    overlay = pd.DataFrame(rows, index=pd.DatetimeIndex(rebalance_dates))
    return overlay.fillna(0.0)
