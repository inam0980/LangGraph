"""LangChain tools used by the workflow.

A "tool" in LangChain is just a function with a description and a typed
signature. We expose five tools as required:

    1. company_information_tool  -> basic company profile (Alpha Vantage)
    2. financial_data_tool       -> key financial metrics (Alpha Vantage)
    3. news_research_tool        -> recent news headlines (Alpha Vantage)
    4. sentiment_analysis_tool   -> classify text sentiment (Gemini)
    5. investment_decision_tool  -> combine everything into a verdict (Gemini)

Each tool returns a plain Python dict/list so it is easy to render in a
Django template and easy to feed into the next node.

Note on design: the heavy lifting lives in private ``_helper`` functions.
The ``@tool``-decorated wrappers are thin. This lets the workflow nodes call
the helpers directly (deterministic, fast) while the tools remain available
for any LLM-driven / agentic use you add later.

Data source: we use Alpha Vantage instead of yfinance. Alpha Vantage uses a
per-account API key, so it works reliably from shared cloud IPs (e.g. the free
PythonAnywhere tier) where Yahoo Finance / yfinance gets rate-limited (429).
"""
from __future__ import annotations

import os
from typing import Any

import requests
import yfinance as yf
from langchain_core.tools import tool

from .llm import ask_json


_AV_BASE = "https://www.alphavantage.co/query"


def _av_get(params: dict[str, str]) -> dict[str, Any]:
    """Call the Alpha Vantage API and return parsed JSON.

    Adds the API key, handles network errors, and detects Alpha Vantage's
    rate-limit / info responses (which come back as ``{"Note": ...}`` or
    ``{"Information": ...}`` instead of real data).
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
    if not api_key:
        return {"error": "ALPHAVANTAGE_API_KEY is not set. Get a free key at "
                         "https://www.alphavantage.co/support/#api-key"}
    params = {**params, "apikey": api_key}
    try:
        resp = requests.get(_AV_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Alpha Vantage request failed: {exc}"}

    # Alpha Vantage signals quota/errors via these keys.
    if "Note" in data or "Information" in data:
        return {"error": data.get("Note") or data.get("Information") or "Rate limited by Alpha Vantage."}
    if "Error Message" in data:
        return {"error": data["Error Message"]}
    return data


# Per-process cache so one analysis never burns the same Alpha Vantage
# request twice (the free tier allows only 25 requests/day).
_av_cache: dict[tuple[str, str], dict[str, Any]] = {}


def _av_get_cached(function: str, cache_key: str, **params: str) -> dict[str, Any]:
    """Like ``_av_get`` but caches successful responses per (function, symbol)."""
    key = (function, cache_key)
    if key in _av_cache:
        return _av_cache[key]
    data = _av_get({"function": function, **params})
    if "error" not in data:
        _av_cache[key] = data
    return data


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------
def _human_money(value: Any) -> str:
    """Format a large number like 3120000000000 as '$3.12T'."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "N/A"

    for unit, threshold in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(num) >= threshold:
            return f"${num / threshold:.2f}{unit}"
    return f"${num:.2f}"


def _pct(value: Any) -> str:
    """Format a fraction like 0.21 as '21.0%'."""
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


_info_cache: dict[str, dict[str, Any]] = {}

def _safe_info(ticker: str, retries: int = 3) -> dict[str, Any]:
    """Fetch yfinance ``.info`` with caching + retry on rate limits.

    Yahoo Finance aggressively rate-limits shared cloud IPs (e.g. the free
    PythonAnywhere tier), returning 429 "Too Many Requests". We retry a few
    times with exponential backoff before giving up.
    """
    import time

    if ticker in _info_cache:
        return _info_cache[ticker]

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            print(f"[agent] fetching yfinance info for {ticker} (attempt {attempt + 1})...", flush=True)
            data = yf.Ticker(ticker).info or {}
            if data and (data.get("longName") or data.get("shortName")):
                print(f"[agent] yfinance info fetched for {ticker}", flush=True)
                _info_cache[ticker] = data
                return data
            # Empty data can also mean a transient block; retry.
            last_exc = RuntimeError("empty response")
        except Exception as exc:  # noqa: BLE001 - includes rate-limit errors
            last_exc = exc
            print(f"[agent] yfinance error for {ticker}: {exc}", flush=True)

        if attempt < retries - 1:
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"[agent] retrying in {wait}s...", flush=True)
            time.sleep(wait)

    print(f"[agent] giving up on {ticker}: {last_exc}", flush=True)
    return {}


# ---------------------------------------------------------------------------
# Alpha Vantage fallback (kicks in when Yahoo rate-limits us, e.g. from
# PythonAnywhere's shared IPs where yfinance reliably gets 429'd)
# ---------------------------------------------------------------------------
def _av_company_information(ticker: str) -> dict[str, Any]:
    """Company profile from Alpha Vantage OVERVIEW."""
    print(f"[agent] Yahoo failed; trying Alpha Vantage for {ticker} company info...", flush=True)
    data = _av_get_cached("OVERVIEW", ticker, symbol=ticker)
    if "error" in data:
        return {"error": f"No company data found for ticker '{ticker}' "
                         f"(Yahoo rate-limited; Alpha Vantage fallback: {data['error']})"}
    if not data.get("Name"):
        return {"error": f"No company data found for ticker '{ticker}'."}
    return {
        "name": data.get("Name") or ticker,
        "industry": (data.get("Industry") or "N/A").title(),
        "sector": (data.get("Sector") or "N/A").title(),
        "market_cap": _human_money(data.get("MarketCapitalization")),
        # Alpha Vantage's OVERVIEW endpoint does not expose company officers.
        "ceo": "N/A",
        "summary": data.get("Description", ""),
    }


def _av_financial_data(ticker: str) -> dict[str, Any]:
    """Key financial metrics from Alpha Vantage OVERVIEW.

    Debt / cash-flow figures live in separate endpoints; we skip them to
    preserve the 25-requests/day free quota and report "N/A" instead.
    """
    print(f"[agent] Yahoo failed; trying Alpha Vantage for {ticker} financials...", flush=True)
    data = _av_get_cached("OVERVIEW", ticker, symbol=ticker)
    if "error" in data:
        return {"error": data["error"]}
    if not data.get("Name"):
        return {"error": f"No financial data found for ticker '{ticker}'."}
    return {
        "revenue": _human_money(data.get("RevenueTTM")),
        "revenue_growth": _pct(data.get("QuarterlyRevenueGrowthYOY")),
        "profit_margin": _pct(data.get("ProfitMargin")),
        "operating_margin": _pct(data.get("OperatingMarginTTM")),
        "pe_ratio": _round(data.get("TrailingPE") or data.get("PERatio")),
        "forward_pe": _round(data.get("ForwardPE")),
        "debt_to_equity": "N/A",
        "total_debt": "N/A",
        "free_cash_flow": "N/A",
        "operating_cash_flow": "N/A",
        "return_on_equity": _pct(data.get("ReturnOnEquityTTM")),
    }


def _av_recent_news(ticker: str, limit: int = 6) -> list[dict[str, str]]:
    """Recent headlines from Alpha Vantage NEWS_SENTIMENT."""
    print(f"[agent] Yahoo failed; trying Alpha Vantage for {ticker} news...", flush=True)
    data = _av_get_cached("NEWS_SENTIMENT", ticker, tickers=ticker, limit="20")
    items = data.get("feed") or []
    return [
        {
            "title": item.get("title", ""),
            "publisher": item.get("source") or "Unknown",
            "link": item.get("url") or "",
        }
        for item in items[:limit]
        if item.get("title")
    ]


def _av_resolve_ticker(query: str) -> str:
    """Name -> ticker via Alpha Vantage SYMBOL_SEARCH ('' if nothing found).

    Alpha Vantage ranks by *symbol* similarity, so "Apple" ranks APLE (Apple
    Hospitality REIT) above AAPL (Apple Inc). We re-rank: among US equities,
    prefer the one whose company name starts with the query, breaking ties
    with the shortest name — "Apple Inc" beats "Apple Hospitality REIT Inc".
    """
    data = _av_get_cached("SYMBOL_SEARCH", query.upper(), keywords=query)
    matches = data.get("bestMatches") or []

    us_equities = [
        m for m in matches
        if m.get("3. type") == "Equity" and m.get("4. region") == "United States" and m.get("1. symbol")
    ]
    q = query.lower()
    name_matches = [m for m in us_equities if (m.get("2. name") or "").lower().startswith(q)]
    if name_matches:
        best = min(name_matches, key=lambda m: len(m.get("2. name") or ""))
        return best["1. symbol"].upper()
    if us_equities:
        return us_equities[0]["1. symbol"].upper()
    for match in matches:
        if match.get("1. symbol"):
            return match["1. symbol"].upper()
    return ""


# ---------------------------------------------------------------------------
# Ticker resolution (lets users type a company NAME or a ticker)
# ---------------------------------------------------------------------------
def resolve_ticker(query: str) -> str:
    """Turn user input into a valid stock ticker.

    Users naturally type "Google" or "Apple" instead of "GOOGL"/"AAPL".
    This helper accepts either:

      1. If the input already looks like a valid ticker, use it as-is.
      2. Otherwise, search Yahoo Finance by name and pick the first equity.

    Args:
        query: Raw user input (a ticker symbol or a company name).

    Returns:
        A best-guess ticker symbol (uppercase). Falls back to the cleaned
        input if nothing better is found.
    """
    query = (query or "").strip()
    if not query:
        return ""

    # 1. Only try the input directly as a ticker if it *looks* like one
    # (short, single token). This avoids noisy 404s for names like "Apple".
    looks_like_ticker = len(query) <= 6 and " " not in query
    if looks_like_ticker:
        try:
            info = yf.Ticker(query.upper()).info or {}
            if info.get("longName") or info.get("shortName"):
                return query.upper()
        except Exception:  # noqa: BLE001 - fall through to search
            pass

    # 2. Search by company name and prefer a real stock (EQUITY).
    try:
        quotes = yf.Search(query).quotes or []
        for quote in quotes:
            if quote.get("quoteType") == "EQUITY" and quote.get("symbol"):
                return quote["symbol"].upper()
        if quotes and quotes[0].get("symbol"):
            return quotes[0]["symbol"].upper()
    except Exception:  # noqa: BLE001 - search may be unavailable
        pass

    # 3. Yahoo search failed (likely rate-limited) — try Alpha Vantage.
    symbol = _av_resolve_ticker(query)
    if symbol:
        return symbol

    # 4. Nothing worked; return the cleaned input so the caller can error out.
    return query.upper()


# ---------------------------------------------------------------------------
# Helper implementations (called by nodes directly)
# ---------------------------------------------------------------------------
def get_company_information(ticker: str) -> dict[str, Any]:
    """Return a company profile: name, industry, sector, market cap, CEO."""
    try:
        info = _safe_info(ticker)
        if not info.get("longName") and not info.get("shortName"):
            # Yahoo gave nothing (bad symbol OR rate-limited) — try Alpha Vantage.
            return _av_company_information(ticker)

        # Find the CEO from the list of officers, if present.
        ceo = "N/A"
        for officer in info.get("companyOfficers", []) or []:
            title = (officer.get("title") or "").lower()
            if "ceo" in title or "chief executive" in title:
                ceo = officer.get("name", "N/A")
                break

        return {
            "name": info.get("longName") or info.get("shortName") or ticker,
            "industry": info.get("industry", "N/A"),
            "sector": info.get("sector", "N/A"),
            "market_cap": _human_money(info.get("marketCap")),
            "ceo": ceo,
            "summary": info.get("longBusinessSummary", ""),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Failed to fetch company info: {exc}"}


def get_financial_data(ticker: str) -> dict[str, Any]:
    """Return key financial metrics used for the financial health score."""
    try:
        info = _safe_info(ticker)
        if not info:
            # Yahoo gave nothing (bad symbol OR rate-limited) — try Alpha Vantage.
            return _av_financial_data(ticker)

        return {
            "revenue": _human_money(info.get("totalRevenue")),
            "revenue_growth": _pct(info.get("revenueGrowth")),
            "profit_margin": _pct(info.get("profitMargins")),
            "operating_margin": _pct(info.get("operatingMargins")),
            "pe_ratio": _round(info.get("trailingPE")),
            "forward_pe": _round(info.get("forwardPE")),
            "debt_to_equity": _round(info.get("debtToEquity")),
            "total_debt": _human_money(info.get("totalDebt")),
            "free_cash_flow": _human_money(info.get("freeCashflow")),
            "operating_cash_flow": _human_money(info.get("operatingCashflow")),
            "return_on_equity": _pct(info.get("returnOnEquity")),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Failed to fetch financial data: {exc}"}


def get_recent_news(ticker: str, limit: int = 6) -> list[dict[str, str]]:
    """Return a list of recent news headlines for the ticker.

    yfinance has changed its news shape across versions, so we defensively
    support both the old flat format and the newer nested ``content`` format.
    """
    try:
        raw_items = yf.Ticker(ticker).news or []
    except Exception:  # noqa: BLE001
        raw_items = []
    if not raw_items:
        # Yahoo gave nothing (rate-limited?) — try Alpha Vantage instead.
        return _av_recent_news(ticker, limit)

    headlines: list[dict[str, str]] = []
    for item in raw_items[:limit]:
        content = item.get("content", item)  # new format nests under "content"
        title = content.get("title") or item.get("title") or ""
        publisher = (
            content.get("provider", {}).get("displayName")
            if isinstance(content.get("provider"), dict)
            else item.get("publisher", "")
        )
        link = (
            content.get("canonicalUrl", {}).get("url")
            if isinstance(content.get("canonicalUrl"), dict)
            else item.get("link", "")
        )
        if title:
            headlines.append(
                {"title": title, "publisher": publisher or "Unknown", "link": link or ""}
            )
    return headlines


def analyze_sentiment(text: str) -> dict[str, Any]:
    """Use Gemini to classify sentiment of the given text.

    Returns a dict like:
        {"sentiment": "positive"|"neutral"|"negative", "sentiment_score": 1-10}
    """
    if not text.strip():
        return {"sentiment": "neutral", "sentiment_score": 5}

    system = (
        "You are a financial news sentiment classifier. Read the headlines and "
        "decide the overall market sentiment for the company. "
        "Reply with JSON ONLY in this exact shape: "
        '{"sentiment": "positive|neutral|negative", "sentiment_score": <integer 1-10>}. '
        "1 means very negative, 10 means very positive, 5 is neutral."
    )
    result = ask_json(system, text)
    # Provide safe defaults if the model misbehaved.
    if "error" in result:
        return {"sentiment": "neutral", "sentiment_score": 5, "note": result["error"]}
    result.setdefault("sentiment", "neutral")
    result.setdefault("sentiment_score", 5)
    return result


def make_investment_decision(context: dict[str, Any]) -> dict[str, Any]:
    """Combine all collected data into a final recommendation using Gemini."""
    system = (
        "You are a senior investment analyst. Using the provided company, "
        "financial, news, and risk data, produce a balanced investment verdict. "
        "Reply with JSON ONLY in this exact shape:\n"
        "{\n"
        '  "recommendation": "INVEST" | "PASS",\n'
        '  "overall_score": <integer 1-10>,\n'
        '  "strengths": [<string>, ...],\n'
        '  "risks": [<string>, ...],\n'
        '  "reasoning": <string, 3-5 sentences>\n'
        "}\n"
        "Be objective. INVEST only when the balance of evidence is favorable."
    )
    import json

    result = ask_json(system, json.dumps(context, indent=2))
    if "error" in result:
        # Degrade gracefully with a conservative default.
        return {
            "recommendation": "PASS",
            "overall_score": 5,
            "strengths": [],
            "risks": ["Analysis incomplete due to a model error."],
            "reasoning": result["error"],
        }
    # Normalize / validate.
    result.setdefault("recommendation", "PASS")
    result["recommendation"] = str(result["recommendation"]).upper()
    result.setdefault("overall_score", 5)
    result.setdefault("strengths", [])
    result.setdefault("risks", [])
    result.setdefault("reasoning", "")
    return result


def _round(value: Any, digits: int = 2) -> Any:
    """Round numbers, leaving non-numbers as 'N/A'."""
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return "N/A"


# ---------------------------------------------------------------------------
# LangChain @tool wrappers (the public "tools" interface)
# ---------------------------------------------------------------------------
@tool
def company_information_tool(ticker: str) -> dict[str, Any]:
    """Get a company's profile (name, industry, sector, market cap, CEO)
    given its stock ticker symbol (e.g. 'AAPL')."""
    return get_company_information(ticker)


@tool
def financial_data_tool(ticker: str) -> dict[str, Any]:
    """Get key financial metrics (revenue, margins, P/E, debt, cash flow)
    for a company given its stock ticker symbol."""
    return get_financial_data(ticker)


@tool
def news_research_tool(ticker: str) -> list[dict[str, str]]:
    """Get recent news headlines for a company given its stock ticker symbol."""
    return get_recent_news(ticker)


@tool
def sentiment_analysis_tool(text: str) -> dict[str, Any]:
    """Classify the financial sentiment of a block of text (e.g. news
    headlines) as positive, neutral, or negative with a 1-10 score."""
    return analyze_sentiment(text)


@tool
def investment_decision_tool(context: dict[str, Any]) -> dict[str, Any]:
    """Produce a final INVEST/PASS recommendation from a dict containing
    company, financial, news, and risk analysis."""
    return make_investment_decision(context)
