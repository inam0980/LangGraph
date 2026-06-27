"""LangChain tools used by the workflow.

A "tool" in LangChain is just a function with a description and a typed
signature. We expose five tools as required:

    1. company_information_tool  -> basic company profile (yfinance)
    2. financial_data_tool       -> key financial metrics (yfinance)
    3. news_research_tool        -> recent news headlines (yfinance)
    4. sentiment_analysis_tool   -> classify text sentiment (Gemini)
    5. investment_decision_tool  -> combine everything into a verdict (Gemini)

Each tool returns a plain Python dict/list so it is easy to render in a
Django template and easy to feed into the next node.

Note on design: the heavy lifting lives in private ``_helper`` functions.
The ``@tool``-decorated wrappers are thin. This lets the workflow nodes call
the helpers directly (deterministic, fast) while the tools remain available
for any LLM-driven / agentic use you add later.
"""
from __future__ import annotations

from typing import Any

import yfinance as yf
from langchain_core.tools import tool

from .llm import ask_json


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


def _safe_info(ticker: str) -> dict[str, Any]:
    """Fetch yfinance ``.info`` with error handling.

    yfinance can raise or return an almost-empty dict for bad tickers, so we
    normalize that into something predictable.
    """
    data = yf.Ticker(ticker).info or {}
    # yfinance returns a near-empty dict for invalid tickers.
    if not data or data.get("regularMarketPrice") is None and not data.get("longName"):
        # Still return whatever we have; caller decides if it's usable.
        return data
    return data


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

    # 3. Nothing worked; return the cleaned input so the caller can error out.
    return query.upper()


# ---------------------------------------------------------------------------
# Helper implementations (called by nodes directly)
# ---------------------------------------------------------------------------
def get_company_information(ticker: str) -> dict[str, Any]:
    """Return a company profile: name, industry, sector, market cap, CEO."""
    try:
        info = _safe_info(ticker)
        if not info.get("longName") and not info.get("shortName"):
            return {"error": f"No company data found for ticker '{ticker}'."}

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
            return {"error": f"No financial data found for ticker '{ticker}'."}

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
        return []

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
