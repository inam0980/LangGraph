"""The workflow nodes.

Each function is one step in the LangGraph workflow. The contract is always
the same:

    def node(state: AgentState) -> AgentState:
        ...compute something...
        return {"some_key": value}   # a *partial* state update

LangGraph merges the returned dict back into the running state, so the next
node can read what previous nodes produced.

Order of execution:
    company_research -> financial_analysis -> news_analysis
                     -> risk_analysis -> investment_decision
"""
from __future__ import annotations

import json
from typing import Any

from .llm import ask_json
from .state import AgentState
from .tools import (
    analyze_sentiment,
    get_company_information,
    get_financial_data,
    get_recent_news,
    make_investment_decision,
)


def _append_error(state: AgentState, message: str) -> list[str]:
    """Return the existing error list with a new message appended."""
    errors = list(state.get("errors", []))
    errors.append(message)
    return errors


# ---------------------------------------------------------------------------
# 1. Company Research Node
# ---------------------------------------------------------------------------
def company_research_node(state: AgentState) -> AgentState:
    """Collect the basic company profile via the company information tool."""
    ticker = state["ticker"]
    company = get_company_information(ticker)

    if "error" in company:
        return {"company": company, "errors": _append_error(state, company["error"])}

    return {"company": company}


# ---------------------------------------------------------------------------
# 2. Financial Analysis Node
# ---------------------------------------------------------------------------
def financial_analysis_node(state: AgentState) -> AgentState:
    """Pull financial metrics, then ask Gemini for a 1-10 health score."""
    ticker = state["ticker"]
    metrics = get_financial_data(ticker)

    if "error" in metrics:
        return {
            "financials": {"metrics": metrics, "analysis": "", "financial_score": 5},
            "errors": _append_error(state, metrics["error"]),
        }

    system = (
        "You are a financial analyst. Given these metrics, assess revenue "
        "growth, profitability, debt levels, valuation (P/E), and cash flow. "
        "Reply with JSON ONLY in this shape: "
        '{"analysis": <2-4 sentence summary>, "financial_score": <integer 1-10>}. '
        "10 = excellent financial health, 1 = very poor."
    )
    company_name = state.get("company", {}).get("name", ticker)
    user = f"Company: {company_name}\nMetrics:\n{json.dumps(metrics, indent=2)}"
    llm_result = ask_json(system, user)

    financials: dict[str, Any] = {
        "metrics": metrics,
        "analysis": llm_result.get("analysis", "Analysis unavailable."),
        "financial_score": _clamp_score(llm_result.get("financial_score", 5)),
    }
    return {"financials": financials}


# ---------------------------------------------------------------------------
# 3. News Analysis Node
# ---------------------------------------------------------------------------
def news_analysis_node(state: AgentState) -> AgentState:
    """Fetch recent news, summarize it, and score the sentiment."""
    ticker = state["ticker"]
    company_name = state.get("company", {}).get("name", ticker)
    headlines = get_recent_news(ticker)

    if not headlines:
        return {
            "news": {
                "headlines": [],
                "summary": "No recent news found.",
                "sentiment": "neutral",
                "sentiment_score": 5,
            }
        }

    headline_text = "\n".join(f"- {h['title']} ({h['publisher']})" for h in headlines)

    # Summarize key events with the LLM.
    summary_system = (
        "You summarize financial news. Given the headlines, write a concise "
        "summary of the key events. Reply with JSON ONLY: "
        '{"summary": <3-5 sentences>}.'
    )
    summary_result = ask_json(summary_system, f"{company_name} headlines:\n{headline_text}")
    summary = summary_result.get("summary", "Summary unavailable.")

    # Score sentiment using the dedicated sentiment tool.
    sentiment = analyze_sentiment(headline_text)

    news = {
        "headlines": headlines,
        "summary": summary,
        "sentiment": sentiment.get("sentiment", "neutral"),
        "sentiment_score": _clamp_score(sentiment.get("sentiment_score", 5)),
    }
    return {"news": news}


# ---------------------------------------------------------------------------
# 4. Risk Analysis Node
# ---------------------------------------------------------------------------
def risk_analysis_node(state: AgentState) -> AgentState:
    """Identify regulatory, competition, financial, and market risks."""
    company = state.get("company", {})
    financials = state.get("financials", {})
    news = state.get("news", {})

    system = (
        "You are a risk analyst. Identify the company's key risks across four "
        "categories. Reply with JSON ONLY in this exact shape:\n"
        "{\n"
        '  "regulatory": <1-2 sentences>,\n'
        '  "competition": <1-2 sentences>,\n'
        '  "financial": <1-2 sentences>,\n'
        '  "market": <1-2 sentences>,\n'
        '  "risk_score": <integer 1-10>\n'
        "}\n"
        "risk_score: 1 = very low risk, 10 = very high risk."
    )
    context = {
        "company": company,
        "financials": financials.get("metrics", {}),
        "news_summary": news.get("summary", ""),
        "news_sentiment": news.get("sentiment", "neutral"),
    }
    result = ask_json(system, json.dumps(context, indent=2))

    risk = {
        "regulatory": result.get("regulatory", "N/A"),
        "competition": result.get("competition", "N/A"),
        "financial": result.get("financial", "N/A"),
        "market": result.get("market", "N/A"),
        "risk_score": _clamp_score(result.get("risk_score", 5)),
    }
    return {"risk": risk}


# ---------------------------------------------------------------------------
# 5. Investment Decision Node
# ---------------------------------------------------------------------------
def investment_decision_node(state: AgentState) -> AgentState:
    """Combine every previous output into the final recommendation."""
    context = {
        "company": state.get("company", {}),
        "financials": {
            "metrics": state.get("financials", {}).get("metrics", {}),
            "analysis": state.get("financials", {}).get("analysis", ""),
            "financial_score": state.get("financials", {}).get("financial_score", 5),
        },
        "news": {
            "summary": state.get("news", {}).get("summary", ""),
            "sentiment": state.get("news", {}).get("sentiment", "neutral"),
            "sentiment_score": state.get("news", {}).get("sentiment_score", 5),
        },
        "risk": state.get("risk", {}),
    }

    decision = make_investment_decision(context)
    decision["overall_score"] = _clamp_score(decision.get("overall_score", 5))
    return {"decision": decision}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _clamp_score(value: Any) -> int:
    """Force a score into the integer range 1..10."""
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return 5
    return max(1, min(10, score))
