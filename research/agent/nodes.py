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
    get_company_information,
    get_financial_data,
    get_recent_news,
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
    print(f"[node 1/5] company research: {ticker}", flush=True)
    company = get_company_information(ticker)

    if "error" in company:
        return {"company": company, "errors": _append_error(state, company["error"])}

    return {"company": company}


# ---------------------------------------------------------------------------
# 2. Financial Analysis Node
# ---------------------------------------------------------------------------
def financial_analysis_node(state: AgentState) -> AgentState:
    """Pull financial metrics — analysis deferred to the decision node."""
    ticker = state["ticker"]
    print(f"[node 2/5] financial analysis: {ticker}", flush=True)
    metrics = get_financial_data(ticker)

    if "error" in metrics:
        return {
            "financials": {"metrics": metrics, "analysis": "", "financial_score": 5},
            "errors": _append_error(state, metrics["error"]),
        }

    return {"financials": {"metrics": metrics, "analysis": "", "financial_score": 5}}


# ---------------------------------------------------------------------------
# 3. News Analysis Node
# ---------------------------------------------------------------------------
def news_analysis_node(state: AgentState) -> AgentState:
    """Fetch recent news headlines — analysis deferred to the decision node."""
    print("[node 3/5] news analysis", flush=True)
    ticker = state["ticker"]
    print("[node 3/5] fetching news...", flush=True)
    headlines = get_recent_news(ticker)
    print(f"[node 3/5] got {len(headlines)} headlines", flush=True)

    return {
        "news": {
            "headlines": headlines,
            "summary": "",
            "sentiment": "neutral",
            "sentiment_score": 5,
        }
    }


# ---------------------------------------------------------------------------
# 4. Risk Analysis Node
# ---------------------------------------------------------------------------
def risk_analysis_node(state: AgentState) -> AgentState:
    """Risk analysis deferred to the decision node (single Gemini call)."""
    print("[node 4/5] risk analysis (pass-through)", flush=True)
    return {"risk": {"regulatory": "", "competition": "", "financial": "", "market": "", "risk_score": 5}}


# ---------------------------------------------------------------------------
# 5. Investment Decision Node
# ---------------------------------------------------------------------------
def investment_decision_node(state: AgentState) -> AgentState:
    """Single Gemini call that produces all scores, analysis, and the verdict."""
    print("[node 5/5] investment decision - single Gemini call", flush=True)

    company = state.get("company", {})
    metrics = state.get("financials", {}).get("metrics", {})
    headlines = [h["title"] for h in state.get("news", {}).get("headlines", [])]

    system = (
        "You are a senior investment analyst. Given the company profile, financial metrics, "
        "and recent news headlines, produce a complete analysis in ONE response.\n"
        "Reply with JSON ONLY in this exact shape:\n"
        "{\n"
        '  "financial_analysis": <2-3 sentence financial health summary>,\n'
        '  "financial_score": <integer 1-10>,\n'
        '  "news_summary": <3-5 sentence summary of recent news>,\n'
        '  "sentiment": <"positive"|"neutral"|"negative">,\n'
        '  "sentiment_score": <integer 1-10>,\n'
        '  "regulatory_risk": <1-2 sentences>,\n'
        '  "competition_risk": <1-2 sentences>,\n'
        '  "financial_risk": <1-2 sentences>,\n'
        '  "market_risk": <1-2 sentences>,\n'
        '  "risk_score": <integer 1-10, 1=very low 10=very high>,\n'
        '  "recommendation": <"INVEST"|"PASS">,\n'
        '  "overall_score": <integer 1-10>,\n'
        '  "strengths": [<string>, ...],\n'
        '  "risks": [<string>, ...],\n'
        '  "reasoning": <3-5 sentence investment reasoning>\n'
        "}"
    )

    context = {
        "company": company,
        "financial_metrics": metrics,
        "news_headlines": headlines,
    }

    result = ask_json(system, json.dumps(context, indent=2))
    print(f"[node 5/5] Gemini keys: {list(result.keys())}", flush=True)

    if "error" in result:
        return {
            "financials": {**state.get("financials", {}), "analysis": "Analysis failed.", "financial_score": 5},
            "news": {**state.get("news", {}), "summary": "Unavailable.", "sentiment": "neutral", "sentiment_score": 5},
            "risk": {"regulatory": "N/A", "competition": "N/A", "financial": "N/A", "market": "N/A", "risk_score": 5},
            "decision": {"recommendation": "PASS", "overall_score": 5, "strengths": [], "risks": ["Analysis error."], "reasoning": result["error"]},
        }

    return {
        "financials": {
            **state.get("financials", {}),
            "analysis": result.get("financial_analysis", ""),
            "financial_score": _clamp_score(result.get("financial_score", 5)),
        },
        "news": {
            **state.get("news", {}),
            "summary": result.get("news_summary", ""),
            "sentiment": result.get("sentiment", "neutral"),
            "sentiment_score": _clamp_score(result.get("sentiment_score", 5)),
        },
        "risk": {
            "regulatory": result.get("regulatory_risk") or result.get("regulatory", "N/A"),
            "competition": result.get("competition_risk") or result.get("competition", "N/A"),
            "financial": result.get("financial_risk") or result.get("financial", "N/A"),
            "market": result.get("market_risk") or result.get("market", "N/A"),
            "risk_score": _clamp_score(result.get("risk_score", 5)),
        },
        "decision": {
            "recommendation": str(result.get("recommendation", "PASS")).upper(),
            "overall_score": _clamp_score(result.get("overall_score", 5)),
            "strengths": result.get("strengths", []),
            "risks": result.get("risks", []),
            "reasoning": result.get("reasoning", ""),
        },
    }


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
