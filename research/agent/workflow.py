"""The LangGraph workflow.

This file builds the graph:

    START
      -> company_research
      -> financial_analysis
      -> news_analysis
      -> risk_analysis
      -> investment_decision
      -> END

and exposes a single ``run_analysis(ticker)`` helper that Django views call.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .nodes import (
    company_research_node,
    financial_analysis_node,
    investment_decision_node,
    news_analysis_node,
    risk_analysis_node,
)
from .state import AgentState
from .tools import resolve_ticker


def build_workflow():
    """Construct and compile the LangGraph workflow.

    Returns:
        A compiled graph object with an ``.invoke(state)`` method.
    """
    # The StateGraph knows the shape of the state it carries.
    graph = StateGraph(AgentState)

    # Register each node under a name.
    graph.add_node("company_research", company_research_node)
    graph.add_node("financial_analysis", financial_analysis_node)
    graph.add_node("news_analysis", news_analysis_node)
    graph.add_node("risk_analysis", risk_analysis_node)
    graph.add_node("investment_decision", investment_decision_node)

    # Wire the nodes in a straight line.
    graph.add_edge(START, "company_research")
    graph.add_edge("company_research", "financial_analysis")
    graph.add_edge("financial_analysis", "news_analysis")
    graph.add_edge("news_analysis", "risk_analysis")
    graph.add_edge("risk_analysis", "investment_decision")
    graph.add_edge("investment_decision", END)

    return graph.compile()


# Compile once at import time and reuse (compiling is cheap but pointless to
# repeat on every request).
_COMPILED_WORKFLOW = build_workflow()


def run_analysis(ticker: str) -> dict[str, Any]:
    """Run the full investment-research workflow for one ticker.

    Args:
        ticker: A stock ticker symbol, e.g. "AAPL". Case-insensitive.

    Returns:
        The final state dict containing company, financials, news, risk,
        decision, and any errors collected along the way.
    """
    raw = (ticker or "").strip()
    if not raw:
        return {"errors": ["No ticker provided."]}

    # Accept either a ticker (AAPL) or a company name (Apple) and resolve it
    # to a real symbol before running the workflow.
    ticker = resolve_ticker(raw)

    initial_state: AgentState = {"ticker": ticker, "errors": []}

    try:
        result = _COMPILED_WORKFLOW.invoke(initial_state)
        return dict(result)
    except Exception as exc:  # noqa: BLE001 - surface any unexpected failure to the UI
        return {"ticker": ticker, "errors": [f"Workflow failed: {exc}"]}
