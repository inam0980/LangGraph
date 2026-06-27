"""The shared state for the LangGraph workflow.

In LangGraph, every node receives the current "state" and returns a partial
update that gets merged back in. We model that state as a TypedDict so the
keys and types are documented and editor-friendly.

The flow fills these fields in order:

    ticker     -> provided by the user (input)
    company    -> filled by the Company Research node
    financials -> filled by the Financial Analysis node
    news       -> filled by the News Analysis node
    risk       -> filled by the Risk Analysis node
    decision   -> filled by the Investment Decision node (final output)
    errors     -> any node may append a human-readable error here
"""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Data passed between workflow nodes.

    ``total=False`` means every key is optional, because each node only adds
    its own slice of the state.
    """

    # --- Input ---
    ticker: str

    # --- Company Research node output ---
    # { name, industry, sector, market_cap, ceo, summary }
    company: dict[str, Any]

    # --- Financial Analysis node output ---
    # { metrics: {...}, analysis: str, financial_score: int }
    financials: dict[str, Any]

    # --- News Analysis node output ---
    # { headlines: [...], summary: str, sentiment: str, sentiment_score: int }
    news: dict[str, Any]

    # --- Risk Analysis node output ---
    # { regulatory, competition, financial, market, risk_score }
    risk: dict[str, Any]

    # --- Investment Decision node output (final) ---
    # { recommendation, overall_score, strengths, risks, reasoning }
    decision: dict[str, Any]

    # --- Diagnostics ---
    errors: list[str]
