"""AI agent package.

This package contains everything related to the LangGraph workflow:

    llm.py       -> Google Gemini integration (the "brain")
    state.py     -> the shared data structure passed between nodes
    tools.py     -> LangChain tools (company, financials, news, sentiment, decision)
    nodes.py     -> the workflow steps (research -> financial -> news -> risk -> decision)
    workflow.py  -> wires the nodes into a LangGraph graph and runs it
"""
