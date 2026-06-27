
from __future__ import annotations

from django.contrib import messages
from django.shortcuts import redirect, render

from .agent.workflow import run_analysis
from .models import AnalysisRecord


def home(request):
    """Render the home page with the company/ticker input form."""
    return render(request, "research/home.html")


def analyze(request):
    """Run the workflow for the submitted ticker and show the results.

    Only POST is allowed (the form posts here). On GET we just redirect home.
    """
    if request.method != "POST":
        return redirect("home")

    ticker = (request.POST.get("ticker") or "").strip().upper()
    if not ticker:
        messages.error(request, "Please enter a ticker symbol (e.g. AAPL).")
        return redirect("home")

    # Run the LangGraph workflow. This may take several seconds because it
    # makes multiple Gemini calls.
    result = run_analysis(ticker)

    # If the company lookup failed entirely, send the user back with a message.
    company = result.get("company", {})
    if "error" in company and not company.get("name"):
        messages.error(request, company["error"])
        return redirect("home")

    # Persist the analysis (optional but nice to have a history).
    decision = result.get("decision", {})
    try:
        AnalysisRecord.objects.create(
            ticker=ticker,
            company_name=company.get("name", ticker),
            recommendation=decision.get("recommendation", ""),
            overall_score=decision.get("overall_score", 0),
            result=result,
        )
    except Exception:  # noqa: BLE001 - saving is best-effort, never block the page
        pass

    context = {
        "ticker": ticker,
        "company": company,
        "financials": result.get("financials", {}),
        "news": result.get("news", {}),
        "risk": result.get("risk", {}),
        "decision": decision,
        "errors": result.get("errors", []),
    }
    return render(request, "research/results.html", context)
