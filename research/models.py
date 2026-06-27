"""Database models.

For this project a database is optional: the analysis is computed live and
rendered immediately. We still keep one small model so every completed
analysis is persisted. This gives you a history page later and is a good
example of storing semi-structured AI output as JSON.
"""
from django.db import models


class AnalysisRecord(models.Model):
    """A single saved investment-analysis run for one ticker."""

    # The stock ticker the user analyzed, e.g. "AAPL".
    ticker = models.CharField(max_length=20)

    # Human-readable company name (filled from the research node).
    company_name = models.CharField(max_length=255, blank=True)

    # Final recommendation: "INVEST" or "PASS".
    recommendation = models.CharField(max_length=10, blank=True)

    # The headline 1-10 score.
    overall_score = models.FloatField(default=0)

    # The full structured result from the LangGraph workflow, stored as JSON
    # so we don't need a column per field.
    result = models.JSONField(default=dict)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.ticker} - {self.recommendation} ({self.overall_score}/10)"
