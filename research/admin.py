"""Register models with the Django admin so you can browse saved analyses."""
from django.contrib import admin

from .models import AnalysisRecord


@admin.register(AnalysisRecord)
class AnalysisRecordAdmin(admin.ModelAdmin):
    list_display = ("ticker", "company_name", "recommendation", "overall_score", "created_at")
    list_filter = ("recommendation",)
    search_fields = ("ticker", "company_name")
