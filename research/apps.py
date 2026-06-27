"""App configuration for the ``research`` app."""
from django.apps import AppConfig


class ResearchConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "research"
    verbose_name = "Investment Research"
