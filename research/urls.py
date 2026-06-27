"""URL routes for the research app."""
from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("analyze/", views.analyze, name="analyze"),
]
