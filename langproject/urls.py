
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    # All app routes live in research/urls.py
    path('', include('research.urls')),
]
