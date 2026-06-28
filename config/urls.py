from django.contrib import admin
from django.urls import include, path
from django.views.generic import TemplateView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("core.urls")),
    # Single-page test console (same-origin with the API; no CORS needed).
    path("", TemplateView.as_view(template_name="console.html"), name="console"),
]
