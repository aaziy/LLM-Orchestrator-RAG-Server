from django.urls import path
from rest_framework.authtoken.views import obtain_auth_token

from core import views

urlpatterns = [
    path("auth/register", views.RegisterView.as_view(), name="register"),
    path("auth/token", obtain_auth_token, name="token"),
    path("documents", views.DocumentListCreateView.as_view(), name="documents"),
    path("documents/<int:pk>", views.DocumentDetailView.as_view(), name="document-detail"),
    path("query", views.QueryView.as_view(), name="query"),
    path("query/stream", views.QueryStreamView.as_view(), name="query-stream"),
    path("usage", views.UsageView.as_view(), name="usage"),
]
