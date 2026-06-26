"""Django settings for the LLM Orchestrator & RAG server."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-insecure-change-me")
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "rest_framework.authtoken",
    "core",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("POSTGRES_DB", "rag"),
        "USER": os.environ.get("POSTGRES_USER", "rag"),
        "PASSWORD": os.environ.get("POSTGRES_PASSWORD", "rag"),
        "HOST": os.environ.get("POSTGRES_HOST", "localhost"),
        "PORT": os.environ.get("POSTGRES_PORT", "5432"),
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.TokenAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.ScopedRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "query": os.environ.get("THROTTLE_QUERY", "30/minute"),
        "ingest": os.environ.get("THROTTLE_INGEST", "10/minute"),
    },
}

# --- RAG / LLM provider configuration -------------------------------------
RAG = {
    "EMBEDDING_MODEL": os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
    "EMBEDDING_DIM": int(os.environ.get("EMBEDDING_DIM", "1536")),
    "CHAT_MODEL": os.environ.get("CHAT_MODEL", "gpt-4o-mini"),
    "CHUNK_TOKENS": int(os.environ.get("CHUNK_TOKENS", "500")),
    "CHUNK_OVERLAP": int(os.environ.get("CHUNK_OVERLAP", "50")),
    "TOP_K": int(os.environ.get("TOP_K", "5")),
    # "openai" (live) or "fake" (deterministic, for tests/offline dev)
    "PROVIDER": os.environ.get("RAG_PROVIDER", "openai"),
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
}
