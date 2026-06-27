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
    "django.contrib.postgres",
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
MEDIA_URL = "media/"
MEDIA_ROOT = os.environ.get("MEDIA_ROOT", str(BASE_DIR / "media"))
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- Celery -----------------------------------------------------------------
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", REDIS_URL)
CELERY_TASK_ALWAYS_EAGER = os.environ.get("CELERY_TASK_ALWAYS_EAGER", "0") == "1"
CELERY_TASK_EAGER_PROPAGATES = True

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
    # Hybrid retrieval: how many candidates each retriever contributes before
    # reciprocal-rank fusion, and the RRF damping constant.
    "HYBRID": os.environ.get("HYBRID", "1") == "1",
    "CANDIDATE_POOL": int(os.environ.get("CANDIDATE_POOL", "20")),
    "RRF_K": int(os.environ.get("RRF_K", "60")),
    # Reranking: pull a candidate pool, rerank with a cross-encoder, keep top-k.
    "RERANK": os.environ.get("RERANK", "1") == "1",
    # "local" (sentence-transformers cross-encoder), "cohere", or "fake" (tests)
    "RERANKER": os.environ.get("RERANKER", "local"),
    "RERANK_CANDIDATES": int(os.environ.get("RERANK_CANDIDATES", "20")),
    "RERANK_MODEL": os.environ.get(
        "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ),
    "COHERE_API_KEY": os.environ.get("COHERE_API_KEY", ""),
    "COHERE_RERANK_MODEL": os.environ.get("COHERE_RERANK_MODEL", "rerank-english-v3.0"),
    # "openai" (live) or "fake" (deterministic, for tests/offline dev)
    "PROVIDER": os.environ.get("RAG_PROVIDER", "openai"),
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
    # Offload ingestion (parse/chunk/embed) to a Celery worker.
    "ASYNC_INGEST": os.environ.get("ASYNC_INGEST", "1") == "1",
    # Tracing backend: "auto" (langfuse if keys present, else none), "none",
    # "langfuse", or "memory" (in-process, for tests).
    "TRACING": os.environ.get("TRACING", "auto"),
    "LANGFUSE_PUBLIC_KEY": os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
    "LANGFUSE_SECRET_KEY": os.environ.get("LANGFUSE_SECRET_KEY", ""),
    "LANGFUSE_HOST": os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
}
