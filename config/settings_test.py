"""Test settings: deterministic backends, eager Celery, isolated media.

Used via DJANGO_SETTINGS_MODULE=config.settings_test (see pytest.ini). Importing
from the base settings keeps everything else identical to production.
"""
import os
import tempfile

from config.settings import *  # noqa: F401,F403
from config.settings import RAG  # noqa: F401

# Run Celery tasks inline; no broker/worker/redis required during tests.
CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True
CELERY_RESULT_BACKEND = None
CELERY_BROKER_URL = "memory://"

# Deterministic, network-free model + reranker backends.
RAG = {**RAG, "PROVIDER": "fake", "RERANKER": "fake"}

# Keep uploaded test files out of the repo.
MEDIA_ROOT = os.path.join(tempfile.gettempdir(), "rag_test_media")
