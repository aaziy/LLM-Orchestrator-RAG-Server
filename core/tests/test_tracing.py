"""Step 7: end-to-end tracing (memory backend stands in for Langfuse)."""
import pytest
from django.conf import settings
from django.contrib.auth import get_user_model

from core.models import Document
from core.services import tracing
from core.services.ingest import ingest_document
from core.services.retrieve import answer_query
from core.services.tracing import NoopTracer, _resolve_backend, get_tracer

pytestmark = pytest.mark.django_db

User = get_user_model()


@pytest.fixture
def memory_tracing():
    """Switch tracing to the in-process memory backend for assertions."""
    original = settings.RAG["TRACING"]
    settings.RAG["TRACING"] = "memory"
    get_tracer.cache_clear()
    tracing.RECORDS.clear()
    yield tracing.RECORDS
    settings.RAG["TRACING"] = original
    get_tracer.cache_clear()
    tracing.RECORDS.clear()


@pytest.fixture
def user():
    return User.objects.create_user(username="owner", password="secret123")


def test_query_emits_full_trace(memory_tracing, user):
    doc = Document.objects.create(owner=user, title="d", mime_type="text/plain")
    ingest_document(doc, b"the speed of light is about 300000 kilometers per second")

    answer_query(user, "speed of light")

    assert len(memory_tracing) == 1
    rec = memory_tracing[0]
    # input -> retrieved chunks -> prompt -> output -> usage are all captured.
    assert rec.name == "query"
    assert rec.input["question"] == "speed of light"
    span_names = [s["name"] for s in rec.spans]
    assert span_names == ["retrieval", "generation"]

    retrieval = rec.spans[0]
    assert retrieval["output"]["count"] >= 1
    generation = rec.spans[1]
    assert "prompt" in generation["input"]
    assert "answer" in generation["output"]

    assert "answer" in rec.output
    assert "citations" in rec.output
    assert rec.usage["completion_tokens"] >= 0


def test_auto_backend_is_noop_without_keys():
    original = settings.RAG["TRACING"]
    settings.RAG["TRACING"] = "auto"
    settings.RAG["LANGFUSE_PUBLIC_KEY"] = ""
    settings.RAG["LANGFUSE_SECRET_KEY"] = ""
    get_tracer.cache_clear()
    try:
        assert _resolve_backend() == "none"
        assert isinstance(get_tracer(), NoopTracer)
    finally:
        settings.RAG["TRACING"] = original
        get_tracer.cache_clear()


def test_auto_backend_selects_langfuse_with_keys():
    original = settings.RAG["TRACING"]
    pk, sk = settings.RAG["LANGFUSE_PUBLIC_KEY"], settings.RAG["LANGFUSE_SECRET_KEY"]
    settings.RAG["TRACING"] = "auto"
    settings.RAG["LANGFUSE_PUBLIC_KEY"] = "pk-test"
    settings.RAG["LANGFUSE_SECRET_KEY"] = "sk-test"
    try:
        assert _resolve_backend() == "langfuse"
    finally:
        settings.RAG["TRACING"] = original
        settings.RAG["LANGFUSE_PUBLIC_KEY"] = pk
        settings.RAG["LANGFUSE_SECRET_KEY"] = sk


def test_noop_tracer_when_disabled(user):
    # Default test settings keep tracing off; query should still work.
    settings.RAG["TRACING"] = "none"
    get_tracer.cache_clear()
    doc = Document.objects.create(owner=user, title="d", mime_type="text/plain")
    ingest_document(doc, b"gravity pulls objects toward earth")
    result = answer_query(user, "gravity")
    assert "answer" in result
