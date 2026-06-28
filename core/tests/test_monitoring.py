"""Step 8: cost + latency monitoring."""
import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from core.models import Document, QueryLog
from core.services.ingest import ingest_document
from core.services.pricing import chat_cost, estimate_cost
from core.services.retrieve import answer_query

pytestmark = pytest.mark.django_db

User = get_user_model()


@pytest.fixture
def user():
    return User.objects.create_user(username="owner", password="secret123")


def _ingest(user, text):
    doc = Document.objects.create(owner=user, title="d", mime_type="text/plain")
    ingest_document(doc, text.encode())
    return doc


def test_chat_cost_known_model():
    # 1000 in + 1000 out on gpt-4o-mini = 0.00015 + 0.00060.
    assert chat_cost("gpt-4o-mini", 1000, 1000) == pytest.approx(0.00075)


def test_unknown_model_is_free():
    assert chat_cost("mystery-model", 1000, 1000) == 0.0


def test_estimate_includes_embedding_cost():
    cost = estimate_cost(
        chat_model="gpt-4o-mini",
        prompt_tokens=0,
        completion_tokens=0,
        embedding_model="text-embedding-3-small",
        embedding_tokens=1000,
    )
    assert cost == pytest.approx(0.00002)


def test_answer_query_records_timings_and_cost(user):
    _ingest(user, "the boiling point of water is 100 degrees celsius")
    result = answer_query(user, "boiling point of water")

    assert "retrieval_ms" in result and "generation_ms" in result
    assert "cost_usd" in result["usage"]

    log = QueryLog.objects.filter(owner=user).latest("created_at")
    assert log.retrieval_ms >= 0
    assert log.generation_ms >= 0
    assert log.cost_usd >= 0.0


def test_usage_endpoint_aggregates(user):
    _ingest(user, "jupiter is the largest planet in the solar system")
    client = APIClient()
    client.force_authenticate(user)

    client.post("/api/query", {"question": "largest planet"}, format="json")
    client.post("/api/query", {"question": "jupiter"}, format="json")

    resp = client.get("/api/usage")
    assert resp.status_code == 200
    body = resp.json()
    assert body["queries"] == 2
    assert body["completion_tokens"] >= 0
    assert body["avg_latency_ms"] >= 0
    assert "total_cost_usd" in body


def test_usage_endpoint_empty_user(user):
    client = APIClient()
    client.force_authenticate(user)
    resp = client.get("/api/usage")
    assert resp.status_code == 200
    assert resp.json()["queries"] == 0
    assert resp.json()["total_cost_usd"] == 0.0
