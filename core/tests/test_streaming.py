"""Step 5: SSE streaming on /query/stream."""
import json

import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from core.models import Document, QueryLog
from core.services.ingest import ingest_document

pytestmark = pytest.mark.django_db

User = get_user_model()


@pytest.fixture
def user():
    return User.objects.create_user(username="owner", password="secret123")


def _ingest(user, text, title="d"):
    doc = Document.objects.create(owner=user, title=title, mime_type="text/plain")
    ingest_document(doc, text.encode())
    return doc


def _parse_sse(raw: str):
    """Parse an SSE stream into a list of (event, data_dict)."""
    events = []
    for block in raw.strip().split("\n\n"):
        if not block.strip():
            continue
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = json.loads(line[len("data:"):].strip())
        events.append((event, data))
    return events


def test_stream_emits_citations_tokens_done(user):
    _ingest(user, "the great barrier reef is off the coast of australia", title="reef")
    client = APIClient()
    client.force_authenticate(user)

    resp = client.post(
        "/api/query/stream", {"question": "great barrier reef"}, format="json"
    )
    assert resp.status_code == 200
    assert resp["Content-Type"] == "text/event-stream"

    raw = b"".join(resp.streaming_content).decode()
    events = _parse_sse(raw)
    event_types = [e for e, _ in events]

    assert event_types[0] == "citations"
    assert "token" in event_types
    assert event_types[-1] == "done"


def test_stream_tokens_reconstruct_answer(user):
    _ingest(user, "mount everest is the tallest mountain on earth", title="geo")
    client = APIClient()
    client.force_authenticate(user)

    resp = client.post("/api/query/stream", {"question": "tallest mountain"}, format="json")
    events = _parse_sse(b"".join(resp.streaming_content).decode())

    tokens = [d["text"] for e, d in events if e == "token"]
    assert tokens, "expected at least one token delta"
    answer = "".join(tokens)
    assert "[fake-answer]" in answer


def test_stream_done_has_usage_and_logs_query(user):
    _ingest(user, "the amazon river is in south america", title="river")
    client = APIClient()
    client.force_authenticate(user)

    resp = client.post("/api/query/stream", {"question": "amazon river"}, format="json")
    events = _parse_sse(b"".join(resp.streaming_content).decode())

    done = [d for e, d in events if e == "done"][0]
    assert done["usage"]["completion_tokens"] > 0
    assert "latency_ms" in done
    # The query is logged exactly once after the stream completes.
    assert QueryLog.objects.filter(owner=user).count() == 1


def test_stream_requires_auth():
    client = APIClient()
    resp = client.post("/api/query/stream", {"question": "x"}, format="json")
    assert resp.status_code == 401
