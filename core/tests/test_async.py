"""Step 6: Celery async ingestion (run eagerly in tests)."""
import pytest
from django.contrib.auth import get_user_model

from core.models import Document
from core.services.ingest import register_document
from core.tasks import ingest_document_task

pytestmark = pytest.mark.django_db

User = get_user_model()


@pytest.fixture
def user():
    return User.objects.create_user(username="owner", password="secret123")


def test_register_is_fast_and_pending_without_embedding(user):
    # Registration stores bytes + metadata but does NOT chunk/embed.
    doc, action = register_document(
        user, b"async pipeline content", source_filename="a.txt", mime_type="text/plain"
    )
    assert action == "created"
    assert doc.status == Document.Status.PENDING
    assert doc.content_hash
    assert not doc.chunks.exists()
    assert doc.source_file  # raw bytes persisted for the worker


def test_task_processes_to_ready(user):
    doc, _ = register_document(
        user, b"the worker will embed this text", source_filename="a.txt",
        mime_type="text/plain",
    )
    result = ingest_document_task.delay(doc.id).get()
    doc.refresh_from_db()
    assert doc.status == Document.Status.READY
    assert doc.chunks.exists()
    assert result["document_id"] == doc.id


def test_task_reads_persisted_file(user):
    doc, _ = register_document(
        user, b"persisted bytes round trip", source_filename="rt.txt",
        mime_type="text/plain",
    )
    # Simulate a cold worker: only the document id is available.
    ingest_document_task.delay(doc.id).get()
    doc.refresh_from_db()
    assert doc.char_count == len("persisted bytes round trip")


def test_endpoint_dispatches_and_completes_eager(user):
    from rest_framework.test import APIClient
    from django.core.files.uploadedfile import SimpleUploadedFile

    client = APIClient()
    client.force_authenticate(user)
    f = SimpleUploadedFile("doc.txt", b"endpoint async path", content_type="text/plain")
    resp = client.post("/api/documents", {"file": f})
    # Eager mode finishes inline, so the document is already ready (201, not 202).
    assert resp.status_code == 201, resp.content
    assert resp.json()["status"] == "ready"
    assert resp.json()["sync_action"] == "created"
