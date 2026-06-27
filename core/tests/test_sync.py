"""Step 2: content-hash auto-sync (skip / re-embed / delete)."""
import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from core.models import Chunk, Document
from core.services.ingest import sync_document

pytestmark = pytest.mark.django_db

User = get_user_model()


@pytest.fixture
def user():
    return User.objects.create_user(username="owner", password="secret123")


def test_first_upload_creates(user):
    doc, action = sync_document(
        user, b"hello world content", source_filename="a.txt", mime_type="text/plain"
    )
    assert action == "created"
    assert doc.status == Document.Status.READY
    assert doc.content_hash
    assert doc.chunks.exists()


def test_identical_reupload_is_skipped(user):
    doc1, _ = sync_document(
        user, b"stable content", source_filename="a.txt", mime_type="text/plain"
    )
    chunk_ids_before = set(doc1.chunks.values_list("id", flat=True))

    doc2, action = sync_document(
        user, b"stable content", source_filename="a.txt", mime_type="text/plain"
    )
    assert action == "unchanged"
    assert doc2.id == doc1.id
    # No re-embedding: the same chunk rows are still present.
    assert set(doc2.chunks.values_list("id", flat=True)) == chunk_ids_before
    assert Document.objects.filter(owner=user).count() == 1


def test_changed_content_reembeds_in_place(user):
    doc1, _ = sync_document(
        user, b"version one text", source_filename="a.txt", mime_type="text/plain"
    )
    old_chunk_ids = set(doc1.chunks.values_list("id", flat=True))
    old_hash = doc1.content_hash

    doc2, action = sync_document(
        user, b"version two completely different",
        source_filename="a.txt", mime_type="text/plain",
    )
    assert action == "updated"
    assert doc2.id == doc1.id
    assert doc2.content_hash != old_hash
    # Old chunks were replaced, not appended.
    new_chunk_ids = set(doc2.chunks.values_list("id", flat=True))
    assert new_chunk_ids.isdisjoint(old_chunk_ids)
    assert Document.objects.filter(owner=user).count() == 1


def test_delete_removes_chunks(user):
    doc, _ = sync_document(
        user, b"to be deleted", source_filename="a.txt", mime_type="text/plain"
    )
    doc_id = doc.id
    doc.delete()
    assert not Chunk.objects.filter(document_id=doc_id).exists()


def test_endpoint_reports_sync_action(user):
    client = APIClient()
    client.force_authenticate(user)
    from django.core.files.uploadedfile import SimpleUploadedFile

    def upload(content):
        f = SimpleUploadedFile("doc.txt", content, content_type="text/plain")
        return client.post("/api/documents", {"file": f})

    r1 = upload(b"alpha content here")
    assert r1.status_code == 201
    assert r1.json()["sync_action"] == "created"

    r2 = upload(b"alpha content here")
    assert r2.status_code == 200
    assert r2.json()["sync_action"] == "unchanged"

    r3 = upload(b"different content now")
    assert r3.status_code == 200
    assert r3.json()["sync_action"] == "updated"
