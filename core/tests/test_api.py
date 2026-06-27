"""End-to-end API tests with the fake provider (no network, real pgvector DB)."""
import pytest
from rest_framework.test import APIClient

pytestmark = pytest.mark.django_db


@pytest.fixture
def client():
    return APIClient()


@pytest.fixture
def auth_client(client):
    resp = client.post(
        "/api/auth/register",
        {"username": "alice", "password": "secret123"},
        format="json",
    )
    assert resp.status_code == 201, resp.content
    token = resp.json()["token"]
    client.credentials(HTTP_AUTHORIZATION=f"Token {token}")
    return client


def _upload(client, text, title="doc"):
    from django.core.files.uploadedfile import SimpleUploadedFile

    upload = SimpleUploadedFile(f"{title}.txt", text.encode(), content_type="text/plain")
    return client.post("/api/documents", {"file": upload, "title": title})


def test_requires_auth(client):
    assert client.get("/api/documents").status_code == 401


def test_upload_ingests_and_chunks(auth_client):
    resp = _upload(
        auth_client,
        "The capital of France is Paris.\n\nThe Eiffel Tower is in Paris.",
    )
    assert resp.status_code == 201, resp.content
    body = resp.json()
    assert body["status"] == "ready"
    assert body["chunk_count"] >= 1


def test_query_returns_answer_and_citations(auth_client):
    _upload(
        auth_client,
        "Photosynthesis converts sunlight into chemical energy in plants. "
        "Chlorophyll absorbs light in the chloroplast.",
        title="bio",
    )
    resp = auth_client.post(
        "/api/query", {"question": "What does chlorophyll do?"}, format="json"
    )
    assert resp.status_code == 200, resp.content
    body = resp.json()
    assert "answer" in body
    assert isinstance(body["citations"], list)
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["chunk_id"]
    assert "usage" in body and "latency_ms" in body


def test_query_only_searches_own_documents(client):
    # Alice uploads.
    a = APIClient()
    tok_a = a.post(
        "/api/auth/register", {"username": "alice", "password": "secret123"}, format="json"
    ).json()["token"]
    a.credentials(HTTP_AUTHORIZATION=f"Token {tok_a}")
    _upload(a, "Alice secret topic: quasars are bright.", title="alice")

    # Bob queries and should retrieve nothing from Alice.
    b = APIClient()
    tok_b = b.post(
        "/api/auth/register", {"username": "bob", "password": "secret123"}, format="json"
    ).json()["token"]
    b.credentials(HTTP_AUTHORIZATION=f"Token {tok_b}")
    resp = b.post("/api/query", {"question": "quasars"}, format="json")
    assert resp.status_code == 200
    assert resp.json()["citations"] == []


def test_unsupported_file_reports_failure(auth_client):
    from django.core.files.uploadedfile import SimpleUploadedFile

    upload = SimpleUploadedFile(
        "weird.bin", b"\x00\x01\x02", content_type="application/octet-stream"
    )
    resp = auth_client.post("/api/documents", {"file": upload})
    assert resp.status_code == 422
    assert resp.json()["document"]["status"] == "failed"
