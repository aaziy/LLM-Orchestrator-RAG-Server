"""Step 1: metadata extraction + filtered retrieval."""
import datetime

import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from core.models import Document
from core.services.ingest import ingest_document
from core.services.retrieve import retrieve

pytestmark = pytest.mark.django_db

User = get_user_model()


@pytest.fixture
def user():
    return User.objects.create_user(username="owner", password="secret123")


def _ingest(user, text, **kwargs):
    doc = Document.objects.create(
        owner=user, title=kwargs.pop("title", "d"), mime_type="text/plain", **kwargs
    )
    ingest_document(doc, text.encode())
    doc.refresh_from_db()
    return doc


def test_client_metadata_persisted(user):
    doc = _ingest(
        user, "alpha beta gamma", author="Ada Lovelace",
        doc_date=datetime.date(2020, 1, 1),
    )
    assert doc.author == "Ada Lovelace"
    assert doc.doc_date == datetime.date(2020, 1, 1)


def test_filter_by_document_ids(user):
    d1 = _ingest(user, "quantum entanglement physics", title="phys")
    _ingest(user, "sourdough bread baking recipe", title="bread")
    results = retrieve(user, "physics", k=10, filters={"document_ids": [d1.id]})
    assert results
    assert all(r.chunk.document_id == d1.id for r in results)


def test_filter_by_author(user):
    _ingest(user, "neural networks deep learning", title="ml", author="Hinton")
    _ingest(user, "neural networks deep learning", title="other", author="Someone")
    results = retrieve(user, "learning", k=10, filters={"author": "hinton"})
    assert results
    assert all(r.chunk.document.author == "Hinton" for r in results)


def test_filter_by_date_range(user):
    _ingest(user, "old report data", title="old", doc_date=datetime.date(2019, 5, 1))
    _ingest(user, "old report data", title="new", doc_date=datetime.date(2023, 5, 1))
    results = retrieve(
        user, "report", k=10, filters={"date_from": datetime.date(2022, 1, 1)}
    )
    assert results
    assert all(r.chunk.document.doc_date >= datetime.date(2022, 1, 1) for r in results)


def test_pdf_metadata_autoextracted(user):
    # Build a tiny PDF carrying author + creation date metadata.
    pypdf = pytest.importorskip("pypdf")
    import io

    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_metadata({"/Author": "Grace Hopper", "/CreationDate": "D:20210314000000"})
    buf = io.BytesIO()
    writer.write(buf)

    doc = Document.objects.create(
        owner=user, title="scanned", mime_type="application/pdf",
        source_filename="scanned.pdf",
    )
    # A blank page yields no text, so ingestion fails; metadata extraction still runs.
    try:
        ingest_document(doc, buf.getvalue())
    except ValueError:
        pass
    doc.refresh_from_db()
    assert doc.author == "Grace Hopper"
    assert doc.doc_date == datetime.date(2021, 3, 14)


def test_query_endpoint_accepts_filters(user):
    client = APIClient()
    client.force_authenticate(user)
    d1 = _ingest(user, "marsupials live in australia", title="zoo")
    resp = client.post(
        "/api/query",
        {"question": "australia", "document_ids": [d1.id]},
        format="json",
    )
    assert resp.status_code == 200, resp.content
    cites = resp.json()["citations"]
    assert cites and all(c["document_id"] == d1.id for c in cites)
