"""Step 3: hybrid search (full-text keyword + vector via RRF)."""
import pytest
from django.contrib.auth import get_user_model

from core.models import Chunk, Document
from core.services.ingest import ingest_document
from core.services.retrieve import (
    _keyword_rank,
    _reciprocal_rank_fusion,
    hybrid_retrieve,
    retrieve,
)

pytestmark = pytest.mark.django_db

User = get_user_model()


@pytest.fixture
def user():
    return User.objects.create_user(username="owner", password="secret123")


def _ingest(user, text, title="d"):
    doc = Document.objects.create(owner=user, title=title, mime_type="text/plain")
    ingest_document(doc, text.encode())
    return doc


def test_rrf_orders_by_combined_score():
    # B appears high in both lists, so it should win overall.
    fused = _reciprocal_rank_fusion([["A", "B", "C"], ["B", "C", "A"]], rrf_k=60)
    order = [cid for cid, _ in fused]
    assert order[0] == "B"
    assert set(order) == {"A", "B", "C"}


def test_search_vector_populated_on_ingest(user):
    _ingest(user, "photosynthesis converts sunlight into energy")
    assert Chunk.objects.filter(search_vector__isnull=False).count() >= 1


def test_keyword_rank_uses_stemming(user):
    # Full-text stems "running" -> "run", so a "run" query matches lexically.
    doc = _ingest(user, "The athlete was running in the marathon.", title="sport")
    ids = _keyword_rank(user, "run", limit=10, filters=None)
    assert ids
    assert all(Chunk.objects.get(id=i).document_id == doc.id for i in ids)


def test_hybrid_recovers_lexical_match_vector_misses(user):
    # Fake embeddings are bag-of-words: "run" != "running" (different tokens),
    # so the vector arm does not connect the query to this doc -- but the
    # keyword arm (with stemming) does. Hybrid must surface it.
    target = _ingest(user, "The athlete was running in the marathon.", title="sport")
    _ingest(user, "A recipe for cooking tomato pasta sauce.", title="food")

    results = hybrid_retrieve(user, "run", k=5, filters=None)
    assert results, "hybrid should return the lexical match"
    assert results[0].chunk.document_id == target.id


def test_hybrid_respects_filters(user):
    keep = _ingest(user, "quantum computing qubits", title="phys")
    _ingest(user, "quantum computing qubits", title="dupe")
    results = hybrid_retrieve(
        user, "quantum", k=5, filters={"document_ids": [keep.id]}
    )
    assert results
    assert all(r.chunk.document_id == keep.id for r in results)


def test_vector_only_retrieve_still_works(user):
    _ingest(user, "the mitochondria is the powerhouse of the cell")
    results = retrieve(user, "mitochondria powerhouse cell", k=3)
    assert results
