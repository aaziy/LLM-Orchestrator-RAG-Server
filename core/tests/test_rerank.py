"""Step 4: pluggable reranker (cross-encoder stage)."""
import pytest
from django.conf import settings
from django.contrib.auth import get_user_model

from core.models import Document
from core.services.ingest import ingest_document
from core.services.rerank import FakeReranker, get_reranker
from core.services.retrieve import RetrievedChunk, rerank_chunks, retrieve_for_answer

pytestmark = pytest.mark.django_db

User = get_user_model()


@pytest.fixture
def user():
    return User.objects.create_user(username="owner", password="secret123")


def _ingest(user, text, title="d"):
    doc = Document.objects.create(owner=user, title=title, mime_type="text/plain")
    ingest_document(doc, text.encode())
    return doc


def test_fake_reranker_scores_by_overlap():
    scores = FakeReranker().rerank("apple banana", ["apple banana cherry", "zzz qqq"])
    assert scores[0] > scores[1]


def test_rerank_reorders_candidates(user):
    d_relevant = _ingest(user, "apple banana cherry fruit", title="fruit")
    d_other = _ingest(user, "engine piston crankshaft motor", title="car")
    chunks = [
        RetrievedChunk(chunk=d_other.chunks.first(), distance=0.1),   # better 1st-stage
        RetrievedChunk(chunk=d_relevant.chunks.first(), distance=0.9),
    ]
    reranked = rerank_chunks("apple banana", chunks, top_k=2)
    # The lexically relevant chunk should now be first despite worse distance.
    assert reranked[0].chunk.document_id == d_relevant.id
    assert reranked[0].rerank_score is not None


def test_rerank_applies_top_k(user):
    docs = [_ingest(user, f"topic word{i} content", title=str(i)) for i in range(5)]
    chunks = [RetrievedChunk(chunk=d.chunks.first(), distance=0.5) for d in docs]
    out = rerank_chunks("topic", chunks, top_k=2)
    assert len(out) == 2


def test_retrieve_for_answer_returns_rerank_scores(user):
    _ingest(user, "the sun is a star at the center of the solar system", title="space")
    settings.RAG["RERANK"] = True
    results = retrieve_for_answer(user, "star solar system", k=3)
    assert results
    assert all(r.rerank_score is not None for r in results)


def test_get_reranker_fake_selected():
    assert isinstance(get_reranker(), FakeReranker)


def test_local_cross_encoder_importable():
    # Auto-skips if sentence-transformers is not installed (heavy, optional in dev).
    st = pytest.importorskip("sentence_transformers")
    assert hasattr(st, "CrossEncoder")
