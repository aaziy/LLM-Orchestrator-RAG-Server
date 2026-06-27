"""Pluggable reranking stage.

After hybrid retrieval produces a candidate pool, a cross-encoder reranker scores
each (query, chunk) pair jointly and reorders them — far more precise than the
bi-encoder similarity used for first-stage retrieval. The interface mirrors the
LLM provider so the backend (local model / Cohere / fake) is a config switch.
"""
from __future__ import annotations

from functools import lru_cache

from django.conf import settings


class Reranker:
    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Return a relevance score per text, higher = more relevant."""
        raise NotImplementedError


class FakeReranker(Reranker):
    """Deterministic, dependency-free reranker for tests/offline dev.

    Scores by lexical token overlap between query and text. Enough to prove the
    reranking stage reorders candidates without pulling in an ML model.
    """

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        q_tokens = set(query.lower().split())
        scores = []
        for text in texts:
            t_tokens = text.lower().split()
            overlap = sum(1 for tok in t_tokens if tok in q_tokens)
            scores.append(float(overlap))
        return scores


class LocalCrossEncoderReranker(Reranker):
    """sentence-transformers CrossEncoder; runs locally, no API key."""

    def __init__(self):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(settings.RAG["RERANK_MODEL"])

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        return [float(s) for s in self._model.predict(pairs)]


class CohereReranker(Reranker):
    """Hosted Cohere Rerank endpoint."""

    def __init__(self):
        import cohere

        self._client = cohere.Client(settings.RAG["COHERE_API_KEY"])
        self._model = settings.RAG["COHERE_RERANK_MODEL"]

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        resp = self._client.rerank(query=query, documents=texts, model=self._model)
        scores = [0.0] * len(texts)
        for result in resp.results:
            scores[result.index] = float(result.relevance_score)
        return scores


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    name = settings.RAG["RERANKER"].lower()
    if name == "fake":
        return FakeReranker()
    if name == "local":
        return LocalCrossEncoderReranker()
    if name == "cohere":
        return CohereReranker()
    raise ValueError(f"Unknown RERANKER: {name!r}")
