"""Retrieval and grounded answer generation."""
from __future__ import annotations

import time
from dataclasses import dataclass

from django.conf import settings
from pgvector.django import CosineDistance

from core.models import Chunk, QueryLog
from core.services.llm import Generation, get_provider

SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the question using ONLY the numbered "
    "context passages provided. Cite the passages you use with their bracketed "
    "numbers, e.g. [1]. If the context does not contain the answer, say exactly: "
    "\"I don't have enough information in the provided documents to answer that.\""
)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    distance: float


def retrieve(owner, query: str, k: int) -> list[RetrievedChunk]:
    """Embed the query and return the k nearest chunks owned by ``owner``."""
    embedding = get_provider().embed([query])[0]
    qs = (
        Chunk.objects.filter(document__owner=owner)
        .annotate(distance=CosineDistance("embedding", embedding))
        .order_by("distance")[:k]
    )
    return [RetrievedChunk(chunk=c, distance=float(c.distance)) for c in qs]


def _build_user_prompt(query: str, retrieved: list[RetrievedChunk]) -> str:
    blocks = []
    for i, r in enumerate(retrieved, start=1):
        blocks.append(f"[{i}] {r.chunk.text}")
    context = "\n\n".join(blocks) if blocks else "(no context found)"
    return f"Context passages:\n{context}\n\nQuestion: {query}"


def answer_query(owner, query: str, k: int | None = None) -> dict:
    """Full query path: retrieve -> grounded prompt -> generate -> log."""
    k = k or settings.RAG["TOP_K"]
    started = time.perf_counter()

    retrieved = retrieve(owner, query, k)
    user_prompt = _build_user_prompt(query, retrieved)
    gen: Generation = get_provider().generate(SYSTEM_PROMPT, user_prompt)

    latency_ms = int((time.perf_counter() - started) * 1000)
    chunk_ids = [r.chunk.id for r in retrieved]

    QueryLog.objects.create(
        owner=owner,
        question=query,
        retrieved_chunk_ids=chunk_ids,
        latency_ms=latency_ms,
        prompt_tokens=gen.prompt_tokens,
        completion_tokens=gen.completion_tokens,
    )

    return {
        "answer": gen.text,
        "citations": [
            {
                "index": i,
                "chunk_id": r.chunk.id,
                "document_id": r.chunk.document_id,
                "ordinal": r.chunk.ordinal,
                "score": round(1.0 - r.distance, 4),
                "preview": r.chunk.text[:200],
            }
            for i, r in enumerate(retrieved, start=1)
        ],
        "latency_ms": latency_ms,
        "usage": {
            "prompt_tokens": gen.prompt_tokens,
            "completion_tokens": gen.completion_tokens,
        },
    }
