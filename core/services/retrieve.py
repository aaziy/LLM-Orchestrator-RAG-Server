"""Retrieval and grounded answer generation."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from django.conf import settings
from django.contrib.postgres.search import SearchQuery, SearchRank
from pgvector.django import CosineDistance

from core.models import Chunk, QueryLog
from core.services.chunking import count_tokens
from core.services.llm import Generation, get_provider
from core.services.pricing import estimate_cost
from core.services.rerank import get_reranker
from core.services.tracing import get_tracer

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
    rerank_score: float | None = None


def _apply_filters(qs, owner, filters: dict | None):
    """Scope a Chunk queryset to the owner plus optional metadata filters."""
    qs = qs.filter(document__owner=owner)
    if not filters:
        return qs
    if filters.get("document_ids"):
        qs = qs.filter(document_id__in=filters["document_ids"])
    if filters.get("author"):
        qs = qs.filter(document__author__icontains=filters["author"])
    if filters.get("date_from"):
        qs = qs.filter(document__doc_date__gte=filters["date_from"])
    if filters.get("date_to"):
        qs = qs.filter(document__doc_date__lte=filters["date_to"])
    return qs


def _vector_rank(owner, query: str, limit: int, filters) -> list[tuple[int, float]]:
    """Return [(chunk_id, cosine_distance)] best-first via pgvector."""
    embedding = get_provider().embed([query])[0]
    qs = _apply_filters(Chunk.objects.all(), owner, filters)
    qs = (
        qs.annotate(distance=CosineDistance("embedding", embedding))
        .order_by("distance")
        .values_list("id", "distance")[:limit]
    )
    return [(cid, float(dist)) for cid, dist in qs]


def _keyword_rank(owner, query: str, limit: int, filters) -> list[int]:
    """Return chunk_ids best-first via Postgres full-text ranking (BM25-like)."""
    search_query = SearchQuery(query, config="english")
    qs = _apply_filters(Chunk.objects.all(), owner, filters)
    qs = (
        qs.filter(search_vector=search_query)
        .annotate(rank=SearchRank("search_vector", search_query))
        .order_by("-rank")
        .values_list("id", flat=True)[:limit]
    )
    return list(qs)


def _reciprocal_rank_fusion(
    ranked_lists: list[list[int]], rrf_k: int
) -> list[tuple[int, float]]:
    """Fuse multiple ranked id lists into one, best-first, via RRF."""
    scores: dict[int, float] = {}
    for ids in ranked_lists:
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def retrieve(owner, query: str, k: int, filters: dict | None = None) -> list[RetrievedChunk]:
    """Vector-only retrieval: the k nearest chunks owned by ``owner``.

    Optional ``filters`` (document_ids, author, date_from, date_to) are applied
    before the vector search so similarity ranks only the eligible chunks.
    """
    ranked = _vector_rank(owner, query, k, filters)
    by_id = Chunk.objects.in_bulk([cid for cid, _ in ranked])
    return [RetrievedChunk(chunk=by_id[cid], distance=dist) for cid, dist in ranked]


def hybrid_retrieve(
    owner, query: str, k: int, filters: dict | None = None
) -> list[RetrievedChunk]:
    """Combine vector similarity and full-text keyword ranking via RRF.

    Each arm contributes up to CANDIDATE_POOL results; the two rankings are fused
    with reciprocal rank fusion and the top-k survivors are returned. The vector
    distance is carried through for display/scoring when available.
    """
    cfg = settings.RAG
    pool = max(cfg["CANDIDATE_POOL"], k)

    vector_ranked = _vector_rank(owner, query, pool, filters)
    keyword_ranked = _keyword_rank(owner, query, pool, filters)

    distance_by_id = {cid: dist for cid, dist in vector_ranked}
    fused = _reciprocal_rank_fusion(
        [[cid for cid, _ in vector_ranked], keyword_ranked], cfg["RRF_K"]
    )[:k]

    by_id = Chunk.objects.in_bulk([cid for cid, _ in fused])
    results = []
    for cid, _score in fused:
        chunk = by_id.get(cid)
        if chunk is None:
            continue
        # Distance is only known if the chunk surfaced in the vector arm.
        results.append(
            RetrievedChunk(chunk=chunk, distance=distance_by_id.get(cid, 1.0))
        )
    return results


def _build_user_prompt(query: str, retrieved: list[RetrievedChunk]) -> str:
    blocks = []
    for i, r in enumerate(retrieved, start=1):
        blocks.append(f"[{i}] {r.chunk.text}")
    context = "\n\n".join(blocks) if blocks else "(no context found)"
    return f"Context passages:\n{context}\n\nQuestion: {query}"


def rerank_chunks(
    query: str, retrieved: list[RetrievedChunk], top_k: int
) -> list[RetrievedChunk]:
    """Reorder candidates with the configured cross-encoder, keep ``top_k``."""
    if not retrieved:
        return retrieved
    scores = get_reranker().rerank(query, [r.chunk.text for r in retrieved])
    for r, s in zip(retrieved, scores):
        r.rerank_score = s
    retrieved.sort(key=lambda r: r.rerank_score, reverse=True)
    return retrieved[:top_k]


def retrieve_for_answer(
    owner, query: str, k: int, filters: dict | None = None
) -> list[RetrievedChunk]:
    """First-stage retrieval (hybrid or vector) followed by optional reranking."""
    cfg = settings.RAG
    if cfg["RERANK"]:
        pool = max(cfg["RERANK_CANDIDATES"], k)
        first_stage = (
            hybrid_retrieve(owner, query, pool, filters=filters)
            if cfg["HYBRID"]
            else retrieve(owner, query, pool, filters=filters)
        )
        return rerank_chunks(query, first_stage, k)

    if cfg["HYBRID"]:
        return hybrid_retrieve(owner, query, k, filters=filters)
    return retrieve(owner, query, k, filters=filters)


def _build_citations(retrieved: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "index": i,
            "chunk_id": r.chunk.id,
            "document_id": r.chunk.document_id,
            "ordinal": r.chunk.ordinal,
            "score": round(1.0 - r.distance, 4),
            "rerank_score": (
                round(r.rerank_score, 4) if r.rerank_score is not None else None
            ),
            "preview": r.chunk.text[:200],
        }
        for i, r in enumerate(retrieved, start=1)
    ]


def answer_query(
    owner, query: str, k: int | None = None, filters: dict | None = None
) -> dict:
    """Full query path: retrieve -> grounded prompt -> generate -> log -> trace."""
    k = k or settings.RAG["TOP_K"]
    started = time.perf_counter()

    cfg = settings.RAG
    with get_tracer().trace(
        "query", input={"question": query, "filters": filters},
        metadata={"user_id": owner.id, "k": k},
    ) as trace:
        retrieved = retrieve_for_answer(owner, query, k, filters=filters)
        retrieval_ms = int((time.perf_counter() - started) * 1000)
        trace.span(
            "retrieval",
            output={"chunk_ids": [r.chunk.id for r in retrieved], "count": len(retrieved)},
            metadata={"latency_ms": retrieval_ms},
        )

        gen_started = time.perf_counter()
        user_prompt = _build_user_prompt(query, retrieved)
        gen: Generation = get_provider().generate(SYSTEM_PROMPT, user_prompt)
        generation_ms = int((time.perf_counter() - gen_started) * 1000)
        trace.span(
            "generation",
            input={"system": SYSTEM_PROMPT, "prompt": user_prompt},
            output={"answer": gen.text},
            metadata={"latency_ms": generation_ms},
        )

        latency_ms = int((time.perf_counter() - started) * 1000)
        cost_usd = estimate_cost(
            chat_model=cfg["CHAT_MODEL"],
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            embedding_model=cfg["EMBEDDING_MODEL"],
            embedding_tokens=count_tokens(query),
        )
        usage = {
            "prompt_tokens": gen.prompt_tokens,
            "completion_tokens": gen.completion_tokens,
            "cost_usd": cost_usd,
        }

        QueryLog.objects.create(
            owner=owner,
            question=query,
            retrieved_chunk_ids=[r.chunk.id for r in retrieved],
            latency_ms=latency_ms,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            cost_usd=cost_usd,
        )

        citations = _build_citations(retrieved)
        trace.end(output={"answer": gen.text, "citations": citations}, usage=usage)

    return {
        "answer": gen.text,
        "citations": citations,
        "latency_ms": latency_ms,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "usage": usage,
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def answer_query_stream(
    owner, query: str, k: int | None = None, filters: dict | None = None
):
    """Server-Sent Events generator: citations first, then token deltas, then done.

    Emits the same retrieval/citation payload as ``answer_query`` but streams the
    answer incrementally and logs the query once generation completes.
    """
    cfg = settings.RAG
    k = k or cfg["TOP_K"]
    started = time.perf_counter()

    retrieved = retrieve_for_answer(owner, query, k, filters=filters)
    retrieval_ms = int((time.perf_counter() - started) * 1000)
    user_prompt = _build_user_prompt(query, retrieved)

    yield _sse("citations", {"citations": _build_citations(retrieved)})

    gen_started = time.perf_counter()
    parts: list[str] = []
    for delta in get_provider().generate_stream(SYSTEM_PROMPT, user_prompt):
        parts.append(delta)
        yield _sse("token", {"text": delta})

    answer = "".join(parts)
    generation_ms = int((time.perf_counter() - gen_started) * 1000)
    latency_ms = int((time.perf_counter() - started) * 1000)
    # Token usage is estimated locally for provider-agnostic streaming accounting.
    prompt_tokens = count_tokens(SYSTEM_PROMPT + "\n" + user_prompt)
    completion_tokens = count_tokens(answer)
    cost_usd = estimate_cost(
        chat_model=cfg["CHAT_MODEL"],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        embedding_model=cfg["EMBEDDING_MODEL"],
        embedding_tokens=count_tokens(query),
    )

    QueryLog.objects.create(
        owner=owner,
        question=query,
        retrieved_chunk_ids=[r.chunk.id for r in retrieved],
        latency_ms=latency_ms,
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
    )

    yield _sse(
        "done",
        {
            "latency_ms": latency_ms,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost_usd,
            },
        },
    )
