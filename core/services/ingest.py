"""Document ingestion pipeline: parse -> chunk -> embed -> store.

Exposed as a single callable, ``ingest_document``, so it can be invoked inline
today and moved behind a task queue (Celery/RQ) later without changing callers.
"""
from __future__ import annotations

import hashlib

from django.conf import settings
from django.contrib.postgres.search import SearchVector
from django.db import transaction

from core.models import Chunk, Document
from core.services import chunking
from core.services.llm import get_provider


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def register_document(
    owner,
    data: bytes,
    *,
    source_filename: str,
    title: str = "",
    author: str = "",
    doc_date=None,
    mime_type: str = "",
) -> tuple[Document, str]:
    """Fast, synchronous registration keyed on (owner, source_filename) + hash.

    Stores the raw bytes and resolves the sync action WITHOUT embedding, so the
    heavy work can be dispatched to a worker. Returns ``(document, action)``:
      - "unchanged": a ready document with identical bytes already exists (no work)
      - "updated":   the source changed; document reset to pending for re-ingest
      - "created":   a brand-new pending document
    """
    from django.core.files.base import ContentFile

    digest = content_hash(data)
    existing = (
        Document.objects.filter(owner=owner, source_filename=source_filename)
        .order_by("-created_at")
        .first()
    )

    if existing and existing.content_hash == digest and existing.status == Document.Status.READY:
        return existing, "unchanged"

    document = existing or Document(owner=owner, source_filename=source_filename)
    action = "updated" if existing else "created"
    document.title = title or document.title or source_filename
    if author:
        document.author = author
    if doc_date is not None:
        document.doc_date = doc_date
    document.mime_type = mime_type or document.mime_type
    document.content_hash = digest
    document.status = Document.Status.PENDING
    document.error = ""
    document.save()
    # Persist raw bytes so an async worker (or re-run) can read them back.
    document.source_file.save(source_filename, ContentFile(data), save=True)
    return document, action


def dispatch_ingest(document: Document, data: bytes) -> None:
    """Run ingestion inline or hand it to Celery, per ASYNC_INGEST."""
    if settings.RAG["ASYNC_INGEST"]:
        from core.tasks import ingest_document_task

        ingest_document_task.delay(document.id)
    else:
        ingest_document(document, data)


def sync_document(
    owner,
    data: bytes,
    *,
    source_filename: str,
    title: str = "",
    author: str = "",
    doc_date=None,
    mime_type: str = "",
) -> tuple[Document, str]:
    """Register + ingest synchronously. Convenience for callers/tests that want
    the document fully processed on return (bypasses the async dispatch)."""
    document, action = register_document(
        owner, data, source_filename=source_filename, title=title,
        author=author, doc_date=doc_date, mime_type=mime_type,
    )
    if action != "unchanged":
        ingest_document(document, data)
    return document, action


def ingest_document(document: Document, data: bytes) -> Document:
    """Run the full pipeline for one document, updating its status as it goes."""
    from core.services.parsers import extract_metadata, parse_document

    cfg = settings.RAG
    document.status = Document.Status.PROCESSING
    document.save(update_fields=["status"])

    try:
        text = parse_document(
            data, mime_type=document.mime_type, filename=document.source_filename
        )
        document.char_count = len(text)

        # Auto-extract document-level metadata, without clobbering client values.
        # Persist immediately so metadata survives even if later stages fail.
        extracted = extract_metadata(
            data, mime_type=document.mime_type, filename=document.source_filename
        )
        if not document.author and extracted.get("author"):
            document.author = extracted["author"]
        if document.doc_date is None and extracted.get("doc_date"):
            document.doc_date = extracted["doc_date"]
        document.save(update_fields=["char_count", "author", "doc_date"])

        chunks = chunking.chunk_text(
            text,
            max_tokens=cfg["CHUNK_TOKENS"],
            overlap_tokens=cfg["CHUNK_OVERLAP"],
            base_metadata={"document_id": document.id},
        )
        if not chunks:
            raise ValueError("Document produced no extractable text")

        embeddings = get_provider().embed([c.text for c in chunks])

        with transaction.atomic():
            document.chunks.all().delete()
            Chunk.objects.bulk_create(
                [
                    Chunk(
                        document=document,
                        ordinal=i,
                        text=c.text,
                        token_count=c.token_count,
                        embedding=emb,
                        metadata=c.metadata,
                    )
                    for i, (c, emb) in enumerate(zip(chunks, embeddings))
                ]
            )
            # Populate the full-text search vector for the keyword arm of hybrid search.
            document.chunks.update(
                search_vector=SearchVector("text", config="english")
            )
            document.status = Document.Status.READY
            document.error = ""
            document.save(update_fields=["status", "error"])
    except Exception as exc:  # noqa: BLE001 — surface any failure on the record
        document.status = Document.Status.FAILED
        document.error = str(exc)
        document.save(update_fields=["status", "error"])
        raise

    return document
