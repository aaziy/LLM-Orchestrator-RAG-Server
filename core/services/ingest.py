"""Document ingestion pipeline: parse -> chunk -> embed -> store.

Exposed as a single callable, ``ingest_document``, so it can be invoked inline
today and moved behind a task queue (Celery/RQ) later without changing callers.
"""
from __future__ import annotations

from django.conf import settings
from django.db import transaction

from core.models import Chunk, Document
from core.services import chunking
from core.services.llm import get_provider


def ingest_document(document: Document, data: bytes) -> Document:
    """Run the full pipeline for one document, updating its status as it goes."""
    from core.services.parsers import parse_document

    cfg = settings.RAG
    document.status = Document.Status.PROCESSING
    document.save(update_fields=["status"])

    try:
        text = parse_document(
            data, mime_type=document.mime_type, filename=document.source_filename
        )
        document.char_count = len(text)

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
            document.status = Document.Status.READY
            document.error = ""
            document.save(update_fields=["status", "char_count", "error"])
    except Exception as exc:  # noqa: BLE001 — surface any failure on the record
        document.status = Document.Status.FAILED
        document.error = str(exc)
        document.save(update_fields=["status", "error"])
        raise

    return document
