"""Celery tasks: heavy ingestion offloaded from the request thread."""
from celery import shared_task

from core.models import Document
from core.services.ingest import ingest_document


@shared_task(bind=True, max_retries=2, default_retry_delay=10)
def ingest_document_task(self, document_id: int):
    """Parse, chunk, embed and index a previously registered document."""
    document = Document.objects.get(id=document_id)
    with document.source_file.open("rb") as fh:
        data = fh.read()
    ingest_document(document, data)
    return {"document_id": document_id, "status": document.status}
