from django.conf import settings
from django.db import models
from pgvector.django import HnswIndex, VectorField

EMBEDDING_DIM = settings.RAG["EMBEDDING_DIM"]


class Document(models.Model):
    """An ingested source document owned by a user."""

    class Status(models.TextChoices):
        PENDING = "pending"
        PROCESSING = "processing"
        READY = "ready"
        FAILED = "failed"

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="documents"
    )
    title = models.CharField(max_length=512)
    source_filename = models.CharField(max_length=512, blank=True)
    mime_type = models.CharField(max_length=128, blank=True)
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.PENDING
    )
    char_count = models.PositiveIntegerField(default=0)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.title} ({self.status})"


class Chunk(models.Model):
    """A token-bounded slice of a document with its embedding vector."""

    document = models.ForeignKey(
        Document, on_delete=models.CASCADE, related_name="chunks"
    )
    ordinal = models.PositiveIntegerField()
    text = models.TextField()
    token_count = models.PositiveIntegerField(default=0)
    embedding = VectorField(dimensions=EMBEDDING_DIM)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["document_id", "ordinal"]
        constraints = [
            models.UniqueConstraint(
                fields=["document", "ordinal"],
                name="core_chunk_document_id_ordinal_uniq",
            )
        ]
        indexes = [
            HnswIndex(
                name="chunk_embedding_hnsw",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            )
        ]

    def __str__(self):
        return f"Chunk {self.ordinal} of doc {self.document_id}"


class QueryLog(models.Model):
    """Audit + observability record for each /query call."""

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="queries"
    )
    question = models.TextField()
    retrieved_chunk_ids = models.JSONField(default=list)
    latency_ms = models.PositiveIntegerField(default=0)
    prompt_tokens = models.PositiveIntegerField(default=0)
    completion_tokens = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
