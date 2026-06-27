from django.conf import settings
from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVectorField
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
    author = models.CharField(max_length=512, blank=True)
    doc_date = models.DateField(null=True, blank=True)
    source_filename = models.CharField(max_length=512, blank=True)
    mime_type = models.CharField(max_length=128, blank=True)
    content_hash = models.CharField(max_length=64, blank=True, db_index=True)
    source_file = models.FileField(upload_to="uploads/", null=True, blank=True)
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
    search_vector = SearchVectorField(null=True)
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
            ),
            GinIndex(fields=["search_vector"], name="chunk_search_vector_gin"),
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
    retrieval_ms = models.PositiveIntegerField(default=0)
    generation_ms = models.PositiveIntegerField(default=0)
    prompt_tokens = models.PositiveIntegerField(default=0)
    completion_tokens = models.PositiveIntegerField(default=0)
    cost_usd = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
