from django.contrib import admin

from core.models import Chunk, Document, QueryLog


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "owner", "status", "char_count", "created_at")
    list_filter = ("status",)
    search_fields = ("title", "source_filename")


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ("id", "document", "ordinal", "token_count")
    list_filter = ("document",)


@admin.register(QueryLog)
class QueryLogAdmin(admin.ModelAdmin):
    list_display = ("id", "owner", "latency_ms", "prompt_tokens", "created_at")
