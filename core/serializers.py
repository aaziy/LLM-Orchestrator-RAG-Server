from rest_framework import serializers

from core.models import Document


class DocumentSerializer(serializers.ModelSerializer):
    chunk_count = serializers.SerializerMethodField()

    class Meta:
        model = Document
        fields = [
            "id",
            "title",
            "author",
            "doc_date",
            "source_filename",
            "mime_type",
            "status",
            "char_count",
            "chunk_count",
            "error",
            "created_at",
        ]
        read_only_fields = fields

    def get_chunk_count(self, obj) -> int:
        return obj.chunks.count()


class QuerySerializer(serializers.Serializer):
    question = serializers.CharField(min_length=1, trim_whitespace=True)
    k = serializers.IntegerField(required=False, min_value=1, max_value=20)
    document_ids = serializers.ListField(
        child=serializers.IntegerField(), required=False
    )
    author = serializers.CharField(required=False)
    date_from = serializers.DateField(required=False)
    date_to = serializers.DateField(required=False)

    def to_filters(self) -> dict:
        data = self.validated_data
        return {
            "document_ids": data.get("document_ids"),
            "author": data.get("author"),
            "date_from": data.get("date_from"),
            "date_to": data.get("date_to"),
        }


class RegisterSerializer(serializers.Serializer):
    username = serializers.CharField(min_length=3, max_length=150)
    password = serializers.CharField(min_length=6, write_only=True)
