from rest_framework import serializers

from core.models import Document


class DocumentSerializer(serializers.ModelSerializer):
    chunk_count = serializers.SerializerMethodField()

    class Meta:
        model = Document
        fields = [
            "id",
            "title",
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


class RegisterSerializer(serializers.Serializer):
    username = serializers.CharField(min_length=3, max_length=150)
    password = serializers.CharField(min_length=6, write_only=True)
