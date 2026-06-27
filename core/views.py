from django.contrib.auth import get_user_model
from django.db.models import Avg, Count, Sum
from django.http import StreamingHttpResponse
from rest_framework import generics, status
from rest_framework.authtoken.models import Token
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import Document, QueryLog
from core.serializers import (
    DocumentSerializer,
    QuerySerializer,
    RegisterSerializer,
)
from core.services.ingest import dispatch_ingest, register_document
from core.services.retrieve import answer_query, answer_query_stream

User = get_user_model()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


class RegisterView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        username = serializer.validated_data["username"]
        if User.objects.filter(username=username).exists():
            return Response(
                {"detail": "Username already taken."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        user = User.objects.create_user(
            username=username, password=serializer.validated_data["password"]
        )
        token, _ = Token.objects.get_or_create(user=user)
        return Response({"token": token.key}, status=status.HTTP_201_CREATED)


class DocumentListCreateView(generics.ListCreateAPIView):
    serializer_class = DocumentSerializer
    throttle_scope = "ingest"

    def get_queryset(self):
        return Document.objects.filter(owner=self.request.user)

    def create(self, request, *args, **kwargs):
        upload = request.FILES.get("file")
        if upload is None:
            return Response(
                {"detail": "Multipart field 'file' is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if upload.size and upload.size > MAX_UPLOAD_BYTES:
            return Response(
                {"detail": "File exceeds 20 MB limit."},
                status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )

        data = upload.read()
        document, action = register_document(
            request.user,
            data,
            source_filename=upload.name,
            title=request.data.get("title") or upload.name,
            author=request.data.get("author", ""),
            doc_date=request.data.get("doc_date") or None,
            mime_type=upload.content_type or "",
        )

        if action == "unchanged":
            body = DocumentSerializer(document).data
            body["sync_action"] = action
            return Response(body, status=status.HTTP_200_OK)

        try:
            dispatch_ingest(document, data)
        except Exception as exc:  # noqa: BLE001 — inline/eager failures surface here
            document.refresh_from_db()
            return Response(
                {
                    "document": DocumentSerializer(document).data,
                    "detail": f"Ingestion failed: {exc}",
                },
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        document.refresh_from_db()
        body = DocumentSerializer(document).data
        body["sync_action"] = action
        # Async + still pending -> 202 Accepted; otherwise reflect created/updated.
        if document.status == Document.Status.PENDING:
            code = status.HTTP_202_ACCEPTED
        else:
            code = status.HTTP_201_CREATED if action == "created" else status.HTTP_200_OK
        return Response(body, status=code)


class DocumentDetailView(generics.RetrieveDestroyAPIView):
    serializer_class = DocumentSerializer

    def get_queryset(self):
        return Document.objects.filter(owner=self.request.user)


class QueryView(APIView):
    throttle_scope = "query"

    def post(self, request):
        serializer = QuerySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        result = answer_query(
            request.user,
            serializer.validated_data["question"],
            serializer.validated_data.get("k"),
            filters=serializer.to_filters(),
        )
        return Response(result, status=status.HTTP_200_OK)


class QueryStreamView(APIView):
    throttle_scope = "query"

    def post(self, request):
        serializer = QuerySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        stream = answer_query_stream(
            request.user,
            serializer.validated_data["question"],
            serializer.validated_data.get("k"),
            filters=serializer.to_filters(),
        )
        response = StreamingHttpResponse(stream, content_type="text/event-stream")
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"  # disable proxy buffering for SSE
        return response


class UsageView(APIView):
    """Aggregate cost/latency/token monitoring for the authenticated user."""

    def get(self, request):
        agg = QueryLog.objects.filter(owner=request.user).aggregate(
            queries=Count("id"),
            prompt_tokens=Sum("prompt_tokens"),
            completion_tokens=Sum("completion_tokens"),
            total_cost_usd=Sum("cost_usd"),
            avg_latency_ms=Avg("latency_ms"),
            avg_retrieval_ms=Avg("retrieval_ms"),
            avg_generation_ms=Avg("generation_ms"),
        )
        # Normalise Nones (no queries yet) and round floats for display.
        return Response(
            {
                "queries": agg["queries"] or 0,
                "prompt_tokens": agg["prompt_tokens"] or 0,
                "completion_tokens": agg["completion_tokens"] or 0,
                "total_cost_usd": round(agg["total_cost_usd"] or 0.0, 6),
                "avg_latency_ms": round(agg["avg_latency_ms"] or 0.0, 1),
                "avg_retrieval_ms": round(agg["avg_retrieval_ms"] or 0.0, 1),
                "avg_generation_ms": round(agg["avg_generation_ms"] or 0.0, 1),
            }
        )
