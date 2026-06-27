from django.contrib.auth import get_user_model
from rest_framework import generics, status
from rest_framework.authtoken.models import Token
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import Document
from core.serializers import (
    DocumentSerializer,
    QuerySerializer,
    RegisterSerializer,
)
from core.services.ingest import ingest_document
from core.services.retrieve import answer_query

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
        document = Document.objects.create(
            owner=request.user,
            title=request.data.get("title") or upload.name,
            source_filename=upload.name,
            mime_type=upload.content_type or "",
        )
        try:
            ingest_document(document, data)
        except Exception as exc:  # noqa: BLE001 — report ingestion failure to client
            return Response(
                {
                    "document": DocumentSerializer(document).data,
                    "detail": f"Ingestion failed: {exc}",
                },
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )
        document.refresh_from_db()
        return Response(
            DocumentSerializer(document).data, status=status.HTTP_201_CREATED
        )


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
        )
        return Response(result, status=status.HTTP_200_OK)
