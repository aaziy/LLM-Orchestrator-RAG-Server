#!/usr/bin/env python
"""Retrieval-quality evaluation harness.

Ingests a small labeled fixture, runs each question through the retriever, and
reports recall@k and Mean Reciprocal Rank (MRR). Use it to tune CHUNK_TOKENS,
CHUNK_OVERLAP, and TOP_K against evidence rather than guesswork.

    python eval/run.py            # uses configured RAG_PROVIDER
    RAG_PROVIDER=fake python eval/run.py   # offline, deterministic

Requires a running pgvector database (the same one the app uses).
"""
import json
import os
import sys
from pathlib import Path

import django

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.conf import settings  # noqa: E402
from django.contrib.auth import get_user_model  # noqa: E402

from core.models import Document  # noqa: E402
from core.services.ingest import ingest_document  # noqa: E402
from core.services.retrieve import retrieve_for_answer  # noqa: E402

FIXTURES = json.loads((Path(__file__).parent / "fixtures.json").read_text())
EVAL_USERNAME = "__eval__"


def setup_corpus():
    User = get_user_model()
    user, _ = User.objects.get_or_create(username=EVAL_USERNAME)
    # Fresh corpus each run for reproducibility.
    Document.objects.filter(owner=user).delete()

    doc_by_key = {}
    for key, text in FIXTURES["documents"].items():
        doc = Document.objects.create(
            owner=user,
            title=key,
            source_filename=f"{key}.txt",
            mime_type="text/plain",
        )
        ingest_document(doc, text.encode())
        doc_by_key[key] = doc
    return user, doc_by_key


def evaluate(k: int):
    user, doc_by_key = setup_corpus()
    hits = 0
    reciprocal_ranks = []
    rows = []

    for item in FIXTURES["questions"]:
        retrieved = retrieve_for_answer(user, item["question"], k)
        rank = None
        for i, r in enumerate(retrieved, start=1):
            same_doc = r.chunk.document_id == doc_by_key[item["doc"]].id
            contains = item["expect"].lower() in r.chunk.text.lower()
            if same_doc and contains:
                rank = i
                break
        if rank:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
        rows.append((item["question"], rank))

    n = len(FIXTURES["questions"])
    recall = hits / n
    mrr = sum(reciprocal_ranks) / n
    return recall, mrr, rows, n


def main():
    k = settings.RAG["TOP_K"]
    print(f"Provider: {settings.RAG['PROVIDER']}  |  k={k}  "
          f"|  hybrid={settings.RAG['HYBRID']}  rerank={settings.RAG['RERANK']}"
          f"  |  chunk_tokens={settings.RAG['CHUNK_TOKENS']}  "
          f"overlap={settings.RAG['CHUNK_OVERLAP']}")
    print("-" * 64)
    recall, mrr, rows, n = evaluate(k)
    for question, rank in rows:
        status = f"rank {rank}" if rank else "MISS"
        print(f"  [{status:>7}]  {question}")
    print("-" * 64)
    print(f"  Questions : {n}")
    print(f"  Recall@{k} : {recall:.3f}")
    print(f"  MRR       : {mrr:.3f}")


if __name__ == "__main__":
    main()
