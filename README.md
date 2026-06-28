# LLM Orchestrator & RAG Server

A backend **control plane** for Retrieval-Augmented Generation. It ingests
documents, chunks them, generates vector embeddings, stores them in
**PostgreSQL + pgvector**, and answers questions grounded in the retrieved
context — with hybrid search, cross-encoder reranking, response streaming,
asynchronous ingestion, end-to-end tracing, and cost/latency monitoring.

The model backend (embeddings + chat) and the reranker sit behind small provider
interfaces, so today's external APIs can be swapped for local models without
touching the pipeline.

---

## Tech stack

| Layer | Technology | Version | Role in this project |
| --- | --- | --- | --- |
| Web framework | **Django** | 5.0.6 | HTTP layer, ORM, auth, migrations, admin, template serving |
| API | **Django REST Framework** | 3.15.1 | Endpoints, serializers, token auth, scoped rate limiting |
| Database | **PostgreSQL** | 16 | Single store for relational data **and** vectors |
| Vector search | **pgvector** | 0.2.5 (ext + `pgvector` Python) | Embedding column + cosine similarity (HNSW index) |
| Keyword search | **PostgreSQL full-text** (`tsvector`/GIN) | built-in | BM25-style lexical arm of hybrid retrieval |
| DB driver | **psycopg** | 3.1.19 | PostgreSQL connectivity |
| Async tasks | **Celery** + **Redis** | 5.4.0 / 5.0.4 | Offload parse→chunk→embed to a worker; Redis is broker/result backend |
| Embeddings + LLM | **OpenAI** SDK | 1.30.5 | `text-embedding-3-small` + `gpt-4o-mini` (pluggable; `fake` provider for offline/tests) |
| Reranking | **sentence-transformers** (cross-encoder) | 3.0.1 | `ms-marco-MiniLM-L-6-v2`; `cohere` 5.5.8 and `fake` are drop-in alternates |
| Tokenization | **tiktoken** | 0.7.0 | Token-aware chunking + token accounting |
| Document parsing | **pypdf**, **python-docx** | 4.2.0 / 1.1.2 | Extract text + metadata from PDF / DOCX (plus txt/md) |
| Observability | **Langfuse** | 2.36.0 | End-to-end trace: input → chunks → prompt → output → tokens |
| Config | **python-dotenv** | 1.0.1 | `.env` loading |
| Server | **Gunicorn** | 22.0.0 | WSGI server in the container |
| Packaging / runtime | **Docker** + Docker Compose | — | Isolates Python/native deps; orchestrates db + redis + web + worker (+ langfuse) |
| Frontend | Vanilla HTML/CSS/JS | — | Single-file test console served by Django at `/` |
| Tests | **pytest** + **pytest-django** | 8.2.2 / 4.8.0 | 56 tests, fully offline/deterministic |

---

## Where each technology is used

### Django — the application backbone

- **HTTP & routing:** [config/urls.py](config/urls.py) mounts the API under `/api/`
  and serves the test console at `/`; [core/urls.py](core/urls.py) defines the routes.
- **ORM & data model:** [core/models.py](core/models.py) — `Document`, `Chunk`,
  `QueryLog`. Models declare the pgvector and full-text **indexes** directly.
- **Migrations:** [core/migrations/](core/migrations/) — schema history, including
  enabling the pgvector extension and creating the HNSW / GIN indexes.
- **Auth:** Django's user model + DRF token auth. Registration/login in
  [core/views.py](core/views.py) (`RegisterView`) and `/api/auth/token`.
- **REST API & throttling:** [core/views.py](core/views.py) +
  [core/serializers.py](core/serializers.py); DRF settings (token auth,
  `IsAuthenticated`, `ScopedRateThrottle` for `query`/`ingest`) live in
  [config/settings.py](config/settings.py).
- **File handling:** `Document.source_file` (`FileField`) persists the upload so a
  Celery worker can read it back.
- **Templates:** the console is a Django template,
  [core/templates/console.html](core/templates/console.html).
- **Settings split:** [config/settings.py](config/settings.py) (runtime) and
  [config/settings_test.py](config/settings_test.py) (deterministic test config).

### PostgreSQL — relational data *and* vectors in one store

- **Connection:** the `postgresql` engine in [config/settings.py](config/settings.py)
  (`DATABASES`), driven by psycopg 3.
- **pgvector extension:** enabled via `VectorExtension()` in
  [core/migrations/0001_initial.py](core/migrations/0001_initial.py).
- **Vector column + ANN index:** `Chunk.embedding` is a `VectorField(1536)` with an
  **HNSW** index (`vector_cosine_ops`) — see [core/models.py](core/models.py).
- **Cosine similarity search:** `CosineDistance(...)` ordering in
  [core/services/retrieve.py](core/services/retrieve.py).
- **Keyword search (no extra engine):** `Chunk.search_vector` is a
  `SearchVectorField` with a **GIN** index, populated on ingest and queried with
  `SearchQuery`/`SearchRank` (Django's `django.contrib.postgres`) — the lexical arm
  of hybrid retrieval.
- **Metadata filtering:** author/date/document filters are plain ORM `WHERE` clauses
  applied **before** the similarity search.
- **Monitoring store:** `QueryLog` rows hold per-query latency, tokens, and cost,
  aggregated by `/api/usage`.

> Why one database: embeddings live next to relational data, so vector similarity,
> full-text keyword search, and metadata filters all run in SQL against a single
> source of truth — no separate vector database to sync.

### Docker — reproducible, isolated runtime

- **Image:** [Dockerfile](Dockerfile) (Python 3.11-slim) installs dependencies and,
  on start, runs `migrate` then Gunicorn.
- **Orchestration:** [docker-compose.yml](docker-compose.yml) defines:
  - `db` — `pgvector/pgvector:pg16` (Postgres with the extension pre-built),
    healthchecked, persistent volume.
  - `redis` — Celery broker/result backend.
  - `web` — Django/Gunicorn API + console.
  - `worker` — Celery worker (shares a `media` volume with `web` to read uploads).
  - `langfuse` + `langfuse-db` — **opt-in** via the `observability` profile.
- **Isolation:** the heavy ML dependency (cross-encoder/torch) is contained in the
  image, keeping local environments clean.

---

## Architecture

```text
            ┌──────────────── Django + DRF (web) ────────────────┐
Client ───▶ │  /auth   /documents   /query   /query/stream  /usage │
            └───────┬───────────────────────┬──────────────────────┘
        register(hash)│ save file            │ embed question
                      ▼                       ▼
                 enqueue ─▶ Redis ─▶ Celery worker        ┌─ vector kNN (pgvector)   ┐
                                     parse→chunk→embed     ├─ keyword (Postgres FTS)  ├─ RRF
                                     store vector+tsvector └─ then cross-encoder rerank┘
                                                                      │
                                                        grounded prompt → LLM
                                                        → answer + citations
                                                        (traced to Langfuse;
                                                         cost/latency → QueryLog)

PostgreSQL + pgvector:  documents · chunks(embedding + search_vector) · query_logs
```

---

## Capability map

| Area | Capability | Implementation |
| --- | --- | --- |
| Ingestion | Recursive, token-aware chunking + overlap | [chunking.py](core/services/chunking.py) |
| Ingestion | Metadata (author/date/filename) + filtered retrieval | [parsers.py](core/services/parsers.py), [retrieve.py](core/services/retrieve.py) |
| Ingestion | Content-hash auto-sync (skip / re-embed / delete) | [ingest.py](core/services/ingest.py) |
| Retrieval | Hybrid vector + keyword via reciprocal rank fusion | [retrieve.py](core/services/retrieve.py) |
| Retrieval | Cross-encoder reranking (local / cohere / fake) | [rerank.py](core/services/rerank.py) |
| Retrieval | Top-K candidate limit | [serializers.py](core/serializers.py) |
| Generation | Source citations on every answer | [retrieve.py](core/services/retrieve.py) |
| Generation | Grounding guardrail ("insufficient context") | `SYSTEM_PROMPT`, [retrieve.py](core/services/retrieve.py) |
| Generation | SSE streaming | `QueryStreamView`, [views.py](core/views.py) |
| Infra | End-to-end tracing | [tracing.py](core/services/tracing.py) |
| Infra | Async ingestion | [tasks.py](core/tasks.py), [config/celery.py](config/celery.py) |
| Infra | Cost + latency monitoring | [pricing.py](core/services/pricing.py), `/api/usage` |

---

## Quick start (Docker)

```bash
cp .env.example .env          # set OPENAI_API_KEY, or RAG_PROVIDER=fake for offline
docker compose up --build     # db + redis + web + worker; migrations run automatically
```

- Test console: <http://localhost:8000/>
- API base: <http://localhost:8000/api>

Add Langfuse tracing UI (<http://localhost:3000>):

```bash
docker compose --profile observability up --build
```

## Run locally without Docker (lightweight, offline)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

docker run -d --name rag-db -e POSTGRES_DB=rag -e POSTGRES_USER=rag \
  -e POSTGRES_PASSWORD=rag -p 5432:5432 pgvector/pgvector:pg16

export RAG_PROVIDER=fake RERANKER=fake ASYNC_INGEST=0 TRACING=none
python manage.py migrate
python manage.py runserver        # console at http://localhost:8000/
```

`RAG_PROVIDER=fake` and `RERANKER=fake` run the whole pipeline with no API key and
no ML model; `ASYNC_INGEST=0` ingests inline so no Redis/worker is needed. For real
answers, set `RAG_PROVIDER=openai` and `OPENAI_API_KEY`.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/auth/register` | Create a user, returns a token |
| POST | `/api/auth/token` | Obtain a token |
| POST | `/api/documents` | Upload (`file`, `title?`, `author?`, `doc_date?`) → register + ingest |
| GET | `/api/documents` | List documents + status |
| GET / DELETE | `/api/documents/{id}` | Detail / delete (cascades chunks) |
| POST | `/api/query` | `{question, k?, document_ids?, author?, date_from?, date_to?}` |
| POST | `/api/query/stream` | Same body, streamed as SSE (`citations` → `token` → `done`) |
| GET | `/api/usage` | Aggregate tokens / cost / latency for the user |

Uploads return `sync_action` (`created` / `updated` / `unchanged`) with HTTP
`201`/`200`, or `202` while async ingestion is still running.

### Example

```bash
TOKEN=$(curl -s -X POST localhost:8000/api/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"username":"demo","password":"secret123"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')

curl -s -X POST localhost:8000/api/documents \
  -H "Authorization: Token $TOKEN" \
  -F file=@whitepaper.pdf -F title="Whitepaper" -F author="Jane Doe"

curl -s -X POST localhost:8000/api/query \
  -H "Authorization: Token $TOKEN" -H 'Content-Type: application/json' \
  -d '{"question":"What problem does this solve?"}'
```

Answers are grounded (the system prompt forbids using anything outside retrieved
passages and refuses when context is insufficient) and always include `citations`.

---

## Test console

A single self-contained page ([core/templates/console.html](core/templates/console.html))
served by Django at `/` — no build step, no CDN, works offline, same-origin with the
API. It exercises every capability: auth, upload with metadata, a live document table
that auto-refreshes during async ingestion, query with filters + top-K, an SSE stream
toggle, citations with vector/rerank scores, and the usage/cost dashboard.

---

## Retrieval evaluation

[eval/run.py](eval/run.py) ingests a labeled fixture and reports **Recall@k** and
**MRR** through the real pipeline (hybrid + rerank):

```bash
RAG_PROVIDER=fake RERANKER=fake python eval/run.py
```

On the bundled fixture, enabling hybrid + reranking improves MRR from ~0.80
(vector-only) to ~0.92 at Recall@5 = 1.0.

---

## Tests

```bash
pytest        # 56 tests: chunking, parsing, sync, hybrid, rerank, streaming,
              #           async (Celery eager), tracing, monitoring, API integration
```

[config/settings_test.py](config/settings_test.py) pins deterministic backends, runs
Celery eagerly, and isolates media — no network, broker, or ML model required. Point
at a pgvector database via `POSTGRES_HOST` / `POSTGRES_PORT`.

---

## Configuration

All tunables are environment variables (see [.env.example](.env.example)):
provider/model, `CHUNK_TOKENS` / `CHUNK_OVERLAP`, `TOP_K`, hybrid
(`HYBRID`, `CANDIDATE_POOL`, `RRF_K`), reranking
(`RERANK`, `RERANKER`, `RERANK_CANDIDATES`), async (`ASYNC_INGEST`, `REDIS_URL`),
tracing (`TRACING`, `LANGFUSE_*`), and throttles.

---

## Possible next steps

Page-accurate citations, query-result caching, multi-tenant quota dashboards, and a
local Llama backend (drop-in `Provider`).

---

© 2026 Aziq Rauf. All rights reserved.
