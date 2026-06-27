"""End-to-end tracing abstraction.

A query trace captures the full chain: user input -> retrieved chunks -> prompt
-> model output -> token usage / latency, as nested spans. The backend is a
config switch:

  - "none":     no-op (default when no Langfuse keys are configured)
  - "langfuse": emit traces to a (self-hosted) Langfuse instance
  - "memory":   record traces in-process for assertions in tests

Usage:

    tracer = get_tracer()
    with tracer.trace("query", input={...}, metadata={...}) as trace:
        trace.span("retrieval", output={...})
        trace.span("generation", input={...}, output={...})
        trace.end(output={...}, usage={...})
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache

from django.conf import settings

# In-process trace store for the "memory" backend (test introspection only).
RECORDS: list["TraceRecord"] = []


@dataclass
class TraceRecord:
    name: str
    input: dict | None = None
    output: dict | None = None
    usage: dict | None = None
    metadata: dict | None = None
    spans: list[dict] = field(default_factory=list)


class Trace:
    """One trace handle. Subclasses implement the side effects."""

    def span(self, name: str, *, input=None, output=None, metadata=None):
        pass

    def end(self, *, output=None, usage=None):
        pass


class _NoopTrace(Trace):
    pass


class _MemoryTrace(Trace):
    def __init__(self, record: TraceRecord):
        self._record = record

    def span(self, name, *, input=None, output=None, metadata=None):
        self._record.spans.append(
            {"name": name, "input": input, "output": output, "metadata": metadata}
        )

    def end(self, *, output=None, usage=None):
        self._record.output = output
        self._record.usage = usage


class _LangfuseTrace(Trace):
    def __init__(self, client, trace):
        self._client = client
        self._trace = trace

    def span(self, name, *, input=None, output=None, metadata=None):
        self._trace.span(name=name, input=input, output=output, metadata=metadata)

    def end(self, *, output=None, usage=None):
        self._trace.update(output=output, metadata={"usage": usage} if usage else None)


class Tracer:
    @contextmanager
    def trace(self, name, *, input=None, metadata=None):
        yield self._begin(name, input=input, metadata=metadata)

    def _begin(self, name, *, input=None, metadata=None) -> Trace:
        raise NotImplementedError


class NoopTracer(Tracer):
    def _begin(self, name, *, input=None, metadata=None):
        return _NoopTrace()


class MemoryTracer(Tracer):
    def _begin(self, name, *, input=None, metadata=None):
        record = TraceRecord(name=name, input=input, metadata=metadata)
        RECORDS.append(record)
        return _MemoryTrace(record)


class LangfuseTracer(Tracer):
    def __init__(self):
        from langfuse import Langfuse

        cfg = settings.RAG
        self._client = Langfuse(
            public_key=cfg["LANGFUSE_PUBLIC_KEY"],
            secret_key=cfg["LANGFUSE_SECRET_KEY"],
            host=cfg["LANGFUSE_HOST"],
        )

    def _begin(self, name, *, input=None, metadata=None):
        trace = self._client.trace(name=name, input=input, metadata=metadata)
        return _LangfuseTrace(self._client, trace)


def _resolve_backend() -> str:
    cfg = settings.RAG
    backend = cfg["TRACING"].lower()
    if backend == "auto":
        keys = cfg["LANGFUSE_PUBLIC_KEY"] and cfg["LANGFUSE_SECRET_KEY"]
        return "langfuse" if keys else "none"
    return backend


@lru_cache(maxsize=1)
def get_tracer() -> Tracer:
    backend = _resolve_backend()
    if backend == "langfuse":
        return LangfuseTracer()
    if backend == "memory":
        return MemoryTracer()
    return NoopTracer()
