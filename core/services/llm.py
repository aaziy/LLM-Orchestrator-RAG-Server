"""Provider abstraction for embeddings and chat generation.

A single interface lets the rest of the app stay ignorant of the backend. Today
there is one live implementation (OpenAI) plus a deterministic ``fake`` provider
used for offline development and tests. Swapping in a local Llama server later is
a matter of adding another class — no caller changes.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache

from django.conf import settings


@dataclass
class Generation:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Provider:
    """Interface for embedding + chat backends."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def generate(self, system: str, user: str) -> Generation:
        raise NotImplementedError


class OpenAIProvider(Provider):
    def __init__(self):
        from openai import OpenAI

        cfg = settings.RAG
        self._client = OpenAI(api_key=cfg["OPENAI_API_KEY"])
        self._embedding_model = cfg["EMBEDDING_MODEL"]
        self._chat_model = cfg["CHAT_MODEL"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(
            model=self._embedding_model, input=texts
        )
        return [item.embedding for item in resp.data]

    def generate(self, system: str, user: str) -> Generation:
        resp = self._client.chat.completions.create(
            model=self._chat_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        usage = resp.usage
        return Generation(
            text=resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
        )


class FakeProvider(Provider):
    """Deterministic, dependency-free provider for offline dev and tests.

    Embeddings are hashed bag-of-words vectors: identical text yields identical
    vectors and similar text yields similar vectors, which is enough to exercise
    the full retrieval path without network access.
    """

    def __init__(self, dim: int | None = None):
        self._dim = dim or settings.RAG["EMBEDDING_DIM"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def generate(self, system: str, user: str) -> Generation:
        # Echo back the first retrieved context line so tests can assert grounding.
        return Generation(
            text="[fake-answer] " + user[:200],
            prompt_tokens=len(user.split()),
            completion_tokens=8,
        )


@lru_cache(maxsize=1)
def get_provider() -> Provider:
    name = settings.RAG["PROVIDER"].lower()
    if name == "openai":
        return OpenAIProvider()
    if name == "fake":
        return FakeProvider()
    raise ValueError(f"Unknown RAG_PROVIDER: {name!r}")
