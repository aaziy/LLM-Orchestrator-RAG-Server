"""Token-aware recursive chunking.

This is the single most important quality lever in the RAG pipeline. We split on
the largest natural boundary that keeps a chunk under the token budget — paragraphs
first, then sentences, then a hard token cut as a last resort — and carry a small
token overlap between consecutive chunks so context is not severed mid-thought.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken

# A single encoder shared across calls; cl100k_base matches OpenAI embedding models.
_ENCODER = tiktoken.get_encoding("cl100k_base")

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


@dataclass
class Chunk:
    text: str
    token_count: int
    metadata: dict = field(default_factory=dict)


def _split_to_units(text: str, max_tokens: int) -> list[str]:
    """Break text into units that each fit within max_tokens.

    Paragraphs are preferred; over-long paragraphs fall back to sentences, and
    over-long sentences fall back to a hard token-window cut.
    """
    units: list[str] = []
    for para in _PARAGRAPH_RE.split(text):
        para = para.strip()
        if not para:
            continue
        if count_tokens(para) <= max_tokens:
            units.append(para)
            continue
        for sentence in _SENTENCE_RE.split(para):
            sentence = sentence.strip()
            if not sentence:
                continue
            if count_tokens(sentence) <= max_tokens:
                units.append(sentence)
            else:
                units.extend(_hard_split(sentence, max_tokens))
    return units


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """Last-resort split of a single oversized unit on token boundaries."""
    token_ids = _ENCODER.encode(text)
    pieces = []
    for start in range(0, len(token_ids), max_tokens):
        window = token_ids[start : start + max_tokens]
        pieces.append(_ENCODER.decode(window).strip())
    return [p for p in pieces if p]


def chunk_text(
    text: str,
    *,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
    base_metadata: dict | None = None,
) -> list[Chunk]:
    """Greedily pack natural units into chunks, with token overlap between them."""
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    base_metadata = base_metadata or {}
    units = _split_to_units(text, max_tokens)
    chunks: list[Chunk] = []

    current: list[str] = []
    current_tokens = 0

    def flush():
        nonlocal current, current_tokens
        if not current:
            return
        body = "\n\n".join(current)
        chunks.append(
            Chunk(
                text=body,
                token_count=count_tokens(body),
                metadata={**base_metadata, "ordinal": len(chunks)},
            )
        )
        # Seed the next chunk with a token-bounded tail of this one for overlap.
        if overlap_tokens > 0:
            tail_ids = _ENCODER.encode(body)[-overlap_tokens:]
            tail = _ENCODER.decode(tail_ids).strip()
            current = [tail] if tail else []
            current_tokens = count_tokens(tail) if tail else 0
        else:
            current = []
            current_tokens = 0

    for unit in units:
        unit_tokens = count_tokens(unit)
        if current and current_tokens + unit_tokens > max_tokens:
            flush()
        current.append(unit)
        current_tokens += unit_tokens

    # Final flush without seeding a new overlap chunk.
    if current:
        body = "\n\n".join(current)
        chunks.append(
            Chunk(
                text=body,
                token_count=count_tokens(body),
                metadata={**base_metadata, "ordinal": len(chunks)},
            )
        )

    return chunks
