"""Unit tests for the token-aware chunker — the core quality lever."""
from core.services import chunking
from core.services.chunking import chunk_text, count_tokens


def test_short_text_is_single_chunk():
    chunks = chunk_text("Hello world.", max_tokens=500, overlap_tokens=50)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert chunks[0].metadata["ordinal"] == 0


def test_respects_max_tokens():
    text = "\n\n".join(f"Paragraph number {i} with some filler words." for i in range(200))
    chunks = chunk_text(text, max_tokens=60, overlap_tokens=10)
    assert len(chunks) > 1
    # Allow a small margin for the seeded overlap tail riding along.
    for c in chunks:
        assert c.token_count <= 60 + 10


def test_overlap_between_consecutive_chunks():
    paras = [f"Sentence {i} carries unique token zeta{i}." for i in range(40)]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, max_tokens=40, overlap_tokens=12)
    assert len(chunks) >= 2
    # The tail tokens of chunk[0] should reappear at the head of chunk[1].
    tail = chunks[0].text.split()[-3:]
    assert any(tok in chunks[1].text for tok in tail)


def test_ordinals_are_sequential():
    text = "\n\n".join(f"Para {i} text here." for i in range(50))
    chunks = chunk_text(text, max_tokens=30, overlap_tokens=5)
    assert [c.metadata["ordinal"] for c in chunks] == list(range(len(chunks)))


def test_oversized_single_sentence_is_hard_split():
    giant = "word " * 1000  # one whitespace-joined blob, no sentence breaks
    chunks = chunk_text(giant, max_tokens=50, overlap_tokens=0)
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c.text) <= 50


def test_overlap_must_be_smaller_than_max():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("x", max_tokens=10, overlap_tokens=10)


def test_count_tokens_nonzero():
    assert chunking.count_tokens("the quick brown fox") > 0
