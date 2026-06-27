import pytest

from core.services.parsers import UnsupportedDocumentError, parse_document


def test_parse_plain_text_by_mime():
    out = parse_document(b"hello there", mime_type="text/plain")
    assert out == "hello there"


def test_parse_markdown_by_extension():
    out = parse_document(b"# Title\n\nbody", filename="notes.md")
    assert "Title" in out


def test_unsupported_raises():
    with pytest.raises(UnsupportedDocumentError):
        parse_document(b"\x00\x01", mime_type="application/octet-stream", filename="x.bin")


def test_mime_with_charset_param():
    out = parse_document(b"data", mime_type="text/plain; charset=utf-8")
    assert out == "data"
