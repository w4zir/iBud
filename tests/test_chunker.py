"""Unit tests for WixQA chunker: section-aware and parent-document chunking."""

import pytest

from backend.rag.chunker import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    chunk_article,
    prepare_article_for_chunking,
)


def test_prepare_article_for_chunking_builds_text_and_metadata():
    text, meta = prepare_article_for_chunking(
        title="How to track orders",
        body="Go to Orders page and click Track.",
        url="https://help.example/orders",
        category="shipping",
        source_id="art-1",
    )
    assert "How to track orders" in text
    assert "Go to Orders page" in text
    assert meta["category"] == "shipping"
    assert meta["url"] == "https://help.example/orders"
    assert meta["source_id"] == "art-1"


def test_prepare_article_for_chunking_empty_title_uses_body_only():
    text, meta = prepare_article_for_chunking(title="", body="Body only.", url="", category="", source_id="")
    assert text.strip() == "Body only."
    assert meta == {}


def test_chunk_article_empty_text_returns_empty_list():
    assert chunk_article("", {"category": "x"}) == []
    assert chunk_article("   \n  ", {"category": "x"}) == []


def test_chunk_article_short_text_returns_single_chunk():
    text, meta = "Short article.", {"category": "test"}
    chunks = chunk_article(text, meta)
    assert len(chunks) == 1
    assert chunks[0][0] == "Short article."
    assert chunks[0][1]["category"] == "test"


def test_chunk_article_long_plain_text_splits_and_preserves_metadata():
    long_text = "Intro. " + ("x" * (CHUNK_SIZE * 2))
    meta = {"category": "shipping", "source_id": "art-99"}
    chunks = chunk_article(long_text, meta, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    assert len(chunks) >= 2
    for _text, chunk_meta in chunks:
        assert chunk_meta["category"] == "shipping"
        assert chunk_meta["source_id"] == "art-99"


def test_chunk_article_custom_size_and_overlap():
    text = "A" * 500
    chunks = chunk_article(text, {}, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2


def test_chunk_article_html_like_uses_header_aware_split():
    html = "<h1>Title</h1><p>Para one.</p><h2>Section</h2><p>Para two.</p>"
    chunks = chunk_article(html, {"category": "kb"})
    assert len(chunks) >= 1
    for _text, m in chunks:
        assert m.get("category") == "kb"
