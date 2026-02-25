"""Tests for WixQA article transform and row mapping (ingest_wixqa)."""

from backend.rag.ingest_wixqa import article_row_to_text_and_meta
from backend.rag.chunker import chunk_article, prepare_article_for_chunking


def test_wixqa_row_to_text_and_meta_maps_corpus_fields():
    row = {
        "id": "art-42",
        "url": "https://wix.com/help/foo",
        "contents": "Full article body here.",
        "article_type": "article",
    }
    text, meta = article_row_to_text_and_meta(row)
    assert text.strip() == "Full article body here."
    assert meta.get("category") == "article"
    assert meta.get("url") == "https://wix.com/help/foo"
    assert meta.get("source_id") == "art-42"


def test_wixqa_row_handles_missing_fields():
    text, meta = article_row_to_text_and_meta({})
    assert text == ""
    assert meta.get("category") is None or meta.get("category") == ""


def test_wixqa_row_then_chunk_article_produces_child_chunks():
    row = {
        "id": "art-1",
        "url": "",
        "contents": "Short answer.",
        "article_type": "known_issue",
    }
    full_text, meta = article_row_to_text_and_meta(row)
    chunks = chunk_article(full_text, meta)
    assert len(chunks) == 1
    assert chunks[0][1].get("source_id") == "art-1"
    assert chunks[0][1].get("category") == "known_issue"


def test_prepare_article_for_chunking_integration():
    text, meta = prepare_article_for_chunking(
        title="Returns policy",
        body="You can return within 30 days.",
        url="https://example/returns",
        category="returns",
        source_id="ret-1",
    )
    assert "Returns policy" in text
    assert "30 days" in text
    chunks = chunk_article(text, meta)
    assert len(chunks) >= 1
    assert all(m.get("category") == "returns" for _, m in chunks)
