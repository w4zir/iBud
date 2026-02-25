"""
Section-aware chunking for WixQA KB articles (Phase 1).

Uses HTMLHeaderTextSplitter for structure and RecursiveCharacterTextSplitter
for size limits. Parent-document pattern: callers create a parent row (full article)
and child rows (chunks) with parent_id. Chunk size/overlap are character-based
approximations of 400 / 50 tokens.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter


# Approximate 400 tokens / 50 tokens as characters (plan P1-3)
CHUNK_SIZE = 1600
CHUNK_OVERLAP = 200


def _build_article_text(title: str, body: str) -> str:
    """Combine title and body into a single text for chunking."""
    parts: List[str] = []
    if title:
        parts.append(title.strip())
    if body:
        parts.append(body.strip())
    return "\n\n".join(parts) if parts else ""


def _split_by_headers(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    If text looks like HTML, split by headers and return (chunk_text, metadata) list.
    Otherwise return a single (text, {}) element.
    """
    if "<" not in text or ">" not in text:
        return [(text, {})]

    try:
        splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h1", "h1"), ("h2", "h2"), ("h3", "h3")])
        docs = splitter.split_text(text)
        return [(d.page_content, dict(d.metadata)) for d in docs]
    except Exception:
        return [(text, {})]


def chunk_article(
    full_text: str,
    metadata: Dict[str, Any],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Produce child chunks from a full article for parent-document storage.

    Uses HTMLHeaderTextSplitter when content looks like HTML, then
    RecursiveCharacterTextSplitter to enforce size. Each returned chunk
    gets a copy of the input metadata (caller adds parent_id on insert).

    Returns:
        List of (chunk_text, metadata) with metadata including any header info.
    """
    if not full_text or not full_text.strip():
        return []

    recursive = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    sections = _split_by_headers(full_text)
    result: List[Tuple[str, Dict[str, Any]]] = []

    for section_text, section_meta in sections:
        meta = dict(metadata)
        meta.update(section_meta)
        for chunk in recursive.split_text(section_text):
            if chunk.strip():
                result.append((chunk.strip(), meta))

    return result


def prepare_article_for_chunking(
    title: str = "",
    body: str = "",
    url: str = "",
    category: str = "",
    source_id: str = "",
) -> Tuple[str, Dict[str, Any]]:
    """
    Build (full_article_text, metadata) for one WixQA article.
    Used by ingest_wixqa to pass into chunk_article.
    """
    text = _build_article_text(title, body)
    metadata: Dict[str, Any] = {}
    if category:
        metadata["category"] = category
    if url:
        metadata["url"] = url
    if source_id:
        metadata["source_id"] = source_id
    return text, metadata


__all__ = [
    "chunk_article",
    "prepare_article_for_chunking",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
]
