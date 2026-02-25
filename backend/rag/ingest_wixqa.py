"""
Primary ingestion: WixQA KB corpus → chunking → pgvector (Phase 1).

Loads Wix/WixQA wix_kb_corpus, extracts article fields, runs section-aware
parent-document chunking, embeds child chunks, and upserts into documents
with source="wixqa", doc_tier=1.
"""

import asyncio
import os
import uuid
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import Document
from ..db.postgres import async_session_factory
from .chunker import chunk_article, prepare_article_for_chunking
from .embeddings import get_client


SOURCE_WIXQA = "wixqa"
DOC_TIER_KB = 1
HF_DATASET_PRIMARY = "Wix/WixQA"
WIX_KB_SPLIT = "wix_kb_corpus"


def _article_row_to_text_and_meta(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Map a wix_kb_corpus row to (full_article_text, metadata).
    Corpus fields: id, url, contents, article_type.
    """
    article_id = row.get("id") or ""
    url = row.get("url") or ""
    contents = row.get("contents") or ""
    article_type = row.get("article_type") or ""

    # No separate title in wix_kb_corpus; use contents as body
    text, meta = prepare_article_for_chunking(
        title="",
        body=contents,
        url=url,
        category=article_type,
        source_id=article_id,
    )
    return text, meta


# Public for tests
article_row_to_text_and_meta = _article_row_to_text_and_meta


async def _insert_parent(
    session: AsyncSession,
    content: str,
    source_id: str,
    category: str | None,
    meta: Dict[str, Any],
) -> str:
    """Insert one parent document (no embedding); return its id."""
    doc_id = str(uuid.uuid4())
    await session.execute(
        insert(Document.__table__).values(
            id=doc_id,
            content=content,
            parent_id=None,
            embedding=None,
            source=SOURCE_WIXQA,
            doc_tier=DOC_TIER_KB,
            category=category,
            source_id=source_id,
            metadata=meta,
        )
    )
    return doc_id


async def _insert_children(
    session: AsyncSession,
    parent_id: str,
    chunks: List[Tuple[str, Dict[str, Any]]],
    embeddings: List[List[float]],
    source_id_prefix: str,
) -> None:
    """Insert child documents with embeddings and parent_id."""
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings length mismatch")

    for i, ((content, meta), embedding) in enumerate(zip(chunks, embeddings)):
        category = meta.get("category")
        await session.execute(
            insert(Document.__table__).values(
                id=str(uuid.uuid4()),
                content=content,
                parent_id=parent_id,
                embedding=embedding,
                source=SOURCE_WIXQA,
                doc_tier=DOC_TIER_KB,
                category=category,
                source_id=f"{source_id_prefix}-chunk-{i}",
                metadata=meta,
            )
        )


async def ingest_wixqa() -> None:
    """
    End-to-end WixQA KB ingestion:
    - Load Wix/WixQA wix_kb_corpus
    - For each article: build text + metadata, chunk, embed children, insert parent + children
    """
    dataset_name = os.getenv("HF_DATASET_PRIMARY", HF_DATASET_PRIMARY)
    print(f"Loading dataset '{dataset_name}' config '{WIX_KB_SPLIT}' from HuggingFace...")
    ds = load_dataset(dataset_name, WIX_KB_SPLIT, split="train")

    client = get_client()
    total_parents = 0
    total_children = 0

    async with async_session_factory() as session:
        for idx, row in enumerate(ds):
            full_text, meta = _article_row_to_text_and_meta(dict(row))
            if not full_text or not full_text.strip():
                continue

            chunks = chunk_article(full_text, meta)
            if not chunks:
                # Store as single parent-only row (no children)
                source_id = meta.get("source_id") or str(idx)
                parent_id = await _insert_parent(
                    session,
                    content=full_text,
                    source_id=source_id,
                    category=meta.get("category"),
                    meta=meta,
                )
                total_parents += 1
                continue

            # Parent row (full article)
            source_id = meta.get("source_id") or str(idx)
            parent_id = await _insert_parent(
                session,
                content=full_text,
                source_id=source_id,
                category=meta.get("category"),
                meta=meta,
            )
            total_parents += 1

            # Child chunks with embeddings
            child_texts = [c[0] for c in chunks]
            vectors = client.embed_documents(child_texts)
            await _insert_children(
                session,
                parent_id=parent_id,
                chunks=chunks,
                embeddings=vectors,
                source_id_prefix=source_id,
            )
            total_children += len(chunks)

        await session.commit()

    # Validation count
    async with async_session_factory() as session:
        r = await session.execute(
            text("SELECT COUNT(*) FROM documents WHERE source = :s"),
            {"s": SOURCE_WIXQA},
        )
        total = r.scalar() or 0

    print(f"WixQA ingestion complete. documents (source=wixqa) count={total} (parents≈{total_parents}, children={total_children})")


async def _main() -> None:
    await ingest_wixqa()


if __name__ == "__main__":
    asyncio.run(_main())
