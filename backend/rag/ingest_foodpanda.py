"""
Ingestion for Foodpanda e-commerce policy markdown docs → chunking → pgvector.

Reads markdown files from data/foodpanda/policy_docs, builds simple metadata,
chunks content, embeds child chunks, and upserts into documents with
source="foodpanda", doc_tier=1, company_id="foodpanda".
"""

from __future__ import annotations

import asyncio
import glob
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import Document
from ..db.postgres import async_session_factory
from .chunker import chunk_article
from .embeddings import get_client


SOURCE_FOODPANDA = "foodpanda"
DOC_TIER_KB = 1
COMPANY_ID_FOODPANDA = "foodpanda"


def _load_markdown_files(base_dir: str) -> List[Tuple[str, str]]:
    """
    Return list of (path, text) for markdown files under base_dir.
    """
    paths = sorted(glob.glob(os.path.join(base_dir, "*.md")))
    results: List[Tuple[str, str]] = []
    for p in paths:
        try:
            text = Path(p).read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        if text.strip():
            results.append((p, text))
    return results


def _markdown_to_text_and_meta(path: str, text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Prepare article-style text and metadata for a markdown policy doc.
    """
    # Use filename (without extension) as a coarse category/title.
    name = Path(path).stem
    title = name.replace("_", " ")
    body = text

    full_text = f"{title}\n\n{body}".strip()
    metadata: Dict[str, Any] = {
        "category": "policy",
        "company": COMPANY_ID_FOODPANDA,
        "policy_file": os.path.basename(path),
        "source_id": name,
    }
    return full_text, metadata


async def _insert_parent(
    session: AsyncSession,
    content: str,
    source_id: str,
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
            company_id=COMPANY_ID_FOODPANDA,
            source=SOURCE_FOODPANDA,
            doc_tier=DOC_TIER_KB,
            category=meta.get("category"),
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
        await session.execute(
            insert(Document.__table__).values(
                id=str(uuid.uuid4()),
                content=content,
                parent_id=parent_id,
                embedding=embedding,
                company_id=COMPANY_ID_FOODPANDA,
                source=SOURCE_FOODPANDA,
                doc_tier=DOC_TIER_KB,
                category=meta.get("category"),
                source_id=f"{source_id_prefix}-chunk-{i}",
                metadata=meta,
            )
        )


async def ingest_foodpanda_policies() -> None:
    """
    End-to-end Foodpanda policy ingestion:
    - Load markdown policy docs from data/foodpanda/policy_docs
    - For each file: build text + metadata, chunk, embed children, insert parent + children
    """
    base_dir = os.getenv(
        "FOODPANDA_POLICY_DIR",
        str(Path(__file__).resolve().parents[2] / "data" / "foodpanda" / "policy_docs"),
    )
    files = _load_markdown_files(base_dir)
    if not files:
        print(f"No Foodpanda policy markdown files found under {base_dir}")
        return

    client = get_client()
    total_parents = 0
    total_children = 0

    async with async_session_factory() as session:
        for path, raw_text in files:
            full_text, meta = _markdown_to_text_and_meta(path, raw_text)
            if not full_text.strip():
                continue

            chunks = chunk_article(full_text, meta)
            if not chunks:
                source_id = meta.get("source_id") or Path(path).stem
                await _insert_parent(session, content=full_text, source_id=source_id, meta=meta)
                total_parents += 1
                continue

            source_id = meta.get("source_id") or Path(path).stem
            parent_id = await _insert_parent(session, content=full_text, source_id=source_id, meta=meta)
            total_parents += 1

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

    print(
        f"Foodpanda policy ingestion complete. parents={total_parents}, children={total_children}, "
        f"source={SOURCE_FOODPANDA}, company_id={COMPANY_ID_FOODPANDA}"
    )


async def _main() -> None:
    await ingest_foodpanda_policies()


if __name__ == "__main__":
    asyncio.run(_main())

