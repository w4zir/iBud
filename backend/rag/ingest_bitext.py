from __future__ import annotations

"""
Ingestion for the Bitext customer-support QA dataset.

This treats each row as a single retrievable support record whose content is
the combination of `instruction` (user request) and `response` (assistant
answer). Category, intent, and flags are preserved in metadata so they can
be used for analysis and filtering.
"""

import asyncio
import os
from typing import Any, Dict, List

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import Document
from ..db.postgres import async_session_factory
from .embeddings import get_client


SOURCE_BITEXT = "bitext"
DOC_TIER_BITEXT = 1
HF_DATASET_BITEXT = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


def _row_to_text_and_meta(row: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """
    Map a Bitext row to (content, metadata).

    Fields:
      - flags: generation / variation tags
      - instruction: user request text
      - category: coarse semantic category
      - intent: fine-grained intent label
      - response: assistant response text
    """
    flags = (row.get("flags") or "").strip()
    instruction = (row.get("instruction") or "").strip()
    category = (row.get("category") or "").strip()
    intent = (row.get("intent") or "").strip()
    response = (row.get("response") or "").strip()

    # Use both instruction and response as the searchable content so that
    # retrieval can match on either the question or the answer text.
    parts: List[str] = []
    if instruction:
        parts.append(f"User: {instruction}")
    if response:
        parts.append(f"Assistant: {response}")
    content = "\n\n".join(parts).strip()

    meta: Dict[str, Any] = {
        "category": category or None,
        "intent": intent or None,
        "flags": flags or None,
        "instruction": instruction or None,
        "response": response or None,
    }
    # Drop keys that ended up as empty/None to keep metadata compact.
    meta = {k: v for k, v in meta.items() if v}
    return content, meta


async def ingest_bitext() -> None:
    """
    End-to-end Bitext ingestion:
    - Load the Bitext customer-support QA dataset from Hugging Face
    - For each row: build text + metadata, embed, and insert into `documents`
      with source="bitext" and doc_tier=1.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Bitext ingestion requires the 'datasets' package. Install backend requirements."
        ) from exc

    dataset_name = os.getenv("HF_DATASET_BITEXT", HF_DATASET_BITEXT)
    print(f"Loading Bitext dataset '{dataset_name}' from HuggingFace...")
    ds = load_dataset(dataset_name, split="train")

    client = get_client()
    total_rows = 0

    async with async_session_factory() as session:
        for idx, row in enumerate(ds):
            content, meta = _row_to_text_and_meta(dict(row))
            if not content or not content.strip():
                continue

            # Embed the whole support record as a single document.
            vectors = client.embed_documents([content])
            vector = vectors[0] if vectors else None
            if vector is None:
                continue

            await _insert_document(
                session=session,
                content=content,
                embedding=vector,
                source_id=f"bitext-{idx}",
                category=meta.get("category"),
                meta=meta,
            )
            total_rows += 1

        await session.commit()

    print(f"Bitext ingestion complete. documents (source={SOURCE_BITEXT}) inserted={total_rows}")


async def _insert_document(
    session: AsyncSession,
    content: str,
    embedding: List[float],
    source_id: str,
    category: str | None,
    meta: Dict[str, Any],
) -> None:
    await session.execute(
        insert(Document.__table__).values(
            content=content,
            parent_id=None,
            embedding=embedding,
            source=SOURCE_BITEXT,
            doc_tier=DOC_TIER_BITEXT,
            category=category,
            source_id=source_id,
            metadata=meta,
        )
    )


async def _main() -> None:
    await ingest_bitext()


if __name__ == "__main__":
    asyncio.run(_main())

