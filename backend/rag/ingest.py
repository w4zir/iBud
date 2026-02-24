import asyncio
import os
from typing import Any, Dict, Iterable, List, Tuple

from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import Document
from ..db.postgres import async_session_factory
from .embeddings import get_client


HF_DATASET_DEFAULT = "rjac/e-commerce-customer-support-qa"


def build_document_text(example: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Map a raw dataset row to a (content, metadata) pair.

    Assumes presence of at least question/answer; category treated as optional.
    """
    question = example.get("question") or example.get("Question") or ""
    answer = example.get("answer") or example.get("Answer") or ""
    category = example.get("category") or example.get("Category")

    content_parts: List[str] = []
    if question:
        content_parts.append(f"Q: {question}")
    if answer:
        content_parts.append(f"A: {answer}")
    content = "\n".join(content_parts).strip()

    metadata: Dict[str, Any] = {}
    if category:
        metadata["category"] = category

    # Preserve original fields for potential future use
    metadata["source_example"] = {
        "question": question,
        "answer": answer,
        "category": category,
    }

    return content, metadata


def chunk_documents(
    contents_and_metadata: Iterable[Tuple[str, Dict[str, Any]]],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Chunk documents using RecursiveCharacterTextSplitter while preserving metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunked: List[Tuple[str, Dict[str, Any]]] = []
    for content, metadata in contents_and_metadata:
        if not content:
            continue
        for chunk in splitter.split_text(content):
            chunked.append((chunk, dict(metadata)))
    return chunked


async def upsert_documents(
    session: AsyncSession,
    chunks: List[Tuple[str, Dict[str, Any]]],
    embeddings: List[List[float]],
    category_fallback: str | None = None,
) -> None:
    """
    Upsert document chunks into the documents table.

    Uses source_id derived from a simple incremental index to avoid duplicates on re-run.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings length mismatch")

    values = []
    for idx, ((content, metadata), embedding) in enumerate(zip(chunks, embeddings)):
        category = metadata.get("category") or category_fallback
        source_id = f"hf-{idx}"
        values.append(
            {
                "content": content,
                "embedding": embedding,
                "category": category,
                "source_id": source_id,
                "metadata": metadata,
            }
        )

    if not values:
        return

    stmt = insert(Document.__table__).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["source_id"],
        set_={
            "content": stmt.excluded.content,
            "embedding": stmt.excluded.embedding,
            "category": stmt.excluded.category,
            "metadata": stmt.excluded.metadata,
        },
    )
    await session.execute(stmt)
    await session.commit()


async def ingest() -> None:
    """
    End-to-end ingestion pipeline:
    - Load HuggingFace dataset
    - Map rows to text + metadata
    - Chunk
    - Embed
    - Upsert into pgvector-backed documents table
    """
    dataset_name = os.getenv("HF_DATASET", HF_DATASET_DEFAULT)
    print(f"Loading dataset '{dataset_name}' from HuggingFace...")
    ds = load_dataset(dataset_name, split="train")

    contents_and_metadata: List[Tuple[str, Dict[str, Any]]] = []
    for example in ds:
        contents_and_metadata.append(build_document_text(example))

    print(f"Preparing {len(contents_and_metadata)} documents for chunking...")
    chunks = chunk_documents(contents_and_metadata)
    print(f"Generated {len(chunks)} chunks. Embedding...")

    client = get_client()
    texts = [c for c, _ in chunks]
    vectors = client.embed_documents(texts)

    async with async_session_factory() as session:
        await upsert_documents(session, chunks, vectors)

    # Basic validation: count documents
    async with async_session_factory() as session:
        result = await session.execute(select(Document))
        total = len(result.scalars().all())
        print(f"Ingestion complete. documents count={total}")


async def _main() -> None:
    await ingest()


if __name__ == "__main__":
    asyncio.run(_main())

