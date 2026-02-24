import pytest
from sqlalchemy import func, select

from backend.db.models import Document
from backend.db.postgres import async_session_factory
from backend.rag.embeddings import get_client


@pytest.mark.asyncio
async def test_vector_query_smoke():
    client = get_client()
    query_vector = client.embed_query("Where is my order?")

    async with async_session_factory() as session:
        stmt = (
            select(Document, func.cosine_distance(Document.embedding, query_vector))
            .order_by(func.cosine_distance(Document.embedding, query_vector))
            .limit(3)
        )
        result = await session.execute(stmt)
        rows = result.all()

    # Smoke expectation: query runs and returns up to 3 rows without error.
    assert rows is not None

