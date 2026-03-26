import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.postgres import async_session_factory
from backend.rag.es_client import ESClient


@pytest.fixture(scope="session", autouse=True)
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session


@pytest.fixture
def mock_es_client() -> ESClient:
    """
    Unit tests for Retriever/ES integration can inject this instead of
    connecting to a real Elasticsearch instance.
    """

    es = MagicMock()
    return ESClient(es=es, index_name="test-index", embedding_dim=4)

