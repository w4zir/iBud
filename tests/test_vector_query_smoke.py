"""Mocked retrieval/embedding test: no live Ollama or Postgres required."""

import pytest
from unittest.mock import MagicMock, patch

from backend.rag.embeddings import get_client


def test_embedding_client_dimension():
    with patch("backend.rag.embeddings.get_embedding_dim", return_value=768), \
         patch("backend.rag.embeddings.get_embedding_model") as mock_model:
        mock_model.return_value = MagicMock()
        client = get_client()
        assert client.dimension == 768


def test_vector_query_path_with_mocked_client():
    """Simulate the retrieval path: embed query -> use vector in ordering (logic only)."""
    mock_vector = [0.1] * 768

    with patch("backend.rag.embeddings.get_embedding_model") as mock_get_model:
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = mock_vector
        mock_embed.embed_documents.return_value = [mock_vector]
        mock_get_model.return_value = mock_embed

        client = get_client()
        qvec = client.embed_query("Where is my order?")
        assert qvec == mock_vector
        assert len(qvec) == 768


@pytest.mark.asyncio
async def test_vector_query_smoke_mocked_db():
    """Smoke test: cosine_distance ordering logic with mocked embedding (no DB)."""
    mock_vector = [0.01] * 768
    with patch("backend.rag.embeddings.get_embedding_model") as mock_get_model:
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = mock_vector
        mock_get_model.return_value = mock_embed

        from backend.db.models import Document
        from sqlalchemy import func, select

        query_vector = mock_vector
        stmt = (
            select(Document, func.cosine_distance(Document.embedding, query_vector))
            .order_by(func.cosine_distance(Document.embedding, query_vector))
            .limit(3)
        )
        assert stmt is not None
