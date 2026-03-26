from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.rag.es_client import ESClient


@pytest.mark.asyncio
async def test_ensure_index_creates_when_missing():
    es = MagicMock()
    es.indices = MagicMock()
    es.indices.exists = AsyncMock(return_value=False)
    es.indices.create = AsyncMock()

    client = ESClient(es=es, index_name="test-index", embedding_dim=4)
    await client.ensure_index()

    es.indices.create.assert_awaited()


@pytest.mark.asyncio
async def test_index_document_single():
    es = MagicMock()
    es.index = AsyncMock()

    client = ESClient(es=es, index_name="test-index", embedding_dim=4)
    await client.index_document(
        doc_id="1",
        content="hello",
        embedding=[0.1, 0.2, 0.3, 0.4],
        company_id=None,
        source="wixqa",
        doc_tier=1,
        category="policy",
        source_id="src1",
        parent_id=None,
        metadata={"k": "v"},
    )

    assert es.index.await_count == 1
    _, kwargs = es.index.await_args
    assert kwargs["id"] == "1"
    assert kwargs["document"]["content"] == "hello"
    assert kwargs["document"]["embedding"] == [0.1, 0.2, 0.3, 0.4]


@pytest.mark.asyncio
async def test_bulk_index_calls_async_bulk():
    es = MagicMock()
    fake_async_bulk = AsyncMock()

    client = ESClient(es=es, index_name="test-index", embedding_dim=4)
    docs: List[Dict[str, Any]] = [
        {
            "id": "1",
            "content": "c1",
            "embedding": [0.0, 0.0, 0.0, 0.0],
            "company_id": None,
            "source": "bitext",
            "doc_tier": 1,
            "category": "qa",
            "source_id": "s1",
            "parent_id": None,
            "metadata": {"a": 1},
        },
        {
            "id": "2",
            "content": "c2",
            "embedding": None,
            "company_id": None,
            "source": "bitext",
            "doc_tier": 1,
            "category": "qa",
            "source_id": "s2",
            "parent_id": None,
            "metadata": {"b": 2},
        },
    ]

    with patch("backend.rag.es_client.async_bulk", fake_async_bulk):
        await client.bulk_index(docs)

    fake_async_bulk.assert_awaited()
    args, _ = fake_async_bulk.call_args
    assert args[0] is es  # ES client passed into async_bulk


@pytest.mark.asyncio
async def test_vector_search_builds_filters_and_parses_hits():
    es = MagicMock()
    es.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [
                    {
                        "_id": "doc-1",
                        "_score": 0.12,
                        "_source": {
                            "content": "c1",
                            "metadata": {"x": 1},
                            "source": "wixqa",
                            "doc_tier": 1,
                            "parent_id": None,
                        },
                    }
                ]
            }
        }
    )

    client = ESClient(es=es, index_name="test-index", embedding_dim=4)
    results = await client.vector_search(
        query_vector=[0.1, 0.2, 0.3, 0.4],
        top_k=5,
        company_id="acme",
        category="policy",
        tier_filter=1,
        source="wixqa",
    )

    assert len(results) == 1
    assert results[0]["document_id"] == "doc-1"
    assert results[0]["score"] == 0.12
    assert results[0]["metadata"] == {"x": 1}

    # Verify the expected ES filter terms are present in the body.
    _, kwargs = es.search.await_args
    body = kwargs["body"]
    filters = body["query"]["script_score"]["query"]["bool"]["filter"]
    assert {"term": {"company_id": "acme"}} in filters
    assert {"term": {"category": "policy"}} in filters
    assert {"term": {"doc_tier": 1}} in filters
    assert {"term": {"source": "wixqa"}} in filters


@pytest.mark.asyncio
async def test_get_documents_by_ids_parses_mget():
    es = MagicMock()
    es.mget = AsyncMock(
        return_value={
            "docs": [
                {
                    "_id": "doc-1",
                    "found": True,
                    "_source": {
                        "content": "c1",
                        "metadata": {"x": 1},
                        "source": "wixqa",
                        "doc_tier": 1,
                        "parent_id": None,
                    },
                },
                {"_id": "missing", "found": False},
            ]
        }
    )

    client = ESClient(es=es, index_name="test-index", embedding_dim=4)
    results = await client.get_documents_by_ids(["doc-1", "missing"])
    assert len(results) == 1
    assert results[0]["document_id"] == "doc-1"
    assert results[0]["content"] == "c1"


@pytest.mark.asyncio
async def test_delete_document_calls_delete():
    es = MagicMock()
    es.delete = AsyncMock()

    client = ESClient(es=es, index_name="test-index", embedding_dim=4)
    await client.delete_document(doc_id="doc-1")

    es.delete.assert_awaited()

