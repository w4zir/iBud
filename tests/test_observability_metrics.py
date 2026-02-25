from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agent import nodes
from backend.rag.retriever import RetrievedDoc, Retriever


class DummyEmbeddingClient:
    def __init__(self) -> None:
        self._called = False

    def embed_query(self, text: str) -> list[float]:
        self._called = True
        return [0.0, 0.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_retriever_observes_cache_hits_and_latency(monkeypatch):
    # Arrange a cache hit path and ensure metrics are touched.
    async def fake_get_cached(key: str):
        return [
            {
                "content": "cached-doc",
                "metadata": {},
                "score": 0.1,
                "source": "wixqa",
                "doc_tier": 1,
                "document_id": "doc-1",
                "parent_id": None,
            }
        ]

    fake_cache_hits = MagicMock()
    fake_latency = MagicMock()

    monkeypatch.setattr("backend.rag.retriever.get_cached", fake_get_cached)
    monkeypatch.setattr("backend.rag.retriever.set_cached", AsyncMock())
    monkeypatch.setattr("backend.rag.retriever.retrieval_latency", fake_latency)
    monkeypatch.setattr("backend.rag.retriever.cache_hits", fake_cache_hits)

    retriever = Retriever(embedding_client=DummyEmbeddingClient())  # type: ignore[arg-type]

    # Act
    docs = await retriever.search("Where is my order?", use_cache=True, rerank=False)

    # Assert
    assert docs and docs[0].content == "cached-doc"
    fake_cache_hits.inc.assert_called_once()
    # Latency is only observed on cache miss (when we do DB + optional rerank).
    fake_latency.observe.assert_not_called()


@pytest.mark.asyncio
async def test_tool_calls_counter_incremented(monkeypatch):
    # Use a tiny state that triggers a single known tool.
    state = {
        "session_id": "sess-1",
        "user_id": "user-1",
        "messages": [{"role": "user", "content": "Where is my order?"}],
        "tool_calls": [
            {"name": "order_lookup", "arguments": {"order_number": "123"}}
        ],
        "tool_results": [],
        "retrieved_docs": [],
        "final_response": None,
        "intent": "order_status",
        "should_escalate": False,
        "ticket_id": None,
    }

    async def fake_tool(order_number=None, user_id=None):
        return {"order_number": order_number, "status": "processing"}

    fake_counter = MagicMock()

    monkeypatch.setattr(
        "backend.tools.order_lookup.order_lookup_tool",
        fake_tool,
    )
    monkeypatch.setattr(
        "backend.agent.nodes.tool_calls",
        fake_counter,
    )

    await nodes.execute_tool(state)  # type: ignore[arg-type]
    fake_counter.labels.assert_called_once_with(tool_name="order_lookup")

