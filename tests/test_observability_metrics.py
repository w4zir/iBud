from __future__ import annotations

from types import SimpleNamespace
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
        "backend.agent.nodes.order_lookup_tool",
        fake_tool,
    )
    monkeypatch.setattr(
        "backend.agent.nodes.tool_calls",
        fake_counter,
    )

    await nodes.execute_tool(state)  # type: ignore[arg-type]
    fake_counter.labels.assert_called_once_with(tool_name="order_lookup")


@pytest.mark.asyncio
async def test_classify_intent_increments_intent_distribution(monkeypatch):
    state = {
        "session_id": "sess-1",
        "request_id": "req-1",
        "messages": [{"role": "user", "content": "Where is my order?"}],
    }

    class DummyLLM:
        async def ainvoke(self, messages):
            return SimpleNamespace(content="order_status", response_metadata={})

    fake_intent_counter = MagicMock()
    monkeypatch.setattr("backend.agent.nodes.get_llm", lambda: DummyLLM())
    monkeypatch.setattr("backend.agent.nodes.intent_distribution", fake_intent_counter)

    next_state = await nodes.classify_intent(state)  # type: ignore[arg-type]
    assert next_state["intent"] == "order_status"
    fake_intent_counter.labels.assert_called_once_with(intent="order_status")


@pytest.mark.asyncio
async def test_execute_tool_records_tool_outcome(monkeypatch):
    state = {
        "session_id": "sess-1",
        "request_id": "req-1",
        "user_id": "user-1",
        "messages": [{"role": "user", "content": "Where is my order?"}],
        "tool_calls": [{"name": "order_lookup", "arguments": {"order_number": "123"}}],
        "tool_results": [],
    }

    async def fake_tool(order_number=None, user_id=None):
        return {"order_number": order_number, "status": "processing"}

    fake_outcome_counter = MagicMock()
    monkeypatch.setattr("backend.agent.nodes.order_lookup_tool", fake_tool)
    monkeypatch.setattr("backend.agent.nodes.tool_outcome", fake_outcome_counter)

    await nodes.execute_tool(state)  # type: ignore[arg-type]
    fake_outcome_counter.labels.assert_called_once_with(
        tool_name="order_lookup",
        outcome="success",
    )


@pytest.mark.asyncio
async def test_check_escalation_records_task_outcome(monkeypatch):
    state = {
        "session_id": "sess-1",
        "request_id": "req-1",
        "intent": "product_qa",
        "messages": [{"role": "user", "content": "thanks"}],
        "tool_results": [],
        "should_escalate": False,
    }

    class DummyLLM:
        async def ainvoke(self, messages):
            return SimpleNamespace(content="false", response_metadata={})

    fake_task_outcome = MagicMock()
    monkeypatch.setattr("backend.agent.nodes.get_llm", lambda: DummyLLM())
    monkeypatch.setattr("backend.agent.nodes.task_outcome", fake_task_outcome)

    result = await nodes.check_escalation(state)  # type: ignore[arg-type]
    assert result["should_escalate"] is False
    fake_task_outcome.labels.assert_called_once_with(outcome="resolved_without_escalation")


@pytest.mark.asyncio
async def test_retriever_records_embedding_and_rerank_latency(monkeypatch):
    fake_embedding_latency = MagicMock()
    fake_rerank_latency = MagicMock()
    monkeypatch.setattr("backend.rag.retriever.embedding_latency", fake_embedding_latency)
    monkeypatch.setattr("backend.rag.retriever.rerank_latency", fake_rerank_latency)
    monkeypatch.setattr("backend.rag.retriever.get_cached", AsyncMock(return_value=None))
    monkeypatch.setattr("backend.rag.retriever.set_cached", AsyncMock())

    async def fake_similarity_search(
        self,
        query_vector,
        top_k,
        category,
        tier_filter,
        source=None,
        company_id=None,
    ):
        return [
            RetrievedDoc(
                content="doc",
                metadata={},
                score=0.1,
                source="wixqa",
                doc_tier=1,
                document_id="doc-1",
                parent_id=None,
            )
        ]

    async def fake_expand(docs, score_threshold=0.4):
        return docs

    monkeypatch.setattr(Retriever, "_similarity_search", fake_similarity_search)
    monkeypatch.setattr(Retriever, "_maybe_expand_parents", staticmethod(fake_expand))

    class FakeCrossEncoder:
        def predict(self, pairs):
            return [0.9 for _ in pairs]

    async def fake_ensure_cross_encoder(self):
        self._cross_encoder = FakeCrossEncoder()

    monkeypatch.setattr(Retriever, "_ensure_cross_encoder", fake_ensure_cross_encoder)

    retriever = Retriever(embedding_client=DummyEmbeddingClient())  # type: ignore[arg-type]
    await retriever.search("Where is my order?", use_cache=True, rerank=True)

    assert fake_embedding_latency.observe.call_count == 1
    assert fake_rerank_latency.observe.call_count == 1

