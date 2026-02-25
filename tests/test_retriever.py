from typing import Any, List

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.rag.retriever import RetrievedDoc, Retriever


class DummyEmbeddingClient:
    def __init__(self, vector: List[float]) -> None:
        self._vector = vector

    def embed_query(self, text: str) -> List[float]:
        return self._vector


@pytest.mark.asyncio
async def test_search_empty_query_returns_empty_list():
    retriever = Retriever(embedding_client=DummyEmbeddingClient([0.0] * 4))  # type: ignore[arg-type]
    docs = await retriever.search("")
    assert docs == []


@pytest.mark.asyncio
async def test_search_uses_cache_when_available():
    cached_payload = [
        {
            "content": "cached",
            "metadata": {},
            "score": 0.1,
            "source": "wixqa",
            "doc_tier": 1,
            "document_id": "doc-1",
            "parent_id": None,
        }
    ]

    with patch("backend.rag.retriever.get_cached", return_value=cached_payload) as mock_get_cached:
        retriever = Retriever(embedding_client=DummyEmbeddingClient([0.0] * 4))  # type: ignore[arg-type]
        docs = await retriever.search("Where is my order?", use_cache=True, rerank=False)

    mock_get_cached.assert_awaited()
    assert len(docs) == 1
    assert isinstance(docs[0], RetrievedDoc)
    assert docs[0].content == "cached"


@pytest.mark.asyncio
async def test_search_populates_cache_on_miss(monkeypatch: pytest.MonkeyPatch):
    # First, force cache miss then capture set_cached
    async def _fake_get_cached(key: str) -> Any:
        return None

    captured_to_cache: list[list[dict[str, Any]]] = []

    async def _fake_set_cached(key: str, value: Any) -> None:
        captured_to_cache.append(value)

    vec = [0.01] * 4
    retriever = Retriever(embedding_client=DummyEmbeddingClient(vec))  # type: ignore[arg-type]

    async def _fake_similarity_search(*args: Any, **kwargs: Any) -> list[RetrievedDoc]:
        return [
            RetrievedDoc(
                content="doc1",
                metadata={},
                score=0.1,
                source="wixqa",
                doc_tier=1,
                document_id="1",
                parent_id=None,
            )
        ]

    monkeypatch.setattr("backend.rag.retriever.get_cached", _fake_get_cached)
    monkeypatch.setattr("backend.rag.retriever.set_cached", _fake_set_cached)
    monkeypatch.setattr(Retriever, "_similarity_search", _fake_similarity_search)
    monkeypatch.setattr(Retriever, "_maybe_expand_parents", AsyncMock(side_effect=lambda docs: docs))
    monkeypatch.setattr(Retriever, "_rerank", AsyncMock(side_effect=lambda q, docs: docs))

    docs = await retriever.search("Where is my order?", use_cache=True, rerank=True)
    assert len(docs) == 1
    assert captured_to_cache
    cached_docs = captured_to_cache[0]
    assert isinstance(cached_docs, list)
    assert cached_docs[0]["content"] == "doc1"


@pytest.mark.asyncio
async def test_parent_expansion_adds_parent_when_below_threshold(monkeypatch: pytest.MonkeyPatch):
    child = RetrievedDoc(
        content="child",
        metadata={},
        score=0.3,
        source="wixqa",
        doc_tier=1,
        document_id="child-id",
        parent_id="parent-id",
    )

    async def _fake_sim_search(*args: Any, **kwargs: Any) -> list[RetrievedDoc]:
        return [child]

    async def _fake_expand(docs: list[RetrievedDoc], score_threshold: float = 0.4) -> list[RetrievedDoc]:
        parent = RetrievedDoc(
            content="parent",
            metadata={},
            score=child.score,
            source="wixqa",
            doc_tier=1,
            document_id="parent-id",
            parent_id=None,
        )
        return [child, parent]

    vec = [0.01] * 4
    retriever = Retriever(embedding_client=DummyEmbeddingClient(vec))  # type: ignore[arg-type]
    monkeypatch.setattr(Retriever, "_similarity_search", _fake_sim_search)
    monkeypatch.setattr(Retriever, "_maybe_expand_parents", _fake_expand)
    monkeypatch.setattr(Retriever, "_rerank", AsyncMock(side_effect=lambda q, docs: docs))
    monkeypatch.setattr("backend.rag.retriever.get_cached", AsyncMock(return_value=None))
    monkeypatch.setattr("backend.rag.retriever.set_cached", AsyncMock())

    docs = await retriever.search("Where is my order?", use_cache=True, rerank=True)
    assert len(docs) == 2
    assert docs[0].content == "child"
    assert docs[1].content == "parent"


@pytest.mark.asyncio
async def test_rerank_changes_order_when_encoder_available(monkeypatch: pytest.MonkeyPatch):
    d1 = RetrievedDoc(
        content="low",
        metadata={},
        score=0.1,
        source="wixqa",
        doc_tier=1,
        document_id="1",
        parent_id=None,
    )
    d2 = RetrievedDoc(
        content="high",
        metadata={},
        score=0.2,
        source="wixqa",
        doc_tier=1,
        document_id="2",
        parent_id=None,
    )

    vec = [0.01] * 4
    retriever = Retriever(embedding_client=DummyEmbeddingClient(vec))  # type: ignore[arg-type]

    async def _fake_sim_search(*args: Any, **kwargs: Any) -> list[RetrievedDoc]:
        return [d1, d2]

    async def _fake_expand(docs: list[RetrievedDoc], score_threshold: float = 0.4) -> list[RetrievedDoc]:
        return docs

    class FakeCrossEncoder:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            # Force d1 to be ranked above d2
            return [0.9, 0.1]

    monkeypatch.setattr(Retriever, "_similarity_search", _fake_sim_search)
    monkeypatch.setattr(Retriever, "_maybe_expand_parents", _fake_expand)
    retriever._cross_encoder = FakeCrossEncoder()
    monkeypatch.setattr("backend.rag.retriever.get_cached", AsyncMock(return_value=None))
    monkeypatch.setattr("backend.rag.retriever.set_cached", AsyncMock())

    docs = await retriever.search("Where is my order?", use_cache=True, rerank=True)
    assert [d.content for d in docs] == ["low", "high"]

