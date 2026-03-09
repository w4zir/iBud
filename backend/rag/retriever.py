from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import log_event
from ..db.models import Document
from ..db.postgres import async_session_factory
from ..db.redis_client import CacheKeyParts, build_cache_key, get_cached, set_cached
from ..observability.prometheus_metrics import (
    cache_hits,
    db_latency,
    embedding_latency,
    error_count,
    rerank_latency,
    retrieval_latency,
)
from .embeddings import EmbeddingClient


@dataclass
class RetrievedDoc:
    content: str
    metadata: Dict[str, Any]
    score: float
    source: Optional[str]
    doc_tier: int
    document_id: str
    parent_id: Optional[str]


class Retriever:
    """
    Core RAG retriever using pgvector similarity, optional Redis caching,
    metadata filters, parent expansion, and optional reranking.
    """

    def __init__(
        self,
        embedding_client: Optional[EmbeddingClient] = None,
    ) -> None:
        self._embedding_client = embedding_client or EmbeddingClient()
        self._cross_encoder = None

    async def search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        tier_filter: Optional[int] = None,
        use_cache: bool = True,
        rerank: bool = True,
        source: Optional[str] = None,
    ) -> List[RetrievedDoc]:
        if not query or not query.strip():
            return []

        cache_key = None
        if use_cache:
            parts = CacheKeyParts(
                query=query,
                category=category,
                tier_filter=tier_filter,
                top_k=top_k,
                rerank=rerank,
                source=source,
            )
            cache_key = build_cache_key(parts)
            cached = await get_cached(cache_key)
            if cached is not None:
                try:
                    cache_hits.inc()
                except Exception:  # pragma: no cover - defensive
                    pass
                return [RetrievedDoc(**item) for item in cached]  # type: ignore[arg-type]

        start_time = time.perf_counter()
        embedding_start = time.perf_counter()
        try:
            query_vector = self._embedding_client.embed_query(query)
        except Exception:
            try:
                error_count.labels(error_type="embedding_failure", component="retriever").inc()
            except Exception:
                pass
            raise
        finally:
            try:
                embedding_latency.observe(time.perf_counter() - embedding_start)
            except Exception:
                pass

        async with async_session_factory() as session:
            db_start = time.perf_counter()
            docs = await self._similarity_search(
                session=session,
                query_vector=query_vector,
                top_k=top_k,
                category=category,
                tier_filter=tier_filter,
                source=source,
            )
            try:
                db_latency.labels(operation="similarity_search").observe(
                    time.perf_counter() - db_start
                )
            except Exception:
                pass

        docs = await type(self)._maybe_expand_parents(docs)

        if rerank:
            docs = await self._rerank(query, docs)

        latency = time.perf_counter() - start_time
        try:
            retrieval_latency.observe(latency)
        except Exception:  # pragma: no cover - defensive
            pass
        log_event(
            "retriever",
            "search_complete",
            query_len=len(query),
            docs_count=len(docs),
            latency_ms=round(latency * 1000, 2),
            category=category,
            tier_filter=tier_filter,
            cache_used=use_cache,
        )

        if cache_key and use_cache and docs:
            serializable = [asdict(d) for d in docs]
            await set_cached(cache_key, serializable)

        return docs

    async def _similarity_search(
        self,
        session: AsyncSession,
        query_vector: List[float],
        top_k: int,
        category: Optional[str],
        tier_filter: Optional[int],
        source: Optional[str],
    ) -> List[RetrievedDoc]:
        distance = Document.embedding.cosine_distance(query_vector)
        stmt = select(
            Document,
            distance.label("score"),
        ).where(Document.embedding.is_not(None))

        if category:
            stmt = stmt.where(Document.category == category)

        if tier_filter is not None:
            stmt = stmt.where(Document.doc_tier == tier_filter)

        if source:
            stmt = stmt.where(Document.source == source)

        stmt = stmt.order_by(distance).limit(top_k)

        result = await session.execute(stmt)
        rows = result.all()

        docs: List[RetrievedDoc] = []
        for doc, score in rows:
            metadata: Dict[str, Any] = doc.metadata_ or {}
            docs.append(
                RetrievedDoc(
                    content=doc.content,
                    metadata=metadata,
                    score=float(score) if score is not None else 0.0,
                    source=doc.source,
                    doc_tier=doc.doc_tier,
                    document_id=str(doc.id),
                    parent_id=doc.parent_id,
                )
            )
        return docs

    @staticmethod
    async def _maybe_expand_parents(
        docs: List[RetrievedDoc],
        score_threshold: float = 0.4,
    ) -> List[RetrievedDoc]:
        """
        If a child chunk scores highly, attach its parent content as additional context.

        For simplicity, this implementation appends a synthetic parent doc after
        each qualifying child, when a parent exists.
        """
        parent_ids = {d.parent_id for d in docs if d.parent_id and d.score <= score_threshold}
        if not parent_ids:
            return docs

        async with async_session_factory() as session:
            result = await session.execute(
                select(Document).where(Document.id.in_(parent_ids))
            )
            parents_by_id = {str(doc.id): doc for (doc,) in result.all()}

        expanded: List[RetrievedDoc] = []
        for d in docs:
            expanded.append(d)
            if d.parent_id and d.score <= score_threshold:
                parent = parents_by_id.get(d.parent_id)
                if parent:
                    metadata: Dict[str, Any] = parent.metadata_ or {}
                    expanded.append(
                        RetrievedDoc(
                            content=parent.content,
                            metadata=metadata,
                            score=d.score,
                            source=parent.source,
                            doc_tier=parent.doc_tier,
                            document_id=str(parent.id),
                            parent_id=None,
                        )
                    )
        return expanded

    async def _ensure_cross_encoder(self) -> None:
        if self._cross_encoder is not None:
            return
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]

            # Lightweight general-purpose cross-encoder; can be tuned later.
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:  # pragma: no cover - optional dependency path
            self._cross_encoder = None

    async def _rerank(
        self,
        query: str,
        docs: List[RetrievedDoc],
    ) -> List[RetrievedDoc]:
        if not docs:
            return docs

        await self._ensure_cross_encoder()
        if self._cross_encoder is None:
            return docs

        try:
            start = time.perf_counter()
            pairs = [(query, d.content) for d in docs]
            scores = self._cross_encoder.predict(pairs)
            try:
                rerank_latency.observe(time.perf_counter() - start)
            except Exception:
                pass
        except Exception:  # pragma: no cover - defensive
            try:
                error_count.labels(error_type="rerank_failure", component="retriever").inc()
            except Exception:
                pass
            return docs

        rescored: List[RetrievedDoc] = []
        for d, s in zip(docs, scores):
            rescored.append(
                RetrievedDoc(
                    content=d.content,
                    metadata=d.metadata,
                    score=float(s),
                    source=d.source,
                    doc_tier=d.doc_tier,
                    document_id=d.document_id,
                    parent_id=d.parent_id,
                )
            )

        rescored.sort(key=lambda d: d.score, reverse=True)
        return rescored


__all__ = ["RetrievedDoc", "Retriever"]

