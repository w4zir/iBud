from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from ..config import (
    get_es_retrieval_top_k,
    get_rerank_model,
    get_rerank_top_k,
    log_event,
)
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
from .es_client import ESClient, get_es_client


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
        es_client: Optional[ESClient] = None,
    ) -> None:
        self._embedding_client = embedding_client or EmbeddingClient()
        # Unit tests often mock `_similarity_search` / `_maybe_expand_parents`.
        # In environments without the `elasticsearch` dependency installed,
        # eager client creation would break those tests.
        if es_client is not None:
            self._es_client = es_client
        else:
            try:
                self._es_client = get_es_client()
            except ImportError:
                self._es_client = None  # type: ignore[assignment]
        self._cross_encoder = None

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        category: Optional[str] = None,
        tier_filter: Optional[int] = None,
        use_cache: bool = True,
        rerank: bool = True,
        source: Optional[str] = None,
        company_id: Optional[str] = None,
    ) -> List[RetrievedDoc]:
        if not query or not query.strip():
            return []

        candidate_top_k = int(top_k if top_k is not None else get_es_retrieval_top_k())
        cache_key = None
        if use_cache:
            parts = CacheKeyParts(
                query=query,
                category=category,
                tier_filter=tier_filter,
                top_k=candidate_top_k,
                rerank=rerank,
                source=source,
                company_id=company_id,
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

        db_start = time.perf_counter()
        docs = await self._similarity_search(
            query_vector=query_vector,
            top_k=candidate_top_k,
            category=category,
            tier_filter=tier_filter,
            source=source,
            company_id=company_id,
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
            rerank_top_k = get_rerank_top_k()
            if rerank_top_k > 0:
                docs = docs[:rerank_top_k]

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
        query_vector: List[float],
        top_k: int,
        category: Optional[str],
        tier_filter: Optional[int],
        source: Optional[str],
        company_id: Optional[str],
    ) -> List[RetrievedDoc]:
        if self._es_client is None:
            raise ImportError(
                "Elasticsearch dependency is not installed. "
                "Install with: pip install elasticsearch"
            )
        rows = await self._es_client.vector_search(
            query_vector=query_vector,
            top_k=top_k,
            company_id=company_id,
            category=category,
            tier_filter=tier_filter,
            source=source,
        )

        return [
            RetrievedDoc(
                content=r.get("content", ""),
                metadata=r.get("metadata") or {},
                score=float(r.get("score") or 0.0),
                source=r.get("source"),
                doc_tier=int(r.get("doc_tier") or 1),
                document_id=str(r.get("document_id") or ""),
                parent_id=r.get("parent_id"),
            )
            for r in rows
        ]

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

        es_client = get_es_client()
        parent_rows = await es_client.get_documents_by_ids(list(parent_ids))
        parents_by_id = {
            str(r.get("document_id") or ""): r for r in parent_rows if r.get("document_id")
        }

        expanded: List[RetrievedDoc] = []
        for d in docs:
            expanded.append(d)
            if d.parent_id and d.score <= score_threshold:
                parent = parents_by_id.get(str(d.parent_id))
                if parent:
                    metadata: Dict[str, Any] = parent.get("metadata") or {}
                    expanded.append(
                        RetrievedDoc(
                            content=parent.get("content", ""),
                            metadata=metadata,
                            score=d.score,
                            source=parent.get("source"),
                            doc_tier=int(parent.get("doc_tier") or 1),
                            document_id=str(parent.get("document_id") or ""),
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
            self._cross_encoder = CrossEncoder(get_rerank_model())
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

