from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

try:
    # Optional import: unit tests can inject a mocked `es` client and avoid
    # requiring the real dependency at import time.
    from elasticsearch import AsyncElasticsearch
    from elasticsearch.helpers import async_bulk
except Exception:  # pragma: no cover - missing dependency path
    AsyncElasticsearch = None  # type: ignore[assignment]
    async_bulk = None  # type: ignore[assignment]

from ..config import (
    get_embedding_dim,
    get_es_host,
    get_es_index_name,
    get_es_port,
)


class ESClient:
    """
    Elasticsearch client wrapper for:
    - index creation
    - (bulk) indexing documents + embeddings
    - exact vector search via `script_score`
    - parent lookups via `_mget`

    Notes:
    - We intentionally use exact vector search (not ANN) to keep the mapping
      and retrieval logic closer to Intercom's initial setup.
    - We return a *distance* score where lower is better (to match pgvector's
      cosine_distance semantics used elsewhere in the codebase).
    """

    def __init__(
        self,
        *,
        index_name: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        es: Optional[AsyncElasticsearch] = None,
    ) -> None:
        self._index_name = index_name or get_es_index_name()
        self._embedding_dim = embedding_dim or get_embedding_dim()
        if es is not None:
            self._es = es
            return
        if AsyncElasticsearch is None:
            raise ImportError(
                "Elasticsearch dependency is not installed. "
                "Install with: pip install elasticsearch"
            )
        self._es = AsyncElasticsearch(
            hosts=[f"http://{get_es_host()}:{get_es_port()}"],
        )

    @property
    def index_name(self) -> str:
        return self._index_name

    async def ensure_index(self) -> None:
        exists = await self._es.indices.exists(index=self._index_name)
        if exists:
            return

        body: Dict[str, Any] = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "2s",
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self._embedding_dim,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "company_id": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "doc_tier": {"type": "integer"},
                    "category": {"type": "keyword"},
                    "source_id": {"type": "keyword"},
                    "parent_id": {"type": "keyword"},
                    # Not indexed; still stored in _source for retrieval payloads.
                    "metadata": {"type": "object", "enabled": False},
                }
            },
        }
        await self._es.indices.create(index=self._index_name, body=body)

    async def index_document(
        self,
        *,
        doc_id: str,
        content: str,
        embedding: Optional[List[float]],
        company_id: Optional[str],
        source: Optional[str],
        doc_tier: int,
        category: Optional[str],
        source_id: Optional[str],
        parent_id: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        doc: Dict[str, Any] = {
            "content": content,
            "company_id": company_id,
            "source": source,
            "doc_tier": int(doc_tier),
            "category": category,
            "source_id": source_id,
            "parent_id": parent_id,
            "metadata": metadata or {},
        }
        if embedding is not None:
            doc["embedding"] = embedding
        await self._es.index(index=self._index_name, id=doc_id, document=doc)

    async def bulk_index(self, documents: Sequence[Dict[str, Any]]) -> None:
        """
        Index many documents in one request.

        Each element must have:
        - id
        - content
        - embedding (optional)
        - company_id, source, doc_tier, category, source_id, parent_id
        - metadata (optional)
        """

        actions: List[Dict[str, Any]] = []
        for d in documents:
            doc = {
                "content": d.get("content", ""),
                "company_id": d.get("company_id"),
                "source": d.get("source"),
                "doc_tier": int(d.get("doc_tier", 1)),
                "category": d.get("category"),
                "source_id": d.get("source_id"),
                "parent_id": d.get("parent_id"),
                "metadata": d.get("metadata") or {},
            }
            if d.get("embedding") is not None:
                doc["embedding"] = d["embedding"]
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self._index_name,
                    "_id": d["id"],
                    "_source": doc,
                }
            )

        if not actions:
            return

        if async_bulk is None:
            raise ImportError(
                "Elasticsearch dependency is not installed (missing elasticsearch.helpers)."
            )

        # We ignore detailed stats here; callers should rely on logs/monitoring.
        await async_bulk(self._es, actions)

    async def vector_search(
        self,
        *,
        query_vector: List[float],
        top_k: int,
        company_id: Optional[str] = None,
        category: Optional[str] = None,
        tier_filter: Optional[int] = None,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Exact vector search returning *distance* where lower is better.
        """

        filters: List[Dict[str, Any]] = [{"exists": {"field": "embedding"}}]
        if company_id:
            filters.append({"term": {"company_id": company_id}})
        if category:
            filters.append({"term": {"category": category}})
        if tier_filter is not None:
            filters.append({"term": {"doc_tier": int(tier_filter)}})
        if source:
            filters.append({"term": {"source": source}})

        body: Dict[str, Any] = {
            "size": int(top_k),
            "query": {
                "script_score": {
                    "query": {"bool": {"filter": filters}},
                    # cosineSimilarity in [-1,1], so `1 - sim` maps closer to smaller.
                    "script": {
                        "source": "1 - cosineSimilarity(params.qv, 'embedding')",
                        "params": {"qv": query_vector},
                    },
                }
            },
            "sort": [{"_score": {"order": "asc"}}],
        }

        resp = await self._es.search(index=self._index_name, body=body)
        hits = resp.get("hits", {}).get("hits", []) or []

        out: List[Dict[str, Any]] = []
        for h in hits:
            src = h.get("_source") or {}
            out.append(
                {
                    "content": src.get("content", ""),
                    "metadata": src.get("metadata") or {},
                    "score": float(h.get("_score") or 0.0),
                    "source": src.get("source"),
                    "doc_tier": int(src.get("doc_tier") or 1),
                    "document_id": str(h.get("_id") or ""),
                    "parent_id": src.get("parent_id"),
                }
            )
        return out

    async def get_documents_by_ids(
        self, ids: Sequence[str]
    ) -> List[Dict[str, Any]]:
        if not ids:
            return []

        resp = await self._es.mget(index=self._index_name, ids=list(ids))
        docs = resp.get("docs") or []

        out: List[Dict[str, Any]] = []
        for item in docs:
            if not item or not item.get("found"):
                continue
            src = item.get("_source") or {}
            out.append(
                {
                    "content": src.get("content", ""),
                    "metadata": src.get("metadata") or {},
                    "source": src.get("source"),
                    "doc_tier": int(src.get("doc_tier") or 1),
                    "document_id": str(item.get("_id") or ""),
                    "parent_id": src.get("parent_id"),
                }
            )
        return out

    async def delete_document(self, *, doc_id: str) -> None:
        await self._es.delete(index=self._index_name, id=doc_id, ignore=[404])


_ES_CLIENT: Optional[ESClient] = None


def get_es_client() -> ESClient:
    global _ES_CLIENT
    if _ES_CLIENT is None:
        _ES_CLIENT = ESClient()
    return _ES_CLIENT


__all__ = ["ESClient", "get_es_client"]

