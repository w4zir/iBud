from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..rag.retriever import RetrievedDoc, Retriever


async def faq_search_tool(
    query: str,
    category: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Thin wrapper over Retriever.search() for use as an agent tool.
    """
    retriever = Retriever()
    docs: List[RetrievedDoc] = await retriever.search(
        query=query,
        top_k=top_k,
        category=category,
        tier_filter=1,
        use_cache=True,
        rerank=True,
    )
    serialized = [
        {
            "content": d.content,
            "metadata": d.metadata,
            "score": d.score,
            "source": d.source,
            "doc_tier": d.doc_tier,
            "document_id": d.document_id,
            "parent_id": d.parent_id,
        }
        for d in docs
    ]
    return {"results": serialized}


__all__ = ["faq_search_tool"]

