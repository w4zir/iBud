from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


Intent = Literal[
    "order_status",
    "return_request",
    "product_qa",
    "account_issue",
    "complaint",
    "other",
]


class ToolCall(TypedDict):
    name: str
    arguments: Dict[str, Any]


class ToolResult(TypedDict):
    name: str
    success: bool
    result: Any
    error: Optional[str]


class RetrievedDocState(TypedDict):
    content: str
    metadata: Dict[str, Any]
    score: float
    source: Optional[str]
    doc_tier: int
    document_id: str
    parent_id: Optional[str]


class AgentState(TypedDict, total=False):
    """
    Canonical LangGraph agent state.

    This is intentionally simple and JSON-serializable so it can be
    persisted by a checkpointer and exposed over the API.
    """

    session_id: Optional[str]
    user_id: Optional[str]
    request_id: Optional[str]
    trace_id: Optional[str]
    # Active knowledge-base dataset key (e.g. "wixqa", "bitext").
    dataset: Optional[str]
    messages: List[Dict[str, Any]]
    intent: Optional[Intent]
    retrieved_docs: List[RetrievedDocState]
    tool_calls: List[ToolCall]
    tool_results: List[ToolResult]
    final_response: Optional[str]
    should_escalate: bool
    ticket_id: Optional[str]


__all__ = [
    "AgentState",
    "Intent",
    "ToolCall",
    "ToolResult",
    "RetrievedDocState",
]

