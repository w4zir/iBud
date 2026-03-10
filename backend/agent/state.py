from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


Intent = Literal[
    # Coarse-grained intents used by the main chat agent.
    "order_status",
    "return_request",
    "product_qa",
    "account_issue",
    "complaint",
    "other",
    # Fine-grained Bitext intents used for intent-eval experiments.
    "cancel_order",
    "change_order",
    "change_shipping_address",
    "check_cancellation_fee",
    "check_invoice",
    "check_payment_methods",
    "check_refund_policy",
    "contact_customer_service",
    "contact_human_agent",
    "create_account",
    "delete_account",
    "delivery_options",
    "delivery_period",
    "edit_account",
    "get_invoice",
    "get_refund",
    "newsletter_subscription",
    "payment_issue",
    "place_order",
    "recover_password",
    "registration_problems",
    "review",
    "set_up_shipping_address",
    "switch_account",
    "track_order",
    "track_refund",
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
    # When True, suppress external observability signals (LangSmith / OTel / Prometheus).
    # Intended for intent-only evaluation flow.
    observability_disabled: bool
    # Active knowledge-base dataset key (e.g. "wixqa", "bitext").
    dataset: Optional[str]
    # Optional prompt profile name for intent classification (e.g. "default", "bitext").
    intent_prompt_profile: Optional[str]
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

