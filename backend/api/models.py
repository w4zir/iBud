from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    session_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("session_id", "sessionId"),
    )
    user_id: str = Field(
        default="anonymous",
        validation_alias=AliasChoices("user_id", "userId"),
    )
    message: str = Field(
        ...,
        validation_alias=AliasChoices("message", "userMessage", "text", "query"),
    )
    # Optional dataset key for retrieval (e.g. "wixqa", "bitext"). Defaults server-side.
    dataset: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("dataset", "datasetId"),
    )
    # Optional company identifier for scoping (e.g. "foodpanda"). Defaults server-side.
    company: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("company", "companyId"),
    )


class IntentClassifyRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    session_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("session_id", "sessionId"),
    )
    user_id: str = Field(
        default="anonymous",
        validation_alias=AliasChoices("user_id", "userId"),
    )
    message: str = Field(
        ...,
        validation_alias=AliasChoices("message", "userMessage", "text", "query"),
    )
    # Optional dataset key for retrieval context (e.g. "wixqa", "bitext").
    dataset: Optional[str] = Field(
        default="wixqa",
        validation_alias=AliasChoices("dataset", "datasetId"),
    )
    # Optional prompt profile override for intent classification (e.g. "default", "bitext").
    intent_prompt_profile: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("intent_prompt_profile", "intentPromptProfile"),
    )
    # Optional company identifier for scoping intent classification.
    company: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("company", "companyId"),
    )


class Source(BaseModel):
    content: str
    category: Optional[str] = None
    score: float
    source: Optional[str] = None
    doc_tier: int
    document_id: str
    parent_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    sources: List[Source]
    tools_used: List[str]
    escalated: bool
    # Optional classified intent for this turn/session; populated by the agent.
    intent: Optional[str] = None
    ticket_id: Optional[str] = None


class AgentStepEvent(BaseModel):
    id: str
    label: str
    detail: Optional[str] = None
    status: Literal["started", "completed", "info", "failed"] = "info"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentStreamEvent(BaseModel):
    type: Literal["step", "final", "error"]
    step: Optional[AgentStepEvent] = None
    final: Optional[ChatResponse] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class IntentClassifyResponse(BaseModel):
    session_id: str
    # Classified intent label (coarse or Bitext fine-grained, depending on profile).
    intent: Optional[str] = None
    # Resolved prompt profile used for this classification.
    intent_prompt_profile: Optional[str] = None


class SessionMessage(BaseModel):
    role: str
    content: str
    created_at: datetime


class SessionHistoryResponse(BaseModel):
    messages: List[SessionMessage]


class HealthResponse(BaseModel):
    status: str
    postgres: bool
    redis: bool
    ollama: bool
    classifier: bool


__all__ = [
    "ChatRequest",
    "IntentClassifyRequest",
    "ChatResponse",
    "AgentStepEvent",
    "AgentStreamEvent",
    "IntentClassifyResponse",
    "Source",
    "SessionMessage",
    "SessionHistoryResponse",
    "HealthResponse",
]

