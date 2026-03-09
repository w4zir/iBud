from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_id: str
    message: str
    # Optional dataset key for retrieval (e.g. "wixqa", "bitext").
    dataset: Optional[str] = "wixqa"


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


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "Source",
    "SessionMessage",
    "SessionHistoryResponse",
    "HealthResponse",
]

