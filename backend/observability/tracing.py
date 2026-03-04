from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class Span:
    """
    Single agent action within a conversation trace.

    Maps to the OTel span concept: each agent node (intent detection,
    retrieval, tool execution, etc.) produces one span per invocation.
    """

    trace_id: str
    span_id: str
    span_name: str
    attributes: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    latency_ms: Optional[float] = None

    def finish(self) -> None:
        self.end_time = datetime.now(timezone.utc)
        if self.start_time:
            delta = (self.end_time - self.start_time).total_seconds() * 1000
            self.latency_ms = round(delta, 2)


@dataclass
class Trace:
    """
    Conversation-level trace collecting child spans.

    In the OTel model a Trace represents the full conversation session.
    Child spans are created for each discrete agent action (classify,
    retrieve, tool_call, synthesize, outcome).
    """

    trace_id: str
    session_id: str
    channel: Optional[str] = None
    user_id: Optional[str] = None
    spans: List[Span] = field(default_factory=list)

    def start_span(self, name: str, **attributes: Any) -> Span:
        span = Span(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4()),
            span_name=name,
            attributes=dict(attributes),
            start_time=datetime.now(timezone.utc),
        )
        self.spans.append(span)
        return span

    @property
    def root_attributes(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "channel": self.channel,
            "user_id": self.user_id,
        }


def create_trace(
    session_id: str,
    *,
    channel: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Trace:
    """Factory for a new conversation trace (trace_id == session_id)."""
    return Trace(
        trace_id=session_id,
        session_id=session_id,
        channel=channel,
        user_id=user_id,
    )


__all__ = ["Span", "Trace", "create_trace"]
