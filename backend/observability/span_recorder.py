from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..db.models import AgentSpan, EvaluationScore, Outcome, Session
from ..db.postgres import async_session_factory
from .tracing import Span, Trace

logger = logging.getLogger(__name__)


async def record_span(
    trace_id: str,
    span_name: str,
    attributes: Dict[str, Any],
    latency_ms: Optional[float] = None,
) -> None:
    """Persist a single agent span to the event warehouse.  Fire-and-forget."""
    try:
        async with async_session_factory() as session:
            row = AgentSpan(
                trace_id=trace_id,
                span_name=span_name,
                attributes=attributes,
                latency_ms=latency_ms,
            )
            session.add(row)
            await session.commit()
    except Exception:
        logger.warning(
            "Failed to record span %s for trace %s",
            span_name,
            trace_id,
            exc_info=True,
        )


async def record_outcome(
    session_id: str,
    task: str,
    completed: bool,
    verified: bool = False,
) -> None:
    """Persist a task outcome to the event warehouse."""
    try:
        async with async_session_factory() as session:
            row = Outcome(
                session_id=session_id,
                task=task,
                completed=completed,
                verified=verified,
            )
            session.add(row)
            await session.commit()
    except Exception:
        logger.warning(
            "Failed to record outcome for session %s",
            session_id,
            exc_info=True,
        )


async def record_evaluation(
    session_id: str,
    groundedness: Optional[float] = None,
    hallucination: Optional[bool] = None,
    helpfulness: Optional[float] = None,
    policy_compliance: Optional[float] = None,
) -> None:
    """Persist evaluation scores to the event warehouse."""
    try:
        async with async_session_factory() as session:
            row = EvaluationScore(
                session_id=session_id,
                groundedness=groundedness,
                hallucination=hallucination,
                helpfulness=helpfulness,
                policy_compliance=policy_compliance,
            )
            session.add(row)
            await session.commit()
    except Exception:
        logger.warning(
            "Failed to record evaluation for session %s",
            session_id,
            exc_info=True,
        )


async def finalize_session(
    session_id: str,
    *,
    intent: Optional[str] = None,
    escalated: Optional[bool] = None,
) -> None:
    """Update the session row with end-of-conversation enrichment."""
    try:
        async with async_session_factory() as session:
            row = await session.get(Session, session_id)
            if row is None:
                return
            row.end_time = datetime.now(timezone.utc)
            if intent is not None:
                row.intent = intent
            if escalated is not None:
                row.escalated = escalated
            await session.commit()
    except Exception:
        logger.warning(
            "Failed to finalize session %s",
            session_id,
            exc_info=True,
        )


async def persist_trace(trace: Trace) -> None:
    """Bulk-persist all spans from a completed trace."""
    for span in trace.spans:
        await record_span(
            trace_id=span.trace_id,
            span_name=span.span_name,
            attributes=span.attributes,
            latency_ms=span.latency_ms,
        )


__all__ = [
    "record_span",
    "record_outcome",
    "record_evaluation",
    "finalize_session",
    "persist_trace",
]
