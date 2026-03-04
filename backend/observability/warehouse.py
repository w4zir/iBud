from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from ..config import log_event
from ..db.models import AgentSpan, EvaluationScore, Outcome, Session
from ..db.postgres import async_session_factory


async def record_span(
    *,
    session_id: Optional[str],
    trace_id: Optional[str],
    span_name: str,
    attributes: Optional[Dict[str, Any]] = None,
    latency_ms: Optional[float] = None,
) -> None:
    try:
        async with async_session_factory() as db:
            db.add(
                AgentSpan(
                    session_id=session_id,
                    trace_id=trace_id,
                    span_name=span_name,
                    attributes=attributes,
                    latency_ms=latency_ms,
                )
            )
            await db.commit()
    except Exception as exc:
        log_event("warehouse", "record_span_failed", span_name=span_name, error=str(exc))


async def record_outcome(
    *,
    session_id: Optional[str],
    task: str,
    completed: bool,
    escalated: bool,
    verified: bool = False,
) -> None:
    try:
        async with async_session_factory() as db:
            db.add(
                Outcome(
                    session_id=session_id,
                    task=task,
                    completed=completed,
                    escalated=escalated,
                    verified=verified,
                )
            )
            await db.commit()
    except Exception as exc:
        log_event("warehouse", "record_outcome_failed", task=task, error=str(exc))


async def record_evaluation_score(
    *,
    session_id: Optional[str],
    groundedness: Optional[float],
    hallucination: Optional[bool],
    helpfulness: Optional[float],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        async with async_session_factory() as db:
            db.add(
                EvaluationScore(
                    session_id=session_id,
                    groundedness=groundedness,
                    hallucination=hallucination,
                    helpfulness=helpfulness,
                    metadata_=metadata,
                )
            )
            await db.commit()
    except Exception as exc:
        log_event(
            "warehouse",
            "record_evaluation_score_failed",
            session_id=session_id,
            error=str(exc),
        )


async def update_session_analytics(
    *,
    session_id: Optional[str],
    intent: Optional[str] = None,
    escalated: Optional[bool] = None,
    resolved_at: Optional[datetime] = None,
) -> None:
    if not session_id:
        return
    try:
        async with async_session_factory() as db:
            session = await db.get(Session, session_id)
            if session is None:
                return
            if intent is not None:
                session.intent = intent
            if escalated is not None:
                session.escalated = escalated
            if resolved_at is not None:
                session.resolved_at = resolved_at
            await db.commit()
    except Exception as exc:
        log_event(
            "warehouse",
            "update_session_analytics_failed",
            session_id=session_id,
            error=str(exc),
        )


__all__ = [
    "record_span",
    "record_outcome",
    "record_evaluation_score",
    "update_session_analytics",
]

