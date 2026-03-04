from __future__ import annotations

from typing import Optional

from sqlalchemy import Float, and_, cast, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import AgentSpan, EvaluationScore, Outcome


async def automation_rate(session: AsyncSession) -> Optional[float]:
    """
    AI resolved / total conversations.

    Automation = completed AND NOT escalated (no human needed).
    """
    total_q = select(func.count()).select_from(Outcome)
    resolved_q = select(func.count()).select_from(Outcome).where(
        and_(Outcome.completed.is_(True), Outcome.verified.is_(False))
    )
    total = (await session.execute(total_q)).scalar() or 0
    if total == 0:
        return None
    resolved = (await session.execute(resolved_q)).scalar() or 0
    return resolved / total


async def escalation_rate(session: AsyncSession) -> Optional[float]:
    """Fraction of outcomes that were escalated (completed=False)."""
    total_q = select(func.count()).select_from(Outcome)
    escalated_q = select(func.count()).select_from(Outcome).where(
        Outcome.completed.is_(False)
    )
    total = (await session.execute(total_q)).scalar() or 0
    if total == 0:
        return None
    escalated = (await session.execute(escalated_q)).scalar() or 0
    return escalated / total


async def tool_success_rate(session: AsyncSession) -> Optional[float]:
    """
    Average tool success across all tool_call spans.

    Each tool_call span stores ``{"success": true/false}`` in its
    attributes JSONB column.
    """
    stmt = (
        select(
            func.avg(
                cast(
                    case(
                        (AgentSpan.attributes["success"].as_boolean().is_(True), 1),
                        else_=0,
                    ),
                    Float,
                )
            )
        )
        .select_from(AgentSpan)
        .where(AgentSpan.span_name == "tool_call")
    )
    result = (await session.execute(stmt)).scalar()
    return float(result) if result is not None else None


async def hallucination_rate(session: AsyncSession) -> Optional[float]:
    """Fraction of evaluated sessions flagged as hallucinated."""
    total_q = (
        select(func.count())
        .select_from(EvaluationScore)
        .where(EvaluationScore.hallucination.is_not(None))
    )
    hallucinated_q = (
        select(func.count())
        .select_from(EvaluationScore)
        .where(EvaluationScore.hallucination.is_(True))
    )
    total = (await session.execute(total_q)).scalar() or 0
    if total == 0:
        return None
    hallucinated = (await session.execute(hallucinated_q)).scalar() or 0
    return hallucinated / total


async def avg_resolution_latency_ms(session: AsyncSession) -> Optional[float]:
    """Mean latency across all recorded spans (milliseconds)."""
    stmt = (
        select(func.avg(AgentSpan.latency_ms))
        .select_from(AgentSpan)
        .where(AgentSpan.latency_ms.is_not(None))
    )
    result = (await session.execute(stmt)).scalar()
    return float(result) if result is not None else None


async def avg_groundedness(session: AsyncSession) -> Optional[float]:
    """Mean groundedness score across evaluated sessions."""
    stmt = (
        select(func.avg(EvaluationScore.groundedness))
        .select_from(EvaluationScore)
        .where(EvaluationScore.groundedness.is_not(None))
    )
    result = (await session.execute(stmt)).scalar()
    return float(result) if result is not None else None


async def avg_helpfulness(session: AsyncSession) -> Optional[float]:
    """Mean helpfulness score across evaluated sessions."""
    stmt = (
        select(func.avg(EvaluationScore.helpfulness))
        .select_from(EvaluationScore)
        .where(EvaluationScore.helpfulness.is_not(None))
    )
    result = (await session.execute(stmt)).scalar()
    return float(result) if result is not None else None


__all__ = [
    "automation_rate",
    "escalation_rate",
    "tool_success_rate",
    "hallucination_rate",
    "avg_resolution_latency_ms",
    "avg_groundedness",
    "avg_helpfulness",
]
