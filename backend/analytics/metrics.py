from __future__ import annotations

from typing import Optional

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import Message, Outcome, Session


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> float:
    num = float(numerator or 0.0)
    den = float(denominator or 0.0)
    if den <= 0:
        return 0.0
    return num / den


async def automation_rate(db: AsyncSession) -> float:
    total = await db.scalar(select(func.count()).select_from(Outcome))
    automated = await db.scalar(
        select(func.count()).select_from(Outcome).where(
            Outcome.completed.is_(True),
            Outcome.escalated.is_(False),
        )
    )
    return _safe_ratio(float(automated or 0), float(total or 0))


async def escalation_rate(db: AsyncSession) -> float:
    total = await db.scalar(select(func.count()).select_from(Outcome))
    escalated = await db.scalar(
        select(func.count()).select_from(Outcome).where(Outcome.escalated.is_(True))
    )
    return _safe_ratio(float(escalated or 0), float(total or 0))


async def fcr_proxy(db: AsyncSession) -> float:
    total_sessions = await db.scalar(select(func.count()).select_from(Session))
    resolved_single = await db.scalar(
        select(func.count())
        .select_from(Outcome)
        .join(Session, Session.id == Outcome.session_id)
        .where(
            Outcome.completed.is_(True),
            Outcome.escalated.is_(False),
            Session.resolved_at.is_not(None),
        )
    )
    return _safe_ratio(float(resolved_single or 0), float(total_sessions or 0))


async def tool_success_rate(db: AsyncSession) -> float:
    # Parse per-tool spans where execute_tool writes {"success_count", "failure_count"}.
    query = text(
        """
        SELECT
          COALESCE(SUM((attributes->>'success_count')::float), 0) AS success_count,
          COALESCE(SUM((attributes->>'failure_count')::float), 0) AS failure_count
        FROM agent_spans
        WHERE span_name = 'execute_tool'
        """
    )
    result = await db.execute(query)
    row = result.first()
    if not row:
        return 0.0
    success_count = float(row.success_count or 0.0)
    failure_count = float(row.failure_count or 0.0)
    return _safe_ratio(success_count, success_count + failure_count)


async def turns_to_resolution(db: AsyncSession) -> float:
    query = text(
        """
        SELECT AVG(message_counts.msg_count)::float AS avg_turns
        FROM (
          SELECT m.session_id, COUNT(*) AS msg_count
          FROM messages m
          JOIN sessions s ON s.id = m.session_id
          WHERE s.resolved_at IS NOT NULL
          GROUP BY m.session_id
        ) AS message_counts
        """
    )
    result = await db.execute(query)
    row = result.first()
    return float(row.avg_turns or 0.0) if row else 0.0


async def recovery_rate(db: AsyncSession) -> float:
    # Recovery proxy: session had tool failures but still completed.
    query = text(
        """
        WITH tool_stats AS (
          SELECT
            session_id,
            COALESCE(SUM((attributes->>'failure_count')::float), 0) AS failures
          FROM agent_spans
          WHERE span_name = 'execute_tool'
          GROUP BY session_id
        ),
        recovered AS (
          SELECT COUNT(*)::float AS count
          FROM tool_stats ts
          JOIN outcomes o ON o.session_id = ts.session_id
          WHERE ts.failures > 0 AND o.completed = TRUE
        ),
        with_failures AS (
          SELECT COUNT(*)::float AS count
          FROM tool_stats
          WHERE failures > 0
        )
        SELECT
          COALESCE((SELECT count FROM recovered), 0) AS recovered_count,
          COALESCE((SELECT count FROM with_failures), 0) AS with_failures_count
        """
    )
    result = await db.execute(query)
    row = result.first()
    if not row:
        return 0.0
    return _safe_ratio(float(row.recovered_count or 0.0), float(row.with_failures_count or 0.0))


__all__ = [
    "automation_rate",
    "escalation_rate",
    "fcr_proxy",
    "tool_success_rate",
    "turns_to_resolution",
    "recovery_rate",
]

