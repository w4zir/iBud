from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from sqlalchemy import select

from ..db.models import AgentSpan
from ..db.postgres import async_session_factory
from ..observability.warehouse import record_span


async def record_intervention(
    *,
    session_id: str | None,
    trace_id: str | None,
    rule_name: str,
    action_taken: str,
    trigger_metrics: Dict[str, Any],
    outcome: str,
) -> None:
    await record_span(
        session_id=session_id,
        trace_id=trace_id,
        span_name="remediation",
        attributes={
            "rule_name": rule_name,
            "action_taken": action_taken,
            "trigger_metrics": trigger_metrics,
            "outcome": outcome,
        },
        latency_ms=None,
    )


async def recent_interventions(*, hours: int = 24) -> List[Dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    async with async_session_factory() as db:
        result = await db.execute(
            select(AgentSpan)
            .where(AgentSpan.span_name == "remediation")
            .where(AgentSpan.created_at >= cutoff)
            .order_by(AgentSpan.created_at.desc())
            .limit(200)
        )
        rows = result.scalars().all()

    payload: List[Dict[str, Any]] = []
    for row in rows:
        attrs = row.attributes or {}
        payload.append(
            {
                "id": str(row.id),
                "session_id": row.session_id,
                "trace_id": row.trace_id,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "rule_name": attrs.get("rule_name"),
                "action_taken": attrs.get("action_taken"),
                "trigger_metrics": attrs.get("trigger_metrics"),
                "outcome": attrs.get("outcome"),
            }
        )
    return payload
