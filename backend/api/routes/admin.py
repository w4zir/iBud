from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.models import Ticket
from ...db.postgres import get_session
from ...evaluation.pipeline import AsyncEvaluator
from ...rag.ingest_wixqa import ingest_wixqa
from ...rag.ingest_bitext import ingest_bitext
from ...remediation.drift import detect_model_data_drift
from ...remediation.engine import RemediationEngine
from ...remediation.events import recent_interventions
from ...remediation.governance import GovernanceConfig


router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/ingest")
async def admin_ingest_wixqa() -> dict:
    await ingest_wixqa()
    return {"status": "ok"}


@router.post("/ingest/bitext")
async def admin_ingest_bitext() -> dict:
    await ingest_bitext()
    return {"status": "ok"}


@router.get("/tickets")
async def admin_list_tickets(
    db: AsyncSession = Depends(get_session),
) -> List[dict]:
    result = await db.execute(
        select(Ticket).where(Ticket.status == "open").order_by(Ticket.created_at.desc())
    )
    tickets = []
    for t in result.scalars().all():
        tickets.append(
            {
                "id": str(t.id),
                "session_id": t.session_id,
                "issue_type": t.issue_type,
                "summary": t.summary,
                "status": t.status,
                "priority": t.priority,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
        )
    return tickets


@router.post("/eval/trigger")
async def admin_trigger_eval(limit: int = 25, min_age_minutes: int = 5) -> dict:
    evaluator = AsyncEvaluator()
    stats = await evaluator.run_batch(limit=limit, min_age_minutes=min_age_minutes)
    return {"status": "ok", **stats}


@router.post("/remediation/check")
async def admin_remediation_check() -> dict:
    engine = RemediationEngine()
    report = await engine.run(dry_run=True)
    drift = await detect_model_data_drift()
    return {"status": "ok", "report": report, "drift": drift.__dict__}


@router.post("/remediation/trigger")
async def admin_remediation_trigger() -> dict:
    engine = RemediationEngine()
    report = await engine.run(dry_run=False)
    drift = await detect_model_data_drift()
    return {"status": "ok", "report": report, "drift": drift.__dict__}


@router.get("/remediation/history")
async def admin_remediation_history(hours: int = 24) -> dict:
    history = await recent_interventions(hours=hours)
    return {"status": "ok", "count": len(history), "events": history}


@router.get("/remediation/config")
async def admin_remediation_config() -> dict:
    cfg = GovernanceConfig.from_env()
    return {
        "status": "ok",
        "config": {
            "global_enabled": cfg.global_enabled,
            "manual_override": cfg.manual_override,
            "min_cooldown_seconds": cfg.min_cooldown_seconds,
            "max_actions_per_hour": cfg.max_actions_per_hour,
            "rule_enabled_overrides": cfg.rule_enabled_overrides or {},
        },
    }


__all__ = ["router"]

