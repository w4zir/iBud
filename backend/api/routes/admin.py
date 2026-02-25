from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.models import Ticket
from ...db.postgres import get_session
from ...rag.ingest_wixqa import ingest_wixqa


router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/ingest")
async def admin_ingest_wixqa() -> dict:
    await ingest_wixqa()
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


__all__ = ["router"]

