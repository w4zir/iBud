from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.models import Message, Session
from ...db.postgres import get_session
from ..models import SessionHistoryResponse, SessionMessage


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(
    session_id: str,
    db: AsyncSession = Depends(get_session),
) -> SessionHistoryResponse:
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
    )
    messages: List[SessionMessage] = []
    for msg in result.scalars().all():
        messages.append(
            SessionMessage(
                role=msg.role,
                content=msg.content,
                created_at=msg.created_at,
            )
        )
    return SessionHistoryResponse(messages=messages)


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_session),
) -> dict:
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    await db.execute(delete(Message).where(Message.session_id == session_id))
    await db.commit()
    return {"status": "deleted"}


__all__ = ["router"]

