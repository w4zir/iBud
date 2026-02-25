from __future__ import annotations

from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import Ticket
from ..db.postgres import async_session_factory


async def _create_ticket(
    session: AsyncSession,
    session_id: Optional[str],
    issue_type: str,
    summary: str,
) -> str:
    ticket = Ticket(
        session_id=session_id,
        issue_type=issue_type,
        summary=summary,
    )
    session.add(ticket)
    await session.flush()
    return str(ticket.id)


async def ticket_create_tool(
    issue_type: str,
    summary: str,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a human escalation ticket linked to a chat session.
    """
    async with async_session_factory() as session:
        ticket_id = await _create_ticket(
            session=session,
            session_id=session_id,
            issue_type=issue_type,
            summary=summary,
        )
        await session.commit()

    return {
        "ticket_id": ticket_id,
        "issue_type": issue_type,
    }


__all__ = ["ticket_create_tool"]

