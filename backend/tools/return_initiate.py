from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import Order
from ..db.postgres import async_session_factory


RETURN_WINDOW_DAYS = 30


async def _get_order_by_number(
    session: AsyncSession,
    order_number: str,
) -> Optional[Order]:
    result = await session.execute(
        select(Order).where(Order.order_number == order_number)
    )
    return result.scalars().first()


async def _is_return_eligible(order: Order) -> bool:
    if order.status != "delivered":
        return False
    if not order.created_at:
        return False
    now = datetime.now(timezone.utc)
    return now - order.created_at <= timedelta(days=RETURN_WINDOW_DAYS)


async def return_initiate_tool(
    order_number: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Initiate a return for an order if it is eligible.

    - Order must be delivered
    - Order must be less than RETURN_WINDOW_DAYS old
    """
    async with async_session_factory() as session:
        order = await _get_order_by_number(session, order_number)
        if order is None:
            return {"success": False, "reason": "order_not_found"}

        if user_id and order.user_id and order.user_id != user_id:
            return {"success": False, "reason": "user_mismatch"}

        if not await _is_return_eligible(order):
            return {"success": False, "reason": "not_eligible"}

        rma_number = f"RMA-{order.order_number}"

        await session.execute(
            update(Order)
            .where(Order.id == order.id)
            .values(status="return_initiated")
        )
        await session.commit()

    return {
        "success": True,
        "order_number": order_number,
        "rma_number": rma_number,
    }


__all__ = ["return_initiate_tool", "RETURN_WINDOW_DAYS"]

