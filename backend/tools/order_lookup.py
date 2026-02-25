from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import Order
from ..db.postgres import async_session_factory


async def _lookup_order(
    session: AsyncSession,
    order_number: Optional[str],
    user_id: Optional[str],
) -> List[Dict[str, Any]]:
    stmt = select(Order)
    if order_number:
        stmt = stmt.where(Order.order_number == order_number)
    if user_id:
        stmt = stmt.where(Order.user_id == user_id)

    result = await session.execute(stmt)
    orders = []
    for order in result.scalars().all():
        orders.append(
            {
                "id": str(order.id),
                "order_number": order.order_number,
                "user_id": order.user_id,
                "status": order.status,
                "items": order.items,
                "total_amount": float(order.total_amount) if order.total_amount else None,
                "created_at": order.created_at.isoformat() if order.created_at else None,
                "estimated_delivery": (
                    order.estimated_delivery.isoformat()
                    if order.estimated_delivery
                    else None
                ),
            }
        )
    return orders


async def order_lookup_tool(
    order_number: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Look up the status and details of a customer order by order number or user_id.
    """
    async with async_session_factory() as session:
        orders = await _lookup_order(session, order_number=order_number, user_id=user_id)
    return {"orders": orders}


__all__ = ["order_lookup_tool"]

