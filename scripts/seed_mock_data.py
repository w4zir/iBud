import asyncio
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List

# Ensure the project root (which contains the `backend` package) is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.db.models import Order
from backend.db.postgres import async_session_factory
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert


STATUSES: List[str] = ["processing", "in-transit", "delivered", "returned"]


def _build_mock_order(i: int) -> Dict[str, Any]:
    order_number = f"ORD-{10000 + i}"
    user_id = f"user-{(i % 10) + 1}"
    status = STATUSES[i % len(STATUSES)]

    items = [
        {
            "sku": f"SKU-{i}-{j}",
            "name": f"Product {j}",
            "quantity": 1 + (j % 3),
            "unit_price": float(19.99 + j * 5),
        }
        for j in range(1, 4)
    ]

    total_amount = sum(item["unit_price"] * item["quantity"] for item in items)

    created_at = datetime.now(timezone.utc) - timedelta(days=random.randint(1, 30))
    estimated_delivery = created_at + timedelta(days=5)

    return {
        "order_number": order_number,
        "user_id": user_id,
        "status": status,
        "items": items,
        "total_amount": Decimal(str(round(total_amount, 2))),
        "created_at": created_at,
        "estimated_delivery": estimated_delivery,
    }


async def seed_orders(total: int = 50) -> int:
    """
    Seed mock orders into the database.

    Uses an upsert on order_number to be idempotent across runs.
    Returns the total number of orders in the table after seeding.
    """
    async with async_session_factory() as session:
        orders = [_build_mock_order(i) for i in range(total)]

        stmt = insert(Order.__table__).values(orders)
        stmt = stmt.on_conflict_do_nothing(index_elements=["order_number"])
        await session.execute(stmt)
        await session.commit()

        result = await session.execute(select(Order))
        return len(result.scalars().all())


async def _main() -> None:
    total = await seed_orders()
    print(f"Seeded mock orders. orders count={total}")


if __name__ == "__main__":
    asyncio.run(_main())

