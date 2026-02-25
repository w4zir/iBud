import asyncio

import pytest
from sqlalchemy import select

from backend.db.models import Order
from backend.db.postgres import async_session_factory
from scripts.seed_mock_data import STATUSES, seed_orders


@pytest.mark.integration
@pytest.mark.asyncio
async def test_seed_creates_50_orders():
    total = await seed_orders(total=50)
    assert total >= 50


@pytest.mark.integration
@pytest.mark.asyncio
async def test_all_statuses_present():
    await seed_orders(total=50)

    async def _get_statuses() -> set[str]:
        async with async_session_factory() as session:
            result = await session.execute(select(Order.status))
            return {row[0] for row in result.all()}

    statuses = await _get_statuses()
    for status in STATUSES:
        assert status in statuses


@pytest.mark.integration
@pytest.mark.asyncio
async def test_order_number_uniqueness_idempotent():
    await seed_orders(total=50)
    second_total = await seed_orders(total=50)
    assert second_total >= 50

    async def _get_order_numbers() -> list[str]:
        async with async_session_factory() as session:
            result = await session.execute(select(Order.order_number))
            return [row[0] for row in result.all()]

    order_numbers = await _get_order_numbers()
    assert len(order_numbers) == len(set(order_numbers))

