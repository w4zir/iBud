from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from backend.analytics import metrics


class _FakeResult:
    def __init__(self, row: Any) -> None:
        self._row = row

    def first(self):
        return self._row


class _FakeDB:
    def __init__(self, scalars: list[Any] | None = None, row: Any = None) -> None:
        self._scalars = scalars or []
        self._row = row

    async def scalar(self, *args: Any, **kwargs: Any) -> Any:
        return self._scalars.pop(0)

    async def execute(self, *args: Any, **kwargs: Any) -> _FakeResult:
        return _FakeResult(self._row)


@pytest.mark.asyncio
async def test_automation_rate():
    db = _FakeDB(scalars=[10, 6])  # total, automated
    value = await metrics.automation_rate(db)  # type: ignore[arg-type]
    assert value == 0.6


@pytest.mark.asyncio
async def test_escalation_rate():
    db = _FakeDB(scalars=[20, 5])  # total, escalated
    value = await metrics.escalation_rate(db)  # type: ignore[arg-type]
    assert value == 0.25


@pytest.mark.asyncio
async def test_tool_success_rate():
    db = _FakeDB(row=SimpleNamespace(success_count=7, failure_count=3))
    value = await metrics.tool_success_rate(db)  # type: ignore[arg-type]
    assert value == 0.7

