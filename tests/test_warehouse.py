from __future__ import annotations

from typing import Any

import pytest

from backend.observability import warehouse


class _FakeDB:
    def __init__(self) -> None:
        self.added: list[Any] = []
        self.committed = False

    def add(self, item: Any) -> None:
        self.added.append(item)

    async def commit(self) -> None:
        self.committed = True

    async def get(self, model: Any, key: str) -> Any:
        class _Session:
            intent = None
            escalated = False
            resolved_at = None

        return _Session()


class _Factory:
    def __init__(self, db: _FakeDB) -> None:
        self.db = db

    async def __aenter__(self) -> _FakeDB:
        return self.db

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


@pytest.mark.asyncio
async def test_record_outcome_writes_row(monkeypatch):
    db = _FakeDB()
    monkeypatch.setattr("backend.observability.warehouse.async_session_factory", lambda: _Factory(db))
    await warehouse.record_outcome(
        session_id="sess-1",
        task="conversation",
        completed=True,
        escalated=False,
        verified=False,
    )
    assert len(db.added) == 1
    assert db.committed is True

