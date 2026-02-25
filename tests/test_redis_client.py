import json
from hashlib import md5
from typing import Any

import pytest
from unittest.mock import AsyncMock, patch

from backend.db.redis_client import CacheKeyParts, build_cache_key, get_cached, set_cached


def test_build_cache_key_is_deterministic():
    parts = CacheKeyParts(
        query="Where is my order?",
        category="shipping",
        tier_filter=1,
        top_k=5,
        rerank=True,
    )
    key1 = build_cache_key(parts)
    key2 = build_cache_key(parts)
    assert key1 == key2
    assert key1.startswith("retrieval:")


def test_build_cache_key_changes_with_filters():
    base = CacheKeyParts(
        query="Where is my order?",
        category="shipping",
        tier_filter=1,
        top_k=5,
        rerank=True,
    )
    k1 = build_cache_key(base)
    k2 = build_cache_key(CacheKeyParts(**{**base.__dict__, "tier_filter": 2}))
    k3 = build_cache_key(CacheKeyParts(**{**base.__dict__, "category": "billing"}))
    assert len({k1, k2, k3}) == 3


@pytest.mark.asyncio
async def test_get_cached_miss_returns_none():
    async_mock = AsyncMock()
    async_mock.get.return_value = None

    with patch("backend.db.redis_client.get_client", return_value=async_mock):
        value = await get_cached("missing-key")

    assert value is None


@pytest.mark.asyncio
async def test_get_cached_hit_deserializes_json():
    payload: dict[str, Any] = {"foo": "bar", "n": 1}
    async_mock = AsyncMock()
    async_mock.get.return_value = json.dumps(payload)

    with patch("backend.db.redis_client.get_client", return_value=async_mock):
        value = await get_cached("some-key")

    assert value == payload


@pytest.mark.asyncio
async def test_get_cached_handles_redis_errors_gracefully():
    async_mock = AsyncMock()
    async_mock.get.side_effect = RuntimeError("boom")

    with patch("backend.db.redis_client.get_client", return_value=async_mock):
        value = await get_cached("some-key")

    assert value is None


@pytest.mark.asyncio
async def test_set_cached_writes_json_with_ttl():
    async_mock = AsyncMock()

    with patch("backend.db.redis_client.get_client", return_value=async_mock), \
         patch("backend.db.redis_client._get_redis_params", return_value={"host": "x", "port": 1, "ttl": 123}):
        await set_cached("k", {"a": 1})

    async_mock.set.assert_awaited_once()
    args, kwargs = async_mock.set.call_args
    assert args[0] == "k"
    assert json.loads(args[1]) == {"a": 1}
    assert kwargs.get("ex") == 123


@pytest.mark.asyncio
async def test_set_cached_handles_errors_gracefully():
    async_mock = AsyncMock()
    async_mock.set.side_effect = RuntimeError("boom")

    with patch("backend.db.redis_client.get_client", return_value=async_mock):
        # Should not raise
        await set_cached("k", {"a": 1})

