import json
import logging
import os
from dataclasses import dataclass
from hashlib import md5
from typing import Any, Dict, Optional

from redis.asyncio import Redis


logger = logging.getLogger(__name__)


def _get_redis_params() -> Dict[str, Any]:
    host = os.getenv("REDIS_HOST", "redis")
    port = int(os.getenv("REDIS_PORT", "6379"))
    ttl = int(os.getenv("REDIS_CACHE_TTL", "300"))
    return {"host": host, "port": port, "ttl": ttl}


_redis_client: Optional[Redis] = None


def get_client() -> Redis:
    """
    Return a singleton asyncio Redis client.
    """
    global _redis_client
    if _redis_client is None:
        params = _get_redis_params()
        _redis_client = Redis(
            host=params["host"],
            port=params["port"],
            decode_responses=True,
        )
    return _redis_client


@dataclass(frozen=True)
class CacheKeyParts:
    query: str
    category: Optional[str]
    tier_filter: Optional[int]
    top_k: int
    rerank: bool


def build_cache_key(parts: CacheKeyParts) -> str:
    """
    Build a deterministic md5 cache key from the query and filters.
    """
    payload = {
        "query": parts.query,
        "category": parts.category,
        "tier_filter": parts.tier_filter,
        "top_k": parts.top_k,
        "rerank": parts.rerank,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = md5(raw.encode("utf-8")).hexdigest()
    return f"retrieval:{digest}"


async def get_cached(key: str) -> Optional[Any]:
    """
    Fetch a cached value by key.

    Returns deserialized JSON or None on miss or Redis error.
    """
    client = get_client()
    try:
        data = await client.get(key)
        if data is None:
            return None
        return json.loads(data)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Redis get failed for key %s: %s", key, exc)
        return None


async def set_cached(key: str, value: Any) -> None:
    """
    Store a JSON-serializable value under key with TTL.

    Fails silently (logs) if Redis is unavailable.
    """
    client = get_client()
    params = _get_redis_params()
    ttl = params["ttl"]
    try:
        payload = json.dumps(value)
        await client.set(key, payload, ex=ttl)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Redis set failed for key %s: %s", key, exc)


__all__ = [
    "CacheKeyParts",
    "build_cache_key",
    "get_client",
    "get_cached",
    "set_cached",
]

