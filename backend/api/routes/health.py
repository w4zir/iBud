from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.postgres import async_engine, get_session
from ...db.redis_client import get_client as get_redis_client
from ...observability.prometheus_metrics import request_count
from ..models import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(db: AsyncSession = Depends(get_session)) -> HealthResponse:
    postgres_ok = False
    redis_ok = False
    ollama_ok = False
    classifier_ok = False

    try:
        await db.execute(text("SELECT 1"))
        postgres_ok = True
    except Exception:
        postgres_ok = False

    try:
        redis = get_redis_client()
        pong = await redis.ping()
        redis_ok = bool(pong)
    except Exception:
        redis_ok = False

    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{base_url}/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception:
        ollama_ok = False

    classifier_url = os.getenv("CLASSIFIER_BENTOML_URL", "").strip()
    if classifier_url:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.post(classifier_url, json={"text": "health"})
                classifier_ok = resp.status_code == 200
        except Exception:
            classifier_ok = False

    status = "ok" if postgres_ok and redis_ok else "degraded"

    try:
        request_count.labels(status=status).inc()
    except Exception:
        pass

    return HealthResponse(
        status=status,
        postgres=postgres_ok,
        redis=redis_ok,
        ollama=ollama_ok,
        classifier=classifier_ok,
    )


@router.get("/metrics")
async def metrics() -> PlainTextResponse:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    data = generate_latest()
    return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


__all__ = ["router"]

