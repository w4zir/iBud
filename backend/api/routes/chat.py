from __future__ import annotations

import inspect
import json
import time
from hashlib import md5
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...agent import graph as agent_graph
from ...agent.state import AgentState
from ...config import debug_print
from ...db.models import Message, Session
from ...db.postgres import get_session
from ...db.redis_client import get_client as get_redis_client
from ...observability.prometheus_metrics import cache_hits, request_count, request_latency
from ..models import ChatRequest, ChatResponse, Source


router = APIRouter(prefix="/chat", tags=["chat"])

# In-process fallback cache used when Redis is unavailable (e.g., in tests).
_local_chat_cache: Dict[str, Dict[str, Any]] = {}


def _build_chat_cache_key(session_id: str, user_id: str, message: str) -> str:
    payload = {
        "session_id": session_id,
        "user_id": user_id,
        "message": message,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = md5(raw.encode("utf-8")).hexdigest()
    return f"chat:{digest}"


async def _get_or_create_session(
    db: AsyncSession,
    session_id: Optional[str],
    user_id: str,
) -> Session:
    if session_id:
        existing = await db.get(Session, session_id)
        if existing:
            return existing

    session = Session(user_id=user_id)
    db.add(session)
    await db.flush()
    return session


async def _load_message_history(db: AsyncSession, session_id: str) -> List[Dict[str, Any]]:
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
    )
    messages: List[Dict[str, Any]] = []
    for msg in result.scalars().all():
        messages.append(
            {
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat(),
            }
        )
    return messages


def _build_sources_from_state(state: AgentState) -> List[Source]:
    docs = state.get("retrieved_docs") or []
    sources: List[Source] = []
    for d in docs:
        metadata = d.get("metadata") or {}
        sources.append(
            Source(
                content=d.get("content", ""),
                category=metadata.get("category"),
                score=float(d.get("score", 0.0)),
                source=d.get("source"),
                doc_tier=int(d.get("doc_tier", 1)),
                document_id=str(d.get("document_id")),
                parent_id=d.get("parent_id"),
            )
        )
    return sources


def _tools_used_from_state(state: AgentState) -> List[str]:
    results = state.get("tool_results") or []
    names = [str(r.get("name")) for r in results if r.get("name")]
    return sorted(set(names))


async def _run_agent_flow(
    db: AsyncSession,
    req: ChatRequest,
    *,
    use_cache: bool = True,
) -> ChatResponse:
    start = time.perf_counter()
    try:
        session = await _get_or_create_session(db, req.session_id, req.user_id)

        debug_print(
            "chat",
            "request",
            session_id=str(session.id),
            user_id=req.user_id,
            message_len=len(req.message),
        )

        user_msg = Message(
            session_id=session.id,
            role="user",
            content=req.message,
        )
        db.add(user_msg)
        await db.flush()

        # Commit so the session (and user message) are visible to tools that use
        # a separate DB connection (e.g. ticket_create_tool) during the agent run.
        await db.commit()

        cache_hit = False
        cached_payload: Optional[Dict[str, Any]] = None
        cache_key = _build_chat_cache_key(str(session.id), req.user_id, req.message)
        if use_cache:
            # First try Redis; on error, fall back to local in-memory cache.
            try:
                redis = get_redis_client()
                data = await redis.get(cache_key)
                if data:
                    cached_payload = json.loads(data)
                    cache_hit = True
            except Exception:
                cached_payload = _local_chat_cache.get(cache_key)
                cache_hit = cached_payload is not None

        debug_print("chat", "cache", cache_hit=cache_hit)

        if cache_hit and cached_payload is not None:
            assistant_content = cached_payload.get("response", "")
            assistant_msg = Message(
                session_id=session.id,
                role="assistant",
                content=assistant_content,
            )
            db.add(assistant_msg)
            await db.commit()
            try:
                cache_hits.inc()
            except Exception:
                pass
            resp = ChatResponse(**cached_payload)
            debug_print("chat", "returning cached response")
            return resp

        debug_print("chat", "agent start")
        history = await _load_message_history(db, session_id=session.id)
        history.append(
            {
                "role": "user",
                "content": req.message,
            }
        )

        state: AgentState = {
            "session_id": str(session.id),
            "user_id": req.user_id,
            "messages": history,
        }
        run_result = agent_graph.run_agent(state, thread_id=str(session.id))
        if inspect.isawaitable(run_result):
            final_state = await run_result
        else:
            final_state = run_result
        response_text = final_state.get("final_response") or ""

        debug_print(
            "chat",
            "agent end",
            response_len=len(response_text),
            sources_count=len(final_state.get("retrieved_docs") or []),
            tools_count=len(final_state.get("tool_results") or []),
        )

        assistant_msg = Message(
            session_id=session.id,
            role="assistant",
            content=response_text,
        )
        db.add(assistant_msg)
        await db.commit()

        sources = _build_sources_from_state(final_state)
        tools_used = _tools_used_from_state(final_state)

        payload: Dict[str, Any] = {
            "session_id": str(session.id),
            "response": response_text,
            "sources": [s.dict() for s in sources],
            "tools_used": tools_used,
            "escalated": bool(final_state.get("should_escalate", False)),
            "ticket_id": final_state.get("ticket_id"),
        }

        if use_cache:
            try:
                redis = get_redis_client()
                ttl = int(
                    __import__("os").getenv("REDIS_CACHE_TTL", "300")  # lazy import
                )
                await redis.set(cache_key, json.dumps(payload), ex=ttl)
            except Exception:
                # Fallback to local in-memory cache when Redis is not available.
                _local_chat_cache[cache_key] = payload

        debug_print(
            "chat",
            "response",
            session_id=str(session.id),
            sources=len(sources),
            tools_used=tools_used,
            escalated=bool(final_state.get("should_escalate", False)),
        )
        resp = ChatResponse(**payload)
        return resp
    finally:
        elapsed = time.perf_counter() - start
        try:
            request_count.labels(status="ok").inc()
            request_latency.observe(elapsed)
        except Exception:
            pass


@router.post("/", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    db: AsyncSession = Depends(get_session),
) -> JSONResponse:
    if not req.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="message must not be empty",
        )

    resp = await _run_agent_flow(db, req, use_cache=True)
    return JSONResponse(content=resp.dict())


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    db: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    if not req.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="message must not be empty",
        )

    async def event_generator() -> AsyncGenerator[bytes, None]:
        resp = await _run_agent_flow(db, req, use_cache=True)
        data = json.dumps(resp.dict())
        yield f"data: {data}\n\n".encode("utf-8")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


__all__ = ["router"]

