from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from hashlib import md5
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...agent.orchestrator import classify_intent_only, run_orchestrated_agent
from ...agent.state import AgentState
from ...config import log_event
from ...db.models import Message, Session
from ...db.postgres import get_session
from ...db.redis_client import get_client as get_redis_client
from ...observability.prometheus_metrics import (
    cache_hits,
    chat_turns,
    error_count,
    request_count,
    request_latency,
)
from ...observability.langsmith_tracer import chat_request_trace
from ...observability.otel import get_current_trace_ids, get_tracer
from ...observability.warehouse import record_outcome, update_session_analytics
from ..models import (
    ChatRequest,
    ChatResponse,
    IntentClassifyRequest,
    IntentClassifyResponse,
    Source,
)


router = APIRouter(prefix="/chat", tags=["chat"])

# In-process fallback cache used when Redis is unavailable (e.g., in tests).
_local_chat_cache: Dict[str, Dict[str, Any]] = {}


def _build_chat_cache_key(session_id: str, user_id: str, message: str, company: str) -> str:
    payload = {
        "session_id": session_id,
        "user_id": user_id,
        "message": message,
        "company": company,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = md5(raw.encode("utf-8")).hexdigest()
    return f"chat:{digest}"


async def _get_or_create_session(
    db: AsyncSession,
    session_id: Optional[str],
    user_id: str,
    company: str,
) -> Session:
    if session_id:
        existing = await db.get(Session, session_id)
        if existing:
            # Enforce per-session company consistency.
            if existing.company_id and existing.company_id != company:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="company does not match existing session",
                )
            return existing

    session = Session(user_id=user_id, company_id=company)
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


def _emit_warehouse_updates(
    *,
    session_id: str,
    intent: Optional[str],
    escalated: bool,
    completed: bool = True,
) -> None:
    asyncio.create_task(
        update_session_analytics(
            session_id=session_id,
            intent=intent,
            escalated=escalated,
            resolved_at=datetime.now(timezone.utc),
        )
    )
    asyncio.create_task(
        record_outcome(
            session_id=session_id,
            task="conversation",
            completed=completed,
            escalated=escalated,
            verified=False,
        )
    )


async def _run_agent_flow(
    db: AsyncSession,
    req: ChatRequest,
    request_id: str,
    *,
    use_cache: bool = True,
) -> ChatResponse:
    start = time.perf_counter()
    span_ctx = None
    root_span = None
    request_status = "ok"
    try:
        session = await _get_or_create_session(db, req.session_id, req.user_id, req.company)
        tracer = get_tracer()
        if tracer is not None:
            span_ctx = tracer.start_as_current_span("conversation")
            root_span = span_ctx.__enter__()
            root_span.set_attribute("session_id", str(session.id))
            root_span.set_attribute("user_id", req.user_id)
            root_span.set_attribute("channel", "api")

        log_event(
            "chat",
            "request_received",
            request_id=request_id,
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
        cache_key = _build_chat_cache_key(str(session.id), req.user_id, req.message, req.company)
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

        log_event(
            "chat",
            "cache_lookup",
            request_id=request_id,
            session_id=str(session.id),
            status="hit" if cache_hit else "miss",
        )

        if cache_hit and cached_payload is not None:
            request_status = "cache_hit"
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
            log_event(
                "chat",
                "return_cached_response",
                request_id=request_id,
                session_id=str(session.id),
            )
            _emit_warehouse_updates(
                session_id=str(session.id),
                intent=cached_payload.get("intent"),
                escalated=bool(cached_payload.get("escalated", False)),
                completed=True,
            )
            return resp

        log_event("chat", "agent_start", request_id=request_id, session_id=str(session.id))
        history = await _load_message_history(db, session_id=session.id)
        try:
            chat_turns.observe(len(history))
        except Exception:
            pass

        state: AgentState = {
            "session_id": str(session.id),
            "user_id": req.user_id,
            "request_id": request_id,
            "messages": history,
            # Default to "wixqa" when the client does not explicitly choose a dataset.
            "dataset": (req.dataset or "wixqa").lower(),
            "company_id": req.company,
        }
        trace_id, _ = get_current_trace_ids()
        if trace_id:
            state["trace_id"] = trace_id
        async with chat_request_trace(state):
            final_state = await run_orchestrated_agent(state)
        response_text = final_state.get("final_response") or ""

        log_event(
            "chat",
            "agent_end",
            request_id=request_id,
            session_id=str(session.id),
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
            "intent": final_state.get("intent"),
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

        log_event(
            "chat",
            "response_ready",
            request_id=request_id,
            session_id=str(session.id),
            sources=len(sources),
            tools_used=tools_used,
            escalated=bool(final_state.get("should_escalate", False)),
        )
        _emit_warehouse_updates(
            session_id=str(session.id),
            intent=final_state.get("intent"),
            escalated=bool(final_state.get("should_escalate", False)),
            completed=True,
        )
        if root_span is not None:
            root_span.set_attribute("intent", str(final_state.get("intent") or ""))
            root_span.set_attribute("escalated", bool(final_state.get("should_escalate", False)))
        resp = ChatResponse(**payload)
        return resp
    except Exception:
        request_status = "agent_error"
        try:
            error_count.labels(error_type="agent_error", component="chat").inc()
        except Exception:
            pass
        raise
    finally:
        elapsed = time.perf_counter() - start
        if root_span is not None:
            root_span.set_attribute("latency_ms", elapsed * 1000.0)
        if span_ctx is not None:
            span_ctx.__exit__(None, None, None)
        try:
            request_count.labels(status=request_status).inc()
            request_latency.observe(elapsed)
        except Exception:
            pass


async def _run_intent_only_flow(
    db: AsyncSession,
    req: IntentClassifyRequest,
    request_id: str,
) -> IntentClassifyResponse:
    """
    Persist a user message like the main chat flow but run only the intent
    classification node and return the classified intent.
    """
    start = time.perf_counter()
    span_ctx = None
    root_span = None
    request_status = "ok"
    try:
        # For intent-only flow, we do not enforce or persist company_id strictly,
        # but we allow it to be threaded through state for future use.
        session = await _get_or_create_session(
            db,
            req.session_id,
            req.user_id,
            req.company or "default",
        )
        # Intent-only eval must not emit OpenTelemetry traces.

        log_event(
            "chat",
            "intent_only_request_received",
            request_id=request_id,
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
        await db.commit()

        history = await _load_message_history(db, session_id=session.id)

        state: AgentState = {
            "session_id": str(session.id),
            "user_id": req.user_id,
            "request_id": request_id,
            "messages": history,
            "dataset": (req.dataset or "wixqa").lower(),
            "observability_disabled": True,
        }
        if req.company:
            state["company_id"] = req.company
        if req.intent_prompt_profile:
            state["intent_prompt_profile"] = req.intent_prompt_profile

        # Avoid LangSmith tracing even if it is globally enabled via env vars.
        try:
            import langsmith as _langsmith  # type: ignore[import-not-found]
        except Exception:
            _langsmith = None

        if _langsmith is not None:
            with _langsmith.tracing_context(enabled=False):
                final_state = await classify_intent_only(state)  # type: ignore[arg-type]
        else:
            final_state = await classify_intent_only(state)  # type: ignore[arg-type]
        intent = final_state.get("intent")
        intent_profile = final_state.get("intent_prompt_profile")

        log_event(
            "chat",
            "intent_only_classified",
            request_id=request_id,
            session_id=str(session.id),
            intent=intent,
            intent_profile=intent_profile,
        )

        return IntentClassifyResponse(
            session_id=str(session.id),
            intent=str(intent) if intent is not None else None,
            intent_prompt_profile=str(intent_profile) if intent_profile is not None else None,
        )
    except Exception:
        request_status = "intent_error"
        try:
            error_count.labels(error_type="agent_error", component="chat").inc()
        except Exception:
            pass
        raise
    finally:
        elapsed = time.perf_counter() - start
        # Intent-only eval must not emit Prometheus metrics.
        _ = elapsed


@router.post("/", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> JSONResponse:
    if not req.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="message must not be empty",
        )

    request_id = getattr(request.state, "request_id", "")
    resp = await _run_agent_flow(db, req, request_id, use_cache=True)
    return JSONResponse(content=resp.dict(), headers={"X-Request-ID": request_id})


@router.post("/intent", response_model=IntentClassifyResponse)
async def chat_intent(
    req: IntentClassifyRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> JSONResponse:
    if not req.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="message must not be empty",
        )

    request_id = getattr(request.state, "request_id", "")
    resp = await _run_intent_only_flow(db, req, request_id)
    return JSONResponse(content=resp.dict(), headers={"X-Request-ID": request_id})

@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    if not req.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="message must not be empty",
        )

    request_id = getattr(request.state, "request_id", "")

    async def event_generator() -> AsyncGenerator[bytes, None]:
        resp = await _run_agent_flow(db, req, request_id, use_cache=True)
        data = json.dumps(resp.dict())
        yield f"data: {data}\n\n".encode("utf-8")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"X-Request-ID": request_id},
    )


__all__ = ["router"]

