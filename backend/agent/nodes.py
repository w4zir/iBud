from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import get_llm, log_event
from ..observability.prometheus_metrics import (
    error_count,
    escalations,
    intent_distribution,
    llm_tokens,
    task_outcome,
    tool_calls,
    tool_outcome,
)
from ..observability.otel import get_tracer
from ..observability.warehouse import record_span
from ..rag.retriever import RetrievedDoc, Retriever
from ..tools.faq_search import faq_search_tool
from ..tools.order_lookup import order_lookup_tool
from ..tools.return_initiate import return_initiate_tool
from ..tools.ticket_create import ticket_create_tool
from .prompts import (
    SYSTEM_ESCALATION,
    SYSTEM_INTENT_CLASSIFIER,
    SYSTEM_PLANNER,
    SYSTEM_RESPONDER,
)
from .state import AgentState, Intent, RetrievedDocState, ToolCall, ToolResult


def _get_latest_user_message(state: AgentState) -> str:
    messages = state.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _record_llm_tokens(response: Any) -> None:
    """
    Best-effort extraction of token usage from LangChain chat responses.

    Different providers expose slightly different metadata keys; this
    helper normalises the most common shapes and records total tokens
    against the configured LLM_PROVIDER.
    """
    try:  # pragma: no cover - defensive metrics path
        provider = os.getenv("LLM_PROVIDER", "ollama")
        metadata = getattr(response, "response_metadata", None) or {}
        usage = metadata.get("token_usage") or metadata.get("usage") or {}

        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        total = usage.get("total_tokens") or (prompt_tokens + completion_tokens)

        total_int = int(total) if total else 0
        if total_int > 0:
            llm_tokens.labels(provider=provider).inc(total_int)
    except Exception:
        # Metrics must never break the agent path.
        return


def _parse_planner_tool_calls(raw: str) -> List[Dict[str, Any]]:
    """
    Parse the planner LLM output into a list of tool-call dicts.
    Tries direct JSON first; if the model returns prose + markdown code block,
    extracts and parses the JSON from the code block or from the first [...] array.
    """
    text = (raw or "").strip()
    # 1) Direct JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
        if isinstance(parsed, dict):
            return [parsed]
        return []
    except (json.JSONDecodeError, TypeError):
        pass

    # 2) Markdown code block: ```json ... ``` or ``` ... ```
    code_block = re.search(r"```(?:json)?\s*\n([\s\S]*?)```", text)
    if code_block:
        try:
            parsed = json.loads(code_block.group(1).strip())
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
            if isinstance(parsed, dict):
                return [parsed]
            return []
        except (json.JSONDecodeError, TypeError):
            pass

    # 3) First top-level JSON array [...]
    start = text.find("[")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                        if isinstance(parsed, list):
                            return [x for x in parsed if isinstance(x, dict)]
                    except (json.JSONDecodeError, TypeError):
                        break
                    break
    return []


def _record_node_span(state: AgentState, span_name: str, *, latency_ms: float, **attrs: Any) -> None:
    session_id = state.get("session_id")
    trace_id = state.get("trace_id")
    asyncio.create_task(
        record_span(
            session_id=session_id,
            trace_id=trace_id,
            span_name=span_name,
            attributes=attrs,
            latency_ms=latency_ms,
        )
    )


def _start_otel_span(span_name: str):
    tracer = get_tracer()
    if tracer is None:
        return None, None
    context_manager = tracer.start_as_current_span(span_name)
    span = context_manager.__enter__()
    return context_manager, span


def _finish_otel_span(context_manager: Any, span: Any, **attrs: Any) -> None:
    if span is not None:
        for key, value in attrs.items():
            if value is None:
                continue
            span.set_attribute(key, value)
    if context_manager is not None:
        context_manager.__exit__(None, None, None)


async def classify_intent(state: AgentState) -> AgentState:
    start = time.perf_counter()
    span_ctx, otel_span = _start_otel_span("intent_detection")
    llm = get_llm()
    user_text = _get_latest_user_message(state)
    messages = [
        SystemMessage(content=SYSTEM_INTENT_CLASSIFIER),
        HumanMessage(content=user_text),
    ]
    resp = await llm.ainvoke(messages)
    _record_llm_tokens(resp)
    raw = (resp.content or "").strip().lower()

    intent: Intent
    if raw in {
        "order_status",
        "return_request",
        "product_qa",
        "account_issue",
        "complaint",
        "other",
    }:
        intent = raw  # type: ignore[assignment]
    else:
        if "status" in raw:
            intent = "order_status"
        elif "return" in raw or "refund" in raw:
            intent = "return_request"
        elif "account" in raw or "login" in raw or "password" in raw:
            intent = "account_issue"
        elif "complain" in raw or "terrible" in raw or "angry" in raw:
            intent = "complaint"
        else:
            intent = "product_qa"

    next_state: AgentState = dict(state)
    next_state["intent"] = intent
    try:
        intent_distribution.labels(intent=intent).inc()
    except Exception:
        pass
    log_event(
        "agent",
        "classify_intent",
        session_id=state.get("session_id"),
        request_id=state.get("request_id"),
        intent=intent,
    )
    _record_node_span(
        state,
        "classify_intent",
        latency_ms=(time.perf_counter() - start) * 1000.0,
        intent=intent,
        status="ok",
    )
    _finish_otel_span(
        span_ctx,
        otel_span,
        intent=intent,
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )
    return next_state


def _intent_to_category(intent: Intent | None) -> str | None:
    if intent in ("order_status", "return_request"):
        return "orders"
    if intent in ("account_issue", "complaint"):
        return "account"
    if intent == "product_qa":
        return "product"
    return None


async def retrieve_context(state: AgentState) -> AgentState:
    start = time.perf_counter()
    span_ctx, otel_span = _start_otel_span("retrieval")
    user_text = _get_latest_user_message(state)
    intent = state.get("intent")
    category = _intent_to_category(intent)

    retriever = Retriever()
    docs: List[RetrievedDoc] = await retriever.search(
        query=user_text,
        top_k=5,
        category=category,
        tier_filter=1,
        use_cache=True,
        rerank=True,
    )
    serialized: List[RetrievedDocState] = [
        {
            "content": d.content,
            "metadata": d.metadata,
            "score": d.score,
            "source": d.source,
            "doc_tier": d.doc_tier,
            "document_id": d.document_id,
            "parent_id": d.parent_id,
        }
        for d in docs
    ]
    next_state: AgentState = dict(state)
    next_state["retrieved_docs"] = serialized
    log_event(
        "agent",
        "retrieve_context",
        session_id=state.get("session_id"),
        request_id=state.get("request_id"),
        intent=intent,
        category=category,
        docs_count=len(docs),
    )
    _record_node_span(
        state,
        "retrieve_context",
        latency_ms=(time.perf_counter() - start) * 1000.0,
        intent=intent,
        category=category,
        docs_count=len(docs),
        status="ok",
    )
    _finish_otel_span(
        span_ctx,
        otel_span,
        intent=intent,
        docs_returned=len(docs),
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )
    return next_state


async def plan_action(state: AgentState) -> AgentState:
    start = time.perf_counter()
    span_ctx, otel_span = _start_otel_span("response_planning")
    llm = get_llm()
    user_text = _get_latest_user_message(state)
    intent = state.get("intent")
    retrieved = state.get("retrieved_docs") or []

    context_snippets = [
        f"- ({d.get('score', 0.0):.2f}) {d.get('content', '')[:300]}"
        for d in retrieved[:5]
    ]
    prompt = (
        SYSTEM_PLANNER
        + "\n\n"
        + "User message:\n"
        + user_text
        + "\n\nIntent: "
        + str(intent)
        + "\n\nTop retrieved snippets:\n"
        + "\n".join(context_snippets)
    )

    messages = [SystemMessage(content=SYSTEM_PLANNER), HumanMessage(content=prompt)]
    resp = await llm.ainvoke(messages)
    _record_llm_tokens(resp)
    raw = (resp.content or "").strip()

    tool_calls: List[ToolCall] = []
    for item in _parse_planner_tool_calls(raw):
        name = str(item.get("name") or "").strip()
        args = item.get("arguments") or {}
        if not name:
            continue
        tool_calls.append(
            ToolCall(
                name=name,
                arguments=dict(args),
            )
        )

    next_state: AgentState = dict(state)
    next_state["tool_calls"] = tool_calls
    log_event(
        "agent",
        "plan_action",
        session_id=state.get("session_id"),
        request_id=state.get("request_id"),
        tool_calls_count=len(tool_calls),
        names=[t.get("name") for t in tool_calls],
    )
    _record_node_span(
        state,
        "plan_action",
        latency_ms=(time.perf_counter() - start) * 1000.0,
        tool_calls_count=len(tool_calls),
        tool_names=[t.get("name") for t in tool_calls],
        status="ok",
    )
    _finish_otel_span(
        span_ctx,
        otel_span,
        tool_calls_count=len(tool_calls),
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )
    return next_state


async def execute_tool(state: AgentState) -> AgentState:
    start = time.perf_counter()
    span_ctx, otel_span = _start_otel_span("tool_calls")
    calls = state.get("tool_calls") or []
    results: List[ToolResult] = []

    for call in calls:
        name = call.get("name")
        args = call.get("arguments", {})
        log_event(
            "agent",
            "execute_tool",
            session_id=state.get("session_id"),
            request_id=state.get("request_id"),
            tool_name=name,
            arguments=args,
        )
        try:
            if name:
                try:
                    tool_calls.labels(tool_name=name).inc()
                except Exception:
                    pass
            if name == "order_lookup":
                result = await order_lookup_tool(
                    order_number=args.get("order_number"),
                    user_id=args.get("user_id") or state.get("user_id"),
                )
            elif name == "return_initiate":
                result = await return_initiate_tool(
                    order_number=args.get("order_number"),
                    user_id=args.get("user_id") or state.get("user_id"),
                )
            elif name == "faq_search":
                result = await faq_search_tool(
                    query=args.get("query") or _get_latest_user_message(state),
                    category=args.get("category"),
                    top_k=int(args.get("top_k") or 5),
                )
            elif name == "ticket_create":
                result = await ticket_create_tool(
                    issue_type=args.get("issue_type") or "escalation",
                    summary=args.get("summary")
                    or _get_latest_user_message(state),
                    session_id=state.get("session_id"),
                )
            else:
                raise ValueError(f"Unknown tool: {name}")

            results.append(
                ToolResult(
                    name=name or "",
                    success=True,
                    result=result,
                    error=None,
                )
            )
            if name:
                try:
                    tool_outcome.labels(tool_name=name, outcome="success").inc()
                except Exception:
                    pass
        except Exception as exc:
            results.append(
                ToolResult(
                    name=name or "",
                    success=False,
                    result={},
                    error=str(exc),
                )
            )
            if name:
                try:
                    tool_outcome.labels(tool_name=name, outcome="failure").inc()
                except Exception:
                    pass
            try:
                error_count.labels(error_type="tool_error", component="agent").inc()
            except Exception:
                pass

    next_state: AgentState = dict(state)
    next_state["tool_results"] = results

    for r in results:
        if r["name"] == "ticket_create" and r["success"]:
            ticket_info = r["result"] or {}
            ticket_id = ticket_info.get("ticket_id")
            if ticket_id:
                next_state["ticket_id"] = str(ticket_id)

    _record_node_span(
        state,
        "execute_tool",
        latency_ms=(time.perf_counter() - start) * 1000.0,
        tool_calls_count=len(calls),
        success_count=sum(1 for r in results if r.get("success")),
        failure_count=sum(1 for r in results if not r.get("success")),
        status="ok",
    )
    _finish_otel_span(
        span_ctx,
        otel_span,
        success_count=sum(1 for r in results if r.get("success")),
        failure_count=sum(1 for r in results if not r.get("success")),
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )
    return next_state


async def synthesize_response(state: AgentState) -> AgentState:
    start = time.perf_counter()
    span_ctx, otel_span = _start_otel_span("response_synthesis")
    llm = get_llm()
    user_text = _get_latest_user_message(state)
    retrieved = state.get("retrieved_docs") or []
    tool_results = state.get("tool_results") or []

    kb_context = "\n".join(
        f"- ({d.get('score', 0.0):.2f}) {d.get('content', '')[:400]}"
        for d in retrieved[:5]
    )
    tools_context = json.dumps(tool_results, indent=2, default=str)

    system = SYSTEM_RESPONDER
    user_prompt = (
        f"User message:\n{user_text}\n\n"
        f"Intent: {state.get('intent')}\n\n"
        f"Retrieved knowledge base context:\n{kb_context or '(none)'}\n\n"
        f"Tool results:\n{tools_context or '(none)'}\n"
    )

    messages = [SystemMessage(content=system), HumanMessage(content=user_prompt)]
    resp = await llm.ainvoke(messages)
    _record_llm_tokens(resp)
    answer = (resp.content or "").strip()

    next_state: AgentState = dict(state)
    next_state["final_response"] = answer
    log_event(
        "agent",
        "synthesize_response",
        session_id=state.get("session_id"),
        request_id=state.get("request_id"),
    )
    _record_node_span(
        state,
        "synthesize_response",
        latency_ms=(time.perf_counter() - start) * 1000.0,
        response_len=len(answer),
        status="ok",
    )
    _finish_otel_span(
        span_ctx,
        otel_span,
        latency_ms=(time.perf_counter() - start) * 1000.0,
        response_len=len(answer),
    )
    return next_state


async def check_escalation(state: AgentState) -> AgentState:
    start = time.perf_counter()
    span_ctx, otel_span = _start_otel_span("outcome_decision")
    llm = get_llm()
    user_text = _get_latest_user_message(state)
    tool_results = state.get("tool_results") or []

    had_failures = any(not r.get("success") for r in tool_results)
    is_complaint = state.get("intent") == "complaint"

    if had_failures and is_complaint:
        should = True
    else:
        context = json.dumps(tool_results, indent=2, default=str)
        system = SYSTEM_ESCALATION
        human = (
            f"User message:\n{user_text}\n\n"
            f"Intent: {state.get('intent')}\n\n"
            f"Tool results:\n{context or '(none)'}\n"
        )
        messages = [SystemMessage(content=system), HumanMessage(content=human)]
        resp = await llm.ainvoke(messages)
        _record_llm_tokens(resp)
        raw = (resp.content or "").strip().lower()
        should = "true" in raw

    next_state: AgentState = dict(state)
    next_state["should_escalate"] = bool(should)
    if should:
        try:
            task_outcome.labels(outcome="escalated").inc()
        except Exception:
            pass
    else:
        try:
            task_outcome.labels(outcome="resolved_without_escalation").inc()
        except Exception:
            pass
    log_event(
        "agent",
        "check_escalation",
        session_id=state.get("session_id"),
        request_id=state.get("request_id"),
        should_escalate=bool(should),
    )
    _record_node_span(
        state,
        "check_escalation",
        latency_ms=(time.perf_counter() - start) * 1000.0,
        should_escalate=bool(should),
        status="ok",
    )
    _finish_otel_span(
        span_ctx,
        otel_span,
        escalated=bool(should),
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )
    return next_state


async def create_ticket(state: AgentState) -> AgentState:
    start = time.perf_counter()
    span_ctx, otel_span = _start_otel_span("outcome")
    if not state.get("should_escalate"):
        log_event(
            "agent",
            "create_ticket_skipped",
            session_id=state.get("session_id"),
            request_id=state.get("request_id"),
            skipped="no_escalation",
        )
        _record_node_span(
            state,
            "create_ticket",
            latency_ms=(time.perf_counter() - start) * 1000.0,
            skipped="no_escalation",
            status="ok",
        )
        _finish_otel_span(
            span_ctx,
            otel_span,
            completed=True,
            escalated=False,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
        return state

    if state.get("ticket_id"):
        log_event(
            "agent",
            "create_ticket_skipped",
            session_id=state.get("session_id"),
            request_id=state.get("request_id"),
            skipped="already_has_ticket",
        )
        _record_node_span(
            state,
            "create_ticket",
            latency_ms=(time.perf_counter() - start) * 1000.0,
            skipped="already_has_ticket",
            status="ok",
        )
        _finish_otel_span(
            span_ctx,
            otel_span,
            completed=True,
            escalated=True,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
        return state

    user_text = _get_latest_user_message(state)
    result = await ticket_create_tool(
        issue_type="escalation",
        summary=user_text,
        session_id=state.get("session_id"),
    )

    next_state: AgentState = dict(state)
    ticket_id = result.get("ticket_id")
    if ticket_id:
        next_state["ticket_id"] = str(ticket_id)
        log_event(
            "agent",
            "create_ticket",
            session_id=state.get("session_id"),
            request_id=state.get("request_id"),
            ticket_id=str(ticket_id),
        )
        try:
            escalations.inc()
        except Exception:
            pass
        try:
            task_outcome.labels(outcome="completed").inc()
        except Exception:
            pass

    tool_results = list(next_state.get("tool_results") or [])
    tool_results.append(
        ToolResult(
            name="ticket_create",
            success=True,
            result=result,
            error=None,
        )
    )
    next_state["tool_results"] = tool_results
    _record_node_span(
        state,
        "create_ticket",
        latency_ms=(time.perf_counter() - start) * 1000.0,
        ticket_id=next_state.get("ticket_id"),
        status="ok",
    )
    _finish_otel_span(
        span_ctx,
        otel_span,
        completed=True,
        escalated=bool(next_state.get("ticket_id")),
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )
    return next_state


__all__ = [
    "classify_intent",
    "retrieve_context",
    "plan_action",
    "execute_tool",
    "synthesize_response",
    "check_escalation",
    "create_ticket",
]

