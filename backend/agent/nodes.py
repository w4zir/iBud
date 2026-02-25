from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import get_llm
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


async def classify_intent(state: AgentState) -> AgentState:
    llm = get_llm()
    user_text = _get_latest_user_message(state)
    messages = [
        SystemMessage(content=SYSTEM_INTENT_CLASSIFIER),
        HumanMessage(content=user_text),
    ]
    resp = await llm.ainvoke(messages)
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
    return next_state


async def plan_action(state: AgentState) -> AgentState:
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
    raw = (resp.content or "").strip()

    tool_calls: List[ToolCall] = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = [parsed]
        if isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
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
    except Exception:
        tool_calls = []

    next_state: AgentState = dict(state)
    next_state["tool_calls"] = tool_calls
    return next_state


async def execute_tool(state: AgentState) -> AgentState:
    tool_calls = state.get("tool_calls") or []
    results: List[ToolResult] = []

    for call in tool_calls:
        name = call.get("name")
        args = call.get("arguments", {})
        try:
            if name == "order_lookup":
                result = await order_lookup_tool(
                    order_number=args.get("order_number"),
                    user_id=args.get("user_id"),
                )
            elif name == "return_initiate":
                result = await return_initiate_tool(
                    order_number=args.get("order_number"),
                    user_id=args.get("user_id"),
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
        except Exception as exc:
            results.append(
                ToolResult(
                    name=name or "",
                    success=False,
                    result={},
                    error=str(exc),
                )
            )

    next_state: AgentState = dict(state)
    next_state["tool_results"] = results

    for r in results:
        if r["name"] == "ticket_create" and r["success"]:
            ticket_info = r["result"] or {}
            ticket_id = ticket_info.get("ticket_id")
            if ticket_id:
                next_state["ticket_id"] = str(ticket_id)

    return next_state


async def synthesize_response(state: AgentState) -> AgentState:
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
    answer = (resp.content or "").strip()

    next_state: AgentState = dict(state)
    next_state["final_response"] = answer
    return next_state


async def check_escalation(state: AgentState) -> AgentState:
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
        raw = (resp.content or "").strip().lower()
        should = "true" in raw

    next_state: AgentState = dict(state)
    next_state["should_escalate"] = bool(should)
    return next_state


async def create_ticket(state: AgentState) -> AgentState:
    if not state.get("should_escalate"):
        return state

    if state.get("ticket_id"):
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

