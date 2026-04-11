from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..config import get_llm, is_human_escalation_enabled, log_event
from ..observability.langsmith_tracer import build_stage_run_config, tool_trace
from ..observability.otel import get_tracer
from ..observability.warehouse import record_span
from ..tools.faq_search import faq_search_tool
from ..tools.order_lookup import order_lookup_tool
from ..tools.return_initiate import return_initiate_tool
from ..tools.ticket_create import ticket_create_tool
from ..tools.human_handoff import human_handoff_tool
from .state import AgentState, ToolResult

StepEmitter = Callable[[str, str, Optional[str], str], Awaitable[None]]


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0.0 else default


PLANNER_MAX_CYCLES = _get_int_env("AGENT_PLANNER_MAX_CYCLES", 5)
SENTIMENT_THRESHOLD = _get_float_env("AGENT_SENTIMENT_THRESHOLD", 0.3)


_USER_ASK_HUMAN_RE = re.compile(
    r"\b(human|agent|representative|support)\b", re.IGNORECASE
)
_ANGRY_MARKERS_RE = re.compile(
    r"\b(angry|furious|pissed|terrible|worst|unacceptable|refund now|lawsuit)\b",
    re.IGNORECASE,
)


def _get_latest_user_message(state: AgentState) -> str:
    messages = state.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _history_as_messages(state: AgentState, *, include_system_prefix: bool = False) -> List[Any]:
    msgs: List[Any] = []
    for msg in state.get("messages") or []:
        role = str(msg.get("role") or "").strip().lower()
        content = str(msg.get("content") or "")
        if not content:
            continue
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
        elif include_system_prefix:
            msgs.append(SystemMessage(content=content))
    return msgs


def _history_as_text(state: AgentState) -> str:
    lines: List[str] = []
    for msg in state.get("messages") or []:
        role = str(msg.get("role") or "").strip().lower()
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            lines.append(f"assistant: {content}")
        else:
            lines.append(f"user: {content}")
    return "\n".join(lines)


async def _emit_step(
    step_emitter: Optional[StepEmitter],
    step_id: str,
    label: str,
    detail: Optional[str] = None,
    status: str = "info",
) -> None:
    if step_emitter is None:
        return
    try:
        await step_emitter(step_id, label, detail, status)
    except Exception:
        # Step emission is best-effort and must not break agent execution.
        return


def _ensure_workflow_state(state: AgentState) -> Dict[str, Any]:
    wf = state.get("workflow_state")
    if isinstance(wf, dict):
        return wf
    wf = {}
    state["workflow_state"] = wf
    return wf


def _inc_planner_cycle(state: AgentState) -> int:
    wf = _ensure_workflow_state(state)
    raw = wf.get("planner_cycle_count", 0)
    try:
        n = int(raw)
    except Exception:
        n = 0
    n += 1
    wf["planner_cycle_count"] = n
    return n


def _current_planner_cycle(state: AgentState) -> int:
    wf = _ensure_workflow_state(state)
    raw = wf.get("planner_cycle_count", 0)
    try:
        return int(raw)
    except Exception:
        return 0


def _extract_json(text: str) -> Optional[Any]:
    """
    Best-effort JSON extraction. Accepts either a raw JSON object/array or a markdown
    code block that contains JSON.
    """
    t = (text or "").strip()
    if not t:
        return None
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*\n([\s\S]*?)```", t)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            return None
    # Try first object
    start = t.find("{")
    if start != -1:
        for end in range(len(t) - 1, start, -1):
            if t[end] == "}":
                try:
                    return json.loads(t[start : end + 1])
                except Exception:
                    continue
    return None


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.5
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _is_angry_text(text: str) -> bool:
    if not text:
        return False
    return bool(_ANGRY_MARKERS_RE.search(text))


def _user_asked_human(text: str) -> bool:
    if not text:
        return False
    if "talk to" in text.lower() and _USER_ASK_HUMAN_RE.search(text):
        return True
    return "human" in text.lower() or "agent" in text.lower()


SYSTEM_INTENT_SENTIMENT = """
You are a lightweight intent classifier for a customer support assistant.

Tasks:
1) Classify the user's message into ONE coarse intent:
   - order_status
   - return_request
   - product_qa
   - account_issue
   - complaint
   - other
2) Produce a sentiment_score in [0, 1] where 0 is very negative/angry and 1 is very positive.
3) Determine if the user explicitly requested to talk to a human agent like "I want to talk to a human agent" or "I want to talk to a human" or "I want to talk to a human support".

Return ONLY valid JSON with this shape:
{
  "intent": "order_status|return_request|product_qa|account_issue|complaint|other",
  "sentiment_score": 0.0,
  "user_requested_human": false
}
""".strip()


SYSTEM_PLANNER_SCHEMA_ISSUE = """
You are the planner. Produce a plan as JSON only, matching exactly this schema:
{
  "plan_id": "<string>",
  "tasks": [
    {
      "task_id": "<string>",
      "action": "<string>",
      "params": { },
      "depends_on": ["<task_id>", "..."]
    }
  ],
  "metadata": {
    "strategy": "<string>",
    "cycle_count": <int>,
    "user_request": "<string>"
  }
}

Rules:
- Choose actions that can be executed without an LLM (deterministic tools/APIs only).
- Keep tasks minimal; include dependencies only when necessary.
- If the user asked for a human, create no tasks and set strategy accordingly.

Available actions:
- kb_search (params: {\"query\": string, \"category\": string|null, \"top_k\": int})
- order_lookup (params: {\"order_number\": string|null, \"user_id\": string|null})
- return_initiate (params: {\"order_number\": string, \"user_id\": string|null})
- faq_search (params: {\"query\": string, \"category\": string|null, \"top_k\": int})
""".strip()


SYSTEM_PLANNER_SCHEMA_NON_ISSUE = """
You are the planner. Produce a plan as JSON only, matching exactly this schema:
{
  "plan_id": "<string>",
  "tasks": [
    {
      "task_id": "<string>",
      "action": "<string>",
      "params": { },
      "depends_on": ["<task_id>", "..."]
    }
  ],
  "metadata": {
    "strategy": "<string>",
    "cycle_count": <int>,
    "user_request": "<string>"
  }
}

Rules:
- Choose actions that can be executed without an LLM (deterministic tools/APIs only).
- Keep tasks minimal; include dependencies only when necessary.
- If the user asked for a human, create no tasks and set strategy accordingly.

Available actions:
- kb_search (params: {\"query\": string, \"category\": string|null, \"top_k\": int})
- faq_search (params: {\"query\": string, \"category\": string|null, \"top_k\": int})
""".strip()


SYSTEM_PLAN_EVALUATOR = """
You are a plan evaluator. Given a user_request and a proposed plan, check:
- schema correctness
- uses only allowed actions
- tasks are minimal and dependencies make sense
- plan is likely to satisfy the user_request

Return ONLY valid JSON:
{ "plan_valid": true|false, "feedback": "<string>" }
""".strip()


SYSTEM_VALIDATOR = """
You are a validator. You only receive a Plan and an Execution Result (not chat history).
Decide whether the user_request in plan.metadata.user_request has been achieved.
Also output a sentiment_score in [0,1] (0 very negative, 1 very positive) based on the result quality.

Return ONLY valid JSON:
{ "achieved": true|false, "feedback": "<string>", "sentiment_score": 0.0 }
""".strip()


SYSTEM_RESPONSE_GENERATOR = """
You are a response generator. Write a final, human-readable response.
You are given a Plan and an Execution Result. Be concise, helpful, and accurate.
If escalation happened, clearly tell the user that a human will follow up.

Return plain text only.
""".strip()


SYSTEM_SIMPLE_CHAT = """
You are a concise and helpful e-commerce customer support assistant.
Use the full conversation history for continuity and respond directly to the latest user message.
If the question is unclear, ask a brief clarifying question.
Return plain text only.
""".strip()


@dataclass(frozen=True)
class EscalationDecision:
    should_escalate: bool
    reason: str


def _decide_escalation(
    *,
    user_text: str,
    classifier_sentiment: float,
    validator_sentiment: Optional[float],
    user_requested_human: bool,
    planner_cycle_count: int,
) -> EscalationDecision:
    if not is_human_escalation_enabled():
        return EscalationDecision(False, "human_escalation_disabled")
    if user_requested_human:
        return EscalationDecision(True, "user_requested_human")
    if _is_angry_text(user_text):
        return EscalationDecision(True, "user_angry")
    if classifier_sentiment < SENTIMENT_THRESHOLD:
        return EscalationDecision(True, "classifier_low_sentiment")
    if validator_sentiment is not None and validator_sentiment < SENTIMENT_THRESHOLD:
        return EscalationDecision(True, "validator_low_sentiment")
    if planner_cycle_count > PLANNER_MAX_CYCLES:
        return EscalationDecision(True, "planner_cycle_exceeded")
    return EscalationDecision(False, "no_escalation")


async def _record_span_nonblocking(
    state: AgentState,
    span_name: str,
    *,
    latency_ms: float,
    **attrs: Any,
) -> None:
    try:
        await record_span(
            session_id=state.get("session_id"),
            trace_id=state.get("trace_id"),
            span_name=span_name,
            attributes=attrs,
            latency_ms=latency_ms,
        )
    except Exception:
        return


def _allowed_actions(query_type: Optional[str]) -> set[str]:
    """
    Coarse pipeline routing:
    - issue: allow order/return related tools
    - non_issue: allow only knowledge-base retrieval tools
    """

    if query_type == "non_issue":
        return {"kb_search", "faq_search"}
    return {"kb_search", "order_lookup", "return_initiate", "faq_search"}


def _normalize_plan(plan: Dict[str, Any], *, cycle_count: int, user_request: str) -> Dict[str, Any]:
    plan = dict(plan or {})
    md = dict(plan.get("metadata") or {})
    md["cycle_count"] = int(cycle_count)
    md.setdefault("user_request", user_request)
    plan["metadata"] = md
    if "tasks" not in plan or not isinstance(plan.get("tasks"), list):
        plan["tasks"] = []
    return plan


def _validate_plan_schema(plan: Any, *, query_type: Optional[str]) -> Tuple[bool, str]:
    if not isinstance(plan, dict):
        return False, "plan is not an object"
    if not plan.get("plan_id"):
        return False, "missing plan_id"
    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        return False, "tasks must be a list"
    for t in tasks:
        if not isinstance(t, dict):
            return False, "task is not an object"
        if not t.get("task_id"):
            return False, "task_id missing"
        if not t.get("action"):
            return False, "action missing"
        if not isinstance(t.get("depends_on", []), list):
            return False, "depends_on must be a list"
        if "params" in t and not isinstance(t.get("params"), dict):
            return False, "params must be an object"
        if str(t.get("action")) not in _allowed_actions(query_type):
            return False, f"unsupported action: {t.get('action')}"
    return True, "ok"


def _toposort(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {str(t.get("task_id")): t for t in tasks}
    remaining = set(by_id.keys())
    done: set[str] = set()
    ordered: List[Dict[str, Any]] = []

    while remaining:
        progress = False
        for tid in list(remaining):
            t = by_id[tid]
            deps = [str(x) for x in (t.get("depends_on") or [])]
            if all(d in done for d in deps):
                ordered.append(t)
                done.add(tid)
                remaining.remove(tid)
                progress = True
        if not progress:
            # Cycle or missing dep: execute remaining in stable order.
            for tid in sorted(remaining):
                ordered.append(by_id[tid])
            break
    return ordered


async def _execute_action(
    action: str,
    params: Dict[str, Any],
    state: AgentState,
    *,
    task_id: str | None = None,
    attempt: int | None = None,
    depends_on: List[str] | None = None,
) -> ToolResult:
    run_name = f"agent.tool.{action}"
    async with tool_trace(
        state,
        run_name,
        task_id=task_id,
        action=action,
        attempt=attempt,
        depends_on=depends_on,
    ):
        try:
            if action == "kb_search":
                query = params.get("query") or _get_latest_user_message(state)
                category = params.get("category")
                top_k_raw = params.get("top_k")
                top_k = int(top_k_raw) if top_k_raw is not None else None
                if top_k is not None and top_k <= 0:
                    top_k = None
                # Map to FAQ search tool (KB abstraction) to keep compatibility with current stack.
                result = await faq_search_tool(query=query, category=category, top_k=top_k)
            elif action == "faq_search":
                query = params.get("query") or _get_latest_user_message(state)
                category = params.get("category")
                top_k_raw = params.get("top_k")
                top_k = int(top_k_raw) if top_k_raw is not None else None
                if top_k is not None and top_k <= 0:
                    top_k = None
                result = await faq_search_tool(query=query, category=category, top_k=top_k)
            elif action == "order_lookup":
                result = await order_lookup_tool(
                    order_number=params.get("order_number"),
                    user_id=params.get("user_id") or state.get("user_id"),
                )
            elif action == "return_initiate":
                result = await return_initiate_tool(
                    order_number=params.get("order_number"),
                    user_id=params.get("user_id") or state.get("user_id"),
                )
            else:
                raise ValueError(f"Unknown action: {action}")
            return ToolResult(name=action, success=True, result=result, error=None)
        except Exception as exc:
            return ToolResult(name=action, success=False, result={}, error=str(exc))


async def run_simple_chat_agent(
    state: AgentState,
    *,
    step_emitter: Optional[StepEmitter] = None,
) -> AgentState:
    """
    Direct conversation response path (no planner/evaluator/executor).
    Always uses persisted session history as LLM context.
    """
    start = time.perf_counter()
    await _emit_step(
        step_emitter,
        "simple_chat_start",
        "Agent generating direct response",
        "Using quick path without planner",
        "started",
    )
    llm = get_llm(role="small")  # type: ignore[call-arg]
    convo = _history_as_messages(state)
    if not convo:
        user_text = _get_latest_user_message(state)
        convo = [HumanMessage(content=user_text)] if user_text else []
    final = await llm.ainvoke(
        [SystemMessage(content=SYSTEM_SIMPLE_CHAT), *convo],
        config=build_stage_run_config(state, "agent.simple_chat"),
    )
    answer = (final.content or "").strip()
    state["final_response"] = answer
    state["should_escalate"] = False
    wf = _ensure_workflow_state(state)
    wf["route"] = "simple_chat"
    await _record_span_nonblocking(
        state,
        "simple_chat",
        latency_ms=(time.perf_counter() - start) * 1000.0,
        response_len=len(answer),
        status="ok",
    )
    await _emit_step(
        step_emitter,
        "simple_chat_done",
        "Direct response ready",
        "Prepared final assistant response",
        "completed",
    )
    return state


async def run_orchestrated_agent(
    state: AgentState,
    *,
    step_emitter: Optional[StepEmitter] = None,
) -> AgentState:
    """
    New runtime pipeline:
      intent_classifier -> planner -> evaluator -> workflow_engine -> executor -> validator -> response_generator

    Notes:
    - Executor is deterministic (no LLM).
    - Validator sees only Plan + Result (Plan embeds user_request in metadata).
    - Planner cycle count is persisted in state['workflow_state'].
    - Immediate escalation triggers: sentiment<0.3, angry, ask human, >3 cycles.
    """
    wf = _ensure_workflow_state(state)
    tracer = get_tracer()
    user_text = _get_latest_user_message(state)

    # 0) Query type is expected to be set by chat route classification.
    query_type: str = str(state.get("query_type") or "issue")

    # 1) Intent classifier (small model)
    await _emit_step(
        step_emitter,
        "intent_classifier_start",
        "Classifying customer intent",
        "Determining request type and sentiment",
        "started",
    )
    start = time.perf_counter()
    classifier_llm = get_llm(role="small")  # type: ignore[call-arg]
    history_text = _history_as_text(state)
    classifier_resp = await classifier_llm.ainvoke(
        [
            SystemMessage(content=SYSTEM_INTENT_SENTIMENT),
            HumanMessage(content=f"Conversation history:\n{history_text}\n\nLatest user message:\n{user_text}"),
        ],
        config=build_stage_run_config(state, "agent.intent_classifier"),
    )
    raw_classifier = (classifier_resp.content or "").strip()
    parsed = _extract_json(raw_classifier) or {}
    intent = str((parsed.get("intent") or "other")).strip()
    classifier_sent = _clamp01(parsed.get("sentiment_score"))
    user_requested_human = bool(parsed.get("user_requested_human")) or _user_asked_human(user_text)

    state["intent"] = intent  # type: ignore[assignment]
    state["classifier_sentiment_score"] = classifier_sent
    state["user_requested_human"] = user_requested_human
    wf["classifier"] = {"raw": raw_classifier}
    await _record_span_nonblocking(
        state,
        "intent_classifier",
        latency_ms=(time.perf_counter() - start) * 1000.0,
        intent=intent,
        sentiment_score=classifier_sent,
        user_requested_human=user_requested_human,
        status="ok",
    )
    await _emit_step(
        step_emitter,
        "intent_classifier_done",
        "Intent classified",
        f"Intent: {intent}, sentiment: {classifier_sent:.2f}",
        "completed",
    )

    # Early escalation check.
    decision = _decide_escalation(
        user_text=user_text,
        classifier_sentiment=classifier_sent,
        validator_sentiment=None,
        user_requested_human=user_requested_human,
        planner_cycle_count=_current_planner_cycle(state),
    )
    if decision.should_escalate:
        return await _handle_escalation(state, reason=decision.reason, step_emitter=step_emitter)

    # Planner / evaluator / workflow / validator loops
    plan: Dict[str, Any] = {}
    execution_results: Dict[str, Any] = {}
    validator_sent: Optional[float] = None
    plan_feedback: str = ""
    validator_feedback: str = ""

    while True:
        # cycle guard + persisted counter
        cycle = _inc_planner_cycle(state)
        if cycle > PLANNER_MAX_CYCLES:
            return await _handle_escalation(
                state,
                reason="planner_cycle_exceeded",
                step_emitter=step_emitter,
            )

        # 2) Planner (large model)
        await _emit_step(
            step_emitter,
            "planner_start",
            "Building action plan",
            f"Planner cycle {cycle}",
            "started",
        )
        start = time.perf_counter()
        planner_llm = get_llm(role="planner")  # type: ignore[call-arg]
        planner_input = json.dumps(
            {
                "intent": intent,
                "query_type": query_type,
                "user_request": user_text,
                "chat_history": history_text,
                "cycle_count": cycle,
                "previous_feedback": plan_feedback or validator_feedback or "",
            },
            ensure_ascii=False,
        )
        planner_resp = await planner_llm.ainvoke(
            [
                SystemMessage(
                    content=(
                        SYSTEM_PLANNER_SCHEMA_NON_ISSUE
                        if query_type == "non_issue"
                        else SYSTEM_PLANNER_SCHEMA_ISSUE
                    )
                ),
                HumanMessage(content=planner_input),
            ],
            config=build_stage_run_config(state, "agent.planner", cycle_count=cycle),
        )
        raw_plan = (planner_resp.content or "").strip()
        parsed_plan = _extract_json(raw_plan) or {}
        plan = _normalize_plan(parsed_plan if isinstance(parsed_plan, dict) else {}, cycle_count=cycle, user_request=user_text)
        wf["planner"] = {"raw": raw_plan}
        await _record_span_nonblocking(
            state,
            "planner",
            latency_ms=(time.perf_counter() - start) * 1000.0,
            cycle_count=cycle,
            status="ok",
        )
        await _emit_step(
            step_emitter,
            "planner_done",
            "Plan created",
            f"Generated {len(plan.get('tasks') or [])} task(s)",
            "completed",
        )

        # 3) Evaluator (small model)
        await _emit_step(
            step_emitter,
            "evaluator_start",
            "Checking plan quality",
            "Validating schema and feasibility",
            "started",
        )
        start = time.perf_counter()
        evaluator_llm = get_llm(role="small")  # type: ignore[call-arg]
        schema_ok, schema_msg = _validate_plan_schema(plan, query_type=query_type)
        if not schema_ok:
            plan_feedback = f"Plan schema invalid: {schema_msg}"
            await _record_span_nonblocking(
                state,
                "evaluator",
                latency_ms=(time.perf_counter() - start) * 1000.0,
                plan_valid=False,
                feedback=plan_feedback,
                status="schema_invalid",
            )
            await _emit_step(
                step_emitter,
                "evaluator_schema_invalid",
                "Plan validation failed",
                "Adjusting plan and retrying",
                "failed",
            )
            continue

        evaluator_input = json.dumps(
            {"user_request": user_text, "chat_history": history_text, "plan": plan},
            ensure_ascii=False,
        )
        evaluator_resp = await evaluator_llm.ainvoke(
            [SystemMessage(content=SYSTEM_PLAN_EVALUATOR), HumanMessage(content=evaluator_input)],
            config=build_stage_run_config(state, "agent.evaluator", cycle_count=cycle),
        )
        raw_eval = (evaluator_resp.content or "").strip()
        eval_parsed = _extract_json(raw_eval) or {}
        plan_valid = bool(eval_parsed.get("plan_valid"))
        plan_feedback = str(eval_parsed.get("feedback") or "").strip()
        wf["evaluator"] = {"raw": raw_eval}
        await _record_span_nonblocking(
            state,
            "evaluator",
            latency_ms=(time.perf_counter() - start) * 1000.0,
            plan_valid=plan_valid,
            status="ok",
        )
        if not plan_valid:
            await _emit_step(
                step_emitter,
                "evaluator_replan",
                "Plan needs revision",
                "Planner is generating an improved plan",
                "info",
            )
            continue
        await _emit_step(
            step_emitter,
            "evaluator_done",
            "Plan approved",
            "Proceeding to execute tasks",
            "completed",
        )

        # 4) Workflow engine + executor (deterministic)
        await _emit_step(
            step_emitter,
            "workflow_start",
            "Executing plan tasks",
            "Running deterministic tool actions",
            "started",
        )
        start = time.perf_counter()
        tasks = list(plan.get("tasks") or [])
        ordered = _toposort(tasks)
        execution_results = {"tasks": [], "by_task_id": {}}

        for task in ordered:
            task_id = str(task.get("task_id"))
            action = str(task.get("action"))
            params = dict(task.get("params") or {})
            deps = [str(x) for x in (task.get("depends_on") or [])]

            # Failure recovery: if any dependency failed/blocked, mark this task as blocked.
            dep_records = [execution_results["by_task_id"].get(d) for d in deps]
            if any((r is None) or (not bool(r.get("success"))) for r in dep_records):
                task_record = {
                    "task_id": task_id,
                    "action": action,
                    "params": params,
                    "attempts": 0,
                    "success": False,
                    "result": {},
                    "error": "blocked_by_failed_dependency",
                    "depends_on": deps,
                }
                execution_results["tasks"].append(task_record)
                execution_results["by_task_id"][task_id] = task_record
                await _emit_step(
                    step_emitter,
                    f"task_blocked_{task_id}",
                    f"Skipped task: {action}",
                    "Blocked by failed dependency",
                    "failed",
                )
                continue

            # Retry policy (simple fixed retries)
            max_attempts = int((wf.get("retry_policy") or {}).get("max_attempts") or 2)
            base_backoff_ms = int((wf.get("retry_policy") or {}).get("base_backoff_ms") or 200)
            attempt = 0
            last: ToolResult | None = None
            while attempt < max_attempts:
                attempt += 1
                await _emit_step(
                    step_emitter,
                    f"task_start_{task_id}_{attempt}",
                    f"Running task: {action}",
                    f"Attempt {attempt}",
                    "started",
                )
                last = await _execute_action(
                    action,
                    params,
                    state,
                    task_id=task_id,
                    attempt=attempt,
                    depends_on=deps,
                )
                if last.get("success"):
                    await _emit_step(
                        step_emitter,
                        f"task_done_{task_id}",
                        f"Task completed: {action}",
                        "Tool action succeeded",
                        "completed",
                    )
                    break
                # small backoff for transient errors
                if attempt < max_attempts:
                    try:
                        import asyncio

                        await asyncio.sleep((base_backoff_ms * attempt) / 1000.0)
                    except Exception:
                        pass
            if not bool(last and last.get("success")):
                await _emit_step(
                    step_emitter,
                    f"task_failed_{task_id}",
                    f"Task failed: {action}",
                    "Continuing with fallback behavior",
                    "failed",
                )
            task_record = {
                "task_id": task_id,
                "action": action,
                "params": params,
                "attempts": attempt,
                "success": bool(last and last.get("success")),
                "result": (last or {}).get("result"),
                "error": (last or {}).get("error"),
                "depends_on": deps,
            }
            execution_results["tasks"].append(task_record)
            execution_results["by_task_id"][task_id] = task_record

        state["execution_results"] = execution_results
        state["plan"] = plan
        state["plan_feedback"] = plan_feedback or None
        await _record_span_nonblocking(
            state,
            "workflow_engine",
            latency_ms=(time.perf_counter() - start) * 1000.0,
            tasks_count=len(ordered),
            status="ok",
        )
        await _emit_step(
            step_emitter,
            "workflow_done",
            "Task execution complete",
            f"Completed {len(ordered)} planned task(s)",
            "completed",
        )

        # 5) Validator (small model) - Plan + Result only
        await _emit_step(
            step_emitter,
            "validator_start",
            "Validating execution results",
            "Checking if user request is fulfilled",
            "started",
        )
        start = time.perf_counter()
        validator_llm = get_llm(role="small")  # type: ignore[call-arg]
        validator_input = json.dumps(
            {"chat_history": history_text, "plan": plan, "result": execution_results},
            ensure_ascii=False,
        )
        validator_resp = await validator_llm.ainvoke(
            [SystemMessage(content=SYSTEM_VALIDATOR), HumanMessage(content=validator_input)],
            config=build_stage_run_config(state, "agent.validator", cycle_count=cycle),
        )
        raw_val = (validator_resp.content or "").strip()
        val_parsed = _extract_json(raw_val) or {}
        achieved = bool(val_parsed.get("achieved"))
        validator_feedback = str(val_parsed.get("feedback") or "").strip()
        validator_sent = _clamp01(val_parsed.get("sentiment_score"))
        state["validator_sentiment_score"] = validator_sent
        wf["validator"] = {"raw": raw_val}
        await _record_span_nonblocking(
            state,
            "validator",
            latency_ms=(time.perf_counter() - start) * 1000.0,
            achieved=achieved,
            sentiment_score=validator_sent,
            status="ok",
        )
        await _emit_step(
            step_emitter,
            "validator_done",
            "Validation complete",
            "Request resolved" if achieved else "Needs another planning pass",
            "completed" if achieved else "info",
        )

        decision = _decide_escalation(
            user_text=user_text,
            classifier_sentiment=classifier_sent,
            validator_sentiment=validator_sent,
            user_requested_human=user_requested_human,
            planner_cycle_count=_current_planner_cycle(state),
        )
        if decision.should_escalate:
            return await _handle_escalation(state, reason=decision.reason, step_emitter=step_emitter)

        if achieved:
            break
        # Replan with validator feedback.
        plan_feedback = validator_feedback or plan_feedback
        continue

    # 6) Response generator (small model)
    await _emit_step(
        step_emitter,
        "response_generator_start",
        "Drafting final response",
        "Summarizing results for the customer",
        "started",
    )
    start = time.perf_counter()
    resp_llm = get_llm(role="small")  # type: ignore[call-arg]
    gen_input = json.dumps(
        {"chat_history": history_text, "plan": plan, "result": execution_results},
        ensure_ascii=False,
    )
    final = await resp_llm.ainvoke(
        [SystemMessage(content=SYSTEM_RESPONSE_GENERATOR), HumanMessage(content=gen_input)],
        config=build_stage_run_config(state, "agent.response_generator"),
    )
    answer = (final.content or "").strip()
    state["final_response"] = answer
    state["should_escalate"] = False
    await _record_span_nonblocking(
        state,
        "response_generator",
        latency_ms=(time.perf_counter() - start) * 1000.0,
        response_len=len(answer),
        status="ok",
    )
    await _emit_step(
        step_emitter,
        "response_generator_done",
        "Final response ready",
        "Sending answer to UI",
        "completed",
    )
    log_event(
        "agent",
        "orchestrated_complete",
        session_id=state.get("session_id"),
        request_id=state.get("request_id"),
        cycle_count=_current_planner_cycle(state),
        escalated=False,
    )
    if tracer is not None:
        try:
            tracer.get_current_span()  # touch to avoid unused
        except Exception:
            pass
    return state


async def classify_intent_only(state: AgentState) -> AgentState:
    """
    Thin compatibility classifier-only path for `/chat/intent`.

    Emits:
    - state['intent']
    - state['classifier_sentiment_score']
    - state['user_requested_human']
    """
    user_text = _get_latest_user_message(state)
    start = time.perf_counter()
    llm = get_llm(role="small")  # type: ignore[call-arg]
    history_text = _history_as_text(state)
    resp = await llm.ainvoke(
        [
            SystemMessage(content=SYSTEM_INTENT_SENTIMENT),
            HumanMessage(content=f"Conversation history:\n{history_text}\n\nLatest user message:\n{user_text}"),
        ],
        config=build_stage_run_config(state, "agent.intent_classifier"),
    )
    raw = (resp.content or "").strip()
    parsed = _extract_json(raw) or {}
    intent = str((parsed.get("intent") or "other")).strip()
    sentiment = _clamp01(parsed.get("sentiment_score"))
    user_requested_human = bool(parsed.get("user_requested_human")) or _user_asked_human(user_text)

    state["intent"] = intent  # type: ignore[assignment]
    state["classifier_sentiment_score"] = sentiment
    state["user_requested_human"] = user_requested_human
    wf = _ensure_workflow_state(state)
    wf["classifier"] = {"raw": raw}
    await _record_span_nonblocking(
        state,
        "intent_classifier",
        latency_ms=(time.perf_counter() - start) * 1000.0,
        intent=intent,
        sentiment_score=sentiment,
        user_requested_human=user_requested_human,
        status="ok",
    )
    return state


async def _handle_escalation(
    state: AgentState,
    *,
    reason: str,
    step_emitter: Optional[StepEmitter] = None,
) -> AgentState:
    """
    Dual-write escalation:
    - external human handoff API/tool
    - local ticket persistence
    """
    if not is_human_escalation_enabled():
        wf = _ensure_workflow_state(state)
        wf["escalation"] = {"reason": reason, "skipped": True, "human_escalation_disabled": True}
        state["should_escalate"] = False
        state["escalation_reason"] = reason
        state["final_response"] = (
            "I wasn't able to resolve this automatically. "
            "Please try rephrasing your question or provide more details."
        )
        log_event(
            "agent",
            "human_escalation_skipped",
            session_id=state.get("session_id"),
            request_id=state.get("request_id"),
            reason=reason,
        )
        await _emit_step(
            step_emitter,
            "escalation_skipped",
            "Human escalation disabled",
            f"Would have escalated: {reason}",
            "completed",
        )
        await _record_span_nonblocking(
            state,
            "human_escalation",
            latency_ms=0.0,
            reason=reason,
            skipped=True,
            status="ok",
        )
        return state

    user_text = _get_latest_user_message(state)
    await _emit_step(
        step_emitter,
        "escalation_start",
        "Escalating to human support",
        f"Reason: {reason}",
        "started",
    )
    plan = state.get("plan") or state.get("workflow_state", {}).get("last_plan") or {}
    result = state.get("execution_results") or {}

    # External handoff first (best-effort)
    external_status: Dict[str, Any] = {}
    async with tool_trace(state, "agent.tool.human_handoff", reason=reason):
        try:
            external_status = await human_handoff_tool(
                session_id=state.get("session_id"),
                user_id=state.get("user_id"),
                reason=reason,
                summary=user_text[:1000],
                plan=plan,
                result=result,
            )
        except Exception as exc:
            external_status = {"success": False, "error": str(exc)}

    # Local ticket for audit/fallback
    ticket_id: Optional[str] = None
    async with tool_trace(state, "agent.tool.ticket_create", reason=reason):
        try:
            ticket = await ticket_create_tool(
                issue_type="human_escalation",
                summary=f"{reason}: {user_text}",
                session_id=state.get("session_id"),
            )
            ticket_id = str(ticket.get("ticket_id")) if ticket.get("ticket_id") else None
        except Exception:
            ticket_id = None

    wf = _ensure_workflow_state(state)
    wf["escalation"] = {
        "reason": reason,
        "external": external_status,
        "ticket_id": ticket_id,
        "cycle_count": _current_planner_cycle(state),
    }
    state["should_escalate"] = True
    state["ticket_id"] = ticket_id
    state["escalation_reason"] = reason
    state["final_response"] = (
        "I’m escalating this to a human agent now. "
        "A support representative will follow up shortly."
        + (f" (Ticket: {ticket_id})" if ticket_id else "")
    )
    log_event(
        "agent",
        "human_escalation",
        session_id=state.get("session_id"),
        request_id=state.get("request_id"),
        reason=reason,
        ticket_id=ticket_id,
        external_success=bool(external_status.get("success")),
    )
    await _record_span_nonblocking(
        state,
        "human_escalation",
        latency_ms=0.0,
        reason=reason,
        ticket_id=ticket_id,
        external_success=bool(external_status.get("success")),
        status="ok",
    )
    await _emit_step(
        step_emitter,
        "escalation_done",
        "Escalation submitted",
        f"Ticket created: {ticket_id}" if ticket_id else "Ticket creation pending",
        "completed",
    )
    return state


__all__ = ["run_simple_chat_agent", "run_orchestrated_agent", "classify_intent_only"]

