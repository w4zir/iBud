from __future__ import annotations

from typing import Any, Dict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .nodes import (
    check_escalation,
    classify_intent,
    create_ticket,
    execute_tool,
    plan_action,
    retrieve_context,
    synthesize_response,
)
from .state import AgentState


def _plan_condition(state: AgentState) -> str:
    calls = state.get("tool_calls") or []
    return "needs_tools" if calls else "no_tools"


def _escalation_condition(state: AgentState) -> str:
    return "escalate" if state.get("should_escalate") else "resolved"


def build_agent_graph():
    """
    Assemble the LangGraph agent according to the Phase 3 plan.
    """
    workflow: StateGraph = StateGraph(AgentState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("plan_action", plan_action)
    workflow.add_node("execute_tool", execute_tool)
    workflow.add_node("synthesize_response", synthesize_response)
    workflow.add_node("check_escalation", check_escalation)
    workflow.add_node("create_ticket", create_ticket)

    workflow.set_entry_point("classify_intent")
    workflow.add_edge("classify_intent", "retrieve_context")
    workflow.add_edge("retrieve_context", "plan_action")

    workflow.add_conditional_edges(
        "plan_action",
        _plan_condition,
        {
            "needs_tools": "execute_tool",
            "no_tools": "synthesize_response",
        },
    )

    workflow.add_edge("execute_tool", "synthesize_response")
    workflow.add_edge("synthesize_response", "check_escalation")

    workflow.add_conditional_edges(
        "check_escalation",
        _escalation_condition,
        {
            "resolved": END,
            "escalate": "create_ticket",
        },
    )

    workflow.add_edge("create_ticket", END)

    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    return graph


_graph = None


def get_agent_graph():
    global _graph
    if _graph is None:
        _graph = build_agent_graph()
    return _graph


async def run_agent(state: AgentState, *, thread_id: str | None = None) -> AgentState:
    """
    Convenience wrapper to run the compiled graph asynchronously.
    """
    graph = get_agent_graph()
    config: Dict[str, Any] = {}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    result = await graph.ainvoke(state, config=config)
    return result


__all__ = ["build_agent_graph", "get_agent_graph", "run_agent"]

