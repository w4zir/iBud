from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
import streamlit as st


BACKEND_URL_DEFAULT = "http://localhost:8000"


def _get_backend_base_url() -> str:
    return os.getenv("BACKEND_BASE_URL", BACKEND_URL_DEFAULT).rstrip("/")


def _init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "user_id" not in st.session_state:
        st.session_state.user_id = "user-1"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = os.getenv("LLM_PROVIDER", "ollama")


def _render_sidebar() -> None:
    st.sidebar.header("Session")
    user_id = st.sidebar.text_input("User ID", value=st.session_state.user_id)
    st.session_state.user_id = user_id or st.session_state.user_id

    st.sidebar.markdown(f"**LLM Provider:** `{st.session_state.llm_provider}`")

    new_session = st.sidebar.button("Start New Session")
    if new_session:
        st.session_state.session_id = None
        st.session_state.messages = []

    st.sidebar.markdown("---")
    st.session_state.show_sources = st.sidebar.checkbox(
        "Show sources", value=st.session_state.show_sources
    )

    if st.sidebar.button("Clear Conversation"):
        st.session_state.messages = []


def _append_message(role: str, content: str) -> None:
    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
            "kind": role,
            "created_at": datetime.utcnow().isoformat(),
        }
    )


def _call_chat_stream_api(
    base_url: str,
    session_id: Optional[str],
    user_id: str,
    message: str,
    on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    payload = {
        "session_id": session_id,
        "user_id": user_id,
        "message": message,
    }
    resp = requests.post(
        f"{base_url}/chat/stream",
        json=payload,
        timeout=120,
        stream=True,
    )
    resp.raise_for_status()

    event_name = "message"
    data_lines: List[str] = []
    step_events: List[Dict[str, Any]] = []
    final_payload: Dict[str, Any] = {}

    for raw_line in resp.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.strip()
        if not line:
            if not data_lines:
                continue
            raw_data = "\n".join(data_lines)
            data_lines = []
            payload_obj = json.loads(raw_data)
            event_type = payload_obj.get("type") or event_name
            if event_type == "step" and payload_obj.get("step"):
                step = payload_obj["step"]
                step_events.append(step)
                if on_step is not None:
                    on_step(step)
            elif event_type == "final" and payload_obj.get("final"):
                final_payload = payload_obj["final"]
            elif event_type == "error":
                raise RuntimeError(payload_obj.get("error") or "streaming error")
            event_name = "message"
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())

    if not final_payload:
        raise RuntimeError("No final response received from streaming endpoint.")
    return final_payload, step_events


def _render_agent_step(step: Dict[str, Any]) -> None:
    status = str(step.get("status") or "info")
    label = str(step.get("label") or "Agent step")
    detail = str(step.get("detail") or "").strip()
    if status == "failed":
        border = "#ef4444"
        badge = "FAILED"
    elif status == "completed":
        border = "#22c55e"
        badge = "DONE"
    elif status == "started":
        border = "#3b82f6"
        badge = "RUNNING"
    else:
        border = "#a855f7"
        badge = "INFO"

    detail_html = f"<div style='opacity:0.85;margin-top:4px'>{detail}</div>" if detail else ""
    st.markdown(
        (
            "<div style='border-left:4px solid {border};padding:8px 10px;"
            "margin:6px 0;background:rgba(59,130,246,0.08);border-radius:6px'>"
            "<div style='font-size:0.75rem;opacity:0.8'>Agent activity · {badge}</div>"
            "<div style='font-weight:600'>{label}</div>"
            "{detail_html}</div>"
        ).format(border=border, badge=badge, label=label, detail_html=detail_html),
        unsafe_allow_html=True,
    )


def _render_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        st.info("No sources returned for this answer.")
        return

    with st.expander("Sources"):
        for idx, src in enumerate(sources, start=1):
            score = src.get("score", 0.0)
            category = src.get("category") or "unknown"
            doc_tier = src.get("doc_tier", 1)
            source = src.get("source") or "unknown"
            st.markdown(
                f"**Source {idx}** — dataset: `{source}`, category: `{category}`, "
                f"tier: `{doc_tier}`, score: `{score:.2f}`"
            )
            st.write(src.get("content", "")[:500])
            st.markdown("---")


def _render_tool_activity(tools_used: List[str]) -> None:
    if not tools_used:
        return

    labels = []
    for name in tools_used:
        if name == "order_lookup":
            labels.append("🔍 Searched orders")
        elif name == "return_initiate":
            labels.append("📦 Return initiated")
        elif name == "faq_search":
            labels.append("📚 Searched FAQ")
        elif name == "ticket_create":
            labels.append("🎫 Ticket created")
        else:
            labels.append(f"🛠️ {name}")

    st.caption("Tools used: " + " · ".join(labels))


def main() -> None:
    st.set_page_config(page_title="E-Commerce Support", page_icon="💬")
    _init_session_state()
    _render_sidebar()

    st.title("E-Commerce Customer Support")
    st.write("Ask about orders, returns, products, or account issues.")

    base_url = _get_backend_base_url()

    for msg in st.session_state.messages:
        kind = msg.get("kind") or msg.get("role")
        if kind == "user":
            st.chat_message("user").markdown(msg["content"])
        elif kind == "agent_step":
            _render_agent_step(msg)
        else:
            st.chat_message("assistant").markdown(msg["content"])

    user_input = st.chat_input("Type your message")
    if user_input:
        _append_message("user", user_input)
        st.chat_message("user").markdown(user_input)

        try:
            resp, agent_steps = _call_chat_stream_api(
                base_url=base_url,
                session_id=st.session_state.session_id,
                user_id=st.session_state.user_id,
                message=user_input,
                on_step=_render_agent_step,
            )
        except Exception as exc:
            st.error(f"Error contacting backend: {exc}")
            return

        st.session_state.session_id = resp.get("session_id", st.session_state.session_id)

        answer = resp.get("response", "")
        escalated = bool(resp.get("escalated", False))
        ticket_id = resp.get("ticket_id")
        sources = resp.get("sources") or []
        tools_used = resp.get("tools_used") or []

        for step in agent_steps:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "kind": "agent_step",
                    "content": step.get("label") or "Agent step",
                    "label": step.get("label"),
                    "detail": step.get("detail"),
                    "status": step.get("status"),
                    "created_at": datetime.utcnow().isoformat(),
                }
            )

        if escalated:
            alert = "This conversation has been escalated to a human agent."
            if ticket_id:
                alert += f" Ticket ID: `{ticket_id}`."
            st.warning(alert)

        with st.chat_message("assistant"):
            st.markdown(answer or "_No response generated._")
            _render_tool_activity(tools_used)
            if st.session_state.show_sources:
                _render_sources(sources)

        _append_message("assistant", answer or "")


if __name__ == "__main__":
    main()

