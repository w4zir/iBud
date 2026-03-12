from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

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
    if "dataset" not in st.session_state:
        # Default KB dataset; matches Document.source="wixqa".
        st.session_state.dataset = "wixqa"
    if "company" not in st.session_state:
        # Default company identifier; can be overridden via sidebar.
        st.session_state.company = "default"


def _render_sidebar() -> None:
    st.sidebar.header("Session")
    user_id = st.sidebar.text_input("User ID", value=st.session_state.user_id)
    st.session_state.user_id = user_id or st.session_state.user_id

    st.sidebar.markdown(f"**LLM Provider:** `{st.session_state.llm_provider}`")

    st.sidebar.markdown("---")
    dataset_label = {
        "wixqa": "WixQA KB (articles)",
        "bitext": "Bitext QA pairs",
        "foodpanda": "Foodpanda policies",
    }
    dataset_key = st.sidebar.selectbox(
        "Knowledge base dataset",
        options=list(dataset_label.keys()),
        format_func=lambda k: dataset_label.get(k, k),
        index=list(dataset_label.keys()).index(st.session_state.get("dataset", "wixqa")),
    )
    st.session_state.dataset = dataset_key

    st.sidebar.markdown("---")
    company_label = {
        "default": "Default demo company",
        "foodpanda": "Foodpanda",
    }
    company_key = st.sidebar.selectbox(
        "Company",
        options=list(company_label.keys()),
        format_func=lambda k: company_label.get(k, k),
        index=list(company_label.keys()).index(st.session_state.get("company", "default")),
    )
    st.session_state.company = company_key

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
            "created_at": datetime.utcnow().isoformat(),
        }
    )


def _call_chat_api(
    base_url: str,
    session_id: Optional[str],
    user_id: str,
    message: str,
) -> Dict[str, Any]:
    payload = {
        "session_id": session_id,
        "user_id": user_id,
        "message": message,
        "dataset": st.session_state.get("dataset", "wixqa"),
        "company": st.session_state.get("company", "default"),
    }
    resp = requests.post(f"{base_url}/chat/", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


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
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    user_input = st.chat_input("Type your message")
    if user_input:
        _append_message("user", user_input)
        st.chat_message("user").markdown(user_input)

        try:
            resp = _call_chat_api(
                base_url=base_url,
                session_id=st.session_state.session_id,
                user_id=st.session_state.user_id,
                message=user_input,
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

