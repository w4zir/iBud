from __future__ import annotations

import types

import pytest


@pytest.mark.asyncio
async def test_classify_intent_skips_prometheus_and_otel_when_observability_disabled(monkeypatch):
    import backend.agent.nodes as nodes

    class DummyLLM:
        async def ainvoke(self, messages):  # type: ignore[no-untyped-def]
            # Return something that would normally trigger token metrics.
            return types.SimpleNamespace(
                content="cancel_order",
                response_metadata={"token_usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
            )

    # If OTel is accidentally invoked, fail loudly.
    def boom_get_tracer():  # type: ignore[no-untyped-def]
        raise AssertionError("get_tracer() should not be called for intent-only eval.")

    monkeypatch.setattr(nodes, "get_llm", lambda: DummyLLM())
    monkeypatch.setattr(nodes, "get_tracer", boom_get_tracer)

    # If Prometheus is accidentally invoked, fail loudly.
    class BoomCounter:
        def labels(self, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("Prometheus metric should not be incremented for intent-only eval.")

    monkeypatch.setattr(nodes, "intent_distribution", BoomCounter())
    monkeypatch.setattr(nodes, "llm_tokens", BoomCounter())

    state = {
        "messages": [{"role": "user", "content": "please cancel my order"}],
        "observability_disabled": True,
    }
    out = await nodes.classify_intent(state)  # type: ignore[arg-type]
    assert out["intent"] == "cancel_order"

