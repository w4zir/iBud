from __future__ import annotations

from datetime import datetime, timezone

from backend.observability.tracing import Span, Trace, create_trace


def test_create_trace_sets_trace_id_to_session_id():
    trace = create_trace("sess-42", channel="web", user_id="u-1")
    assert trace.trace_id == "sess-42"
    assert trace.session_id == "sess-42"
    assert trace.channel == "web"
    assert trace.user_id == "u-1"
    assert trace.spans == []


def test_start_span_appends_to_trace():
    trace = create_trace("sess-1")
    span = trace.start_span("intent_detection", intent="order_status", confidence=0.95)

    assert len(trace.spans) == 1
    assert span is trace.spans[0]
    assert span.trace_id == "sess-1"
    assert span.span_name == "intent_detection"
    assert span.attributes == {"intent": "order_status", "confidence": 0.95}
    assert span.start_time is not None
    assert span.end_time is None
    assert span.latency_ms is None


def test_span_finish_records_latency():
    span = Span(
        trace_id="t-1",
        span_id="s-1",
        span_name="retrieval",
        attributes={},
        start_time=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    )
    span.finish()

    assert span.end_time is not None
    assert span.latency_ms is not None
    assert span.latency_ms >= 0


def test_multiple_spans_on_trace():
    trace = create_trace("sess-2")
    trace.start_span("intent_detection", intent="product_qa")
    trace.start_span("retrieval", docs_returned=3)
    trace.start_span("tool_call", tool_name="faq_search", success=True)

    assert len(trace.spans) == 3
    names = [s.span_name for s in trace.spans]
    assert names == ["intent_detection", "retrieval", "tool_call"]


def test_root_attributes():
    trace = create_trace("sess-3", channel="mobile", user_id="u-5")
    attrs = trace.root_attributes
    assert attrs["session_id"] == "sess-3"
    assert attrs["channel"] == "mobile"
    assert attrs["user_id"] == "u-5"


def test_span_ids_are_unique():
    trace = create_trace("sess-4")
    s1 = trace.start_span("a")
    s2 = trace.start_span("b")
    assert s1.span_id != s2.span_id
