from __future__ import annotations

import os
from typing import Optional, Tuple

from fastapi import FastAPI

_tracer = None
_initialized = False


def _env_flag(name: str) -> bool:
    return (os.getenv(name, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def is_otel_enabled() -> bool:
    return _env_flag("OTEL_ENABLED")


def init_tracing(app: FastAPI) -> None:
    global _tracer, _initialized
    if _initialized or not is_otel_enabled():
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception:
        return

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    service_name = os.getenv("OTEL_SERVICE_NAME", "ibud-backend")

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
    _tracer = trace.get_tracer("ibud.backend")
    _initialized = True


def get_tracer():
    global _tracer
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace
    except Exception:
        return None
    _tracer = trace.get_tracer("ibud.backend")
    return _tracer


def get_current_trace_ids() -> Tuple[Optional[str], Optional[str]]:
    try:
        from opentelemetry import trace
    except Exception:
        return None, None

    span = trace.get_current_span()
    if span is None:
        return None, None
    context = span.get_span_context()
    if not context or not context.is_valid:
        return None, None
    trace_id = format(context.trace_id, "032x")
    span_id = format(context.span_id, "016x")
    return trace_id, span_id


__all__ = ["is_otel_enabled", "init_tracing", "get_tracer", "get_current_trace_ids"]

