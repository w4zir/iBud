from __future__ import annotations

import uuid
from typing import List

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes.admin import router as admin_router
from .api.routes.chat import router as chat_router
from .api.routes.health import router as health_router
from .api.routes.sessions import router as sessions_router
from .config import log_event
from .observability.otel import init_tracing
from .observability.prometheus_metrics import error_count, request_count


def create_app() -> FastAPI:
    app = FastAPI(title="E-Commerce Support RAG API")
    init_tracing(app)

    origins_env = __import__("os").getenv("CORS_ORIGINS", "")
    origins: List[str] = []
    if origins_env:
        origins = [o.strip() for o in origins_env.split(",") if o.strip()]

    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    app.include_router(chat_router)
    app.include_router(sessions_router)
    app.include_router(health_router)
    app.include_router(admin_router)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ):
        request_id = getattr(request.state, "request_id", "")
        try:
            request_count.labels(status="validation_error").inc()
            error_count.labels(error_type="validation", component="api").inc()
        except Exception:
            pass
        log_event(
            "api",
            "validation_error",
            request_id=request_id,
            error_type="validation",
        )
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors()},
            headers={"X-Request-ID": request_id},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "")
        try:
            request_count.labels(status="error").inc()
            error_count.labels(error_type="unhandled", component="api").inc()
        except Exception:
            pass
        log_event(
            "api",
            "unhandled_error",
            request_id=request_id,
            error_type=type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
            headers={"X-Request-ID": request_id},
        )

    return app


app = create_app()

