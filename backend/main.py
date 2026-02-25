from __future__ import annotations

from typing import List

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes.admin import router as admin_router
from .api.routes.chat import router as chat_router
from .api.routes.health import router as health_router
from .api.routes.sessions import router as sessions_router
from .observability.prometheus_metrics import request_count


def create_app() -> FastAPI:
    app = FastAPI(title="E-Commerce Support RAG API")

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

    app.include_router(chat_router)
    app.include_router(sessions_router)
    app.include_router(health_router)
    app.include_router(admin_router)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ):
        try:
            request_count.labels(status="validation_error").inc()
        except Exception:
            pass
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        try:
            request_count.labels(status="error").inc()
        except Exception:
            pass
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app


app = create_app()

