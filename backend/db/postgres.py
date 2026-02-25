import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool


def _build_database_url() -> str:
    """
    Construct the async PostgreSQL URL from environment variables.

    Falls back to a DATABASE_URL env var if provided explicitly.
    """
    explicit = os.getenv("DATABASE_URL")
    if explicit:
        return explicit

    user = os.getenv("POSTGRES_USER", "admin")
    password = os.getenv("POSTGRES_PASSWORD", "changeme")

    # Default to the Docker service hostname when running in containers,
    # but transparently talk to the published port on the host when running
    # tools/tests outside Docker (no /.dockerenv present).
    host = os.getenv("POSTGRES_HOST", "postgres")
    if host == "postgres" and not os.path.exists("/.dockerenv"):
        host = "localhost"

    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "ecom_support")

    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"


DATABASE_URL = _build_database_url()

async_engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=False,
    # Use NullPool to avoid reusing connections across event loops in tests.
    poolclass=NullPool,
    pool_pre_ping=False,
)

async_session_factory = async_sessionmaker(
    async_engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI-style dependency for acquiring an async session.
    """
    async with async_session_factory() as session:
        yield session


__all__ = [
    "DATABASE_URL",
    "async_engine",
    "async_session_factory",
    "get_session",
]

