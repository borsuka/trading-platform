"""Async engine and session management.

Schema is **never** auto-created outside development and test environments. Production schema
changes go through Alembic so that they are reviewable, revertible and auditable.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.config import AppEnv, Settings, get_settings
from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger
from app.database.base import Base

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def _engine_kwargs(settings: Settings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "echo": settings.database_echo,
        "future": True,
        "pool_pre_ping": True,
    }
    if settings.database_url.startswith("sqlite"):
        # SQLite has no meaningful server-side pooling; NullPool avoids cross-loop reuse
        # problems in tests where each test gets its own event loop.
        kwargs["poolclass"] = NullPool
        kwargs.pop("pool_pre_ping")
        _ensure_sqlite_directory(settings.database_url)
    else:
        kwargs["pool_size"] = settings.database_pool_size
        kwargs["max_overflow"] = settings.database_max_overflow
        kwargs["pool_recycle"] = 1800
    return kwargs


def _ensure_sqlite_directory(url: str) -> None:
    """Create the parent directory of a SQLite file so the first connection succeeds."""
    _, _, path_part = url.partition(":///")
    if not path_part or path_part == ":memory:":
        return
    path = Path(path_part)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    """Return the process-wide async engine, creating it on first use."""
    global _engine
    if _engine is None:
        resolved = settings or get_settings()
        if not resolved.database_url:
            raise ConfigurationError("DATABASE_URL is not configured")
        _engine = create_async_engine(resolved.database_url, **_engine_kwargs(resolved))
        logger.info(
            "database.engine_created",
            dialect=_engine.dialect.name,
            pool=type(_engine.pool).__name__,
        )
    return _engine


def get_sessionmaker(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    """Return the process-wide session factory."""
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            bind=get_engine(settings),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _sessionmaker


@asynccontextmanager
async def session_scope(
    settings: Settings | None = None,
) -> AsyncIterator[AsyncSession]:
    """Transactional scope: commit on success, roll back on any exception."""
    factory = get_sessionmaker(settings)
    session = factory()
    try:
        yield session
        await session.commit()
    except SQLAlchemyError:
        await session.rollback()
        logger.exception("database.transaction_failed")
        raise
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a request-scoped session."""
    async with session_scope() as session:
        yield session


async def create_all(settings: Settings | None = None) -> None:
    """Create the schema directly.

    Only permitted in development and test. Any other environment must use Alembic.
    """
    resolved = settings or get_settings()
    if resolved.app_env not in {AppEnv.DEVELOPMENT, AppEnv.TEST}:
        raise ConfigurationError(
            f"Refusing to auto-create schema in {resolved.app_env.value}; "
            "run 'alembic upgrade head' instead."
        )
    # Import for the side effect of registering every mapper before create_all runs.
    import app.database.models  # noqa: F401

    engine = get_engine(resolved)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    logger.info("database.schema_created", tables=len(Base.metadata.tables))


async def drop_all(settings: Settings | None = None) -> None:
    """Drop the schema. Test fixtures only."""
    resolved = settings or get_settings()
    if resolved.app_env is not AppEnv.TEST:
        raise ConfigurationError("drop_all is only permitted with APP_ENV=test")
    import app.database.models  # noqa: F401

    engine = get_engine(resolved)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)


async def check_connection(settings: Settings | None = None) -> bool:
    """Health probe. Returns False instead of raising so probes stay cheap."""
    from sqlalchemy import text

    try:
        engine = get_engine(settings)
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("database.health_check_failed", error=str(exc))
        return False
    return True


async def dispose_engine() -> None:
    """Close pooled connections. Called on shutdown and between test sessions."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
        logger.info("database.engine_disposed")
    _engine = None
    _sessionmaker = None
