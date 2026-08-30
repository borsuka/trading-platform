"""Alembic environment.

The database URL is read from application settings rather than ``alembic.ini``, so migrations
always target the same database the application does. A checked-in URL is how a migration ends
up running against the wrong environment.

The async engine is driven through ``run_sync`` because Alembic's migration API is synchronous.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.config import get_settings
from app.database.base import Base

# Import for the side effect of registering every mapper before autogenerate compares.
import app.database.models  # noqa: F401  isort:skip

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)


def _render_item(type_, obj, autogen_context) -> str | bool:  # noqa: ANN001
    """Render application-defined column types as their underlying SQL types.

    A migration that imports ``app.database.base`` would break the moment that module is
    refactored - and a migration must keep running years after the code around it changed.
    ``JSONDict`` only differs from ``JSON`` in Python-side default handling, so the DDL is
    identical.
    """
    if type_ == "type" and obj.__class__.__name__ == "JSONDict":
        return "sa.JSON()"
    return False


def _include_object(obj, name, type_, reflected, compare_to) -> bool:  # noqa: ANN001
    """Skip objects Alembic should not manage.

    SQLite's internal tables show up in reflection and would otherwise generate spurious
    drop operations.
    """
    if type_ == "table" and name.startswith("sqlite_"):
        return False
    return True


def run_migrations_offline() -> None:
    """Emit SQL without a database connection, for review or manual application."""
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_object=_include_object,
        render_item=_render_item,
        render_as_batch=settings.database_url.startswith("sqlite"),
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_object=_include_object,
        render_item=_render_item,
        # SQLite cannot ALTER most things; batch mode rebuilds the table instead.
        render_as_batch=connection.dialect.name == "sqlite",
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
