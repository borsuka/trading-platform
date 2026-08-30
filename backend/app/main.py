"""FastAPI application factory.

Startup deliberately does **not** create the database schema outside development: production
schema changes go through Alembic so they are reviewable and revertible.

Shutdown stops every running bot. It does not close their positions — a deploy or a restart
must not become a forced liquidation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api import api_router, system_router
from app.config import AppEnv, Settings, get_settings
from app.core.exceptions import TradingPlatformError
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown."""
    settings: Settings = app.state.settings
    # `force` because uvicorn installs its own logging config after the module is imported;
    # without this, structlog's formatting and - more importantly - its secret redaction would
    # be silently replaced.
    configure_logging(force=True)

    logger.warning(
        "app.starting",
        version=settings.app_version,
        environment=settings.app_env.value,
        mode=settings.describe_mode(),
        live_trading_enabled=settings.live_trading_enabled,
    )
    if settings.is_live:
        logger.warning(
            "app.live_trading_active",
            message=(
                "This instance is configured for LIVE trading. Real orders will be placed "
                "with real money."
            ),
        )

    if settings.email_verification_required and not settings.email_delivery_configured:
        logger.error(
            "app.email_verification_unreachable",
            message=(
                "Email verification is required but no SMTP host is configured. New accounts "
                "will be created in pending_verification and will NOT be able to sign in. "
                "Configure SMTP_HOST/SMTP_FROM, or set REQUIRE_EMAIL_VERIFICATION=false if "
                "this is a single-operator install."
            ),
        )

    if settings.app_env in {AppEnv.DEVELOPMENT, AppEnv.TEST}:
        from app.database.session import create_all

        await create_all(settings)
        logger.info("app.schema_ensured", environment=settings.app_env.value)
    else:
        logger.info(
            "app.schema_not_managed",
            message="Run 'alembic upgrade head' to apply migrations.",
        )

    await _reconcile_bot_status()

    try:
        yield
    finally:
        from app.database.session import dispose_engine
        from app.paper_trading.runtime import bot_registry

        running = [b.config.name for b in bot_registry.all() if b.is_running]
        if running:
            logger.warning(
                "app.stopping_bots",
                bots=running,
                message="Positions are left open; stopping a bot is not a liquidation.",
            )
        await bot_registry.stop_all()
        await dispose_engine()
        logger.warning("app.stopped")


async def _reconcile_bot_status() -> None:
    """Correct bots the database still believes are running.

    A bot's *record* is persistent; its *runtime* is not - it lives in this process and dies
    with it. So a restart, a crash, or a machine reboot leaves rows saying RUNNING with
    nothing behind them, and the dashboard then shows a green RUNNING badge next to a bot that
    will never take another action. That is the worst kind of wrong: it looks like everything
    is fine.

    The registry is empty by definition at startup, so any such row is stale. They are marked
    STOPPED rather than restarted: the user started those bots in a session that has ended,
    and silently resuming trading on their behalf is not a decision this code should make.
    """
    from sqlalchemy import select, update

    from app.core.enums import BotStatus
    from app.database.models import Bot
    from app.database.session import session_scope

    try:
        async with session_scope() as session:
            stale = (
                await session.execute(
                    select(Bot.id, Bot.name).where(
                        Bot.status.in_([BotStatus.RUNNING, BotStatus.STARTING])
                    )
                )
            ).all()
            if not stale:
                return
            await session.execute(
                update(Bot)
                .where(Bot.status.in_([BotStatus.RUNNING, BotStatus.STARTING]))
                .values(
                    status=BotStatus.STOPPED,
                    last_error=(
                        "Marked stopped on startup: the process that was running this bot "
                        "exited. Start it again when you want it trading."
                    ),
                )
            )
        logger.warning(
            "app.stale_bot_status_cleared",
            bots=[name for _, name in stale],
            message=(
                "These bots were recorded as running but no runtime survived the restart. "
                "They are now marked stopped and must be started again."
            ),
        )
    except Exception:
        # A reconciliation failure must not stop the application from starting. The rows stay
        # wrong, which is bad, but an API that will not boot is worse.
        logger.exception("app.bot_reconciliation_failed")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Baseline security headers.

    HSTS is set only over HTTPS: sending it over plain HTTP is meaningless and would break
    local development.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy", "geolocation=(), microphone=(), camera=()"
        )
        if request.url.scheme == "https":
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response


class TradingModeMiddleware(BaseHTTPMiddleware):
    """Stamp the trading mode on every response.

    The frontend reads this to render the PAPER/LIVE banner. Putting it on every response
    means the banner cannot go stale after a configuration change.
    """

    def __init__(self, app: Any, settings: Settings) -> None:
        super().__init__(app)
        self.mode = settings.describe_mode()
        self.is_live = settings.is_live

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["X-Trading-Mode"] = self.mode
        response.headers["X-Live-Trading"] = "true" if self.is_live else "false"
        return response


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application."""
    resolved = settings or get_settings()
    configure_logging()

    app = FastAPI(
        title=resolved.app_name,
        version=resolved.app_version,
        description=(
            "Algorithmic trading platform API.\n\n"
            "**This software does not provide investment advice and does not guarantee any "
            "return.** It executes trades according to rules you configure. Past performance "
            "and backtest results do not guarantee future performance. You can lose money.\n\n"
            f"Current mode: **{resolved.describe_mode()}**"
        ),
        lifespan=lifespan,
        docs_url="/docs" if not resolved.is_production else None,
        redoc_url="/redoc" if not resolved.is_production else None,
        openapi_url="/openapi.json" if not resolved.is_production else None,
    )
    app.state.settings = resolved

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(TradingModeMiddleware, settings=resolved)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["X-Trading-Mode", "X-Live-Trading"],
    )

    _register_error_handlers(app)

    app.include_router(system_router)
    app.include_router(api_router, prefix=resolved.api_prefix)

    @app.get("/api", include_in_schema=False)
    async def api_root() -> dict[str, Any]:
        return {
            "name": resolved.app_name,
            "version": resolved.app_version,
            "mode": resolved.describe_mode(),
            "docs": "/docs" if not resolved.is_production else None,
            "api": resolved.api_prefix,
        }

    # Last, deliberately: this is a catch-all mount at "/" and anything registered after it
    # would never be reached. When the dashboard has not been built the API simply runs
    # without one.
    from app.web import mount_ui

    mount_ui(app)

    return app


def _register_error_handlers(app: FastAPI) -> None:
    """Map exceptions onto responses without leaking internals."""

    @app.exception_handler(TradingPlatformError)
    async def platform_error(request: Request, exc: TradingPlatformError) -> JSONResponse:
        # `public_message()` decides what a caller may see; the full detail goes to the log.
        if exc.status_code >= 500:
            logger.error(
                "api.internal_error",
                path=request.url.path,
                error_code=exc.error_code,
                detail=exc.message,
            )
        else:
            logger.info(
                "api.client_error",
                path=request.url.path,
                error_code=exc.error_code,
                status=exc.status_code,
            )
        headers: dict[str, str] = {}
        retry_after = getattr(exc, "retry_after_seconds", None)
        if retry_after:
            headers["Retry-After"] = str(int(retry_after))
        return JSONResponse(
            status_code=exc.status_code, content=exc.to_dict(), headers=headers
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "message": "The request body failed validation",
                "context": {
                    "errors": [
                        {
                            "field": ".".join(str(p) for p in err.get("loc", [])[1:]),
                            "message": err.get("msg", ""),
                        }
                        for err in exc.errors()[:20]
                    ]
                },
            },
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        """Last resort.

        The response is deliberately opaque: an unexpected exception's message can contain
        connection strings, file paths or query fragments.
        """
        logger.exception("api.unhandled_exception", path=request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_error",
                "message": "An internal error occurred. It has been logged.",
                "context": {},
            },
        )


app = create_app()


def main() -> None:
    """Run the development server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=settings.app_env is AppEnv.DEVELOPMENT,
        log_config=None,  # structlog owns logging
    )


if __name__ == "__main__":
    main()
