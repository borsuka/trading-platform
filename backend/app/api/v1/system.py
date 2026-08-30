"""Health, readiness and system information endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status

from app.api.dependencies import CurrentUser, SettingsDep
from app.api.schemas import HealthResponse, ReadinessResponse
from app.monitoring.health import (
    build_health_report,
    runtime_metrics,
    uptime_seconds,
)

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(response: Response, settings: SettingsDep) -> HealthResponse:
    """Full health report.

    Returns 503 when a required dependency is down, so an orchestrator can act on the status
    code without parsing the body.
    """
    report = await build_health_report(settings)
    if report.status == "unhealthy":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status=report.status,  # type: ignore[arg-type]
        version=settings.app_version,
        mode=settings.describe_mode(),
        uptime_seconds=round(uptime_seconds(), 1),
        checks=report.to_dict(),
    )


@router.get("/live", response_model=dict)
async def liveness() -> dict[str, Any]:
    """Liveness probe.

    Deliberately checks nothing external: a database outage must not cause every container to
    be killed and restarted, which would turn a recoverable incident into an outage.
    """
    return {"alive": True, "uptime_seconds": round(uptime_seconds(), 1)}


@router.get("/ready", response_model=ReadinessResponse)
async def readiness(response: Response, settings: SettingsDep) -> ReadinessResponse:
    """Readiness probe. Fails when a required dependency is unavailable."""
    report = await build_health_report(settings)
    if not report.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        ready=report.ready,
        checks={c.name: c.healthy for c in report.components},
    )


@router.get("/metrics", response_model=dict)
async def metrics(user: CurrentUser) -> dict[str, Any]:
    """Runtime metrics. Authenticated: it exposes process detail."""
    return runtime_metrics()


@router.get("/info", response_model=dict)
async def info(settings: SettingsDep) -> dict[str, Any]:
    """Public build and mode information.

    Unauthenticated on purpose: the frontend needs the trading mode before login so it can show
    the PAPER/LIVE banner from the very first screen.
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "mode": settings.describe_mode(),
        "is_live": settings.is_live,
        "environment": settings.app_env.value,
        "disclaimer": (
            "This software executes trades according to configured rules. It does not "
            "provide investment advice, and no return is guaranteed. Past performance and "
            "backtest results do not guarantee future performance."
        ),
    }
