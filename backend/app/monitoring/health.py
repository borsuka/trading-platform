"""Health, readiness and runtime metrics.

Three distinct probes, because conflating them causes real outages:

* ``/health`` (liveness) — is the process alive? Must never depend on a downstream service, or
  a database blip restarts every container at once.
* ``/ready`` (readiness) — can this instance serve traffic? Depends on the database and, when
  required, Redis. Failing readiness removes the instance from the load balancer without
  killing it.
* ``/metrics`` — operational detail for humans and dashboards.
"""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.config import Settings, get_settings
from app.core.clock import utcnow
from app.core.logging import get_logger

logger = get_logger(__name__)

_START_TIME = time.monotonic()
_START_WALL = utcnow()


def uptime_seconds() -> float:
    return time.monotonic() - _START_TIME


def started_at() -> datetime:
    return _START_WALL


@dataclass(slots=True)
class ComponentHealth:
    """One dependency's health."""

    name: str
    healthy: bool
    detail: str = ""
    latency_ms: float | None = None
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "detail": self.detail,
            "latency_ms": round(self.latency_ms, 2) if self.latency_ms else None,
            "required": self.required,
        }


@dataclass(slots=True)
class HealthReport:
    components: list[ComponentHealth] = field(default_factory=list)

    @property
    def status(self) -> str:
        if any(not c.healthy for c in self.components if c.required):
            return "unhealthy"
        if any(not c.healthy for c in self.components):
            return "degraded"
        return "ok"

    @property
    def ready(self) -> bool:
        return all(c.healthy for c in self.components if c.required)

    def to_dict(self) -> dict[str, Any]:
        return {c.name: c.to_dict() for c in self.components}


async def check_database(settings: Settings | None = None) -> ComponentHealth:
    from app.database.session import check_connection

    started = time.monotonic()
    healthy = await check_connection(settings)
    return ComponentHealth(
        name="database",
        healthy=healthy,
        detail="connected" if healthy else "cannot reach the database",
        latency_ms=(time.monotonic() - started) * 1000,
        required=True,
    )


async def check_redis(settings: Settings | None = None) -> ComponentHealth:
    """Redis is optional unless ``REDIS_REQUIRED`` is set.

    The single-node desktop deployment runs perfectly well without it; a multi-worker
    deployment needs it for shared rate limiting and the job queue.
    """
    resolved = settings or get_settings()
    if not resolved.redis_url:
        return ComponentHealth(
            name="redis",
            healthy=True,
            detail="not configured (optional)",
            required=False,
        )
    started = time.monotonic()
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(resolved.redis_url, socket_connect_timeout=2.0)
        await client.ping()
        await client.aclose()
    except Exception as exc:
        return ComponentHealth(
            name="redis",
            healthy=False,
            detail=f"unreachable: {type(exc).__name__}",
            latency_ms=(time.monotonic() - started) * 1000,
            required=resolved.redis_required,
        )
    return ComponentHealth(
        name="redis",
        healthy=True,
        detail="connected",
        latency_ms=(time.monotonic() - started) * 1000,
        required=resolved.redis_required,
    )


def check_bots() -> ComponentHealth:
    """Report running bots and whether any have halted.

    A halted bot is reported as *degraded*, never unhealthy: halting is the system working
    correctly, and restarting the container would not fix it.
    """
    from app.core.enums import BotStatus
    from app.paper_trading.runtime import bot_registry

    bots = bot_registry.all()
    running = [b for b in bots if b.status is BotStatus.RUNNING]
    halted = [b for b in bots if b.status in {BotStatus.HALTED, BotStatus.ERROR}]

    return ComponentHealth(
        name="bots",
        healthy=not halted,
        detail=(
            f"{len(running)} running, {len(halted)} halted: "
            + ", ".join(f"{b.config.name} ({b.status.value})" for b in halted)
            if halted
            else f"{len(running)} running, {len(bots)} total"
        ),
        required=False,
    )


def check_trading_mode(settings: Settings | None = None) -> ComponentHealth:
    """Surface the trading mode as a health component.

    Live mode is not unhealthy, but it is the single most important fact about a running
    instance, so it belongs where an operator will actually see it.
    """
    resolved = settings or get_settings()
    return ComponentHealth(
        name="trading_mode",
        healthy=True,
        detail=(
            "LIVE - real orders with real money"
            if resolved.is_live
            else f"{resolved.describe_mode()} - no real money at risk"
        ),
        required=False,
    )


async def build_health_report(settings: Settings | None = None) -> HealthReport:
    """Full health report across every dependency."""
    resolved = settings or get_settings()
    return HealthReport(
        components=[
            await check_database(resolved),
            await check_redis(resolved),
            check_bots(),
            check_trading_mode(resolved),
        ]
    )


def runtime_metrics() -> dict[str, Any]:
    """Process and platform metrics.

    Uses only the standard library. ``psutil`` would give richer numbers but adds a compiled
    dependency to a desktop build for information an operator rarely needs.
    """
    from app.paper_trading.runtime import bot_registry

    metrics: dict[str, Any] = {
        "uptime_seconds": round(uptime_seconds(), 1),
        "started_at": started_at().isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "pid": os.getpid(),
        "bots_total": len(bot_registry.all()),
        "bots_running": sum(1 for b in bot_registry.all() if b.is_running),
    }

    # `resource` is Unix-only; Windows deployments simply omit these fields.
    try:
        import resource  # type: ignore[import-not-found,unused-ignore]

        usage = resource.getrusage(resource.RUSAGE_SELF)
        metrics["max_rss_kb"] = usage.ru_maxrss
        metrics["cpu_seconds"] = round(usage.ru_utime + usage.ru_stime, 2)
    except (ImportError, AttributeError):
        pass

    return metrics
