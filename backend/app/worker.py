"""Background worker.

Runs the periodic maintenance that must happen whether or not anyone is using the API: retrying
undelivered notifications, purging expired tokens, warning about expiring licences, and
checking on running bots.

Design notes:

* Every job is idempotent, because a worker restart re-runs whatever was in flight.
* A job that fails is logged and the loop continues. One broken job must never stop the others,
  and it must never stop the trading engine, which runs in a different process.
* Jobs are scheduled by elapsed time rather than a cron expression, so a restart does not
  either skip a window or fire everything at once.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.core.clock import utcnow
from app.core.logging import configure_logging, get_logger
from app.database.session import dispose_engine, session_scope

logger = get_logger(__name__)

JobFunc = Callable[[], Awaitable[str]]


@dataclass(slots=True)
class Job:
    """A periodic task."""

    name: str
    func: JobFunc
    interval: timedelta
    last_run: datetime | None = field(default=None)
    failures: int = 0

    def is_due(self, now: datetime) -> bool:
        return self.last_run is None or now - self.last_run >= self.interval


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
async def retry_notifications() -> str:
    """Redeliver notifications that failed the first time."""
    from app.notifications.dispatcher import NotificationDispatcher

    async with session_scope() as session:
        retried = await NotificationDispatcher(session).retry_pending(limit=100)
    return f"redelivered {retried} notification(s)"


async def purge_expired_tokens() -> str:
    """Remove refresh tokens that have expired."""
    from app.auth.service import AuthService

    async with session_scope() as session:
        removed = await AuthService(session).purge_expired_tokens()
    return f"purged {removed} expired token(s)"


async def warn_expiring_licenses() -> str:
    """Notify users whose licence expires within a week."""
    from sqlalchemy import select

    from app.core.enums import LicenseStatus, NotificationEvent
    from app.database.models import License, User
    from app.notifications.dispatcher import NotificationDispatcher

    cutoff = utcnow() + timedelta(days=7)
    notified = 0

    async with session_scope() as session:
        result = await session.execute(
            select(License).where(
                License.status == LicenseStatus.ACTIVE,
                License.expires_at.is_not(None),
                License.expires_at <= cutoff,
                License.expires_at > utcnow(),
            )
        )
        dispatcher = NotificationDispatcher(session)
        for license_record in result.scalars().all():
            user = await session.get(User, license_record.user_id)
            if user is None or license_record.expires_at is None:
                continue
            days = max(0, (license_record.expires_at - utcnow()).days)
            await dispatcher.dispatch(
                user=user,
                event=NotificationEvent.LICENSE_EXPIRING,
                title=f"Your licence expires in {days} day(s)",
                body=(
                    f"Your {license_record.plan.value} licence expires on "
                    f"{license_record.expires_at.date().isoformat()}. Renew it to avoid "
                    "interruption."
                ),
            )
            notified += 1
    return f"warned {notified} user(s) about expiring licences"


async def expire_licenses() -> str:
    """Mark licences past their expiry date as expired."""
    from sqlalchemy import select

    from app.core.enums import LicenseStatus
    from app.database.models import License

    expired = 0
    async with session_scope() as session:
        result = await session.execute(
            select(License).where(
                License.status == LicenseStatus.ACTIVE,
                License.expires_at.is_not(None),
                License.expires_at <= utcnow(),
            )
        )
        for record in result.scalars().all():
            record.status = LicenseStatus.EXPIRED
            expired += 1
    return f"expired {expired} licence(s)"


async def check_bot_health() -> str:
    """Report bots that have halted or errored.

    Reporting only. The worker deliberately does not restart a halted bot: halting is the
    system working correctly, and an automatic restart would defeat the mechanism.
    """
    from app.core.enums import BotStatus
    from app.paper_trading.runtime import bot_registry

    bots = bot_registry.all()
    troubled = [b for b in bots if b.status in {BotStatus.HALTED, BotStatus.ERROR}]
    for bot in troubled:
        logger.error(
            "worker.bot_needs_attention",
            bot_id=bot.config.bot_id,
            name=bot.config.name,
            status=bot.status.value,
            halt_reason=bot.snapshot().halt_reason,
        )
    running = sum(1 for b in bots if b.is_running)
    return f"{running} running, {len(troubled)} needing attention"


async def prune_order_history() -> str:
    """Bound memory in long-running bots by dropping old terminal orders."""
    from app.paper_trading.runtime import bot_registry

    removed = sum(bot.order_manager.prune(keep=1000) for bot in bot_registry.all())
    return f"pruned {removed} terminal order(s)"


DEFAULT_JOBS: tuple[Job, ...] = (
    Job("retry_notifications", retry_notifications, timedelta(minutes=1)),
    Job("check_bot_health", check_bot_health, timedelta(minutes=1)),
    Job("purge_expired_tokens", purge_expired_tokens, timedelta(hours=6)),
    Job("expire_licenses", expire_licenses, timedelta(hours=1)),
    Job("warn_expiring_licenses", warn_expiring_licenses, timedelta(hours=24)),
    Job("prune_order_history", prune_order_history, timedelta(hours=1)),
)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
class Worker:
    """Runs periodic jobs until told to stop."""

    def __init__(
        self, jobs: tuple[Job, ...] | None = None, *, tick_seconds: float = 10.0
    ) -> None:
        self.jobs = list(jobs or DEFAULT_JOBS)
        self.tick_seconds = tick_seconds
        self._stop = asyncio.Event()

    def request_stop(self) -> None:
        self._stop.set()

    async def run_due_jobs(self, *, now: datetime | None = None) -> int:
        """Run every job that is due. Returns how many ran."""
        moment = now or utcnow()
        ran = 0
        for job in self.jobs:
            if not job.is_due(moment):
                continue
            job.last_run = moment
            ran += 1
            try:
                detail = await job.func()
            except Exception:
                job.failures += 1
                # One failing job must not stop the others, and must never affect the
                # trading engine, which runs in a different process.
                logger.exception(
                    "worker.job_failed", job=job.name, consecutive_failures=job.failures
                )
                continue
            job.failures = 0
            logger.info("worker.job_completed", job=job.name, detail=detail)
        return ran

    async def run(self) -> None:
        logger.warning(
            "worker.started", jobs=[j.name for j in self.jobs], tick=self.tick_seconds
        )
        try:
            while not self._stop.is_set():
                await self.run_due_jobs()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=self.tick_seconds)
        finally:
            await dispose_engine()
            logger.warning("worker.stopped")


async def main() -> None:
    configure_logging()
    worker = Worker()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, AttributeError):
            # Signal handlers are unavailable on Windows event loops; Ctrl-C still raises
            # KeyboardInterrupt there, which asyncio.run surfaces.
            loop.add_signal_handler(sig, worker.request_stop)

    await worker.run()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
