"""Notification dispatch.

Notifications are **persisted first, delivered second**. A notification that exists only in a
send attempt is lost when that attempt fails, and the events this platform notifies about —
a stop-loss hit, a kill switch trip, a crashed bot — are exactly the ones a user must not miss.

So: write the row, then try to deliver, then record the outcome on the row. A failed delivery
leaves a visible record with its reason, and the in-app feed always has the event regardless of
whether any external channel was configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import utcnow
from app.core.enums import (
    NotificationChannel,
    NotificationEvent,
    NotificationStatus,
)
from app.core.logging import get_logger
from app.database.models import Notification, User
from app.database.repositories import NotificationRepository
from app.notifications.channels import (
    NotificationChannelBase,
    NotificationMessage,
    build_channels,
)

logger = get_logger(__name__)

#: Which events go to which channels by default. Users can narrow this per account.
DEFAULT_ROUTING: dict[NotificationEvent, tuple[NotificationChannel, ...]] = {
    NotificationEvent.TRADE_OPENED: (NotificationChannel.IN_APP,),
    NotificationEvent.TRADE_CLOSED: (NotificationChannel.IN_APP,),
    NotificationEvent.TAKE_PROFIT_HIT: (NotificationChannel.IN_APP,),
    NotificationEvent.STOP_LOSS_HIT: (
        NotificationChannel.IN_APP,
        NotificationChannel.TELEGRAM,
    ),
    NotificationEvent.RISK_ALERT: (
        NotificationChannel.IN_APP,
        NotificationChannel.TELEGRAM,
        NotificationChannel.EMAIL,
    ),
    NotificationEvent.DAILY_LOSS_LIMIT: (
        NotificationChannel.IN_APP,
        NotificationChannel.TELEGRAM,
        NotificationChannel.EMAIL,
    ),
    NotificationEvent.BOT_CRASHED: (
        NotificationChannel.IN_APP,
        NotificationChannel.TELEGRAM,
        NotificationChannel.EMAIL,
    ),
    NotificationEvent.EXCHANGE_DISCONNECTED: (
        NotificationChannel.IN_APP,
        NotificationChannel.TELEGRAM,
    ),
    NotificationEvent.LICENSE_EXPIRING: (
        NotificationChannel.IN_APP,
        NotificationChannel.EMAIL,
    ),
    NotificationEvent.SYSTEM_ERROR: (NotificationChannel.IN_APP,),
    NotificationEvent.ACCOUNT_SECURITY: (
        NotificationChannel.IN_APP,
        NotificationChannel.EMAIL,
    ),
}

SEVERITY: dict[NotificationEvent, str] = {
    NotificationEvent.STOP_LOSS_HIT: "warning",
    NotificationEvent.RISK_ALERT: "warning",
    NotificationEvent.DAILY_LOSS_LIMIT: "critical",
    NotificationEvent.BOT_CRASHED: "critical",
    NotificationEvent.EXCHANGE_DISCONNECTED: "critical",
    NotificationEvent.SYSTEM_ERROR: "critical",
    NotificationEvent.LICENSE_EXPIRING: "warning",
}


@dataclass(slots=True)
class DispatchReport:
    """What happened to one notification across its channels."""

    notification_ids: list[str]
    delivered: list[NotificationChannel]
    failed: list[tuple[NotificationChannel, str]]
    skipped: list[tuple[NotificationChannel, str]]

    @property
    def any_delivered(self) -> bool:
        return bool(self.delivered)


class NotificationDispatcher:
    """Persists and delivers notifications."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        channels: dict[NotificationChannel, NotificationChannelBase] | None = None,
        routing: dict[NotificationEvent, tuple[NotificationChannel, ...]] | None = None,
    ) -> None:
        self.session = session
        self.repository = NotificationRepository(session)
        self.channels = channels or build_channels()
        self.routing = routing or DEFAULT_ROUTING

    async def dispatch(
        self,
        *,
        user: User,
        event: NotificationEvent,
        title: str,
        body: str,
        bot_id: str | None = None,
        payload: dict[str, Any] | None = None,
        channels: tuple[NotificationChannel, ...] | None = None,
    ) -> DispatchReport:
        """Persist and attempt delivery on every routed channel."""
        targets = channels or self.routing.get(event, (NotificationChannel.IN_APP,))
        severity = SEVERITY.get(event, "info")
        message = NotificationMessage(
            event=event, title=title, body=body, severity=severity, payload=payload
        )

        report = DispatchReport(
            notification_ids=[], delivered=[], failed=[], skipped=[]
        )

        for channel_name in targets:
            channel = self.channels.get(channel_name)
            record = Notification(
                user_id=user.id,
                bot_id=bot_id,
                event=event,
                channel=channel_name,
                status=NotificationStatus.PENDING,
                title=title[:255],
                body=body,
                payload=payload or {},
                created_at=utcnow(),
            )
            await self.repository.add(record)
            report.notification_ids.append(record.id)

            if channel is None or not channel.is_configured:
                record.status = NotificationStatus.SUPPRESSED
                record.error = f"{channel_name.value} is not configured"
                report.skipped.append((channel_name, record.error))
                continue

            record.attempts += 1
            result = await channel.send(message, recipient=self._recipient(user, channel_name))
            if result.delivered:
                record.status = NotificationStatus.SENT
                record.sent_at = utcnow()
                report.delivered.append(channel_name)
            else:
                record.status = NotificationStatus.FAILED
                record.error = result.detail
                report.failed.append((channel_name, result.detail))
                logger.warning(
                    "notifications.delivery_failed",
                    channel=channel_name.value,
                    notification_event=event.value,
                    detail=result.detail,
                )

        await self.session.flush()
        return report

    @staticmethod
    def _recipient(user: User, channel: NotificationChannel) -> str:
        """Where to deliver, per channel.

        Per-user channel addresses live in ``user.settings`` so adding a channel does not
        require a migration.
        """
        settings = user.settings or {}
        if channel is NotificationChannel.EMAIL:
            return user.email
        if channel is NotificationChannel.TELEGRAM:
            return str(settings.get("telegram_chat_id", ""))
        return user.id

    async def retry_pending(self, *, limit: int = 50) -> int:
        """Retry undelivered notifications. Called by the worker."""
        pending = await self.repository.pending(limit=limit)
        retried = 0
        for record in pending:
            channel = self.channels.get(record.channel)
            if channel is None or not channel.is_configured:
                record.status = NotificationStatus.SUPPRESSED
                continue
            if record.attempts >= 5:
                record.status = NotificationStatus.FAILED
                record.error = "gave up after 5 attempts"
                continue
            record.attempts += 1
            result = await channel.send(
                NotificationMessage(
                    event=record.event,
                    title=record.title,
                    body=record.body,
                    severity=SEVERITY.get(record.event, "info"),
                ),
                recipient=record.user_id,
            )
            if result.delivered:
                record.status = NotificationStatus.SENT
                record.sent_at = utcnow()
                retried += 1
            else:
                record.error = result.detail
        await self.session.flush()
        return retried


# --------------------------------------------------------------------------- #
# Convenience helpers used by the API and the bot runtime
# --------------------------------------------------------------------------- #
async def queue_verification_email(
    session: AsyncSession, user: User, token: str
) -> DispatchReport:
    """Queue an email-verification message.

    The link is built by the caller's frontend; the token is what matters here.
    """
    return await NotificationDispatcher(session).dispatch(
        user=user,
        event=NotificationEvent.ACCOUNT_SECURITY,
        title="Verify your email address",
        body=(
            "Use this verification token to activate your account:\n\n"
            f"{token}\n\nIf you did not create an account, ignore this message."
        ),
        channels=(NotificationChannel.EMAIL,),
    )


async def queue_password_reset_email(
    session: AsyncSession, user: User, token: str
) -> DispatchReport:
    return await NotificationDispatcher(session).dispatch(
        user=user,
        event=NotificationEvent.ACCOUNT_SECURITY,
        title="Reset your password",
        body=(
            "Use this token to reset your password. It expires in one hour.\n\n"
            f"{token}\n\nIf you did not request this, no action is needed - your password "
            "has not changed."
        ),
        channels=(NotificationChannel.EMAIL,),
    )


async def notify_trade_closed(
    session: AsyncSession,
    user: User,
    *,
    symbol: str,
    net_pnl: float,
    reason: str,
    bot_id: str | None = None,
) -> DispatchReport:
    won = net_pnl >= 0
    event = (
        NotificationEvent.TAKE_PROFIT_HIT
        if reason == "take_profit"
        else NotificationEvent.STOP_LOSS_HIT
        if reason == "stop_loss"
        else NotificationEvent.TRADE_CLOSED
    )
    return await NotificationDispatcher(session).dispatch(
        user=user,
        event=event,
        bot_id=bot_id,
        title=f"{symbol} closed {'+' if won else ''}{net_pnl:.2f}",
        body=f"Position on {symbol} closed for {net_pnl:+.2f} ({reason}).",
        payload={"symbol": symbol, "net_pnl": net_pnl, "reason": reason},
    )


async def notify_risk_event(
    session: AsyncSession,
    user: User,
    *,
    title: str,
    body: str,
    bot_id: str | None = None,
    critical: bool = False,
) -> DispatchReport:
    return await NotificationDispatcher(session).dispatch(
        user=user,
        event=(
            NotificationEvent.DAILY_LOSS_LIMIT if critical else NotificationEvent.RISK_ALERT
        ),
        bot_id=bot_id,
        title=title,
        body=body,
    )
