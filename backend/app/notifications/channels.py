"""Notification channels.

Every channel implements the same interface, so a deployment can enable any subset without the
sending code branching. The default is in-app only: a fresh install must not silently fail to
notify, so a channel that is not configured reports itself as unavailable rather than throwing
at send time.

BLOCKED BY EXTERNAL DEPENDENCY: Email, Telegram and Discord require credentials
(SMTP, bot token, webhook URL). Each is implemented and tested against a mock transport;
without credentials they report ``is_configured = False`` and are skipped.
"""

from __future__ import annotations

import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.core.enums import NotificationChannel, NotificationEvent
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class NotificationMessage:
    """One notification, rendered for delivery."""

    event: NotificationEvent
    title: str
    body: str
    severity: str = "info"
    payload: dict[str, Any] | None = None

    @property
    def is_urgent(self) -> bool:
        return self.severity in {"warning", "critical"}


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    channel: NotificationChannel
    delivered: bool
    detail: str = ""


class NotificationChannelBase(ABC):
    """Base for delivery channels."""

    channel: NotificationChannel

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """True when this channel has everything it needs to deliver."""

    @abstractmethod
    async def send(self, message: NotificationMessage, *, recipient: str) -> DeliveryResult:
        """Attempt delivery. Must never raise; failures are returned, not thrown.

        A notification channel failing must not be able to take down a trading loop, and the
        dispatcher needs to record the failure rather than lose the event.
        """


class InAppChannel(NotificationChannelBase):
    """Always available. Persistence is handled by the dispatcher."""

    channel = NotificationChannel.IN_APP

    @property
    def is_configured(self) -> bool:
        return True

    async def send(self, message: NotificationMessage, *, recipient: str) -> DeliveryResult:
        return DeliveryResult(self.channel, True, "stored for in-app display")


class EmailChannel(NotificationChannelBase):
    """SMTP email.

    Uses blocking ``smtplib`` inside a thread rather than an async SMTP client: the dependency
    surface is smaller, and notification volume is low enough that a thread per message is not
    a concern.
    """

    channel = NotificationChannel.EMAIL

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.smtp_host and self.settings.smtp_from)

    async def send(self, message: NotificationMessage, *, recipient: str) -> DeliveryResult:
        if not self.is_configured:
            return DeliveryResult(self.channel, False, "SMTP is not configured")
        import asyncio

        try:
            await asyncio.to_thread(self._send_sync, message, recipient)
        except (smtplib.SMTPException, OSError) as exc:
            logger.warning("notifications.email_failed", error=type(exc).__name__)
            return DeliveryResult(self.channel, False, f"SMTP error: {type(exc).__name__}")
        return DeliveryResult(self.channel, True, "sent")

    def _send_sync(self, message: NotificationMessage, recipient: str) -> None:
        email = EmailMessage()
        email["Subject"] = message.title
        email["From"] = self.settings.smtp_from
        email["To"] = recipient
        email.set_content(message.body)

        with smtplib.SMTP(
            self.settings.smtp_host or "", self.settings.smtp_port, timeout=15
        ) as server:
            server.starttls()
            if self.settings.smtp_username and self.settings.smtp_password:
                server.login(
                    self.settings.smtp_username,
                    self.settings.smtp_password.get_secret_value(),
                )
            server.send_message(email)


class TelegramChannel(NotificationChannelBase):
    """Telegram bot messages."""

    channel = NotificationChannel.TELEGRAM

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.telegram_bot_token)

    async def send(self, message: NotificationMessage, *, recipient: str) -> DeliveryResult:
        if not self.is_configured:
            return DeliveryResult(self.channel, False, "Telegram bot token is not configured")
        token = self.settings.telegram_bot_token
        assert token is not None
        chat_id = recipient or self.settings.telegram_chat_id
        if not chat_id:
            return DeliveryResult(self.channel, False, "no Telegram chat id configured")

        client = self._client or httpx.AsyncClient(timeout=10.0)
        try:
            response = await client.post(
                f"https://api.telegram.org/bot{token.get_secret_value()}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": f"*{message.title}*\n{message.body}",
                    "parse_mode": "Markdown",
                },
            )
            if response.status_code >= 400:
                # Never echo the body: Telegram reflects the bot token in some error paths.
                return DeliveryResult(
                    self.channel, False, f"Telegram returned HTTP {response.status_code}"
                )
        except httpx.HTTPError as exc:
            return DeliveryResult(self.channel, False, f"HTTP error: {type(exc).__name__}")
        finally:
            if self._client is None:
                await client.aclose()
        return DeliveryResult(self.channel, True, "sent")


class DiscordChannel(NotificationChannelBase):
    """Discord webhook messages."""

    channel = NotificationChannel.DISCORD

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.discord_webhook_url)

    async def send(self, message: NotificationMessage, *, recipient: str) -> DeliveryResult:
        webhook = self.settings.discord_webhook_url
        if webhook is None:
            return DeliveryResult(self.channel, False, "Discord webhook is not configured")

        colour = {"info": 0x3B82F6, "warning": 0xF59E0B, "critical": 0xEF4444}.get(
            message.severity, 0x6B7280
        )
        client = self._client or httpx.AsyncClient(timeout=10.0)
        try:
            response = await client.post(
                webhook.get_secret_value(),
                json={
                    "embeds": [
                        {
                            "title": message.title[:256],
                            "description": message.body[:4000],
                            "color": colour,
                        }
                    ]
                },
            )
            if response.status_code >= 400:
                return DeliveryResult(
                    self.channel, False, f"Discord returned HTTP {response.status_code}"
                )
        except httpx.HTTPError as exc:
            return DeliveryResult(self.channel, False, f"HTTP error: {type(exc).__name__}")
        finally:
            if self._client is None:
                await client.aclose()
        return DeliveryResult(self.channel, True, "sent")


class WebPushChannel(NotificationChannelBase):
    """Browser push notifications.

    BLOCKED BY EXTERNAL DEPENDENCY: requires VAPID keys and a subscription store. The channel
    is present so the dispatcher's shape is complete and reports itself unconfigured rather
    than pretending to deliver.
    """

    channel = NotificationChannel.WEB_PUSH

    @property
    def is_configured(self) -> bool:
        return False

    async def send(self, message: NotificationMessage, *, recipient: str) -> DeliveryResult:
        return DeliveryResult(
            self.channel,
            False,
            "web push requires VAPID keys and a subscription store, which are not configured",
        )


def build_channels(
    settings: Settings | None = None,
) -> dict[NotificationChannel, NotificationChannelBase]:
    """Every channel, configured or not.

    Unconfigured channels are included deliberately: the dispatcher records "skipped, not
    configured" rather than silently dropping the event, so the reason is visible in the UI.
    """
    resolved = settings or get_settings()
    return {
        NotificationChannel.IN_APP: InAppChannel(),
        NotificationChannel.EMAIL: EmailChannel(resolved),
        NotificationChannel.TELEGRAM: TelegramChannel(resolved),
        NotificationChannel.DISCORD: DiscordChannel(resolved),
        NotificationChannel.WEB_PUSH: WebPushChannel(),
    }
