"""Repositories.

The multi-tenancy guarantee lives here. :class:`OwnedRepository` **cannot** build a query
without an owner scope: every read and write goes through :meth:`OwnedRepository._scoped`,
which unconditionally adds ``WHERE user_id = :owner``. There is no method that returns rows
across users, so "user A sees user B's trades" cannot be caused by a forgotten filter at a call
site — the filter is not the caller's responsibility.

Admin-wide queries, where they exist at all, live on separate explicitly-named methods.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Generic, TypeVar

from sqlalchemy import Select, case, delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import utcnow
from app.core.exceptions import ConflictError, NotFoundError
from app.core.logging import get_logger
from app.database.base import Base
from app.database.models import (
    AuditLog,
    Backtest,
    Bot,
    BotEvent,
    ExchangeAccount,
    License,
    Notification,
    OrderRecord,
    PortfolioSnapshotRecord,
    PositionRecord,
    RefreshToken,
    RiskEventRecord,
    SignalRecord,
    Strategy,
    Subscription,
    TradeRecord,
    User,
)

logger = get_logger(__name__)

ModelT = TypeVar("ModelT", bound=Base)


class Repository(Generic[ModelT]):  # noqa: UP046 - explicit TypeVar keeps 3.12 compatibility
    """Base repository for models without an owner (reference data, audit log)."""

    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, instance: ModelT) -> ModelT:
        self.session.add(instance)
        try:
            await self.session.flush()
        except IntegrityError as exc:
            await self.session.rollback()
            raise ConflictError(
                f"{self.model.__name__} violates a uniqueness constraint",
                context={"model": self.model.__name__},
            ) from exc
        return instance

    async def get(self, record_id: str) -> ModelT | None:
        return await self.session.get(self.model, record_id)

    async def require(self, record_id: str) -> ModelT:
        instance = await self.get(record_id)
        if instance is None:
            raise NotFoundError(f"{self.model.__name__} {record_id} was not found")
        return instance

    async def delete(self, instance: ModelT) -> None:
        await self.session.delete(instance)
        await self.session.flush()

    async def count(self) -> int:
        result = await self.session.execute(
            select(func.count()).select_from(self.model)
        )
        return int(result.scalar_one())


class OwnedRepository(Repository[ModelT]):
    """Repository for user-owned models. Every query is owner-scoped, without exception."""

    #: Name of the ownership column on the model.
    owner_column: str = "user_id"

    def _scoped(self, owner_id: str) -> Select[tuple[ModelT]]:
        """The only way to start a query in this class."""
        if not owner_id:
            raise ValueError(
                f"{type(self).__name__} requires an owner id; refusing to build an "
                "unscoped query"
            )
        column = getattr(self.model, self.owner_column)
        return select(self.model).where(column == owner_id)

    async def get_for_owner(self, record_id: str, owner_id: str) -> ModelT | None:
        result = await self.session.execute(
            self._scoped(owner_id).where(self.model.id == record_id)  # type: ignore[attr-defined]
        )
        return result.scalar_one_or_none()

    async def require_for_owner(self, record_id: str, owner_id: str) -> ModelT:
        instance = await self.get_for_owner(record_id, owner_id)
        if instance is None:
            # Deliberately identical to "does not exist": a 403 here would confirm that
            # someone else's record with this id is real.
            raise NotFoundError(f"{self.model.__name__} {record_id} was not found")
        return instance

    async def list_for_owner(
        self,
        owner_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        order_by: Any | None = None,
    ) -> list[ModelT]:
        query = self._scoped(owner_id)
        if order_by is not None:
            query = query.order_by(order_by)
        result = await self.session.execute(query.limit(limit).offset(offset))
        return list(result.scalars().all())

    async def count_for_owner(self, owner_id: str) -> int:
        column = getattr(self.model, self.owner_column)
        result = await self.session.execute(
            select(func.count()).select_from(self.model).where(column == owner_id)
        )
        return int(result.scalar_one())

    async def delete_for_owner(self, record_id: str, owner_id: str) -> None:
        instance = await self.require_for_owner(record_id, owner_id)
        await self.delete(instance)


# =========================================================================== #
# Identity
# =========================================================================== #
class UserRepository(Repository[User]):
    model = User

    async def get_by_email(self, email: str) -> User | None:
        result = await self.session.execute(
            select(User).where(User.email == email.strip().lower())
        )
        return result.scalar_one_or_none()

    async def email_exists(self, email: str) -> bool:
        return await self.get_by_email(email) is not None

    async def get_by_verification_token(self, token: str) -> User | None:
        result = await self.session.execute(
            select(User).where(User.email_verification_token == token)
        )
        return result.scalar_one_or_none()

    async def get_by_reset_token(self, token: str) -> User | None:
        result = await self.session.execute(
            select(User).where(User.password_reset_token == token)
        )
        return result.scalar_one_or_none()


class RefreshTokenRepository(Repository[RefreshToken]):
    model = RefreshToken

    async def get_by_hash(self, token_hash: str) -> RefreshToken | None:
        result = await self.session.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        )
        return result.scalar_one_or_none()

    async def revoke_all_for_user(self, user_id: str, *, now: datetime | None = None) -> int:
        """Revoke every session. Used on password change and on suspicious activity."""
        moment = now or utcnow()
        result = await self.session.execute(
            select(RefreshToken).where(
                RefreshToken.user_id == user_id,
                RefreshToken.revoked_at.is_(None),
            )
        )
        tokens = list(result.scalars().all())
        for token in tokens:
            token.revoked_at = moment
        await self.session.flush()
        return len(tokens)

    async def purge_expired(self, *, now: datetime | None = None) -> int:
        moment = now or utcnow()
        result = await self.session.execute(
            delete(RefreshToken).where(RefreshToken.expires_at < moment)
        )
        return int(getattr(result, "rowcount", 0) or 0)


# =========================================================================== #
# Trading entities
# =========================================================================== #
class ExchangeAccountRepository(OwnedRepository[ExchangeAccount]):
    model = ExchangeAccount

    async def get_by_name(self, owner_id: str, name: str) -> ExchangeAccount | None:
        result = await self.session.execute(
            self._scoped(owner_id).where(ExchangeAccount.name == name)
        )
        return result.scalar_one_or_none()

    async def active_for_owner(self, owner_id: str) -> list[ExchangeAccount]:
        result = await self.session.execute(
            self._scoped(owner_id).where(ExchangeAccount.is_active.is_(True))
        )
        return list(result.scalars().all())


class StrategyRepository(OwnedRepository[Strategy]):
    model = Strategy

    async def get_by_name(self, owner_id: str, name: str) -> Strategy | None:
        result = await self.session.execute(
            self._scoped(owner_id).where(Strategy.name == name)
        )
        return result.scalar_one_or_none()


class BotRepository(OwnedRepository[Bot]):
    model = Bot

    async def get_by_name(self, owner_id: str, name: str) -> Bot | None:
        result = await self.session.execute(self._scoped(owner_id).where(Bot.name == name))
        return result.scalar_one_or_none()

    async def running_for_owner(self, owner_id: str) -> list[Bot]:
        from app.core.enums import BotStatus

        result = await self.session.execute(
            self._scoped(owner_id).where(
                Bot.status.in_([BotStatus.RUNNING, BotStatus.PAUSED])
            )
        )
        return list(result.scalars().all())


class BotEventRepository(OwnedRepository[BotEvent]):
    model = BotEvent

    async def recent_for_bot(
        self, owner_id: str, bot_id: str, *, limit: int = 100
    ) -> list[BotEvent]:
        result = await self.session.execute(
            self._scoped(owner_id)
            .where(BotEvent.bot_id == bot_id)
            .order_by(BotEvent.occurred_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


class OrderRepository(OwnedRepository[OrderRecord]):
    model = OrderRecord

    async def get_by_client_id(
        self, owner_id: str, client_order_id: str
    ) -> OrderRecord | None:
        """Look up by idempotency key. The database enforces uniqueness per user."""
        result = await self.session.execute(
            self._scoped(owner_id).where(OrderRecord.client_order_id == client_order_id)
        )
        return result.scalar_one_or_none()

    async def open_for_owner(
        self, owner_id: str, symbol: str | None = None
    ) -> list[OrderRecord]:
        from app.core.enums import OrderStatus

        query = self._scoped(owner_id).where(
            OrderRecord.status.in_(
                [
                    OrderStatus.PENDING,
                    OrderStatus.SUBMITTED,
                    OrderStatus.OPEN,
                    OrderStatus.PARTIALLY_FILLED,
                ]
            )
        )
        if symbol:
            query = query.where(OrderRecord.symbol == symbol)
        result = await self.session.execute(query.order_by(OrderRecord.created_at.desc()))
        return list(result.scalars().all())

    async def recent_for_owner(
        self, owner_id: str, *, limit: int = 100, bot_id: str | None = None
    ) -> list[OrderRecord]:
        query = self._scoped(owner_id)
        if bot_id:
            query = query.where(OrderRecord.bot_id == bot_id)
        result = await self.session.execute(
            query.order_by(OrderRecord.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())


class PositionRepository(OwnedRepository[PositionRecord]):
    model = PositionRecord

    async def open_for_owner(self, owner_id: str) -> list[PositionRecord]:
        result = await self.session.execute(
            self._scoped(owner_id).where(PositionRecord.is_open.is_(True))
        )
        return list(result.scalars().all())

    async def get_open(self, owner_id: str, symbol: str) -> PositionRecord | None:
        result = await self.session.execute(
            self._scoped(owner_id).where(
                PositionRecord.symbol == symbol, PositionRecord.is_open.is_(True)
            )
        )
        return result.scalar_one_or_none()


class TradeRepository(OwnedRepository[TradeRecord]):
    model = TradeRecord

    async def closed_for_owner(
        self,
        owner_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        bot_id: str | None = None,
        symbol: str | None = None,
        since: datetime | None = None,
    ) -> list[TradeRecord]:
        query = self._scoped(owner_id).where(TradeRecord.exit_time.is_not(None))
        if bot_id:
            query = query.where(TradeRecord.bot_id == bot_id)
        if symbol:
            query = query.where(TradeRecord.symbol == symbol)
        if since:
            query = query.where(TradeRecord.exit_time >= since)
        result = await self.session.execute(
            query.order_by(TradeRecord.exit_time.desc()).limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def performance_summary(self, owner_id: str) -> dict[str, float | int]:
        """Aggregate closed-trade statistics, computed in the database."""
        result = await self.session.execute(
            select(
                func.count(TradeRecord.id),
                func.coalesce(func.sum(TradeRecord.net_pnl), 0.0),
                func.coalesce(func.sum(TradeRecord.fees), 0.0),
                # `case`, not `func.case`: the latter emits a SQL function literally
                # named "case", which no database has.
                func.coalesce(
                    func.sum(case((TradeRecord.net_pnl > 0, 1), else_=0)),
                    0,
                ),
            ).where(
                TradeRecord.user_id == owner_id, TradeRecord.exit_time.is_not(None)
            )
        )
        total, net_pnl, fees, wins = result.one()
        total = int(total or 0)
        wins = int(wins or 0)
        return {
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": (wins / total) if total else 0.0,
            "net_pnl": float(net_pnl or 0.0),
            "fees": float(fees or 0.0),
        }


class SignalRepository(OwnedRepository[SignalRecord]):
    model = SignalRecord

    async def recent_for_owner(
        self, owner_id: str, *, limit: int = 100, bot_id: str | None = None
    ) -> list[SignalRecord]:
        query = self._scoped(owner_id)
        if bot_id:
            query = query.where(SignalRecord.bot_id == bot_id)
        result = await self.session.execute(
            query.order_by(SignalRecord.generated_at.desc()).limit(limit)
        )
        return list(result.scalars().all())


class RiskEventRepository(OwnedRepository[RiskEventRecord]):
    model = RiskEventRecord

    async def recent_for_owner(
        self, owner_id: str, *, limit: int = 100
    ) -> list[RiskEventRecord]:
        result = await self.session.execute(
            self._scoped(owner_id)
            .order_by(RiskEventRecord.occurred_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


class PortfolioSnapshotRepository(OwnedRepository[PortfolioSnapshotRecord]):
    model = PortfolioSnapshotRecord

    async def curve_for_owner(
        self,
        owner_id: str,
        *,
        portfolio_id: str | None = None,
        since: datetime | None = None,
        limit: int = 2000,
    ) -> list[PortfolioSnapshotRecord]:
        query = self._scoped(owner_id)
        if portfolio_id:
            query = query.where(
                PortfolioSnapshotRecord.portfolio_id == portfolio_id
            )
        if since:
            query = query.where(PortfolioSnapshotRecord.timestamp >= since)
        result = await self.session.execute(
            query.order_by(PortfolioSnapshotRecord.timestamp.asc()).limit(limit)
        )
        return list(result.scalars().all())


class BacktestRepository(OwnedRepository[Backtest]):
    model = Backtest

    async def recent_for_owner(self, owner_id: str, *, limit: int = 50) -> list[Backtest]:
        result = await self.session.execute(
            self._scoped(owner_id).order_by(Backtest.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())


# =========================================================================== #
# Licensing and notifications
# =========================================================================== #
class LicenseRepository(OwnedRepository[License]):
    model = License

    async def get_by_key(self, license_key: str) -> License | None:
        """Look up by key without an owner scope.

        Legitimate because activation happens *before* the caller is known to own it, and the
        key itself is the credential. Named distinctly so it cannot be mistaken for a scoped
        read.
        """
        result = await self.session.execute(
            select(License).where(License.license_key == license_key)
        )
        return result.scalar_one_or_none()

    async def active_for_owner(self, owner_id: str) -> License | None:
        from app.core.enums import LicenseStatus

        result = await self.session.execute(
            self._scoped(owner_id).where(License.status == LicenseStatus.ACTIVE)
        )
        return result.scalars().first()


class SubscriptionRepository(OwnedRepository[Subscription]):
    model = Subscription

    async def current_for_owner(self, owner_id: str) -> Subscription | None:
        from app.core.enums import SubscriptionStatus

        result = await self.session.execute(
            self._scoped(owner_id)
            .where(
                Subscription.status.in_(
                    [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]
                )
            )
            .order_by(Subscription.created_at.desc())
        )
        return result.scalars().first()


class NotificationRepository(OwnedRepository[Notification]):
    model = Notification

    async def unread_for_owner(
        self, owner_id: str, *, limit: int = 50
    ) -> list[Notification]:
        result = await self.session.execute(
            self._scoped(owner_id)
            .where(Notification.read_at.is_(None))
            .order_by(Notification.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def pending(self, *, limit: int = 100) -> list[Notification]:
        """Undelivered notifications across all users.

        Unscoped by necessity: the delivery worker is not acting on behalf of a user. It is a
        background job, never reachable from a request handler.
        """
        from app.core.enums import NotificationStatus

        result = await self.session.execute(
            select(Notification)
            .where(Notification.status == NotificationStatus.PENDING)
            .order_by(Notification.created_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def mark_read(self, owner_id: str, notification_ids: Sequence[str]) -> int:
        if not notification_ids:
            return 0
        result = await self.session.execute(
            self._scoped(owner_id).where(Notification.id.in_(notification_ids))
        )
        items = list(result.scalars().all())
        now = utcnow()
        for item in items:
            item.read_at = now
        await self.session.flush()
        return len(items)


class AuditLogRepository(Repository[AuditLog]):
    """Append-only. There is no update or delete method, by design."""

    model = AuditLog

    async def record(
        self,
        *,
        action: Any,
        user_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        success: bool = True,
        ip_address: str | None = None,
        user_agent: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> AuditLog:
        entry = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            detail=_redact(detail or {}),
            occurred_at=utcnow(),
        )
        return await self.add(entry)

    async def for_user(self, user_id: str, *, limit: int = 100) -> list[AuditLog]:
        result = await self.session.execute(
            select(AuditLog)
            .where(AuditLog.user_id == user_id)
            .order_by(AuditLog.occurred_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


def _redact(detail: dict[str, Any]) -> dict[str, Any]:
    """Strip secrets before they reach the audit table.

    The audit log is read by support staff and exported for compliance, so it is one of the
    likeliest places for a credential to end up somewhere it should not be.
    """
    from app.core.logging import is_sensitive_key

    return {
        key: ("***REDACTED***" if is_sensitive_key(str(key)) else value)
        for key, value in detail.items()
    }
