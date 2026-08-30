"""SQLAlchemy ORM models.

Every user-owned table carries ``user_id`` with an index, and the repository layer refuses to
build a query without an owner scope. That is the mechanism that keeps user A out of user B's
data — it is enforced in code, not by convention.

Enum columns are stored as strings (``native_enum=False``) so that adding a new enum member is a
code change rather than a Postgres ``ALTER TYPE`` migration.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import (
    AuditAction,
    BacktestStatus,
    BotEventType,
    BotStatus,
    ExitReason,
    InstrumentType,
    KillSwitchReason,
    LicensePlan,
    LicenseStatus,
    LiquidityRole,
    MarketRegime,
    NewsEventType,
    NewsImpact,
    NewsSentiment,
    NotificationChannel,
    NotificationEvent,
    NotificationStatus,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    RiskEventType,
    RiskSeverity,
    SignalAction,
    SubscriptionStatus,
    TimeInForce,
    TradeStatus,
    UserRole,
    UserStatus,
)
from app.database.base import Base, JSONDict, TimestampMixin, UUIDMixin


def _enum(enum_cls: type, name: str) -> SAEnum:
    """String-backed enum column."""
    return SAEnum(
        enum_cls,
        name=name,
        native_enum=False,
        length=48,
        values_callable=lambda cls: [member.value for member in cls],
    )


def _utc(**kwargs: Any) -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), **kwargs)


# =========================================================================== #
# Identity
# =========================================================================== #
class User(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(200))
    role: Mapped[UserRole] = mapped_column(
        _enum(UserRole, "user_role"), default=UserRole.USER, nullable=False
    )
    status: Mapped[UserStatus] = mapped_column(
        _enum(UserStatus, "user_status"),
        default=UserStatus.PENDING_VERIFICATION,
        nullable=False,
    )
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    email_verification_token: Mapped[str | None] = mapped_column(String(255), index=True)
    password_reset_token: Mapped[str | None] = mapped_column(String(255), index=True)
    password_reset_expires_at: Mapped[datetime | None] = _utc()
    totp_secret_encrypted: Mapped[str | None] = mapped_column(Text)
    totp_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    last_login_at: Mapped[datetime | None] = _utc()
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[datetime | None] = _utc()
    settings: Mapped[dict] = mapped_column(JSONDict, default=dict)

    exchange_accounts: Mapped[list[ExchangeAccount]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )
    bots: Mapped[list[Bot]] = relationship(back_populates="user", cascade="all, delete-orphan")
    licenses: Mapped[list[License]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    @property
    def is_active(self) -> bool:
        return self.status is UserStatus.ACTIVE


class RefreshToken(UUIDMixin, TimestampMixin, Base):
    """Opaque refresh token. Only the hash is stored so a database leak is not a session leak."""

    __tablename__ = "refresh_tokens"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    expires_at: Mapped[datetime] = _utc(nullable=False)
    revoked_at: Mapped[datetime | None] = _utc()
    user_agent: Mapped[str | None] = mapped_column(String(400))
    ip_address: Mapped[str | None] = mapped_column(String(64))


# =========================================================================== #
# Licensing & subscriptions
# =========================================================================== #
class License(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "licenses"

    license_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    plan: Mapped[LicensePlan] = mapped_column(
        _enum(LicensePlan, "license_plan"), default=LicensePlan.TRIAL, nullable=False
    )
    status: Mapped[LicenseStatus] = mapped_column(
        _enum(LicenseStatus, "license_status"), default=LicenseStatus.INACTIVE, nullable=False
    )
    issued_at: Mapped[datetime | None] = _utc()
    expires_at: Mapped[datetime | None] = _utc(index=True)
    device_limit: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    meta: Mapped[dict] = mapped_column(JSONDict, default=dict)

    user: Mapped[User] = relationship(back_populates="licenses")
    devices: Mapped[list[LicenseDevice]] = relationship(
        back_populates="license", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        CheckConstraint("device_limit >= 1", name="device_limit_positive"),
    )


class LicenseDevice(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "license_devices"

    license_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("licenses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    device_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    device_name: Mapped[str | None] = mapped_column(String(200))
    platform: Mapped[str | None] = mapped_column(String(64))
    app_version: Mapped[str | None] = mapped_column(String(32))
    activated_at: Mapped[datetime | None] = _utc()
    last_seen_at: Mapped[datetime | None] = _utc()
    deactivated_at: Mapped[datetime | None] = _utc()

    license: Mapped[License] = relationship(back_populates="devices")

    __table_args__ = (
        UniqueConstraint("license_id", "device_fingerprint", name="uq_license_device"),
    )


class Subscription(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "subscriptions"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    license_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("licenses.id", ondelete="SET NULL"), index=True
    )
    plan: Mapped[LicensePlan] = mapped_column(
        _enum(LicensePlan, "subscription_plan"), default=LicensePlan.TRIAL, nullable=False
    )
    status: Mapped[SubscriptionStatus] = mapped_column(
        _enum(SubscriptionStatus, "subscription_status"),
        default=SubscriptionStatus.TRIALING,
        nullable=False,
    )
    current_period_start: Mapped[datetime | None] = _utc()
    current_period_end: Mapped[datetime | None] = _utc(index=True)
    cancel_at_period_end: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    external_customer_id: Mapped[str | None] = mapped_column(String(128), index=True)
    external_subscription_id: Mapped[str | None] = mapped_column(String(128), index=True)
    meta: Mapped[dict] = mapped_column(JSONDict, default=dict)


# =========================================================================== #
# Exchange connectivity
# =========================================================================== #
class ExchangeAccount(UUIDMixin, TimestampMixin, Base):
    """A customer's exchange API connection.

    ``api_secret_encrypted`` holds Fernet ciphertext. The plaintext secret never exists outside
    an in-memory adapter instance, is never logged, and is never returned by the API.
    """

    __tablename__ = "exchange_accounts"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    is_testnet: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    api_key_encrypted: Mapped[str | None] = mapped_column(Text)
    api_secret_encrypted: Mapped[str | None] = mapped_column(Text)
    api_key_masked: Mapped[str | None] = mapped_column(String(64))
    api_key_fingerprint: Mapped[str | None] = mapped_column(String(32), index=True)
    permissions: Mapped[dict] = mapped_column(JSONDict, default=dict)
    can_withdraw: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_validated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    validated_at: Mapped[datetime | None] = _utc()
    last_error: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    user: Mapped[User] = relationship(back_populates="exchange_accounts")

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_exchange_account_name"),
    )


# =========================================================================== #
# Strategies
# =========================================================================== #
class Strategy(UUIDMixin, TimestampMixin, Base):
    """A user's configured instance of a registered strategy type."""

    __tablename__ = "strategies"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    strategy_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    parameters: Mapped[dict] = mapped_column(JSONDict, default=dict)
    current_version: Mapped[str] = mapped_column(String(32), default="1.0.0", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    versions: Mapped[list[StrategyVersion]] = relationship(
        back_populates="strategy", cascade="all, delete-orphan"
    )

    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_strategy_name"),)


class StrategyVersion(UUIDMixin, TimestampMixin, Base):
    """Immutable snapshot of a strategy's parameters.

    A running bot pins a version. Editing a strategy creates a new version rather than silently
    changing the behaviour of a bot that is already live.
    """

    __tablename__ = "strategy_versions"

    strategy_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    parameters: Mapped[dict] = mapped_column(JSONDict, default=dict)
    changelog: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str | None] = mapped_column(String(36))

    strategy: Mapped[Strategy] = relationship(back_populates="versions")

    __table_args__ = (
        UniqueConstraint("strategy_id", "version", name="uq_strategy_version"),
    )


# =========================================================================== #
# Bots
# =========================================================================== #
class Bot(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "bots"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    strategy_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("strategies.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    strategy_version_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("strategy_versions.id", ondelete="SET NULL")
    )
    exchange_account_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("exchange_accounts.id", ondelete="SET NULL"), index=True
    )
    portfolio_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("portfolios.id", ondelete="SET NULL"), index=True
    )
    trading_mode: Mapped[str] = mapped_column(String(16), default="paper", nullable=False)
    symbols: Mapped[dict] = mapped_column(JSONDict, default=dict)
    interval: Mapped[str] = mapped_column(String(8), default="15m", nullable=False)
    status: Mapped[BotStatus] = mapped_column(
        _enum(BotStatus, "bot_status"), default=BotStatus.CREATED, nullable=False, index=True
    )
    risk_config: Mapped[dict] = mapped_column(JSONDict, default=dict)
    kill_switch_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    kill_switch_reason: Mapped[KillSwitchReason | None] = mapped_column(
        _enum(KillSwitchReason, "kill_switch_reason")
    )
    kill_switch_triggered_at: Mapped[datetime | None] = _utc()
    last_heartbeat_at: Mapped[datetime | None] = _utc()
    last_error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = _utc()
    stopped_at: Mapped[datetime | None] = _utc()
    state: Mapped[dict] = mapped_column(JSONDict, default=dict)

    user: Mapped[User] = relationship(back_populates="bots")

    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_bot_name"),)


class BotEvent(UUIDMixin, Base):
    """Append-only bot lifecycle log. Powers the activity feed and post-mortems."""

    __tablename__ = "bot_events"

    bot_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("bots.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    event_type: Mapped[BotEventType] = mapped_column(
        _enum(BotEventType, "bot_event_type"), nullable=False, index=True
    )
    message: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(String(16), default="info", nullable=False)
    payload: Mapped[dict] = mapped_column(JSONDict, default=dict)
    occurred_at: Mapped[datetime] = _utc(nullable=False, index=True)

    __table_args__ = (Index("ix_bot_events_bot_time", "bot_id", "occurred_at"),)


# =========================================================================== #
# Portfolio
# =========================================================================== #
class Portfolio(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "portfolios"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(120), default="Default", nullable=False)
    quote_currency: Mapped[str] = mapped_column(String(16), default="USDT", nullable=False)
    trading_mode: Mapped[str] = mapped_column(String(16), default="paper", nullable=False)
    starting_balance: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    cash: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    equity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    peak_equity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    fees_paid: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    last_reconciled_at: Mapped[datetime | None] = _utc()
    reconciliation_ok: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class PortfolioSnapshotRecord(UUIDMixin, Base):
    """One point on the equity curve."""

    __tablename__ = "portfolio_snapshots"

    portfolio_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    timestamp: Mapped[datetime] = _utc(nullable=False, index=True)
    cash: Mapped[float] = mapped_column(Float, nullable=False)
    equity: Mapped[float] = mapped_column(Float, nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_exposure: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    position_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    drawdown: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    __table_args__ = (
        Index("ix_portfolio_snapshots_pf_time", "portfolio_id", "timestamp"),
    )


# =========================================================================== #
# Orders, fills, positions, trades
# =========================================================================== #
class OrderRecord(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "orders"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    bot_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("bots.id", ondelete="SET NULL"), index=True
    )
    portfolio_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("portfolios.id", ondelete="SET NULL"), index=True
    )
    client_order_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    exchange_order_id: Mapped[str | None] = mapped_column(String(64), index=True)
    exchange: Mapped[str] = mapped_column(String(32), default="paper", nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    side: Mapped[OrderSide] = mapped_column(_enum(OrderSide, "order_side"), nullable=False)
    order_type: Mapped[OrderType] = mapped_column(_enum(OrderType, "order_type"), nullable=False)
    time_in_force: Mapped[TimeInForce] = mapped_column(
        _enum(TimeInForce, "time_in_force"), default=TimeInForce.GTC, nullable=False
    )
    status: Mapped[OrderStatus] = mapped_column(
        _enum(OrderStatus, "order_status"),
        default=OrderStatus.PENDING,
        nullable=False,
        index=True,
    )
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float | None] = mapped_column(Float)
    trigger_price: Mapped[float | None] = mapped_column(Float)
    filled_quantity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    average_fill_price: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    fees_paid: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    reduce_only: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    reject_reason: Mapped[str | None] = mapped_column(Text)
    submitted_at: Mapped[datetime | None] = _utc()
    closed_at: Mapped[datetime | None] = _utc()
    meta: Mapped[dict] = mapped_column(JSONDict, default=dict)

    fills: Mapped[list[OrderFill]] = relationship(
        back_populates="order", cascade="all, delete-orphan"
    )

    __table_args__ = (
        # The idempotency guarantee: one client_order_id per user, enforced by the database.
        UniqueConstraint("user_id", "client_order_id", name="uq_order_client_id"),
        Index("ix_orders_user_symbol_status", "user_id", "symbol", "status"),
        CheckConstraint("quantity > 0", name="quantity_positive"),
    )


class OrderFill(UUIDMixin, Base):
    __tablename__ = "order_fills"

    order_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    exchange_fill_id: Mapped[str | None] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[OrderSide] = mapped_column(_enum(OrderSide, "fill_side"), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    fee: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    fee_asset: Mapped[str] = mapped_column(String(16), default="USDT", nullable=False)
    role: Mapped[LiquidityRole] = mapped_column(
        _enum(LiquidityRole, "liquidity_role"), default=LiquidityRole.TAKER, nullable=False
    )
    filled_at: Mapped[datetime] = _utc(nullable=False, index=True)

    order: Mapped[OrderRecord] = relationship(back_populates="fills")


class PositionRecord(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "positions"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    bot_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("bots.id", ondelete="SET NULL"), index=True
    )
    portfolio_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("portfolios.id", ondelete="SET NULL"), index=True
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    side: Mapped[PositionSide] = mapped_column(_enum(PositionSide, "position_side"), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    mark_price: Mapped[float | None] = mapped_column(Float)
    leverage: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    stop_loss: Mapped[float | None] = mapped_column(Float)
    take_profit: Mapped[float | None] = mapped_column(Float)
    trailing_stop: Mapped[float | None] = mapped_column(Float)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    fees_paid: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    is_open: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    opened_at: Mapped[datetime] = _utc(nullable=False)
    closed_at: Mapped[datetime | None] = _utc()
    strategy_name: Mapped[str | None] = mapped_column(String(64))
    meta: Mapped[dict] = mapped_column(JSONDict, default=dict)

    __table_args__ = (
        Index("ix_positions_user_open", "user_id", "is_open"),
        CheckConstraint("quantity >= 0", name="quantity_non_negative"),
    )


class TradeRecord(UUIDMixin, TimestampMixin, Base):
    """A completed round-trip."""

    __tablename__ = "trades"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    bot_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("bots.id", ondelete="SET NULL"), index=True
    )
    portfolio_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("portfolios.id", ondelete="SET NULL"), index=True
    )
    backtest_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("backtests.id", ondelete="CASCADE"), index=True
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    side: Mapped[PositionSide] = mapped_column(_enum(PositionSide, "trade_side"), nullable=False)
    status: Mapped[TradeStatus] = mapped_column(
        _enum(TradeStatus, "trade_status"), default=TradeStatus.OPEN, nullable=False
    )
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float | None] = mapped_column(Float)
    entry_time: Mapped[datetime] = _utc(nullable=False, index=True)
    exit_time: Mapped[datetime | None] = _utc(index=True)
    gross_pnl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    net_pnl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    fees: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    slippage_cost: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    return_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    r_multiple: Mapped[float | None] = mapped_column(Float)
    stop_loss: Mapped[float | None] = mapped_column(Float)
    take_profit: Mapped[float | None] = mapped_column(Float)
    exit_reason: Mapped[ExitReason | None] = mapped_column(_enum(ExitReason, "exit_reason"))
    strategy_name: Mapped[str | None] = mapped_column(String(64), index=True)
    meta: Mapped[dict] = mapped_column(JSONDict, default=dict)

    __table_args__ = (
        Index("ix_trades_user_exit_time", "user_id", "exit_time"),
    )


# =========================================================================== #
# Signals and risk
# =========================================================================== #
class SignalRecord(UUIDMixin, Base):
    """Every decision the pipeline made, including the ones that produced no trade.

    Storing rejections is what makes it possible to answer "why didn't the bot trade?" without
    reproducing the whole market session.
    """

    __tablename__ = "signals"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    bot_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("bots.id", ondelete="CASCADE"), index=True
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    action: Mapped[SignalAction] = mapped_column(
        _enum(SignalAction, "signal_action"), nullable=False, index=True
    )
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    entry_price: Mapped[float | None] = mapped_column(Float)
    stop_loss: Mapped[float | None] = mapped_column(Float)
    take_profit: Mapped[float | None] = mapped_column(Float)
    regime: Mapped[MarketRegime | None] = mapped_column(_enum(MarketRegime, "signal_regime"))
    news_score: Mapped[float | None] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text, default="", nullable=False)
    executed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    blocked_by: Mapped[str | None] = mapped_column(String(64))
    generated_at: Mapped[datetime] = _utc(nullable=False, index=True)
    meta: Mapped[dict] = mapped_column(JSONDict, default=dict)


class RiskEventRecord(UUIDMixin, Base):
    __tablename__ = "risk_events"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    bot_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("bots.id", ondelete="CASCADE"), index=True
    )
    event_type: Mapped[RiskEventType] = mapped_column(
        _enum(RiskEventType, "risk_event_type"), nullable=False, index=True
    )
    severity: Mapped[RiskSeverity] = mapped_column(
        _enum(RiskSeverity, "risk_severity"), default=RiskSeverity.WARNING, nullable=False
    )
    symbol: Mapped[str | None] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(Text, nullable=False)
    limit_value: Mapped[float | None] = mapped_column(Float)
    observed_value: Mapped[float | None] = mapped_column(Float)
    action_taken: Mapped[str | None] = mapped_column(String(64))
    occurred_at: Mapped[datetime] = _utc(nullable=False, index=True)
    meta: Mapped[dict] = mapped_column(JSONDict, default=dict)


# =========================================================================== #
# Market data
# =========================================================================== #
class MarketCandle(Base):
    """OHLCV bar. Composite primary key makes re-ingestion naturally idempotent."""

    __tablename__ = "market_candles"

    exchange: Mapped[str] = mapped_column(String(32), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    interval: Mapped[str] = mapped_column(String(8), primary_key=True)
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    quote_volume: Mapped[float | None] = mapped_column(Float)
    trade_count: Mapped[int | None] = mapped_column(Integer)
    is_closed: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    __table_args__ = (
        Index("ix_candles_lookup", "exchange", "symbol", "interval", "open_time"),
        CheckConstraint("high >= low", name="high_ge_low"),
    )


class MarketTick(UUIDMixin, Base):
    """Latest-price sample. Retained for a short window; not the historical source of truth."""

    __tablename__ = "market_ticks"

    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    bid: Mapped[float | None] = mapped_column(Float)
    ask: Mapped[float | None] = mapped_column(Float)
    bid_size: Mapped[float | None] = mapped_column(Float)
    ask_size: Mapped[float | None] = mapped_column(Float)
    volume_24h: Mapped[float | None] = mapped_column(Float)
    funding_rate: Mapped[float | None] = mapped_column(Float)
    open_interest: Mapped[float | None] = mapped_column(Float)
    timestamp: Mapped[datetime] = _utc(nullable=False, index=True)

    __table_args__ = (Index("ix_ticks_symbol_time", "symbol", "timestamp"),)


class MarketRegimeRecord(UUIDMixin, Base):
    __tablename__ = "market_regimes"

    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    interval: Mapped[str] = mapped_column(String(8), nullable=False)
    regime: Mapped[MarketRegime] = mapped_column(
        _enum(MarketRegime, "market_regime"), nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    adx: Mapped[float | None] = mapped_column(Float)
    atr_pct: Mapped[float | None] = mapped_column(Float)
    trend_slope: Mapped[float | None] = mapped_column(Float)
    detected_at: Mapped[datetime] = _utc(nullable=False, index=True)
    meta: Mapped[dict] = mapped_column(JSONDict, default=dict)

    __table_args__ = (
        Index("ix_regimes_symbol_time", "symbol", "interval", "detected_at"),
    )


class InstrumentRecord(UUIDMixin, TimestampMixin, Base):
    """Cached venue instrument metadata (tick size, lot size, fees)."""

    __tablename__ = "instruments"

    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    base_asset: Mapped[str] = mapped_column(String(16), nullable=False)
    quote_asset: Mapped[str] = mapped_column(String(16), nullable=False)
    instrument_type: Mapped[InstrumentType] = mapped_column(
        _enum(InstrumentType, "instrument_type"), default=InstrumentType.SPOT, nullable=False
    )
    tick_size: Mapped[float] = mapped_column(Float, nullable=False)
    lot_size: Mapped[float] = mapped_column(Float, nullable=False)
    min_quantity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    min_notional: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    max_leverage: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    maker_fee: Mapped[float] = mapped_column(Float, default=0.0002, nullable=False)
    taker_fee: Mapped[float] = mapped_column(Float, default=0.00055, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    __table_args__ = (UniqueConstraint("exchange", "symbol", name="uq_instrument"),)


# =========================================================================== #
# News
# =========================================================================== #
class NewsArticle(UUIDMixin, Base):
    __tablename__ = "news_articles"

    external_id: Mapped[str | None] = mapped_column(String(128), index=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    source_reputation: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    url: Mapped[str | None] = mapped_column(String(1024))
    published_at: Mapped[datetime] = _utc(nullable=False, index=True)
    ingested_at: Mapped[datetime] = _utc(nullable=False)
    assets: Mapped[dict] = mapped_column(JSONDict, default=dict)
    is_duplicate: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    duplicate_of: Mapped[str | None] = mapped_column(String(36), index=True)
    raw_metadata: Mapped[dict] = mapped_column(JSONDict, default=dict)

    analyses: Mapped[list[NewsAnalysis]] = relationship(
        back_populates="article", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("content_hash", name="uq_news_content_hash"),
    )


class NewsAnalysis(UUIDMixin, Base):
    __tablename__ = "news_analyses"

    article_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("news_articles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    asset: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    event_type: Mapped[NewsEventType] = mapped_column(
        _enum(NewsEventType, "news_event_type"), default=NewsEventType.OTHER, nullable=False
    )
    sentiment: Mapped[NewsSentiment] = mapped_column(
        _enum(NewsSentiment, "news_sentiment"), default=NewsSentiment.NEUTRAL, nullable=False
    )
    impact: Mapped[NewsImpact] = mapped_column(
        _enum(NewsImpact, "news_impact"), default=NewsImpact.NONE, nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    novelty: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    urgency: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    rationale: Mapped[str | None] = mapped_column(Text)
    analyzed_at: Mapped[datetime] = _utc(nullable=False, index=True)

    article: Mapped[NewsArticle] = relationship(back_populates="analyses")

    __table_args__ = (
        UniqueConstraint("article_id", "asset", name="uq_news_analysis_asset"),
    )


# =========================================================================== #
# Backtests
# =========================================================================== #
class Backtest(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "backtests"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    strategy_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("strategies.id", ondelete="SET NULL"), index=True
    )
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    strategy_type: Mapped[str] = mapped_column(String(64), nullable=False)
    parameters: Mapped[dict] = mapped_column(JSONDict, default=dict)
    risk_config: Mapped[dict] = mapped_column(JSONDict, default=dict)
    symbols: Mapped[dict] = mapped_column(JSONDict, default=dict)
    interval: Mapped[str] = mapped_column(String(8), nullable=False)
    start_date: Mapped[datetime] = _utc(nullable=False)
    end_date: Mapped[datetime] = _utc(nullable=False)
    initial_balance: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[BacktestStatus] = mapped_column(
        _enum(BacktestStatus, "backtest_status"),
        default=BacktestStatus.PENDING,
        nullable=False,
        index=True,
    )
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict] = mapped_column(JSONDict, default=dict)
    equity_curve: Mapped[dict] = mapped_column(JSONDict, default=dict)
    started_at: Mapped[datetime | None] = _utc()
    completed_at: Mapped[datetime | None] = _utc()


# =========================================================================== #
# Notifications & audit
# =========================================================================== #
class Notification(UUIDMixin, Base):
    __tablename__ = "notifications"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    bot_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("bots.id", ondelete="CASCADE"), index=True
    )
    event: Mapped[NotificationEvent] = mapped_column(
        _enum(NotificationEvent, "notification_event"), nullable=False, index=True
    )
    channel: Mapped[NotificationChannel] = mapped_column(
        _enum(NotificationChannel, "notification_channel"), nullable=False
    )
    status: Mapped[NotificationStatus] = mapped_column(
        _enum(NotificationStatus, "notification_status"),
        default=NotificationStatus.PENDING,
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONDict, default=dict)
    read_at: Mapped[datetime | None] = _utc()
    sent_at: Mapped[datetime | None] = _utc()
    error: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = _utc(nullable=False, index=True)


class AuditLog(UUIDMixin, Base):
    """Append-only security log. Never updated, never deleted by application code."""

    __tablename__ = "audit_logs"

    user_id: Mapped[str | None] = mapped_column(String(36), index=True)
    action: Mapped[AuditAction] = mapped_column(
        _enum(AuditAction, "audit_action"), nullable=False, index=True
    )
    resource_type: Mapped[str | None] = mapped_column(String(64))
    resource_id: Mapped[str | None] = mapped_column(String(36), index=True)
    success: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    ip_address: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(String(400))
    detail: Mapped[dict] = mapped_column(JSONDict, default=dict)
    occurred_at: Mapped[datetime] = _utc(nullable=False, index=True)

    __table_args__ = (Index("ix_audit_user_time", "user_id", "occurred_at"),)
