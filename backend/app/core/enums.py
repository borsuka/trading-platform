"""Domain enumerations.

These are pure Python enums with no ORM or framework dependency, so the strategy, risk and
execution layers can be unit-tested without a database.
"""

from __future__ import annotations

import enum


class StrEnum(enum.StrEnum):
    """Base for the platform's string enums.

    ``enum.StrEnum`` members are real ``str`` instances, so they serialise directly to JSON,
    compare equal to their values, and round-trip through the database without a converter.
    """


# --------------------------------------------------------------------------- #
# Instruments and market structure
# --------------------------------------------------------------------------- #
class InstrumentType(StrEnum):
    SPOT = "spot"
    LINEAR_PERPETUAL = "linear_perpetual"
    INVERSE_PERPETUAL = "inverse_perpetual"
    FUTURES = "futures"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> OrderSide:
        return OrderSide.SELL if self is OrderSide.BUY else OrderSide.BUY

    @property
    def sign(self) -> int:
        """+1 for buy, -1 for sell. Used in PnL and exposure maths."""
        return 1 if self is OrderSide.BUY else -1


class PositionSide(StrEnum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

    @property
    def sign(self) -> int:
        if self is PositionSide.LONG:
            return 1
        if self is PositionSide.SHORT:
            return -1
        return 0

    @classmethod
    def from_side(cls, side: OrderSide) -> PositionSide:
        return cls.LONG if side is OrderSide.BUY else cls.SHORT

    @property
    def closing_side(self) -> OrderSide:
        if self is PositionSide.LONG:
            return OrderSide.SELL
        if self is PositionSide.SHORT:
            return OrderSide.BUY
        raise ValueError("A flat position has no closing side")


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    TRAILING_STOP = "trailing_stop"

    @property
    def is_conditional(self) -> bool:
        return self in {
            OrderType.STOP_MARKET,
            OrderType.STOP_LIMIT,
            OrderType.TAKE_PROFIT_MARKET,
            OrderType.TAKE_PROFIT_LIMIT,
            OrderType.TRAILING_STOP,
        }

    @property
    def requires_limit_price(self) -> bool:
        return self in {OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT}

    @property
    def requires_trigger_price(self) -> bool:
        return self in {
            OrderType.STOP_MARKET,
            OrderType.STOP_LIMIT,
            OrderType.TAKE_PROFIT_MARKET,
            OrderType.TAKE_PROFIT_LIMIT,
        }


class TimeInForce(StrEnum):
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    POST_ONLY = "post_only"


class OrderStatus(StrEnum):
    PENDING = "pending"           # created locally, not yet acknowledged
    SUBMITTED = "submitted"       # acknowledged by the venue
    OPEN = "open"                 # resting on the book
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNKNOWN = "unknown"           # state query needed before any further action

    @property
    def is_terminal(self) -> bool:
        return self in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }

    @property
    def is_active(self) -> bool:
        return self in {
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED,
        }


class LiquidityRole(StrEnum):
    MAKER = "maker"
    TAKER = "taker"


# --------------------------------------------------------------------------- #
# Decisions
# --------------------------------------------------------------------------- #
class SignalAction(StrEnum):
    """What the decision pipeline concluded."""

    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    HOLD = "hold"
    NO_TRADE = "no_trade"

    @property
    def is_entry(self) -> bool:
        return self in {SignalAction.BUY, SignalAction.SELL}

    @property
    def order_side(self) -> OrderSide:
        if self is SignalAction.BUY:
            return OrderSide.BUY
        if self is SignalAction.SELL:
            return OrderSide.SELL
        raise ValueError(f"{self} does not map to an order side")


class MarketRegime(StrEnum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"

    @property
    def is_trending(self) -> bool:
        return self in {MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR}

    @property
    def allows_trading(self) -> bool:
        """UNKNOWN regime is a hard no-trade condition."""
        return self is not MarketRegime.UNKNOWN


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #
class RiskDecision(StrEnum):
    APPROVED = "approved"
    REDUCED = "reduced"      # approved at a smaller size than requested
    REJECTED = "rejected"


class RiskEventType(StrEnum):
    LIMIT_BREACH = "limit_breach"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    WEEKLY_LOSS_LIMIT = "weekly_loss_limit"
    MAX_DRAWDOWN = "max_drawdown"
    EXPOSURE_LIMIT = "exposure_limit"
    POSITION_LIMIT = "position_limit"
    LEVERAGE_LIMIT = "leverage_limit"
    LOSS_STREAK = "loss_streak"
    COOLDOWN = "cooldown"
    LIQUIDITY = "liquidity"
    SPREAD = "spread"
    SLIPPAGE = "slippage"
    STALE_DATA = "stale_data"
    CLOCK_DRIFT = "clock_drift"
    RECONCILIATION = "reconciliation"
    KILL_SWITCH = "kill_switch"
    SIZING_FAILED = "sizing_failed"
    TRADE_COUNT_LIMIT = "trade_count_limit"


class RiskSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class KillSwitchReason(StrEnum):
    MANUAL = "manual"
    DAILY_LOSS = "daily_loss"
    WEEKLY_LOSS = "weekly_loss"
    MAX_DRAWDOWN = "max_drawdown"
    STATE_CORRUPTION = "state_corruption"
    EXCHANGE_DESYNC = "exchange_desync"
    STALE_MARKET_DATA = "stale_market_data"
    API_FAILURES = "api_failures"
    ABNORMAL_VOLATILITY = "abnormal_volatility"
    CLOCK_DRIFT = "clock_drift"
    LICENSE_INVALID = "license_invalid"


# --------------------------------------------------------------------------- #
# Bots
# --------------------------------------------------------------------------- #
class BotStatus(StrEnum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    HALTED = "halted"   # kill switch / reconciliation failure

    @property
    def is_operational(self) -> bool:
        return self in {BotStatus.RUNNING, BotStatus.PAUSED}


class BotEventType(StrEnum):
    STARTED = "started"
    STOPPED = "stopped"
    PAUSED = "paused"
    RESUMED = "resumed"
    EMERGENCY_STOP = "emergency_stop"
    SIGNAL_GENERATED = "signal_generated"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_REJECTED = "order_rejected"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    RISK_BLOCKED = "risk_blocked"
    KILL_SWITCH_TRIPPED = "kill_switch_tripped"
    RECONCILIATION_MISMATCH = "reconciliation_mismatch"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


# --------------------------------------------------------------------------- #
# News
# --------------------------------------------------------------------------- #
class NewsEventType(StrEnum):
    REGULATION = "regulation"
    ETF = "etf"
    MACRO = "macro"
    INTEREST_RATES = "interest_rates"
    INFLATION = "inflation"
    EARNINGS = "earnings"
    EXCHANGE_INCIDENT = "exchange_incident"
    HACK = "hack"
    TOKEN_UNLOCK = "token_unlock"
    PARTNERSHIP = "partnership"
    ACQUISITION = "acquisition"
    LEADERSHIP_CHANGE = "leadership_change"
    LEGAL = "legal"
    GEOPOLITICAL = "geopolitical"
    OTHER = "other"


class NewsSentiment(StrEnum):
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"

    @property
    def score(self) -> float:
        """Map to [-1, 1]."""
        return {
            NewsSentiment.VERY_NEGATIVE: -1.0,
            NewsSentiment.NEGATIVE: -0.5,
            NewsSentiment.NEUTRAL: 0.0,
            NewsSentiment.POSITIVE: 0.5,
            NewsSentiment.VERY_POSITIVE: 1.0,
        }[self]


class NewsImpact(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def weight(self) -> float:
        return {
            NewsImpact.NONE: 0.0,
            NewsImpact.LOW: 0.25,
            NewsImpact.MEDIUM: 0.5,
            NewsImpact.HIGH: 0.8,
            NewsImpact.CRITICAL: 1.0,
        }[self]


# --------------------------------------------------------------------------- #
# Users, licensing, notifications
# --------------------------------------------------------------------------- #
class UserRole(StrEnum):
    USER = "user"
    ADMIN = "admin"


class UserStatus(StrEnum):
    PENDING_VERIFICATION = "pending_verification"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class LicenseStatus(StrEnum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    REVOKED = "revoked"

    @property
    def permits_use(self) -> bool:
        return self is LicenseStatus.ACTIVE


class LicensePlan(StrEnum):
    TRIAL = "trial"
    STARTER = "starter"
    PRO = "pro"
    ENTERPRISE = "enterprise"

    @property
    def default_device_limit(self) -> int:
        return {
            LicensePlan.TRIAL: 1,
            LicensePlan.STARTER: 1,
            LicensePlan.PRO: 3,
            LicensePlan.ENTERPRISE: 10,
        }[self]

    @property
    def max_bots(self) -> int:
        return {
            LicensePlan.TRIAL: 1,
            LicensePlan.STARTER: 2,
            LicensePlan.PRO: 10,
            LicensePlan.ENTERPRISE: 100,
        }[self]


class SubscriptionStatus(StrEnum):
    TRIALING = "trialing"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class NotificationChannel(StrEnum):
    EMAIL = "email"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    WEB_PUSH = "web_push"
    IN_APP = "in_app"


class NotificationEvent(StrEnum):
    ACCOUNT_SECURITY = "account_security"
    TRADE_OPENED = "trade_opened"
    TRADE_CLOSED = "trade_closed"
    TAKE_PROFIT_HIT = "take_profit_hit"
    STOP_LOSS_HIT = "stop_loss_hit"
    RISK_ALERT = "risk_alert"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    BOT_CRASHED = "bot_crashed"
    EXCHANGE_DISCONNECTED = "exchange_disconnected"
    LICENSE_EXPIRING = "license_expiring"
    SYSTEM_ERROR = "system_error"


class NotificationStatus(StrEnum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    SUPPRESSED = "suppressed"


class AuditAction(StrEnum):
    LOGIN = "login"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    USER_REGISTERED = "user_registered"
    PASSWORD_CHANGED = "password_changed"
    LICENSE_ACTIVATED = "license_activated"
    LICENSE_DEACTIVATED = "license_deactivated"
    EXCHANGE_ACCOUNT_CONNECTED = "exchange_account_connected"
    EXCHANGE_ACCOUNT_REMOVED = "exchange_account_removed"
    STRATEGY_CREATED = "strategy_created"
    STRATEGY_UPDATED = "strategy_updated"
    RISK_CONFIG_CHANGED = "risk_config_changed"
    BOT_STARTED = "bot_started"
    BOT_STOPPED = "bot_stopped"
    LIVE_TRADING_ACTIVATED = "live_trading_activated"
    ORDER_PLACED = "order_placed"
    ORDER_CANCELLED = "order_cancelled"
    EMERGENCY_STOP = "emergency_stop"
    KILL_SWITCH_RESET = "kill_switch_reset"


class BacktestStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TradeStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class ExitReason(StrEnum):
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    SIGNAL = "signal"
    MANUAL = "manual"
    RISK = "risk"
    KILL_SWITCH = "kill_switch"
    END_OF_BACKTEST = "end_of_backtest"
    LIQUIDATION = "liquidation"
