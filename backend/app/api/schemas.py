"""API request and response schemas.

Response models are explicit rather than serialising ORM objects directly. That is what keeps
``password_hash``, ``api_secret_encrypted`` and ``totp_secret_encrypted`` off the wire: a field
appears in a response only because someone wrote it here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.core.enums import (
    BacktestStatus,
    BotStatus,
    LicensePlan,
    LicenseStatus,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    SignalAction,
    UserRole,
    UserStatus,
)


class APIModel(BaseModel):
    """Base for every schema."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")


# =========================================================================== #
# Auth
# =========================================================================== #
class RegisterRequest(APIModel):
    email: EmailStr
    password: str = Field(min_length=12, max_length=256)
    full_name: str | None = Field(default=None, max_length=200)


class LoginRequest(APIModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class RefreshRequest(APIModel):
    refresh_token: str = Field(min_length=10, max_length=512)


class PasswordResetRequest(APIModel):
    email: EmailStr


class PasswordResetConfirm(APIModel):
    token: str = Field(min_length=10, max_length=256)
    new_password: str = Field(min_length=12, max_length=256)


class PasswordChangeRequest(APIModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=12, max_length=256)


class TokenResponse(APIModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(APIModel):
    id: str
    email: str
    full_name: str | None
    role: UserRole
    status: UserStatus
    email_verified: bool
    totp_enabled: bool
    created_at: datetime
    last_login_at: datetime | None


# =========================================================================== #
# Strategies
# =========================================================================== #
class StrategyCatalogEntry(APIModel):
    name: str
    version: str
    description: str
    allowed_regimes: list[str]
    default_parameters: dict[str, Any]
    parameters_schema: dict[str, Any]


class StrategyCreateRequest(APIModel):
    name: str = Field(min_length=1, max_length=120)
    strategy_type: str = Field(min_length=1, max_length=64)
    parameters: dict[str, Any] = Field(default_factory=dict)
    description: str | None = Field(default=None, max_length=2000)


class StrategyUpdateRequest(APIModel):
    parameters: dict[str, Any] | None = None
    description: str | None = Field(default=None, max_length=2000)
    is_active: bool | None = None
    changelog: str | None = Field(default=None, max_length=1000)


class StrategyResponse(APIModel):
    id: str
    name: str
    strategy_type: str
    description: str | None
    parameters: dict[str, Any]
    current_version: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


# =========================================================================== #
# Risk
# =========================================================================== #
class RiskLimitsPayload(APIModel):
    """Per-bot risk configuration. Always clamped to the platform ceiling server-side."""

    risk_per_trade: float = Field(default=0.005, gt=0.0, le=0.05)
    max_position_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    max_leverage: float = Field(default=1.0, ge=1.0, le=20.0)
    max_portfolio_exposure: float = Field(default=1.0, gt=0.0, le=10.0)
    max_asset_exposure: float = Field(default=0.35, gt=0.0, le=1.0)
    max_concurrent_positions: int = Field(default=3, ge=1, le=100)
    max_daily_loss: float = Field(default=0.02, gt=0.0, le=0.5)
    max_weekly_loss: float = Field(default=0.06, gt=0.0, le=0.8)
    max_drawdown: float = Field(default=0.10, gt=0.0, le=0.9)
    max_loss_streak: int = Field(default=4, ge=1, le=50)
    max_daily_trades: int = Field(default=30, ge=1, le=10_000)
    cooldown_seconds: int = Field(default=300, ge=0, le=86_400)
    min_reward_risk: float = Field(default=1.0, ge=0.0, le=20.0)


class RiskStatusResponse(APIModel):
    limits: dict[str, Any]
    state: dict[str, Any]
    kill_switch: dict[str, Any]
    headroom: dict[str, Any]


class KillSwitchResetRequest(APIModel):
    note: str = Field(default="", max_length=500)


# =========================================================================== #
# Bots
# =========================================================================== #
class BotCreateRequest(APIModel):
    name: str = Field(min_length=1, max_length=120)
    strategy_type: str = Field(min_length=1, max_length=64)
    symbols: list[str] = Field(min_length=1, max_length=20)
    interval: str = Field(default="15m")
    starting_balance: float = Field(default=10_000.0, gt=0, le=100_000_000)
    strategy_parameters: dict[str, Any] = Field(default_factory=dict)
    risk: RiskLimitsPayload = Field(default_factory=RiskLimitsPayload)

    @field_validator("symbols")
    @classmethod
    def _normalise(cls, value: list[str]) -> list[str]:
        symbols = [s.strip().upper() for s in value if s.strip()]
        if not symbols:
            raise ValueError("At least one symbol is required")
        if len(set(symbols)) != len(symbols):
            raise ValueError("Duplicate symbols are not allowed")
        return symbols


class BotActionRequest(APIModel):
    close_positions: bool = Field(
        default=False,
        description=(
            "Close open positions as part of this action. Off by default: stopping a bot "
            "should not force a sale at whatever price is available."
        ),
    )
    note: str = Field(default="", max_length=500)


class BotResponse(APIModel):
    id: str
    name: str
    strategy_id: str
    trading_mode: str
    symbols: list[str]
    interval: str
    status: BotStatus
    kill_switch_active: bool
    last_error: str | None
    created_at: datetime
    started_at: datetime | None
    stopped_at: datetime | None


class BotRuntimeResponse(APIModel):
    """Live runtime state, distinct from the persisted bot record."""

    bot_id: str
    name: str
    status: str
    mode: str
    symbols: list[str]
    equity: float
    cash: float
    unrealized_pnl: float
    realized_pnl: float
    open_positions: int
    drawdown: float
    cycles: int
    last_cycle_at: datetime | None
    kill_switch: dict[str, Any]
    halt_reason: str | None
    last_error: str | None


class BotEventResponse(APIModel):
    event_type: str
    message: str
    occurred_at: datetime
    severity: str
    payload: dict[str, Any]


# =========================================================================== #
# Trading data
# =========================================================================== #
class OrderResponse(APIModel):
    id: str
    client_order_id: str
    exchange_order_id: str | None
    symbol: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    quantity: float
    price: float | None
    filled_quantity: float
    average_fill_price: float
    fees_paid: float
    reject_reason: str | None
    created_at: datetime


class PositionResponse(APIModel):
    id: str
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    mark_price: float | None
    unrealized_pnl: float
    realized_pnl: float
    stop_loss: float | None
    take_profit: float | None
    leverage: float
    opened_at: datetime
    strategy_name: str | None


class TradeResponse(APIModel):
    id: str
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    exit_price: float | None
    entry_time: datetime
    exit_time: datetime | None
    net_pnl: float
    fees: float
    return_pct: float
    r_multiple: float | None
    exit_reason: str | None
    strategy_name: str | None


class SignalResponse(APIModel):
    id: str
    symbol: str
    action: SignalAction
    strategy_name: str
    confidence: float
    entry_price: float | None
    stop_loss: float | None
    take_profit: float | None
    regime: str | None
    reason: str
    executed: bool
    blocked_by: str | None
    generated_at: datetime


class PortfolioResponse(APIModel):
    equity: float
    cash: float
    realized_pnl: float
    unrealized_pnl: float
    fees_paid: float
    total_return: float
    drawdown: float
    peak_equity: float
    open_positions: int
    total_exposure: float
    closed_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float


# =========================================================================== #
# Backtests
# =========================================================================== #
class BacktestRequest(APIModel):
    name: str = Field(min_length=1, max_length=160)
    strategy_type: str = Field(min_length=1, max_length=64)
    symbol: str = Field(min_length=1, max_length=32)
    interval: str = Field(default="1h")
    bars: int = Field(default=2000, ge=300, le=50_000)
    initial_balance: float = Field(default=10_000.0, gt=0, le=100_000_000)
    parameters: dict[str, Any] = Field(default_factory=dict)
    risk: RiskLimitsPayload = Field(default_factory=RiskLimitsPayload)
    data_source: Literal["synthetic", "csv"] = Field(
        default="synthetic",
        description=(
            "'synthetic' generates a deterministic sample series for demonstration. It is "
            "NOT market data and its results are not a performance claim."
        ),
    )
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    run_validation: bool = Field(
        default=False, description="Also run walk-forward and Monte Carlo analysis"
    )


class BacktestResponse(APIModel):
    id: str
    name: str
    strategy_type: str
    symbol: str
    interval: str
    status: BacktestStatus
    progress: float
    metrics: dict[str, Any]
    error_message: str | None
    created_at: datetime
    completed_at: datetime | None


class BacktestDetailResponse(BacktestResponse):
    parameters: dict[str, Any]
    equity_curve: list[dict[str, Any]]
    monthly_returns: dict[str, float]
    warnings: list[str]
    disclaimer: str = (
        "Past performance and backtest results do not guarantee future performance. "
        "Backtests are simulations and cannot capture every real-world execution cost."
    )


# =========================================================================== #
# Exchange accounts
# =========================================================================== #
class ExchangeAccountCreateRequest(APIModel):
    name: str = Field(min_length=1, max_length=120)
    exchange: Literal["bybit", "binance", "coinbase", "cryptocom"]
    api_key: str = Field(min_length=8, max_length=256)
    api_secret: str = Field(min_length=8, max_length=256)
    testnet: bool = True


class ExchangeAccountResponse(APIModel):
    """Never carries the key or secret. Only a mask and a fingerprint."""

    id: str
    name: str
    exchange: str
    is_testnet: bool
    api_key_masked: str | None
    api_key_fingerprint: str | None
    can_withdraw: bool
    is_validated: bool
    validated_at: datetime | None
    last_error: str | None
    is_active: bool
    created_at: datetime


# =========================================================================== #
# Live trading
# =========================================================================== #
class LivePreflightRequest(APIModel):
    exchange_account_id: str
    symbols: list[str] = Field(min_length=1, max_length=20)
    interval: str = "15m"


class LiveActivationRequest(APIModel):
    exchange_account_id: str
    confirmation: str = Field(
        description='Must be exactly "I UNDERSTAND THE RISKS"', max_length=100
    )
    acknowledge_no_guarantee: bool = Field(
        description="Must be true. Confirms the user understands returns are not guaranteed."
    )


class PreflightResponse(APIModel):
    passed: bool
    performed_at: datetime
    checks: list[dict[str, Any]]
    summary: str


# =========================================================================== #
# Licensing
# =========================================================================== #
class LicenseActivateRequest(APIModel):
    license_key: str = Field(min_length=8, max_length=64)
    device_fingerprint: str = Field(min_length=8, max_length=128)
    device_name: str | None = Field(default=None, max_length=200)
    platform: str | None = Field(default=None, max_length=64)
    app_version: str | None = Field(default=None, max_length=32)


class LicenseResponse(APIModel):
    id: str
    license_key_masked: str
    plan: LicensePlan
    status: LicenseStatus
    expires_at: datetime | None
    device_limit: int
    activated_devices: int
    days_remaining: int | None


class SubscriptionResponse(APIModel):
    id: str
    plan: LicensePlan
    status: str
    current_period_end: datetime | None
    cancel_at_period_end: bool


# =========================================================================== #
# News
# =========================================================================== #
class NewsAssessmentResponse(APIModel):
    asset: str
    timestamp: datetime
    directional_score: float
    confidence: float
    max_impact: float
    article_count: int
    duplicates_removed: int
    dominant_event: str
    dominant_sentiment: str
    # Present in NewsAssessment.to_dict(). `extra="forbid"` means omitting it here is not a
    # missing field but a 500 on every call to the endpoint.
    urgency: float
    headline_summary: str


# =========================================================================== #
# System
# =========================================================================== #
class HealthResponse(APIModel):
    status: Literal["ok", "degraded", "unhealthy"]
    version: str
    mode: str
    uptime_seconds: float
    checks: dict[str, Any]


class ReadinessResponse(APIModel):
    ready: bool
    checks: dict[str, bool]


class PaginatedResponse(APIModel):
    items: list[Any]
    total: int
    limit: int
    offset: int


class MessageResponse(APIModel):
    message: str


class ErrorResponse(APIModel):
    error: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)
