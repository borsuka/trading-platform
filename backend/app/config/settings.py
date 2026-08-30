"""Application configuration.

All configuration is sourced from environment variables (optionally via a ``.env`` file).
Secrets are held in :class:`pydantic.SecretStr` so they cannot be accidentally logged or
serialised into an API response.
"""

from __future__ import annotations

import enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]


class AppEnv(enum.StrEnum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class TradingMode(enum.StrEnum):
    """How orders reach a venue.

    ``PAPER`` is the only safe default. ``LIVE`` sends real orders with real money and is
    additionally gated by :attr:`Settings.live_trading_enabled` plus a runtime preflight.
    """

    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class ExchangeName(enum.StrEnum):
    """Supported venues."""

    PAPER = "paper"
    BYBIT = "bybit"
    BINANCE = "binance"
    COINBASE = "coinbase"
    CRYPTOCOM = "cryptocom"


class RiskSettings(BaseSettings):
    """Default risk limits.

    These are the *platform* defaults. A bot may tighten them, but the
    :class:`~app.risk.manager.RiskManager` clamps any per-bot value so it is never looser than
    the ceilings configured here.
    """

    model_config = SettingsConfigDict(env_prefix="RISK_", extra="ignore")

    per_trade: float = Field(
        default=0.005, gt=0, le=0.05, description="Fraction of equity risked per trade"
    )
    max_daily_loss: float = Field(default=0.02, gt=0, le=0.5)
    max_weekly_loss: float = Field(default=0.06, gt=0, le=0.8)
    max_drawdown: float = Field(default=0.10, gt=0, le=0.9)
    max_position_fraction: float = Field(
        default=0.25, gt=0, le=1.0, description="Max notional of one position / equity"
    )
    max_portfolio_exposure: float = Field(
        default=1.0, gt=0, le=10.0, description="Max total notional / equity"
    )
    max_asset_exposure: float = Field(default=0.35, gt=0, le=1.0)
    max_concurrent_positions: int = Field(default=5, ge=1, le=100)
    max_leverage: float = Field(default=3.0, ge=1.0, le=20.0)
    max_spread_bps: float = Field(default=15.0, gt=0, le=1000.0)
    max_slippage_bps: float = Field(default=25.0, gt=0, le=1000.0)
    min_liquidity_multiple: float = Field(
        default=10.0, ge=1.0, description="Book depth must be N x intended order size"
    )
    cooldown_seconds: int = Field(default=300, ge=0)
    max_loss_streak: int = Field(default=4, ge=1, le=50)
    max_daily_trades: int = Field(default=30, ge=1, le=10_000)


class Settings(BaseSettings):
    """Root application settings."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- application -------------------------------------------------------
    app_env: AppEnv = AppEnv.DEVELOPMENT
    app_name: str = "Trading Platform"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = True
    api_prefix: str = "/api/v1"
    # `NoDecode` stops pydantic-settings from JSON-parsing this before the validator below
    # runs. Without it, the natural `CORS_ORIGINS=http://a,http://b` form is a startup crash,
    # because a bare URL is not valid JSON.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )

    # --- trading -----------------------------------------------------------
    trading_mode: TradingMode = TradingMode.PAPER
    live_trading_enabled: bool = Field(
        default=False,
        description="Master switch. Live mode ALSO requires this to be explicitly true.",
    )
    exchange: ExchangeName = ExchangeName.PAPER
    exchange_testnet: bool = True
    exchange_api_key: SecretStr | None = None
    exchange_api_secret: SecretStr | None = None
    paper_starting_balance: float = Field(default=10_000.0, gt=0)
    paper_quote_currency: str = "USDT"
    # Where a paper bot's prices come from. `bybit` and `binance` read that venue's PUBLIC
    # endpoints - no API key, no account, and no path from here to an order. `synthetic`
    # generates a random walk instead, which exercises the plumbing but tells you nothing
    # about a strategy. Real prices are the default because a paper run against invented data
    # is not the rehearsal the live-trading checklist asks for.
    paper_market_data: Literal["bybit", "binance", "synthetic"] = "bybit"

    risk: RiskSettings = Field(default_factory=RiskSettings)

    # --- infrastructure ----------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./data/trading.db"
    database_echo: bool = False
    database_pool_size: int = Field(default=10, ge=1, le=100)
    database_max_overflow: int = Field(default=20, ge=0, le=200)
    redis_url: str | None = None
    redis_required: bool = False

    # --- security ----------------------------------------------------------
    secret_key: SecretStr = SecretStr("dev-only-insecure-key-change-me")
    encryption_key: SecretStr | None = Field(
        default=None,
        description="Fernet key used to encrypt exchange credentials at rest.",
    )
    access_token_ttl_seconds: int = Field(default=900, ge=60)
    refresh_token_ttl_seconds: int = Field(default=60 * 60 * 24 * 14, ge=3600)
    password_min_length: int = Field(default=12, ge=8)
    auth_rate_limit_per_minute: int = Field(default=10, ge=1)
    require_email_verification: bool | None = Field(
        default=None,
        description=(
            "Require email verification before sign-in. Defaults to true in production and "
            "false elsewhere. Leaving it true without SMTP configured means new accounts "
            "cannot sign in at all, so startup warns loudly about that combination."
        ),
    )

    # --- market data -------------------------------------------------------
    market_data_max_staleness_seconds: int = Field(default=120, ge=5)
    max_clock_drift_seconds: float = Field(default=2.0, gt=0)

    # --- news --------------------------------------------------------------
    news_enabled: bool = True
    # `rss` is the default because it works with no credentials: public feeds, no account.
    # `null` remains fully supported and every strategy must behave correctly under it.
    news_provider: Literal["null", "rss", "file", "http"] = "rss"
    news_api_url: str | None = None
    news_api_key: SecretStr | None = None
    news_max_age_minutes: int = Field(default=180, ge=1)
    news_cache_seconds: float = Field(
        default=60.0,
        ge=5.0,
        description="How long fetched articles are reused before the feeds are polled again.",
    )

    # --- licensing ---------------------------------------------------------
    license_server_url: str | None = None
    license_key: SecretStr | None = None
    license_grace_period_hours: int = Field(default=72, ge=0)
    license_enforcement: bool = False

    # --- notifications -----------------------------------------------------
    notifications_enabled: bool = True
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None
    discord_webhook_url: SecretStr | None = None
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: SecretStr | None = None
    smtp_from: str = "noreply@localhost"

    # --- storage -----------------------------------------------------------
    data_dir: Path = Field(default_factory=lambda: REPO_ROOT / "data")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: Any) -> Any:
        """Accept a comma-separated list or a JSON array.

        Comma-separated is what people actually write in a ``.env`` file; the JSON form is
        supported because it is what pydantic-settings would have accepted by default, and
        silently breaking it would be a nasty upgrade surprise.
        """
        if not isinstance(value, str):
            return value
        text = value.strip()
        if text.startswith("["):
            import json

            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"CORS_ORIGINS looks like JSON but is not valid: {exc}"
                ) from exc
            if not isinstance(parsed, list):
                raise ValueError("CORS_ORIGINS JSON must be an array of strings")
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [item.strip() for item in text.split(",") if item.strip()]

    @field_validator("paper_quote_currency")
    @classmethod
    def _upper_currency(cls, value: str) -> str:
        return value.upper()

    @model_validator(mode="after")
    def _validate_consistency(self) -> Settings:
        if self.trading_mode is TradingMode.LIVE and not self.live_trading_enabled:
            raise ValueError(
                "TRADING_MODE=live requires LIVE_TRADING_ENABLED=true. "
                "Live trading is never enabled implicitly."
            )
        if self.trading_mode is TradingMode.LIVE and self.exchange is ExchangeName.PAPER:
            raise ValueError("TRADING_MODE=live is incompatible with EXCHANGE=paper.")
        if self.is_production:
            if self.secret_key.get_secret_value().startswith("dev-only"):
                raise ValueError("SECRET_KEY must be set to a strong value in production.")
            if self.database_url.startswith("sqlite"):
                raise ValueError("SQLite is not supported in production; use PostgreSQL.")
        return self

    @property
    def is_production(self) -> bool:
        return self.app_env is AppEnv.PRODUCTION

    @property
    def is_testing(self) -> bool:
        return self.app_env is AppEnv.TEST

    @property
    def email_verification_required(self) -> bool:
        """Whether a new account must verify its email before signing in."""
        if self.require_email_verification is not None:
            return self.require_email_verification
        return self.is_production

    @property
    def email_delivery_configured(self) -> bool:
        return bool(self.smtp_host and self.smtp_from)

    @property
    def is_live(self) -> bool:
        """True only when live order routing is fully switched on."""
        return self.trading_mode is TradingMode.LIVE and self.live_trading_enabled

    def describe_mode(self) -> str:
        return "LIVE" if self.is_live else self.trading_mode.value.upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor used as a FastAPI dependency."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache (tests only)."""
    get_settings.cache_clear()
