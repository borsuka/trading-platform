"""Application exception hierarchy.

Every error raised deliberately by the platform derives from :class:`TradingPlatformError`.
Exceptions carry structured ``context`` so that logs are useful without string-formatting, and
:meth:`TradingPlatformError.public_message` guarantees that nothing secret leaks to an API
consumer.
"""

from __future__ import annotations

from typing import Any


class TradingPlatformError(Exception):
    """Base class for all deliberate platform errors."""

    #: HTTP status used when this error surfaces through the API.
    status_code: int = 500
    #: Stable machine-readable code for clients.
    error_code: str = "internal_error"
    #: Message shown to API clients when the detail may be sensitive.
    default_public_message: str = "An internal error occurred."
    #: When true the raw message is safe to return to clients.
    safe_to_expose: bool = False

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context or {})
        if cause is not None:
            self.__cause__ = cause

    def public_message(self) -> str:
        """Message that is safe to hand to an untrusted caller."""
        return self.message if self.safe_to_expose else self.default_public_message

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.error_code,
            "message": self.public_message(),
            "context": {k: v for k, v in self.context.items() if not _is_sensitive_key(k)},
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.message!r}, context={self.context!r})"


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in ("secret", "password", "token", "api_key", "apikey"))


# --------------------------------------------------------------------------- #
# Configuration / startup
# --------------------------------------------------------------------------- #
class ConfigurationError(TradingPlatformError):
    error_code = "configuration_error"
    status_code = 500
    default_public_message = "The service is misconfigured."


# --------------------------------------------------------------------------- #
# Validation & client errors
# --------------------------------------------------------------------------- #
class ValidationError(TradingPlatformError):
    error_code = "validation_error"
    status_code = 422
    safe_to_expose = True


class NotFoundError(TradingPlatformError):
    error_code = "not_found"
    status_code = 404
    safe_to_expose = True


class ConflictError(TradingPlatformError):
    error_code = "conflict"
    status_code = 409
    safe_to_expose = True


class AuthenticationError(TradingPlatformError):
    error_code = "authentication_failed"
    status_code = 401
    default_public_message = "Authentication failed."


class AuthorizationError(TradingPlatformError):
    error_code = "not_authorized"
    status_code = 403
    default_public_message = "You do not have access to this resource."


class RateLimitError(TradingPlatformError):
    error_code = "rate_limited"
    status_code = 429
    safe_to_expose = True

    def __init__(
        self,
        message: str = "Too many requests.",
        *,
        retry_after_seconds: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.retry_after_seconds = retry_after_seconds


# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #
class MarketDataError(TradingPlatformError):
    error_code = "market_data_error"
    status_code = 503
    safe_to_expose = True


class StaleDataError(MarketDataError):
    error_code = "stale_market_data"


class DataValidationError(MarketDataError):
    error_code = "invalid_market_data"


class InsufficientDataError(MarketDataError):
    error_code = "insufficient_market_data"
    status_code = 422


# --------------------------------------------------------------------------- #
# Exchange
# --------------------------------------------------------------------------- #
class ExchangeError(TradingPlatformError):
    error_code = "exchange_error"
    status_code = 502
    safe_to_expose = True


class ExchangeConnectionError(ExchangeError):
    """Transport-level failure. Safe to retry read operations."""

    error_code = "exchange_unreachable"


class ExchangeTimeoutError(ExchangeError):
    """The request may or may not have been applied. NEVER blindly retry writes."""

    error_code = "exchange_timeout"


class ExchangeRateLimitError(ExchangeError):
    error_code = "exchange_rate_limited"

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.retry_after_seconds = retry_after_seconds


class ExchangeAuthError(ExchangeError):
    error_code = "exchange_auth_failed"
    status_code = 401
    safe_to_expose = False
    default_public_message = "The exchange rejected the supplied credentials."


class UnsafeCredentialsError(ExchangeError):
    """Raised when an API key carries withdrawal permission."""

    error_code = "unsafe_exchange_credentials"
    status_code = 400
    safe_to_expose = True


class OrderRejectedError(ExchangeError):
    error_code = "order_rejected"
    status_code = 400


class InsufficientBalanceError(OrderRejectedError):
    error_code = "insufficient_balance"


class InstrumentNotSupportedError(ExchangeError):
    error_code = "instrument_not_supported"
    status_code = 400


# --------------------------------------------------------------------------- #
# Trading decision path
# --------------------------------------------------------------------------- #
class StrategyError(TradingPlatformError):
    error_code = "strategy_error"
    status_code = 400
    safe_to_expose = True


class StrategyConfigurationError(StrategyError):
    error_code = "strategy_configuration_error"
    status_code = 422


class RiskViolationError(TradingPlatformError):
    """A proposed action breaches a risk limit. Never bypassable."""

    error_code = "risk_violation"
    status_code = 409
    safe_to_expose = True


class KillSwitchActiveError(RiskViolationError):
    error_code = "kill_switch_active"


class PositionSizingError(RiskViolationError):
    error_code = "position_sizing_failed"


class ExecutionError(TradingPlatformError):
    error_code = "execution_error"
    status_code = 500
    safe_to_expose = True


class ReconciliationError(ExecutionError):
    """Local state and exchange state disagree. Trading must halt."""

    error_code = "reconciliation_failed"


class PortfolioStateError(ExecutionError):
    error_code = "portfolio_state_invalid"


# --------------------------------------------------------------------------- #
# Backtesting
# --------------------------------------------------------------------------- #
class BacktestError(TradingPlatformError):
    error_code = "backtest_error"
    status_code = 400
    safe_to_expose = True


class LookaheadError(BacktestError):
    """A component tried to read data from the future."""

    error_code = "lookahead_detected"


# --------------------------------------------------------------------------- #
# Licensing
# --------------------------------------------------------------------------- #
class LicenseError(TradingPlatformError):
    error_code = "license_error"
    status_code = 403
    safe_to_expose = True


class LicenseExpiredError(LicenseError):
    error_code = "license_expired"


class LicenseDeviceLimitError(LicenseError):
    error_code = "license_device_limit_reached"


class LicenseInvalidError(LicenseError):
    error_code = "license_invalid"


# --------------------------------------------------------------------------- #
# Live-trading gate
# --------------------------------------------------------------------------- #
class LiveTradingDisabledError(TradingPlatformError):
    error_code = "live_trading_disabled"
    status_code = 403
    safe_to_expose = True


class PreflightFailedError(TradingPlatformError):
    error_code = "preflight_failed"
    status_code = 409
    safe_to_expose = True
