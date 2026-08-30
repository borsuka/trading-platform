"""Live-trading preflight.

Live trading is disabled by default and cannot be enabled implicitly. Before any real order can
be placed, every check below must pass, and each returns a specific, actionable reason when it
fails. A check that cannot be performed counts as a failure — silence is never a pass.

The nine checks, in the order the spec requires:

1. Platform configuration explicitly enables live trading.
2. API credentials are valid.
3. API permissions allow trading and **forbid withdrawal**.
4. The account holds a usable balance.
5. Risk configuration is present and coherent.
6. Market data is flowing and fresh.
7. Local and exchange clocks agree.
8. Exchange connectivity is stable across repeated calls.
9. The user has explicitly confirmed intent, in writing, for this specific bot.

The result is a :class:`PreflightReport`. The API renders it as the confirmation screen; the
bot runtime re-runs the machine-checkable parts at startup, because configuration can change
between confirmation and launch.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime

from app.config import Settings, TradingMode, get_settings
from app.core.clock import measure_drift, utcnow
from app.core.exceptions import ExchangeError, PreflightFailedError
from app.core.logging import get_logger
from app.exchanges.base import ExchangeAdapter
from app.market_data.validation import MarketDataValidator
from app.risk.limits import RiskLimits

logger = get_logger(__name__)

#: The exact phrase a user must type to confirm live trading.
CONFIRMATION_PHRASE = "I UNDERSTAND THE RISKS"


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    """One check and its outcome."""

    name: str
    passed: bool
    detail: str
    blocking: bool = True

    def describe(self) -> str:
        mark = "PASS" if self.passed else ("FAIL" if self.blocking else "WARN")
        return f"[{mark}] {self.name}: {self.detail}"


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Result of the full preflight."""

    checks: tuple[PreflightCheck, ...] = ()
    performed_at: datetime = field(default_factory=utcnow)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.blocking)

    @property
    def failures(self) -> tuple[PreflightCheck, ...]:
        return tuple(c for c in self.checks if c.blocking and not c.passed)

    @property
    def warnings(self) -> tuple[PreflightCheck, ...]:
        return tuple(c for c in self.checks if not c.blocking and not c.passed)

    def require_pass(self) -> None:
        if not self.passed:
            raise PreflightFailedError(
                "Live trading preflight failed:\n"
                + "\n".join(c.describe() for c in self.failures),
                context={"failed_checks": [c.name for c in self.failures]},
            )

    def summary(self) -> str:
        lines = ["LIVE TRADING PREFLIGHT", "=" * 60]
        lines.extend(c.describe() for c in self.checks)
        lines.append("=" * 60)
        lines.append("RESULT: " + ("PASSED" if self.passed else "BLOCKED"))
        if not self.passed:
            lines.append(
                "Live trading remains disabled. Resolve every FAIL above and re-run."
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "performed_at": self.performed_at.isoformat(),
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "detail": c.detail,
                    "blocking": c.blocking,
                }
                for c in self.checks
            ],
            "summary": self.summary(),
        }


async def run_preflight(
    exchange: ExchangeAdapter,
    *,
    symbols: list[str],
    interval: str = "15m",
    risk_limits: RiskLimits | None = None,
    confirmation: str | None = None,
    settings: Settings | None = None,
    min_balance: float = 0.0,
) -> PreflightReport:
    """Run every live-trading precondition and report the results."""
    resolved = settings or get_settings()
    checks: list[PreflightCheck] = [
        _check_configuration(resolved),
        _check_confirmation(confirmation),
        _check_risk_configuration(risk_limits),
    ]

    checks.append(await _check_credentials(exchange))
    checks.append(await _check_permissions(exchange))
    checks.append(await _check_balance(exchange, resolved, min_balance))
    checks.append(await _check_connectivity(exchange))
    checks.append(await _check_clock(exchange, resolved))
    checks.append(await _check_market_data(exchange, symbols, interval, resolved))

    report = PreflightReport(checks=tuple(checks))
    logger.warning(
        "live_gate.preflight",
        passed=report.passed,
        failures=[c.name for c in report.failures],
    )
    return report


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #
def _check_configuration(settings: Settings) -> PreflightCheck:
    if not settings.live_trading_enabled:
        return PreflightCheck(
            "platform_configuration",
            False,
            "LIVE_TRADING_ENABLED is false. Live trading cannot be enabled from the UI "
            "alone; it requires an explicit configuration change on the host.",
        )
    if settings.trading_mode is not TradingMode.LIVE:
        return PreflightCheck(
            "platform_configuration",
            False,
            f"TRADING_MODE is '{settings.trading_mode.value}', not 'live'.",
        )
    return PreflightCheck(
        "platform_configuration", True, "live trading is explicitly enabled"
    )


def _check_confirmation(confirmation: str | None) -> PreflightCheck:
    if confirmation is None:
        return PreflightCheck(
            "user_confirmation",
            False,
            f'No confirmation supplied. Type "{CONFIRMATION_PHRASE}" to proceed.',
        )
    if confirmation.strip().upper() != CONFIRMATION_PHRASE:
        return PreflightCheck(
            "user_confirmation",
            False,
            f'Confirmation phrase did not match. Expected "{CONFIRMATION_PHRASE}".',
        )
    return PreflightCheck("user_confirmation", True, "user confirmed in writing")


def _check_risk_configuration(limits: RiskLimits | None) -> PreflightCheck:
    if limits is None:
        return PreflightCheck(
            "risk_configuration", False, "no risk limits configured for this bot"
        )
    simultaneous = limits.max_simultaneous_risk
    if simultaneous > limits.max_drawdown:
        return PreflightCheck(
            "risk_configuration",
            False,
            f"{simultaneous:.1%} of equity can be at risk simultaneously, exceeding the "
            f"{limits.max_drawdown:.1%} drawdown limit",
        )
    detail = (
        f"risking {limits.risk_per_trade:.2%} per trade, "
        f"max {limits.max_concurrent_positions} positions "
        f"({simultaneous:.1%} simultaneous), "
        f"daily loss limit {limits.max_daily_loss:.1%}, "
        f"drawdown limit {limits.max_drawdown:.1%}"
    )
    return PreflightCheck("risk_configuration", True, detail)


async def _check_credentials(exchange: ExchangeAdapter) -> PreflightCheck:
    try:
        await exchange.validate_credentials()
    except ExchangeError as exc:
        return PreflightCheck(
            "api_credentials", False, f"the exchange rejected the credentials: {exc.message}"
        )
    except Exception as exc:
        return PreflightCheck(
            "api_credentials", False, f"credential check failed: {type(exc).__name__}"
        )
    return PreflightCheck("api_credentials", True, "credentials accepted by the exchange")


async def _check_permissions(exchange: ExchangeAdapter) -> PreflightCheck:
    try:
        permissions = await exchange.validate_credentials()
    except Exception as exc:
        return PreflightCheck(
            "api_permissions", False, f"could not read permissions: {type(exc).__name__}"
        )
    problem = permissions.rejection_reason()
    if problem is not None:
        return PreflightCheck("api_permissions", False, problem)
    return PreflightCheck(
        "api_permissions",
        True,
        "key can read and trade, and cannot withdraw"
        + (" (IP restricted)" if permissions.ip_restricted else ""),
    )


async def _check_balance(
    exchange: ExchangeAdapter, settings: Settings, minimum: float
) -> PreflightCheck:
    try:
        balance = await exchange.get_balance()
    except Exception as exc:
        return PreflightCheck(
            "account_balance", False, f"could not read the balance: {type(exc).__name__}"
        )
    asset = settings.paper_quote_currency
    total = balance.total(asset)
    if total <= 0:
        return PreflightCheck(
            "account_balance", False, f"the account holds no {asset}"
        )
    if total < minimum:
        return PreflightCheck(
            "account_balance",
            False,
            f"balance {total:.2f} {asset} is below the required minimum {minimum:.2f}",
        )
    return PreflightCheck(
        "account_balance", True, f"{total:.2f} {asset} available"
    )


async def _check_connectivity(
    exchange: ExchangeAdapter, *, attempts: int = 3
) -> PreflightCheck:
    """Repeated calls, because one lucky response is not connectivity."""
    latencies: list[float] = []
    for _ in range(attempts):
        started = utcnow()
        try:
            await exchange.get_info()
        except Exception as exc:
            return PreflightCheck(
                "exchange_connectivity",
                False,
                f"connectivity check failed: {type(exc).__name__}",
            )
        latencies.append((utcnow() - started).total_seconds())
        await asyncio.sleep(0)
    average = sum(latencies) / len(latencies)
    if average > 2.0:
        return PreflightCheck(
            "exchange_connectivity",
            False,
            f"average latency {average * 1000:.0f}ms is too high for reliable execution",
        )
    return PreflightCheck(
        "exchange_connectivity",
        True,
        f"{attempts} successful calls, average latency {average * 1000:.0f}ms",
    )


async def _check_clock(exchange: ExchangeAdapter, settings: Settings) -> PreflightCheck:
    try:
        info = await exchange.get_info()
    except Exception as exc:
        return PreflightCheck(
            "clock_synchronisation",
            False,
            f"could not read exchange server time: {type(exc).__name__}",
        )
    report = measure_drift(utcnow(), info.server_time, settings.max_clock_drift_seconds)
    if not report.within_tolerance:
        return PreflightCheck(
            "clock_synchronisation",
            False,
            f"{report.describe()}. Synchronise the system clock (NTP) before trading: "
            "signed orders are rejected when timestamps drift.",
        )
    return PreflightCheck("clock_synchronisation", True, report.describe())


async def _check_market_data(
    exchange: ExchangeAdapter,
    symbols: list[str],
    interval: str,
    settings: Settings,
) -> PreflightCheck:
    if not symbols:
        return PreflightCheck("market_data", False, "no symbols configured")
    validator = MarketDataValidator(
        max_staleness_seconds=settings.market_data_max_staleness_seconds,
        min_required_bars=50,
    )
    problems: list[str] = []
    for symbol in symbols:
        try:
            candles = await exchange.get_candles(symbol, interval, limit=100)
        except Exception as exc:
            problems.append(f"{symbol}: {type(exc).__name__}")
            continue
        if not candles:
            problems.append(f"{symbol}: no candles returned")
            continue
        result = validator.validate_candles(candles, required_bars=50)
        if not result.is_valid:
            problems.append(f"{symbol}: {result.reason()}")

    if problems:
        return PreflightCheck("market_data", False, "; ".join(problems))
    return PreflightCheck(
        "market_data", True, f"fresh data available for {len(symbols)} symbol(s)"
    )
