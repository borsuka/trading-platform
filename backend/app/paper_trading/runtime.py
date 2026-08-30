"""Bot runtime.

Drives one bot: fetch data, decide, check risk, execute, record, repeat. Despite living in
``paper_trading``, this is the runtime for **both** paper and live trading — the only
difference between them is which :class:`~app.exchanges.base.ExchangeAdapter` was constructed.
Duplicating this loop for live mode would guarantee the two drift apart, and the whole value of
paper trading rests on them being the same code.

Lifecycle
---------
``CREATED -> STARTING -> RUNNING -> (PAUSED) -> STOPPING -> STOPPED``, with ``HALTED`` and
``ERROR`` reachable from any running state.

* **STOP** stops opening new positions and stops the loop. It does *not* close open positions:
  a scheduled stop should not become a forced liquidation at whatever price is available.
* **PAUSE** stops decisions but keeps marking positions and monitoring stops.
* **EMERGENCY STOP** trips the kill switch, cancels resting orders and — only when explicitly
  requested — flattens positions.

Startup sequence
----------------
For live mode this is not optional::

    LOCAL STATE -> EXCHANGE STATE -> RECONCILE -> VALIDATE -> TRADING ENABLED

A reconciliation mismatch leaves the bot ``HALTED`` with the difference reported.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from app.core.clock import Clock, SystemClock, interval_to_timedelta, measure_drift, utcnow
from app.core.domain import Fill, InstrumentSpec, Order, Position
from app.core.enums import (
    BotEventType,
    BotStatus,
    ExitReason,
    SignalAction,
)
from app.core.exceptions import (
    ExchangeError,
    LiveTradingDisabledError,
    TradingPlatformError,
)
from app.core.logging import get_logger
from app.exchanges.base import ExchangeAdapter
from app.execution.order_manager import OrderManager
from app.market_data.models import MarketSnapshot
from app.market_data.providers import LiveMarketDataProvider
from app.news.providers import NewsProvider, NullNewsProvider
from app.news.scoring import NewsAggregator, NewsAssessment
from app.portfolio.manager import PortfolioManager, ReconciliationScheduler
from app.risk.limits import RiskLimits
from app.risk.manager import RiskManager, TradeProposal
from app.risk.state import RiskState, TradeOutcome
from app.signals.engine import Signal, SignalEngine, SignalEngineConfig
from app.strategies.base import Strategy

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BotEvent:
    """Something the bot did or observed. Persisted for the activity feed."""

    event_type: BotEventType
    message: str
    occurred_at: datetime
    severity: str = "info"
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "message": self.message,
            "occurred_at": self.occurred_at.isoformat(),
            "severity": self.severity,
            "payload": self.payload,
        }


@dataclass(slots=True)
class BotConfig:
    """Runtime configuration for one bot."""

    bot_id: str
    user_id: str
    name: str
    symbols: tuple[str, ...]
    interval: str = "15m"
    #: Bars of history requested per decision cycle.
    lookback: int = 300
    #: Seconds between cycles. Defaults to a fraction of the bar interval.
    poll_seconds: float | None = None
    #: Seconds between portfolio/exchange reconciliations.
    reconcile_seconds: float = 300.0
    #: Seconds between clock-drift checks.
    clock_check_seconds: float = 60.0
    #: Consecutive cycle errors tolerated before the bot halts.
    max_consecutive_errors: int = 5
    #: Close every position when the bot stops. Off by default, deliberately.
    close_positions_on_stop: bool = False
    #: Reconcile against the exchange before enabling trading.
    reconcile_on_start: bool = True

    def cycle_interval(self) -> float:
        if self.poll_seconds is not None:
            return self.poll_seconds
        return max(5.0, interval_to_timedelta(self.interval).total_seconds() / 4.0)


@dataclass(slots=True)
class BotSnapshot:
    """Point-in-time view of a bot, for the API and dashboard."""

    bot_id: str
    name: str
    status: BotStatus
    mode: str
    symbols: tuple[str, ...]
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "bot_id": self.bot_id,
            "name": self.name,
            "status": self.status.value,
            "mode": self.mode,
            "symbols": list(self.symbols),
            "equity": round(self.equity, 4),
            "cash": round(self.cash, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "realized_pnl": round(self.realized_pnl, 4),
            "open_positions": self.open_positions,
            "drawdown": round(self.drawdown, 6),
            "cycles": self.cycles,
            "last_cycle_at": (
                self.last_cycle_at.isoformat() if self.last_cycle_at else None
            ),
            "kill_switch": self.kill_switch,
            "halt_reason": self.halt_reason,
            "last_error": self.last_error,
        }


EventHandler = Callable[[BotEvent], Awaitable[None]]


class TradingBot:
    """One running strategy against one exchange adapter."""

    def __init__(
        self,
        config: BotConfig,
        *,
        strategy: Strategy,
        exchange: ExchangeAdapter,
        market_data: LiveMarketDataProvider,
        portfolio: PortfolioManager,
        risk_limits: RiskLimits,
        signal_config: SignalEngineConfig | None = None,
        news_provider: NewsProvider | None = None,
        news_aggregator: NewsAggregator | None = None,
        clock: Clock | None = None,
        event_handler: EventHandler | None = None,
    ) -> None:
        self.config = config
        self.strategy = strategy
        self.exchange = exchange
        self.market_data = market_data
        self.portfolio = portfolio
        self.clock = clock or SystemClock()
        self.event_handler = event_handler

        self.risk_manager = RiskManager(
            risk_limits, RiskState(starting_equity=portfolio.starting_balance)
        )
        self.order_manager = OrderManager(exchange)
        self.signal_engine = SignalEngine(
            strategy, config=signal_config or SignalEngineConfig()
        )
        self.news_provider = news_provider or NullNewsProvider()
        self.news_aggregator = news_aggregator or NewsAggregator()
        self.reconciler = ReconciliationScheduler(
            portfolio=portfolio,
            exchange=exchange,
            interval_seconds=config.reconcile_seconds,
        )

        self._status = BotStatus.CREATED
        self._task: asyncio.Task[None] | None = None
        self._stop_requested = asyncio.Event()
        self._pause_requested = False
        self._cycles = 0
        self._last_cycle_at: datetime | None = None
        self._last_error: str | None = None
        self._consecutive_errors = 0
        self._events: list[BotEvent] = []
        self._instruments: dict[str, InstrumentSpec] = {}
        self._last_clock_check: datetime | None = None
        self._bars_since_trade: dict[str, int] = {}
        #: Fill ids already reflected in the portfolio. Prevents double-counting a fill that
        #: is both returned by a submission and seen again by the exchange sync.
        self._applied_fills: set[str] = set()

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    @property
    def status(self) -> BotStatus:
        return self._status

    @property
    def is_running(self) -> bool:
        return self._status in {BotStatus.RUNNING, BotStatus.PAUSED}

    @property
    def mode(self) -> str:
        return "LIVE" if self.exchange.is_live else "PAPER"

    @property
    def events(self) -> list[BotEvent]:
        return list(self._events)

    def drain_events(self) -> list[BotEvent]:
        drained = list(self._events)
        self._events.clear()
        return drained

    def snapshot(self) -> BotSnapshot:
        return BotSnapshot(
            bot_id=self.config.bot_id,
            name=self.config.name,
            status=self._status,
            mode=self.mode,
            symbols=self.config.symbols,
            equity=self.portfolio.equity(),
            cash=self.portfolio.cash,
            unrealized_pnl=self.portfolio.unrealized_pnl(),
            realized_pnl=self.portfolio.realized_pnl,
            open_positions=len(self.portfolio.positions()),
            drawdown=self.portfolio.drawdown(),
            cycles=self._cycles,
            last_cycle_at=self._last_cycle_at,
            kill_switch=self.risk_manager.kill_switch.describe(),
            halt_reason=self.portfolio.halt_reason or self.order_manager.halt_reason,
            last_error=self._last_error,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        """Run the startup sequence and begin the decision loop."""
        if self.is_running:
            raise TradingPlatformError(f"Bot {self.config.name} is already running")

        self._status = BotStatus.STARTING
        self._stop_requested.clear()
        await self._emit(BotEventType.STARTED, f"Starting bot in {self.mode} mode")

        try:
            await self._startup_sequence()
        except Exception as exc:
            self._status = BotStatus.ERROR
            self._last_error = str(exc)
            await self._emit(
                BotEventType.ERROR, f"Startup failed: {exc}", severity="critical"
            )
            raise

        self._status = BotStatus.RUNNING
        self._task = asyncio.create_task(self._run_loop(), name=f"bot-{self.config.bot_id}")

    async def _startup_sequence(self) -> None:
        """LOCAL -> EXCHANGE -> RECONCILE -> VALIDATE -> ENABLE."""
        await self.exchange.connect()

        instruments = await self.exchange.get_instruments()
        for symbol in self.config.symbols:
            spec = instruments.get(symbol)
            if spec is None:
                spec = await self.exchange.get_instrument(symbol)
            self._instruments[symbol] = spec

        await self._check_clock_drift()

        if self.exchange.is_live:
            permissions = await self.exchange.validate_credentials()
            problem = permissions.rejection_reason()
            if problem is not None:
                raise LiveTradingDisabledError(problem)

        if self.config.reconcile_on_start:
            report = await self.portfolio.reconcile(self.exchange)
            if not report.is_clean:
                self._status = BotStatus.HALTED
                self.risk_manager.on_reconciliation_failure(report.summary())
                await self._emit(
                    BotEventType.RECONCILIATION_MISMATCH,
                    f"Startup reconciliation failed: {report.summary()}",
                    severity="critical",
                )
                raise TradingPlatformError(
                    f"Bot cannot start: {report.summary()}. Reconcile manually, then resume."
                )

        problems = self.portfolio.validate_invariants()
        if problems:
            self.risk_manager.on_state_corruption("; ".join(problems))
            raise TradingPlatformError(
                f"Portfolio state is invalid: {'; '.join(problems)}"
            )

        self.risk_manager.state.mark_equity(self.portfolio.equity(), now=self.clock.now())
        logger.info(
            "bot.started",
            bot_id=self.config.bot_id,
            mode=self.mode,
            symbols=list(self.config.symbols),
            strategy=self.strategy.name,
            equity=round(self.portfolio.equity(), 2),
        )

    async def stop(self, *, close_positions: bool | None = None) -> None:
        """Stop the loop. Open positions are left alone unless explicitly requested."""
        if not self.is_running:
            self._status = BotStatus.STOPPED
            return
        self._status = BotStatus.STOPPING
        self._stop_requested.set()

        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=30.0)
            except TimeoutError:
                logger.warning("bot.stop_timeout", bot_id=self.config.bot_id)
                self._task.cancel()
            except asyncio.CancelledError:
                pass
            self._task = None

        should_close = (
            self.config.close_positions_on_stop
            if close_positions is None
            else close_positions
        )
        if should_close:
            await self._flatten_all(ExitReason.MANUAL)

        await self.exchange.close()
        self._status = BotStatus.STOPPED
        await self._emit(
            BotEventType.STOPPED,
            f"Bot stopped after {self._cycles} cycles"
            + ("; positions closed" if should_close else "; positions left open"),
        )

    async def pause(self) -> None:
        """Stop making decisions but keep monitoring."""
        if self._status is not BotStatus.RUNNING:
            return
        self._pause_requested = True
        self._status = BotStatus.PAUSED
        await self._emit(BotEventType.PAUSED, "Bot paused; no new decisions")

    async def resume(self) -> None:
        if self._status is not BotStatus.PAUSED:
            return
        self._pause_requested = False
        self._status = BotStatus.RUNNING
        await self._emit(BotEventType.RESUMED, "Bot resumed")

    async def emergency_stop(
        self, *, actor: str, close_positions: bool = False, note: str = ""
    ) -> None:
        """Trip the kill switch, cancel resting orders, optionally flatten.

        Closing positions is opt-in even here. An emergency is frequently the worst possible
        moment to be a forced seller, and the operator is better placed than the bot to judge
        whether exiting now beats holding through it.
        """
        self.risk_manager.emergency_stop(actor, note)
        self._status = BotStatus.HALTED
        await self._emit(
            BotEventType.EMERGENCY_STOP,
            f"Emergency stop by {actor}{f': {note}' if note else ''}",
            severity="critical",
        )
        try:
            await self.order_manager.cancel_all()
        except ExchangeError as exc:
            logger.error("bot.cancel_all_failed", bot_id=self.config.bot_id, error=str(exc))
        if close_positions:
            await self._flatten_all(ExitReason.KILL_SWITCH)
        self._stop_requested.set()

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    async def _run_loop(self) -> None:
        interval = self.config.cycle_interval()
        while not self._stop_requested.is_set():
            try:
                await self.run_cycle()
                self._consecutive_errors = 0
                self.risk_manager.on_api_success()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._handle_cycle_error(exc)
                if self._consecutive_errors >= self.config.max_consecutive_errors:
                    self._status = BotStatus.HALTED
                    await self._emit(
                        BotEventType.ERROR,
                        f"Halting after {self._consecutive_errors} consecutive errors",
                        severity="critical",
                    )
                    break
            try:
                await asyncio.wait_for(self._stop_requested.wait(), timeout=interval)
            except TimeoutError:
                continue

    async def run_cycle(self) -> list[Signal]:
        """One decision cycle across every configured symbol.

        Exposed publicly so tests and the acceptance scenario can step the bot
        deterministically instead of waiting on wall-clock timers.
        """
        self._cycles += 1
        self._last_cycle_at = self.clock.now()

        # Orders can fill without the bot asking - a venue-side stop-loss or take-profit
        # triggers while the process is between cycles. Those fills must reach the portfolio
        # before anything else runs, or reconciliation will (correctly) see a mismatch and
        # halt the bot for what is actually normal operation.
        await self._sync_exchange_fills()
        await self._periodic_checks()

        signals: list[Signal] = []
        for symbol in self.config.symbols:
            if self._pause_requested or self._stop_requested.is_set():
                break
            signal = await self._process_symbol(symbol)
            if signal is not None:
                signals.append(signal)

        self.portfolio.snapshot(at=self.clock.now())
        self.risk_manager.state.mark_equity(self.portfolio.equity(), now=self.clock.now())
        return signals

    async def _process_symbol(self, symbol: str) -> Signal | None:
        snapshot = await self.market_data.get_snapshot(
            symbol, self.config.interval, lookback=self.config.lookback
        )
        price = snapshot.price
        if price is not None:
            self.portfolio.mark(symbol, price, at=self.clock.now())

        position = self.portfolio.position(symbol)
        news = await self._news_for(symbol)

        signal = self.signal_engine.evaluate(
            snapshot,
            position=position,
            news=news,
            bars_since_last_trade=self._bars_since_trade.get(symbol),
            now=self.clock.now(),
        )
        if symbol in self._bars_since_trade:
            self._bars_since_trade[symbol] += 1

        await self._emit(
            BotEventType.SIGNAL_GENERATED,
            f"{symbol}: {signal.action.value} - {signal.reason}",
            payload=signal.to_dict(),
        )

        if signal.action is SignalAction.CLOSE and position is not None:
            await self._close(position, ExitReason.SIGNAL)
            return signal
        if not signal.is_entry:
            return signal

        if not self._may_open_positions():
            return signal

        await self._attempt_entry(signal, snapshot)
        return signal

    def _may_open_positions(self) -> bool:
        """Every condition that must hold before any new order is placed."""
        if self._status is not BotStatus.RUNNING:
            return False
        if not self.portfolio.trading_enabled:
            return False
        if self.order_manager.is_halted:
            return False
        return not self.risk_manager.kill_switch.is_active

    async def _attempt_entry(self, signal: Signal, snapshot: MarketSnapshot) -> None:
        spec = self._instruments.get(signal.symbol)
        if spec is None:
            spec = await self.exchange.get_instrument(signal.symbol)
            self._instruments[signal.symbol] = spec

        if signal.entry is None or signal.stop_loss is None:
            logger.warning("bot.signal_missing_levels", symbol=signal.symbol)
            return

        proposal = TradeProposal(
            symbol=signal.symbol,
            action=signal.action,
            entry_price=signal.entry,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            instrument=spec,
            confidence=signal.confidence,
            strategy_name=signal.strategy_name,
        )
        assessment = self.risk_manager.evaluate(
            proposal, self.portfolio.to_view(), snapshot, now=self.clock.now()
        )
        if not assessment.approved:
            await self._emit(
                BotEventType.RISK_BLOCKED,
                f"{signal.symbol}: risk blocked the trade - {assessment.reason}",
                severity="warning",
                payload={"blocked_by": (
                    assessment.blocked_by.value if assessment.blocked_by else None
                )},
            )
            return

        result = await self.order_manager.open_position(
            signal.symbol,
            signal.action.order_side,
            assessment.quantity,
            assessment=assessment,
            metadata={"strategy": signal.strategy_name, "bot_id": self.config.bot_id},
        )
        if result.requires_halt:
            self._status = BotStatus.HALTED
            self.risk_manager.on_state_corruption(result.reason)
            await self._emit(
                BotEventType.ERROR,
                f"Halting: {result.reason}",
                severity="critical",
            )
            return
        if not result.succeeded or result.order is None:
            await self._emit(
                BotEventType.ORDER_REJECTED,
                f"{signal.symbol}: {result.reason}",
                severity="warning",
            )
            return

        await self._emit(
            BotEventType.ORDER_SUBMITTED,
            f"{signal.symbol}: {signal.action.value} {assessment.quantity:g} submitted",
            payload={"client_order_id": result.order.client_order_id},
        )

        for order_fill in result.order.fills:
            self._applied_fills.add(order_fill.fill_id)
            self.portfolio.apply_fill(
                order_fill,
                strategy_name=signal.strategy_name,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )
        if result.order.fills:
            self.risk_manager.state.record_order_submitted(now=self.clock.now())
            self._bars_since_trade[signal.symbol] = 0
            await self._emit(
                BotEventType.POSITION_OPENED,
                f"{signal.symbol}: position opened at "
                f"{result.order.average_fill_price:g}",
                payload={"quantity": result.order.filled_quantity},
            )
            await self._place_protection(signal)

    async def _place_protection(self, signal: Signal) -> None:
        """Attach venue-side stops so protection survives a bot crash."""
        position = self.portfolio.position(signal.symbol)
        if position is None:
            return
        results = await self.order_manager.place_protective_orders(
            position, stop_loss=signal.stop_loss, take_profit=signal.take_profit
        )
        if signal.stop_loss is not None and not any(r.succeeded for r in results):
            await self._emit(
                BotEventType.ERROR,
                f"{signal.symbol}: could not place a stop-loss; closing the position "
                "rather than leaving it unprotected",
                severity="critical",
            )
            await self._close(position, ExitReason.RISK)

    async def close_position(
        self, symbol: str, *, reason: ExitReason = ExitReason.MANUAL
    ) -> bool:
        """Close one position on demand. Returns False when there was nothing to close.

        Public because closing is a legitimate operator action, and it is never gated on risk
        approval: reducing exposure is not the risky direction.
        """
        position = self.portfolio.position(symbol.upper())
        if position is None:
            return False
        await self._close(position, reason)
        return True

    async def _close(self, position: Position, reason: ExitReason) -> None:
        result = await self.order_manager.close_position(position, reason=reason.value)
        if not result.succeeded or result.order is None:
            await self._emit(
                BotEventType.ERROR,
                f"{position.symbol}: close failed - {result.reason}",
                severity="critical",
            )
            return
        for order_fill in result.order.fills:
            self._applied_fills.add(order_fill.fill_id)
            trade = self.portfolio.apply_fill(order_fill, exit_reason=reason)
            if trade is None:
                continue
            self.risk_manager.state.record_trade(
                TradeOutcome(
                    symbol=trade.symbol,
                    net_pnl=trade.net_pnl,
                    closed_at=trade.exit_time or self.clock.now(),
                    strategy_name=trade.strategy_name,
                )
            )
            await self._emit(
                BotEventType.POSITION_CLOSED,
                f"{trade.symbol}: closed for {trade.net_pnl:+.2f} ({reason.value})",
                payload={"net_pnl": trade.net_pnl, "reason": reason.value},
            )
            await self._cancel_protective_orders(trade.symbol)

    async def _flatten_all(self, reason: ExitReason) -> None:
        for position in list(self.portfolio.positions()):
            try:
                await self._close(position, reason)
            except ExchangeError as exc:
                logger.error(
                    "bot.flatten_failed", symbol=position.symbol, error=str(exc)
                )

    # ------------------------------------------------------------------ #
    # Exchange-driven fills
    # ------------------------------------------------------------------ #
    async def _sync_exchange_fills(self) -> None:
        """Apply fills that happened on the venue since the last cycle.

        Chiefly protective orders: a stop-loss or take-profit resting at the exchange fills on
        its own, and the platform learns about it here rather than by being told. Without this
        the portfolio would still show a position the venue has already closed - which is
        exactly the divergence reconciliation exists to catch, so the bot would halt on
        entirely normal operation.
        """
        for tracked in list(self.order_manager.known_orders()):
            if tracked.status.is_terminal and all(
                f.fill_id in self._applied_fills for f in tracked.fills
            ):
                continue
            try:
                order = await self.order_manager.refresh(
                    tracked.client_order_id, symbol=tracked.symbol
                )
            except ExchangeError as exc:
                logger.warning(
                    "bot.order_refresh_failed",
                    client_order_id=tracked.client_order_id,
                    error=str(exc),
                )
                continue
            if order is None:
                continue

            for order_fill in order.fills:
                if order_fill.fill_id in self._applied_fills:
                    continue
                self._applied_fills.add(order_fill.fill_id)
                await self._apply_external_fill(order, order_fill)

    async def _apply_external_fill(self, order: Order, order_fill: Fill) -> None:
        """Record a fill the bot did not initiate in this cycle."""
        protective = (order.metadata or {}).get("protective")
        reason = {
            "stop_loss": ExitReason.STOP_LOSS,
            "take_profit": ExitReason.TAKE_PROFIT,
        }.get(str(protective), ExitReason.SIGNAL)

        trade = self.portfolio.apply_fill(order_fill, exit_reason=reason)
        if trade is None:
            return

        self.risk_manager.state.record_trade(
            TradeOutcome(
                symbol=trade.symbol,
                net_pnl=trade.net_pnl,
                closed_at=trade.exit_time or self.clock.now(),
                strategy_name=trade.strategy_name,
            )
        )
        await self._emit(
            BotEventType.POSITION_CLOSED,
            f"{trade.symbol}: closed for {trade.net_pnl:+.2f} ({reason.value})",
            payload={"net_pnl": trade.net_pnl, "reason": reason.value},
        )
        await self._cancel_protective_orders(trade.symbol)

    async def _cancel_protective_orders(self, symbol: str) -> None:
        """Cancel remaining protective orders on a symbol that is now flat.

        One protective order filling makes its sibling dangerous: a reduce-only order resting
        against a flat position is at best noise, and on venues that ignore reduce-only for
        conditional orders it can open an unintended position.
        """
        if self.portfolio.position(symbol) is not None:
            return
        for order in self.order_manager.known_orders():
            if (
                order.symbol == symbol
                and order.is_active
                and (order.metadata or {}).get("protective")
            ):
                await self.order_manager.cancel(order.client_order_id, symbol=symbol)

    # ------------------------------------------------------------------ #
    # Periodic checks
    # ------------------------------------------------------------------ #
    async def _periodic_checks(self) -> None:
        now = self.clock.now()
        if (
            self._last_clock_check is None
            or now - self._last_clock_check
            >= timedelta(seconds=self.config.clock_check_seconds)
        ):
            await self._check_clock_drift()
            self._last_clock_check = now

        report = await self.reconciler.run_if_due(now=now)
        if report is not None and not report.is_clean:
            self.risk_manager.on_reconciliation_failure(report.summary())
            self._status = BotStatus.HALTED
            await self._emit(
                BotEventType.RECONCILIATION_MISMATCH,
                f"Reconciliation mismatch: {report.summary()}",
                severity="critical",
            )

        problems = self.portfolio.validate_invariants()
        if problems:
            self.risk_manager.on_state_corruption("; ".join(problems))
            self._status = BotStatus.HALTED
            await self._emit(
                BotEventType.ERROR,
                f"Portfolio invariants violated: {'; '.join(problems)}",
                severity="critical",
            )

        if self.risk_manager.kill_switch.is_active and self._status is BotStatus.RUNNING:
            self._status = BotStatus.HALTED
            trip = self.risk_manager.kill_switch.trip
            await self._emit(
                BotEventType.KILL_SWITCH_TRIPPED,
                f"Kill switch: {trip.message if trip else 'engaged'}",
                severity="critical",
            )

    async def _check_clock_drift(self) -> None:
        """Compare local time to the exchange's clock.

        Signed order timestamps make drift a hard failure on most venues, so this is a
        trading-blocking check rather than an observation.
        """
        try:
            info = await self.exchange.get_info()
        except ExchangeError as exc:
            logger.warning("bot.clock_check_failed", error=str(exc))
            return
        report = measure_drift(
            self.clock.now(),
            info.server_time,
            self.risk_manager.limits.max_clock_drift_seconds,
        )
        if not report.within_tolerance:
            self.risk_manager.on_clock_drift(report.drift_seconds, now=self.clock.now())
            await self._emit(
                BotEventType.ERROR,
                f"Clock drift: {report.describe()}",
                severity="critical",
            )

    async def _news_for(self, symbol: str) -> NewsAssessment | None:
        if isinstance(self.news_provider, NullNewsProvider):
            return None
        spec = self._instruments.get(symbol)
        asset = spec.base_asset if spec else symbol
        try:
            articles = await self.news_provider.fetch(assets=[asset], limit=50)
        except TradingPlatformError as exc:
            # News is an enhancement. Losing it must never stop the bot managing positions.
            logger.warning("bot.news_fetch_failed", symbol=symbol, error=str(exc))
            return None
        if not articles:
            return None
        return self.news_aggregator.assess(articles, asset, now=self.clock.now())

    async def _handle_cycle_error(self, exc: Exception) -> None:
        self._consecutive_errors += 1
        self._last_error = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, ExchangeError):
            self.risk_manager.on_api_failure(str(exc), now=self.clock.now())
        logger.exception(
            "bot.cycle_failed",
            bot_id=self.config.bot_id,
            consecutive_errors=self._consecutive_errors,
        )
        await self._emit(
            BotEventType.ERROR,
            f"Cycle {self._cycles} failed: {self._last_error}",
            severity="warning",
        )

    async def _emit(
        self,
        event_type: BotEventType,
        message: str,
        *,
        severity: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> None:
        event = BotEvent(
            event_type=event_type,
            message=message,
            occurred_at=self.clock.now(),
            severity=severity,
            payload=payload or {},
        )
        self._events.append(event)
        if len(self._events) > 5_000:
            del self._events[:1_000]
        if self.event_handler is not None:
            try:
                await self.event_handler(event)
            except Exception:
                # An event sink failing must never take the trading loop down with it.
                logger.exception("bot.event_handler_failed", failed_event=event_type.value)


class BotRegistry:
    """Tracks the bots running in this process.

    A single-process registry, which is the right shape for the desktop/VPS deployment where
    one installation runs one customer's bots. A multi-tenant server deployment would replace
    this with a distributed scheduler; the :class:`TradingBot` interface would not change.
    """

    def __init__(self) -> None:
        self._bots: dict[str, TradingBot] = {}
        self._lock = asyncio.Lock()

    async def register(self, bot: TradingBot) -> None:
        async with self._lock:
            if bot.config.bot_id in self._bots:
                raise TradingPlatformError(
                    f"Bot {bot.config.bot_id} is already registered"
                )
            self._bots[bot.config.bot_id] = bot

    async def unregister(self, bot_id: str) -> None:
        async with self._lock:
            bot = self._bots.pop(bot_id, None)
        if bot is not None and bot.is_running:
            await bot.stop()

    def get(self, bot_id: str) -> TradingBot | None:
        return self._bots.get(bot_id)

    def for_user(self, user_id: str) -> list[TradingBot]:
        """Every bot owned by one user. The only enumeration the API exposes."""
        return [b for b in self._bots.values() if b.config.user_id == user_id]

    def all(self) -> list[TradingBot]:
        return list(self._bots.values())

    def snapshots(self, user_id: str | None = None) -> list[dict[str, Any]]:
        bots = self.for_user(user_id) if user_id else self.all()
        return [bot.snapshot().to_dict() for bot in bots]

    async def stop_all(self) -> None:
        """Stop every bot. Called on application shutdown."""
        for bot in list(self._bots.values()):
            if bot.is_running:
                try:
                    await bot.stop()
                except Exception:
                    logger.exception("bot_registry.stop_failed", bot_id=bot.config.bot_id)

    async def emergency_stop_all(self, *, actor: str, close_positions: bool = False) -> None:
        for bot in list(self._bots.values()):
            if bot.is_running:
                await bot.emergency_stop(actor=actor, close_positions=close_positions)


#: Process-wide registry used by the API layer.
bot_registry = BotRegistry()


def build_symbols(raw: Sequence[str] | str) -> tuple[str, ...]:
    """Normalise a symbol list from config or an API payload."""
    if isinstance(raw, str):
        items = [s.strip().upper() for s in raw.split(",")]
    else:
        items = [str(s).strip().upper() for s in raw]
    symbols = tuple(s for s in items if s)
    if not symbols:
        raise ValueError("At least one symbol is required")
    return symbols


def utc_now() -> datetime:
    """Convenience re-export so runtime callers do not import from two places."""
    return utcnow()
