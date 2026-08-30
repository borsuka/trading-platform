"""Event-driven backtester.

The engine's defining property: it uses **the same objects as live trading**. The same
:class:`~app.signals.engine.SignalEngine`, the same
:class:`~app.risk.manager.RiskManager`, the same
:class:`~app.execution.order_manager.OrderManager`, the same
:class:`~app.portfolio.manager.PortfolioManager`, and the same
:class:`~app.exchanges.paper.PaperExchange` that backs paper trading. Only the source of
market data differs.

That is what makes a backtest a prediction rather than a separate program that happens to
produce numbers. A backtester with its own simplified fill logic and its own risk shortcuts
tests a strategy that will never actually run.

Lookahead prevention
--------------------
The single most important guarantee. Enforced structurally, not by discipline:

* Bars are replayed one at a time; the snapshot handed to the pipeline is sliced to
  ``candles[: index + 1]`` and cannot reach later bars.
* Decisions are made on the *close* of bar N and executed on bar N+1 (configurable, and this
  is the default). Filling at the same bar's close means trading on information that was not
  available while the bar was forming.
* :meth:`BacktestEngine._assert_no_lookahead` re-checks the invariant on every bar and raises
  :class:`~app.core.exceptions.LookaheadError` if it is ever violated.
* News is replayed by publication timestamp, so an article cannot influence a decision made
  before it existed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Any

from app.backtesting.metrics import (
    PerformanceMetrics,
    compute_metrics,
    drawdown_curve,
    equity_curve_points,
    monthly_returns,
    periods_per_year_for,
    trade_distribution,
)
from app.core.clock import ensure_utc, interval_to_timedelta
from app.core.domain import InstrumentSpec, OrderRequest, Trade
from app.core.enums import (
    ExitReason,
    OrderSide,
    OrderType,
    SignalAction,
)
from app.core.exceptions import BacktestError, InsufficientDataError, LookaheadError
from app.core.logging import get_logger
from app.exchanges.paper import PaperExchange, PaperExchangeConfig
from app.execution.order_manager import OrderManager, make_client_order_id
from app.market_data.models import Candle, Ticker
from app.market_data.providers import build_snapshot
from app.news.providers import InMemoryNewsProvider
from app.news.scoring import NewsAggregator, NewsAssessment
from app.portfolio.manager import PortfolioManager
from app.risk.limits import RiskLimits
from app.risk.manager import RiskManager, TradeProposal
from app.risk.state import RiskState, TradeOutcome
from app.signals.engine import Signal, SignalEngine, SignalEngineConfig
from app.strategies.base import Strategy

logger = get_logger(__name__)


@dataclass(slots=True)
class BacktestConfig:
    """Backtest execution parameters."""

    initial_balance: float = 10_000.0
    quote_asset: str = "USDT"
    #: Execute on the bar after the signal. Setting this False is a lookahead cheat and is
    #: only permitted so that its effect can be demonstrated in tests.
    execute_next_bar: bool = True
    #: Warm-up bars fed to the simulator before any decision is made.
    warmup_bars: int = 200
    #: Snapshot the portfolio on every bar (needed for a smooth equity curve).
    snapshot_every_bar: bool = True
    #: Close all open positions at the end of the run, so PnL is fully realised.
    close_at_end: bool = True
    #: Cap on bars processed, as a guard against runaway loops.
    max_bars: int = 500_000
    #: Progress callback, invoked with a fraction in [0, 1].
    progress_callback: Callable[[float], None] | None = None
    #: Paper exchange realism settings.
    exchange_config: PaperExchangeConfig | None = None
    #: Emit a warning when the strategy trades more than this fraction of bars.
    overtrading_threshold: float = 0.25
    #: Extra bars kept beyond what the strategy and regime detector require.
    #:
    #: The decision window is bounded rather than growing with the run. That is not only an
    #: O(n) vs O(n^2) performance matter: live trading also works from a bounded rolling
    #: cache, so a backtest that fed indicators an ever-growing history would compute
    #: different values than the live bot on the same bar.
    window_buffer_bars: int = 50


@dataclass(slots=True)
class BacktestResult:
    """Everything a backtest produced."""

    metrics: PerformanceMetrics
    trades: list[Trade] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    drawdown_curve: list[dict[str, Any]] = field(default_factory=list)
    monthly_returns: dict[str, float] = field(default_factory=dict)
    trade_distribution: dict[str, list[float]] = field(default_factory=dict)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    risk_events: list[dict[str, Any]] = field(default_factory=list)
    bars_processed: int = 0
    strategy_name: str = ""
    strategy_version: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    symbol: str = ""
    interval: str = ""
    start: datetime | None = None
    end: datetime | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "bars_processed": self.bars_processed,
            "trade_count": len(self.trades),
            "signal_count": len(self.signals),
            "strategy": self.strategy_name,
            "strategy_version": self.strategy_version,
            "parameters": self.parameters,
            "symbol": self.symbol,
            "interval": self.interval,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "monthly_returns": {k: round(v, 6) for k, v in self.monthly_returns.items()},
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        header = (
            f"{self.strategy_name} v{self.strategy_version} on {self.symbol} "
            f"{self.interval} ({self.bars_processed} bars)"
        )
        body = self.metrics.summary()
        footer = ""
        if self.warnings:
            footer = "\nBacktest warnings: " + "; ".join(self.warnings)
        disclaimer = (
            "\n\nPast performance and backtest results do not guarantee future performance."
        )
        return f"{header}\n{body}{footer}{disclaimer}"


class BacktestEngine:
    """Replays historical bars through the live decision pipeline."""

    def __init__(
        self,
        strategy: Strategy,
        *,
        config: BacktestConfig | None = None,
        risk_limits: RiskLimits | None = None,
        signal_config: SignalEngineConfig | None = None,
        news_provider: InMemoryNewsProvider | None = None,
        news_aggregator: NewsAggregator | None = None,
    ) -> None:
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.risk_limits = risk_limits or RiskLimits()
        self.signal_config = signal_config or SignalEngineConfig()
        self.news_provider = news_provider
        self.news_aggregator = news_aggregator or NewsAggregator()

    # ------------------------------------------------------------------ #
    # Entry points
    # ------------------------------------------------------------------ #
    async def run(
        self,
        candles: Sequence[Candle],
        *,
        instrument: InstrumentSpec | None = None,
    ) -> BacktestResult:
        """Run the backtest over a candle series."""
        self._validate_input(candles)
        symbol = candles[0].symbol
        interval = candles[0].interval
        spec = instrument or _default_instrument(symbol)

        exchange = PaperExchange(
            self.config.exchange_config or PaperExchangeConfig(
                starting_balance=self.config.initial_balance,
                quote_asset=self.config.quote_asset,
            ),
            instruments={spec.symbol: spec},
        )
        await exchange.connect()

        portfolio = PortfolioManager(
            starting_balance=self.config.initial_balance,
            quote_asset=self.config.quote_asset,
        )
        risk_state = RiskState(starting_equity=self.config.initial_balance)
        risk_manager = RiskManager(self.risk_limits, risk_state)
        order_manager = OrderManager(exchange)
        signal_engine = SignalEngine(
            self.strategy,
            config=self.signal_config,
        )

        session = _BacktestSession(
            engine=self,
            candles=list(candles),
            spec=spec,
            exchange=exchange,
            portfolio=portfolio,
            risk_manager=risk_manager,
            order_manager=order_manager,
            signal_engine=signal_engine,
        )
        result = await session.execute()
        await exchange.close()
        result.symbol = symbol
        result.interval = interval
        result.strategy_name = self.strategy.name
        result.strategy_version = self.strategy.version
        result.parameters = self.strategy.params.model_dump()
        return result

    def run_sync(
        self, candles: Sequence[Candle], *, instrument: InstrumentSpec | None = None
    ) -> BacktestResult:
        """Synchronous wrapper, for CLI use and scripts."""
        return asyncio.run(self.run(candles, instrument=instrument))

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _validate_input(self, candles: Sequence[Candle]) -> None:
        if not candles:
            raise InsufficientDataError("Cannot run a backtest with no candles")
        if len(candles) > self.config.max_bars:
            raise BacktestError(
                f"{len(candles)} bars exceeds the {self.config.max_bars} limit"
            )
        required = self.config.warmup_bars + self.strategy.required_history + 1
        if len(candles) < required:
            raise InsufficientDataError(
                f"{len(candles)} bars is not enough: the strategy needs "
                f"{self.strategy.required_history} bars of history plus "
                f"{self.config.warmup_bars} warm-up bars ({required} total)",
                context={"provided": len(candles), "required": required},
            )
        symbols = {c.symbol for c in candles}
        if len(symbols) > 1:
            raise BacktestError(
                f"All candles must be for one symbol; got {sorted(symbols)}"
            )
        intervals = {c.interval for c in candles}
        if len(intervals) > 1:
            raise BacktestError(
                f"All candles must share one interval; got {sorted(intervals)}"
            )
        for previous, current in pairwise(candles):
            if current.open_time <= previous.open_time:
                raise BacktestError(
                    f"Candles must be strictly increasing in time; "
                    f"{current.open_time.isoformat()} follows "
                    f"{previous.open_time.isoformat()}"
                )


@dataclass(slots=True)
class _PendingEntry:
    """A signal approved on bar N, to be executed on bar N+1."""

    signal: Signal
    quantity: float
    side: OrderSide
    stop_loss: float | None
    take_profit: float | None


class _BacktestSession:
    """One backtest run. Holds all mutable state so :class:`BacktestEngine` stays reusable."""

    def __init__(
        self,
        *,
        engine: BacktestEngine,
        candles: list[Candle],
        spec: InstrumentSpec,
        exchange: PaperExchange,
        portfolio: PortfolioManager,
        risk_manager: RiskManager,
        order_manager: OrderManager,
        signal_engine: SignalEngine,
    ) -> None:
        self.engine = engine
        self.config = engine.config
        self.candles = candles
        self.spec = spec
        self.exchange = exchange
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.order_manager = order_manager
        self.signal_engine = signal_engine

        self.signals: list[Signal] = []
        self.rejections: list[dict[str, Any]] = []
        self.pending: _PendingEntry | None = None
        self.bars_since_trade: int | None = None
        self.warnings: list[str] = []
        self._last_snapshot_end: datetime | None = None
        self._entry_bars: list[int] = []
        self.window_size = (
            max(
                engine.strategy.required_history,
                signal_engine.regime_detector.config.min_history,
                signal_engine.config.min_history,
            )
            + engine.config.window_buffer_bars
        )

    async def execute(self) -> BacktestResult:
        config = self.config
        total = len(self.candles)
        warmup = max(config.warmup_bars, self.window_size)

        for index, candle in enumerate(self.candles):
            # 1. The bar opens. Anything decided on the previous close is executed here, at
            #    the first price actually reachable after that decision.
            if index >= warmup and self.pending is not None:
                await self._execute_pending(index, candle)

            # 2. The bar plays out: resting orders, stops and targets are evaluated against
            #    its full range.
            fills = self.exchange.process_candle(candle)
            self._sync_fills(fills)

            # 3. The bar closes. Only now is its close price known, so only now may it inform
            #    a decision - which will execute on the next bar's open.
            if index >= warmup:
                await self._decide(index, candle)

            self.portfolio.mark(candle.symbol, candle.close, at=candle.close_time)
            if config.snapshot_every_bar and index >= warmup:
                self.portfolio.snapshot(at=candle.close_time)

            if config.progress_callback and index % 500 == 0:
                config.progress_callback(index / max(total, 1))

        if config.close_at_end:
            await self._close_all(self.candles[-1])

        return self._build_result(total)

    # ------------------------------------------------------------------ #
    # Decision
    # ------------------------------------------------------------------ #
    async def _decide(self, index: int, candle: Candle) -> None:
        start = max(0, index + 1 - self.window_size)
        window = self.candles[start : index + 1]
        self._assert_no_lookahead(window, candle)

        snapshot = build_snapshot(window, now=candle.close_time)
        news = self._news_for(candle)
        position = self.portfolio.position(candle.symbol)

        signal = self.signal_engine.evaluate(
            snapshot,
            position=position,
            news=news,
            bars_since_last_trade=self.bars_since_trade,
            now=candle.close_time,
        )
        self.signals.append(signal)
        if self.bars_since_trade is not None:
            self.bars_since_trade += 1

        if signal.action is SignalAction.CLOSE and position is not None:
            await self._close_position(candle, ExitReason.SIGNAL)
            return
        if not signal.is_entry:
            return

        proposal = TradeProposal(
            symbol=signal.symbol,
            action=signal.action,
            entry_price=signal.entry or candle.close,
            stop_loss=signal.stop_loss or candle.close * 0.99,
            take_profit=signal.take_profit,
            instrument=self.spec,
            confidence=signal.confidence,
            strategy_name=signal.strategy_name,
        )
        assessment = self.risk_manager.evaluate(
            proposal, self.portfolio.to_view(), snapshot, now=candle.close_time
        )
        if not assessment.approved:
            self.rejections.append(
                {
                    "timestamp": candle.close_time.isoformat(),
                    "symbol": signal.symbol,
                    "action": signal.action.value,
                    "reason": assessment.reason,
                    "blocked_by": (
                        assessment.blocked_by.value if assessment.blocked_by else None
                    ),
                }
            )
            return

        entry = _PendingEntry(
            signal=signal,
            quantity=assessment.quantity,
            side=signal.action.order_side,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )
        if self.config.execute_next_bar:
            self.pending = entry
        else:
            await self._enter(entry, candle, index)

    async def _execute_pending(self, index: int, candle: Candle) -> None:
        """Fill a signal from the previous bar, at this bar's open.

        The simulator is stepped to the opening price first, so the fill happens against the
        open rather than against a close the decision could not have seen.
        """
        if self.pending is None:
            return
        entry = self.pending
        self.pending = None
        self.exchange.process_ticker(
            Ticker(
                symbol=candle.symbol,
                price=candle.open,
                timestamp=candle.open_time,
            )
        )
        await self._enter(entry, candle, index)

    async def _enter(
        self, entry: _PendingEntry, candle: Candle, index: int
    ) -> None:
        request = OrderRequest(
            symbol=candle.symbol,
            side=entry.side,
            order_type=OrderType.MARKET,
            quantity=entry.quantity,
            client_order_id=make_client_order_id("bt"),
            metadata={"strategy": entry.signal.strategy_name},
        )
        result = await self.order_manager.submit(request)
        if not result.succeeded or result.order is None:
            self.rejections.append(
                {
                    "timestamp": candle.close_time.isoformat(),
                    "symbol": candle.symbol,
                    "action": entry.side.value,
                    "reason": result.reason,
                    "blocked_by": "execution",
                }
            )
            return

        for order_fill in result.order.fills:
            self.portfolio.apply_fill(
                order_fill,
                strategy_name=entry.signal.strategy_name,
                stop_loss=entry.stop_loss,
                take_profit=entry.take_profit,
            )
        if entry.stop_loss is not None or entry.take_profit is not None:
            try:
                self.exchange.set_position_stops(
                    candle.symbol,
                    stop_loss=entry.stop_loss,
                    take_profit=entry.take_profit,
                )
            except Exception as exc:  # a stop through the market is a rejection, not a crash
                logger.warning(
                    "backtest.stop_attachment_failed",
                    symbol=candle.symbol,
                    error=str(exc),
                )
        self.risk_manager.state.record_order_submitted(now=candle.close_time)
        self.bars_since_trade = 0
        self._entry_bars.append(index)

    # ------------------------------------------------------------------ #
    # Exits
    # ------------------------------------------------------------------ #
    async def _close_position(self, candle: Candle, reason: ExitReason) -> None:
        position = self.portfolio.position(candle.symbol)
        if position is None:
            return
        result = await self.order_manager.close_position(position, reason=reason.value)
        if not result.succeeded or result.order is None:
            return
        for order_fill in result.order.fills:
            trade = self.portfolio.apply_fill(order_fill, exit_reason=reason)
            if trade is not None:
                self._record_outcome(trade)

    async def _close_all(self, last: Candle) -> None:
        for position in list(self.portfolio.positions()):
            result = await self.order_manager.close_position(
                position, reason=ExitReason.END_OF_BACKTEST.value
            )
            if result.order is None:
                continue
            for order_fill in result.order.fills:
                trade = self.portfolio.apply_fill(
                    order_fill, exit_reason=ExitReason.END_OF_BACKTEST
                )
                if trade is not None:
                    self._record_outcome(trade)
        self.portfolio.mark(last.symbol, last.close, at=last.close_time)
        self.portfolio.snapshot(at=last.close_time)

    def _sync_fills(self, fills: Sequence[Any]) -> None:
        """Apply fills the simulator generated on its own (stop-loss / take-profit)."""
        for exchange_fill in fills:
            auto_exit = self.exchange.order_metadata(exchange_fill.order_id).get("auto_exit")
            if auto_exit is None:
                continue  # entry fills are applied by the caller that placed them
            reason = (
                ExitReason.TAKE_PROFIT
                if auto_exit == "take_profit"
                else ExitReason.STOP_LOSS
            )
            trade = self.portfolio.apply_fill(exchange_fill, exit_reason=reason)
            if trade is not None:
                self._record_outcome(trade)

    def _record_outcome(self, trade: Trade) -> None:
        self.risk_manager.state.record_trade(
            TradeOutcome(
                symbol=trade.symbol,
                net_pnl=trade.net_pnl,
                closed_at=trade.exit_time or trade.entry_time,
                strategy_name=trade.strategy_name,
            )
        )
        self.risk_manager.state.mark_equity(
            self.portfolio.equity(), now=trade.exit_time or trade.entry_time
        )

    # ------------------------------------------------------------------ #
    # News replay
    # ------------------------------------------------------------------ #
    def _news_for(self, candle: Candle) -> NewsAssessment | None:
        """News published strictly before this bar closed.

        The ``<`` boundary is load-bearing: an article timestamped exactly at the close of the
        bar being decided on could not have been read while that bar was forming.
        """
        provider = self.engine.news_provider
        if provider is None:
            return None
        lookback = timedelta(
            minutes=self.engine.news_aggregator.config.max_age_minutes
        )
        articles = provider.window(candle.close_time - lookback, candle.close_time)
        if not articles:
            return None
        asset = self.spec.base_asset
        return self.engine.news_aggregator.assess(
            articles, asset, now=candle.close_time
        )

    # ------------------------------------------------------------------ #
    # Lookahead guard
    # ------------------------------------------------------------------ #
    def _assert_no_lookahead(self, window: Sequence[Candle], current: Candle) -> None:
        """Verify the decision window ends at the current bar and never extends past it."""
        if not window:
            raise LookaheadError("Decision window is empty")
        last = window[-1]
        if last.open_time != current.open_time:
            raise LookaheadError(
                f"Decision window ends at {last.open_time.isoformat()} but the current bar "
                f"is {current.open_time.isoformat()}"
            )
        if self._last_snapshot_end is not None and last.close_time < self._last_snapshot_end:
            raise LookaheadError(
                "Decision window moved backwards in time; bars are out of order"
            )
        self._last_snapshot_end = last.close_time

    # ------------------------------------------------------------------ #
    # Result assembly
    # ------------------------------------------------------------------ #
    def _build_result(self, total_bars: int) -> BacktestResult:
        snapshots = self.portfolio.snapshots()
        trades = self.portfolio.closed_trades()
        interval_seconds = interval_to_timedelta(
            self.candles[0].interval
        ).total_seconds()

        metrics = compute_metrics(
            snapshots,
            trades,
            initial_equity=self.config.initial_balance,
            periods_per_year=periods_per_year_for(interval_seconds),
        )
        self._collect_warnings(total_bars, trades)

        return BacktestResult(
            metrics=metrics,
            trades=trades,
            signals=self.signals,
            equity_curve=equity_curve_points(snapshots),
            drawdown_curve=drawdown_curve(snapshots),
            monthly_returns=monthly_returns(snapshots),
            trade_distribution=trade_distribution(trades),
            rejections=self.rejections,
            risk_events=[e.to_dict() for e in self.risk_manager.events],
            bars_processed=total_bars,
            start=self.candles[0].open_time,
            end=self.candles[-1].close_time,
            warnings=self.warnings,
        )

    def _collect_warnings(self, total_bars: int, trades: Sequence[Trade]) -> None:
        if not self.config.execute_next_bar:
            self.warnings.append(
                "execute_next_bar is disabled: orders filled on the same bar that produced "
                "the signal. This is lookahead and inflates results."
            )
        if trades and total_bars:
            frequency = len(trades) / total_bars
            if frequency > self.config.overtrading_threshold:
                self.warnings.append(
                    f"traded on {frequency:.0%} of bars; fee assumptions dominate the result "
                    "at this frequency"
                )
        problems = self.portfolio.validate_invariants()
        if problems:
            self.warnings.append(
                "portfolio invariants were violated: " + "; ".join(problems)
            )
        if self.order_manager.is_halted:
            self.warnings.append(
                f"order manager halted during the run: {self.order_manager.halt_reason}"
            )


def _default_instrument(symbol: str) -> InstrumentSpec:
    """Reasonable venue metadata when the caller does not supply any."""
    for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return InstrumentSpec(
                symbol=symbol,
                base_asset=symbol[: -len(quote)],
                quote_asset=quote,
                tick_size=0.01,
                lot_size=0.00001,
                min_quantity=0.00001,
                min_notional=5.0,
                maker_fee=0.0002,
                taker_fee=0.00055,
            )
    return InstrumentSpec(
        symbol=symbol, base_asset=symbol, quote_asset="USDT", min_notional=5.0
    )


def split_candles(
    candles: Sequence[Candle], train_fraction: float = 0.7
) -> tuple[list[Candle], list[Candle]]:
    """Chronological train/test split.

    Chronological, never random: shuffling time series data leaks future information into the
    training set and is the most common way an overfit strategy passes validation.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    if len(candles) < 2:
        raise InsufficientDataError("Need at least two candles to split")
    cut = int(len(candles) * train_fraction)
    cut = max(1, min(cut, len(candles) - 1))
    return list(candles[:cut]), list(candles[cut:])


def date_range_slice(
    candles: Sequence[Candle], start: datetime, end: datetime
) -> list[Candle]:
    """Candles with ``start <= open_time < end``."""
    lower, upper = ensure_utc(start), ensure_utc(end)
    return [c for c in candles if lower <= c.open_time < upper]
