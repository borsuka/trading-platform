"""Order manager and portfolio tests.

The centrepiece is the timeout-recovery suite: a submission that fails ambiguously must never
result in a second order. Those tests use a fault-injecting exchange that can time out *after*
recording the order, which is the exact real-world scenario that causes doubled positions.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.core.domain import (
    Fill,
    InstrumentSpec,
    Order,
    OrderRequest,
    Position,
)
from app.core.enums import (
    ExitReason,
    InstrumentType,
    LiquidityRole,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    RiskDecision,
)
from app.core.exceptions import (
    ExchangeConnectionError,
    ExchangeError,
    ExchangeTimeoutError,
    RiskViolationError,
)
from app.exchanges.paper import PaperExchange, PaperExchangeConfig
from app.execution.order_manager import (
    ExecutionResult,
    OrderManager,
    RetryPolicy,
    make_client_order_id,
)
from app.market_data.models import Candle
from app.portfolio.manager import PortfolioManager
from app.risk.manager import RiskAssessment

START = datetime(2024, 1, 1, tzinfo=UTC)


@pytest.fixture
def spec() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        instrument_type=InstrumentType.LINEAR_PERPETUAL,
        tick_size=0.1,
        lot_size=0.001,
        min_quantity=0.001,
        min_notional=10.0,
        max_leverage=10.0,
        maker_fee=0.0002,
        taker_fee=0.00055,
    )


class FaultyExchange(PaperExchange):
    """Paper exchange that can inject transport faults.

    ``fail_after_recording`` reproduces the dangerous case: the venue accepted the order, but
    the client never learned that.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.timeout_on_create = 0
        self.fail_after_recording = False
        self.connection_error_on_create = 0
        self.query_failures = 0
        self.hide_orders = False
        self.create_calls = 0
        self.query_calls = 0

    async def create_order(self, request: OrderRequest) -> Order:
        self.create_calls += 1
        if self.connection_error_on_create > 0:
            self.connection_error_on_create -= 1
            if self.fail_after_recording:
                await super().create_order(request)
            raise ExchangeConnectionError("connection reset by peer")
        if self.timeout_on_create > 0:
            self.timeout_on_create -= 1
            if self.fail_after_recording:
                await super().create_order(request)
            raise ExchangeTimeoutError("request timed out after 10s")
        return await super().create_order(request)

    async def get_order(
        self, client_order_id: str, *, symbol: str | None = None
    ) -> Order | None:
        self.query_calls += 1
        if self.query_failures > 0:
            self.query_failures -= 1
            raise ExchangeError("venue unavailable")
        if self.hide_orders:
            return None
        return await super().get_order(client_order_id, symbol=symbol)


@pytest.fixture
def exchange(spec: InstrumentSpec) -> FaultyExchange:
    ex = FaultyExchange(
        PaperExchangeConfig(starting_balance=100_000.0, base_slippage_bps=0.0),
        instruments={spec.symbol: spec},
    )
    ex.process_candle(
        Candle(
            symbol="BTCUSDT", interval="1h", open_time=START,
            open=100.0, high=101.0, low=99.0, close=100.0, volume=1_000.0,
        )
    )
    return ex


@pytest.fixture
def manager(exchange: FaultyExchange) -> OrderManager:
    return OrderManager(
        exchange,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_seconds=0.0),
        state_query_delay=0.0,
    )


def approved() -> RiskAssessment:
    return RiskAssessment(decision=RiskDecision.APPROVED, quantity=1.0, reason="ok")


def rejected() -> RiskAssessment:
    return RiskAssessment(decision=RiskDecision.REJECTED, reason="limit breached")


def market_request(quantity: float = 1.0, client_id: str | None = None) -> OrderRequest:
    return OrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=quantity,
        client_order_id=client_id or make_client_order_id("test"),
    )


# =========================================================================== #
# Basic submission
# =========================================================================== #
class TestSubmission:
    async def test_successful_submission(self, manager: OrderManager) -> None:
        result = await manager.submit(market_request(), assessment=approved())
        assert result.succeeded
        assert result.order is not None
        assert result.order.status is OrderStatus.FILLED
        assert manager.stats.submitted == 1

    async def test_rejected_assessment_blocks_submission(
        self, manager: OrderManager, exchange: FaultyExchange
    ) -> None:
        with pytest.raises(RiskViolationError):
            await manager.submit(market_request(), assessment=rejected())
        assert exchange.create_calls == 0

    async def test_venue_rejection_is_reported_not_raised(
        self, manager: OrderManager
    ) -> None:
        # Below the 10.0 minimum notional.
        result = await manager.submit(market_request(quantity=0.01), assessment=approved())
        assert not result.succeeded
        assert "notional" in result.reason.lower()

    async def test_open_position_helper(self, manager: OrderManager) -> None:
        result = await manager.open_position(
            "BTCUSDT", OrderSide.BUY, 1.0, assessment=approved()
        )
        assert result.succeeded
        assert result.order is not None
        assert result.order.client_order_id.startswith("open-")

    async def test_client_order_id_fits_venue_limits(self) -> None:
        assert len(make_client_order_id("open")) <= 32


# =========================================================================== #
# Idempotency - the critical suite
# =========================================================================== #
class TestIdempotency:
    async def test_same_client_id_is_never_submitted_twice(
        self, manager: OrderManager, exchange: FaultyExchange
    ) -> None:
        request = market_request(client_id="fixed-1")
        first = await manager.submit(request, assessment=approved())
        second = await manager.submit(request, assessment=approved())
        assert first.succeeded
        assert not second.submitted
        assert manager.stats.duplicates_prevented == 1
        assert exchange.create_calls == 1

    async def test_timeout_after_venue_accepted_does_not_double_submit(
        self, manager: OrderManager, exchange: FaultyExchange
    ) -> None:
        """The scenario that doubles positions in naive implementations."""
        exchange.timeout_on_create = 1
        exchange.fail_after_recording = True

        result = await manager.submit(market_request(quantity=1.0), assessment=approved())

        assert result.recovered_from_timeout
        assert result.succeeded
        assert result.order is not None
        assert result.order.status is OrderStatus.FILLED
        # Exactly one order reached the venue, and the position reflects one order.
        positions = await exchange.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity == pytest.approx(1.0)
        assert manager.stats.timeouts == 1
        assert manager.stats.timeouts_recovered == 1

    async def test_timeout_before_venue_saw_it_reports_safe_to_resubmit(
        self, manager: OrderManager, exchange: FaultyExchange
    ) -> None:
        exchange.timeout_on_create = 1
        exchange.fail_after_recording = False

        result = await manager.submit(market_request(), assessment=approved())

        assert not result.submitted
        assert result.recovered_from_timeout
        assert "safe to submit a new order" in result.reason
        assert await exchange.get_positions() == []

    async def test_connection_error_is_treated_as_ambiguous(
        self, manager: OrderManager, exchange: FaultyExchange
    ) -> None:
        exchange.connection_error_on_create = 1
        exchange.fail_after_recording = True
        result = await manager.submit(market_request(), assessment=approved())
        assert result.recovered_from_timeout
        assert result.succeeded
        assert len(await exchange.get_positions()) == 1

    async def test_unresolvable_state_halts_the_manager(
        self, manager: OrderManager, exchange: FaultyExchange
    ) -> None:
        """If the venue cannot be asked, the bot stops rather than guessing."""
        exchange.timeout_on_create = 1
        exchange.fail_after_recording = True
        exchange.query_failures = 99

        result = await manager.submit(market_request(), assessment=approved())

        assert result.requires_halt
        assert manager.is_halted
        assert result.order is not None
        assert result.order.status is OrderStatus.UNKNOWN
        assert manager.stats.halts == 1

    async def test_halted_manager_refuses_further_orders(
        self, manager: OrderManager, exchange: FaultyExchange
    ) -> None:
        exchange.timeout_on_create = 1
        exchange.fail_after_recording = True
        exchange.query_failures = 99
        await manager.submit(market_request(), assessment=approved())

        follow_up = await manager.submit(market_request(), assessment=approved())
        assert not follow_up.submitted
        assert "halted" in follow_up.reason

    async def test_halt_requires_attributable_clearance(
        self, manager: OrderManager, exchange: FaultyExchange
    ) -> None:
        exchange.timeout_on_create = 1
        exchange.query_failures = 99
        await manager.submit(market_request(), assessment=approved())
        with pytest.raises(ValueError, match="cleared_by is required"):
            manager.clear_halt(cleared_by="")
        manager.clear_halt(cleared_by="operator@example.com")
        assert not manager.is_halted

    async def test_state_query_retries_before_halting(
        self, manager: OrderManager, exchange: FaultyExchange
    ) -> None:
        exchange.timeout_on_create = 1
        exchange.fail_after_recording = True
        exchange.query_failures = 2  # recovers on the third attempt

        result = await manager.submit(market_request(), assessment=approved())
        assert result.succeeded
        assert not manager.is_halted
        assert exchange.query_calls == 3


# =========================================================================== #
# Cancellation and sync
# =========================================================================== #
class TestCancellation:
    async def test_cancel_marks_order_cancelled(self, manager: OrderManager) -> None:
        request = OrderRequest(
            symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            quantity=1.0, price=90.0,
        )
        await manager.submit(request, assessment=approved())
        cancelled = await manager.cancel(request.client_order_id, symbol="BTCUSDT")
        assert cancelled is not None
        assert cancelled.status is OrderStatus.CANCELLED
        assert manager.stats.cancelled == 1

    async def test_cancel_all(self, manager: OrderManager) -> None:
        for price in (90.0, 91.0, 92.0):
            await manager.submit(
                OrderRequest(
                    symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                    quantity=1.0, price=price,
                ),
                assessment=approved(),
            )
        cancelled = await manager.cancel_all("BTCUSDT")
        assert len(cancelled) == 3

    async def test_refresh_reads_venue_state(self, manager: OrderManager) -> None:
        request = market_request()
        await manager.submit(request, assessment=approved())
        refreshed = await manager.refresh(request.client_order_id, symbol="BTCUSDT")
        assert refreshed is not None
        assert refreshed.status is OrderStatus.FILLED

    async def test_prune_drops_old_terminal_orders(self, manager: OrderManager) -> None:
        for _ in range(5):
            await manager.submit(market_request(quantity=0.1), assessment=approved())
        removed = manager.prune(keep=2)
        assert removed >= 0
        assert len(manager.known_orders()) <= 5

    async def test_protective_orders_are_reduce_only(
        self, manager: OrderManager
    ) -> None:
        await manager.submit(market_request(), assessment=approved())
        position = Position(
            symbol="BTCUSDT", side=PositionSide.LONG, quantity=1.0,
            entry_price=100.0, mark_price=100.0,
        )
        results = await manager.place_protective_orders(
            position, stop_loss=95.0, take_profit=110.0
        )
        assert len(results) == 2
        for result in results:
            assert result.order is not None
            assert result.order.reduce_only is True


# =========================================================================== #
# Portfolio manager
# =========================================================================== #
def fill(
    side: OrderSide,
    quantity: float,
    price: float,
    *,
    fee: float = 0.0,
    at: datetime = START,
    symbol: str = "BTCUSDT",
) -> Fill:
    return Fill(
        fill_id=f"f-{side.value}-{quantity}-{price}-{at.timestamp()}",
        order_id="o-1",
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        fee=fee,
        fee_asset="USDT",
        role=LiquidityRole.TAKER,
        timestamp=at,
    )


class TestPortfolioManager:
    def test_opening_a_position(self) -> None:
        portfolio = PortfolioManager(starting_balance=10_000.0)
        portfolio.apply_fill(fill(OrderSide.BUY, 1.0, 100.0, fee=0.055))
        position = portfolio.position("BTCUSDT")
        assert position is not None
        assert position.side is PositionSide.LONG
        assert portfolio.cash == pytest.approx(10_000.0 - 0.055)
        assert portfolio.locked_margin == pytest.approx(100.0)

    def test_round_trip_produces_a_trade(self) -> None:
        portfolio = PortfolioManager(starting_balance=10_000.0)
        portfolio.apply_fill(fill(OrderSide.BUY, 1.0, 100.0))
        trade = portfolio.apply_fill(
            fill(OrderSide.SELL, 1.0, 110.0, at=START + timedelta(hours=1)),
            exit_reason=ExitReason.TAKE_PROFIT,
        )
        assert trade is not None
        assert trade.net_pnl == pytest.approx(10.0)
        assert trade.exit_reason is ExitReason.TAKE_PROFIT
        assert portfolio.realized_pnl == pytest.approx(10.0)
        assert portfolio.position("BTCUSDT") is None

    def test_partial_close_keeps_position_open(self) -> None:
        portfolio = PortfolioManager(starting_balance=10_000.0)
        portfolio.apply_fill(fill(OrderSide.BUY, 2.0, 100.0))
        trade = portfolio.apply_fill(fill(OrderSide.SELL, 1.0, 110.0))
        assert trade is None
        position = portfolio.position("BTCUSDT")
        assert position is not None and position.quantity == pytest.approx(1.0)
        assert portfolio.realized_pnl == pytest.approx(10.0)

    def test_averaging_recomputes_entry(self) -> None:
        portfolio = PortfolioManager(starting_balance=100_000.0)
        portfolio.apply_fill(fill(OrderSide.BUY, 1.0, 100.0))
        portfolio.apply_fill(fill(OrderSide.BUY, 1.0, 120.0))
        position = portfolio.position("BTCUSDT")
        assert position is not None
        assert position.entry_price == pytest.approx(110.0)

    def test_flip_through_flat(self) -> None:
        portfolio = PortfolioManager(starting_balance=100_000.0)
        portfolio.apply_fill(fill(OrderSide.BUY, 1.0, 100.0))
        portfolio.apply_fill(fill(OrderSide.SELL, 3.0, 110.0))
        position = portfolio.position("BTCUSDT")
        assert position is not None
        assert position.side is PositionSide.SHORT
        assert position.quantity == pytest.approx(2.0)

    def test_equity_tracks_marks(self) -> None:
        portfolio = PortfolioManager(starting_balance=10_000.0)
        portfolio.apply_fill(fill(OrderSide.BUY, 1.0, 100.0))
        portfolio.mark("BTCUSDT", 120.0)
        assert portfolio.unrealized_pnl() == pytest.approx(20.0)
        assert portfolio.equity() == pytest.approx(10_020.0)

    def test_drawdown_from_peak(self) -> None:
        portfolio = PortfolioManager(starting_balance=10_000.0)
        portfolio.apply_fill(fill(OrderSide.BUY, 1.0, 100.0))
        portfolio.mark("BTCUSDT", 200.0)
        portfolio.snapshot()
        portfolio.mark("BTCUSDT", 150.0)
        assert portfolio.drawdown() == pytest.approx(50.0 / 10_100.0, rel=1e-3)

    def test_invariants_hold_after_many_fills(self) -> None:
        portfolio = PortfolioManager(starting_balance=100_000.0)
        for i in range(40):
            side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
            price = 100.0 + (i % 9)
            portfolio.apply_fill(
                fill(side, 1.0, price, fee=price * 0.00055,
                     at=START + timedelta(hours=i))
            )
        assert portfolio.validate_invariants() == []

    def test_invariant_violation_is_detected(self) -> None:
        portfolio = PortfolioManager(starting_balance=10_000.0)
        portfolio._cash = -50.0  # simulate corruption
        problems = portfolio.validate_invariants()
        assert any("negative" in p for p in problems)

    def test_statistics(self) -> None:
        portfolio = PortfolioManager(starting_balance=10_000.0)
        portfolio.apply_fill(fill(OrderSide.BUY, 1.0, 100.0))
        portfolio.apply_fill(fill(OrderSide.SELL, 1.0, 110.0))
        portfolio.apply_fill(fill(OrderSide.BUY, 1.0, 100.0, at=START + timedelta(hours=2)))
        portfolio.apply_fill(fill(OrderSide.SELL, 1.0, 95.0, at=START + timedelta(hours=3)))
        stats = portfolio.statistics()
        assert stats["closed_trades"] == 2
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["win_rate"] == pytest.approx(0.5)
        assert stats["profit_factor"] == pytest.approx(2.0)

    def test_to_view_is_a_projection(self) -> None:
        portfolio = PortfolioManager(starting_balance=10_000.0)
        portfolio.apply_fill(fill(OrderSide.BUY, 1.0, 100.0))
        view = portfolio.to_view()
        assert view.equity == pytest.approx(portfolio.equity())
        assert view.position_count == 1

    def test_halt_and_resume(self) -> None:
        portfolio = PortfolioManager(starting_balance=10_000.0)
        portfolio.halt("test")
        assert not portfolio.trading_enabled
        with pytest.raises(ValueError, match="resumed_by is required"):
            portfolio.resume(resumed_by="")
        portfolio.resume(resumed_by="operator")
        assert portfolio.trading_enabled


# =========================================================================== #
# Reconciliation
# =========================================================================== #
class TestReconciliation:
    async def test_matching_state_is_clean(
        self, exchange: FaultyExchange, manager: OrderManager
    ) -> None:
        portfolio = PortfolioManager(starting_balance=100_000.0)
        result = await manager.submit(market_request(), assessment=approved())
        assert result.order is not None
        for f in result.order.fills:
            portfolio.apply_fill(f)

        report = await portfolio.reconcile(exchange)
        assert not report.discrepancies
        assert portfolio.trading_enabled

    async def test_position_missing_locally_halts(
        self, exchange: FaultyExchange, manager: OrderManager
    ) -> None:
        """The venue has a position the platform does not know about."""
        portfolio = PortfolioManager(starting_balance=100_000.0)
        await manager.submit(market_request(), assessment=approved())

        report = await portfolio.reconcile(exchange)
        assert report.discrepancies
        assert report.discrepancies[0].kind == "missing_locally"
        assert not portfolio.trading_enabled
        assert "missing_locally" in (portfolio.halt_reason or "")

    async def test_position_missing_on_exchange_halts(
        self, exchange: FaultyExchange
    ) -> None:
        portfolio = PortfolioManager(starting_balance=100_000.0)
        portfolio.apply_fill(fill(OrderSide.BUY, 1.0, 100.0))
        report = await portfolio.reconcile(exchange)
        assert report.discrepancies[0].kind == "missing_on_exchange"
        assert not portfolio.trading_enabled

    async def test_quantity_mismatch_halts(
        self, exchange: FaultyExchange, manager: OrderManager
    ) -> None:
        portfolio = PortfolioManager(starting_balance=100_000.0)
        await manager.submit(market_request(quantity=2.0), assessment=approved())
        portfolio.apply_fill(fill(OrderSide.BUY, 1.0, 100.0))  # local thinks 1, venue has 2

        report = await portfolio.reconcile(exchange)
        assert report.discrepancies[0].kind == "quantity"
        assert report.discrepancies[0].difference == pytest.approx(1.0)
        assert not portfolio.trading_enabled

    async def test_unreachable_venue_halts(self, spec: InstrumentSpec) -> None:
        class DeadExchange(PaperExchange):
            async def get_positions(self, symbol: str | None = None) -> list[Position]:
                raise ExchangeError("venue unreachable")

        portfolio = PortfolioManager(starting_balance=100_000.0)
        report = await portfolio.reconcile(
            DeadExchange(instruments={spec.symbol: spec})
        )
        assert report.error is not None
        assert not portfolio.trading_enabled

    async def test_adoption_requires_attribution(self) -> None:
        portfolio = PortfolioManager(starting_balance=100_000.0)
        with pytest.raises(ValueError, match="adopted_by is required"):
            portfolio.adopt_exchange_state([], adopted_by="")

    async def test_adoption_replaces_local_state(
        self, exchange: FaultyExchange, manager: OrderManager
    ) -> None:
        portfolio = PortfolioManager(starting_balance=100_000.0)
        await manager.submit(market_request(), assessment=approved())
        venue_positions = await exchange.get_positions()

        portfolio.adopt_exchange_state(venue_positions, adopted_by="operator@example.com")
        assert portfolio.position("BTCUSDT") is not None
        report = await portfolio.reconcile(exchange)
        assert not report.discrepancies


# =========================================================================== #
# Execution result helpers
# =========================================================================== #
def test_execution_result_semantics() -> None:
    assert not ExecutionResult(order=None, submitted=False).succeeded
    order = Order(
        client_order_id="x", symbol="BTCUSDT", side=OrderSide.BUY,
        order_type=OrderType.MARKET, quantity=1.0, status=OrderStatus.FILLED,
    )
    result = ExecutionResult(order=order, submitted=True)
    assert result.succeeded and result.filled
