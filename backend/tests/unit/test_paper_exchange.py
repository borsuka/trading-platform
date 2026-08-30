"""Paper exchange simulator tests.

These are the tests that decide whether a backtest means anything. They check that money is
conserved, that fees and slippage are actually charged, that the venue's constraints are
enforced, and that ambiguous intrabar situations resolve pessimistically.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.core.domain import InstrumentSpec, OrderRequest
from app.core.enums import (
    InstrumentType,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    TimeInForce,
)
from app.exchanges.paper import PaperExchange, PaperExchangeConfig
from app.market_data.models import Candle, OrderBook, OrderBookLevel

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


@pytest.fixture
def exchange(spec: InstrumentSpec) -> PaperExchange:
    return PaperExchange(
        PaperExchangeConfig(starting_balance=100_000.0, base_slippage_bps=0.0),
        instruments={spec.symbol: spec},
    )


def candle(
    index: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 1_000.0,
    symbol: str = "BTCUSDT",
) -> Candle:
    return Candle(
        symbol=symbol,
        interval="1h",
        open_time=START + timedelta(hours=index),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        quote_volume=volume * close,
    )


def feed(exchange: PaperExchange, *candles: Candle) -> None:
    for c in candles:
        exchange.process_candle(c)


# --------------------------------------------------------------------------- #
# Setup and market orders
# --------------------------------------------------------------------------- #
class TestMarketOrders:
    async def test_buy_opens_long_position(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=1.0,
            )
        )
        assert order.status is OrderStatus.FILLED
        assert order.filled_quantity == pytest.approx(1.0)

        positions = await exchange.get_positions()
        assert len(positions) == 1
        assert positions[0].side is PositionSide.LONG
        assert positions[0].quantity == pytest.approx(1.0)

    async def test_taker_fee_is_charged(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=1.0,
            )
        )
        # notional 100 * taker fee 0.00055 = 0.055
        assert exchange.fees_paid == pytest.approx(0.055)
        assert exchange.cash == pytest.approx(100_000.0 - 0.055)

    async def test_round_trip_pnl_is_exact(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=1.0,
            )
        )
        feed(exchange, candle(1, 100, 111, 99, 110))
        await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.SELL,
                order_type=OrderType.MARKET, quantity=1.0, reduce_only=True,
            )
        )
        assert exchange.realized_pnl == pytest.approx(10.0)
        # Fees: 100 * 0.00055 + 110 * 0.00055
        expected_fees = 100 * 0.00055 + 110 * 0.00055
        assert exchange.fees_paid == pytest.approx(expected_fees)
        assert exchange.cash == pytest.approx(100_000.0 + 10.0 - expected_fees)
        assert await exchange.get_positions() == []

    async def test_short_then_cover(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.SELL,
                order_type=OrderType.MARKET, quantity=2.0,
            )
        )
        position = (await exchange.get_positions())[0]
        assert position.side is PositionSide.SHORT

        feed(exchange, candle(1, 100, 100, 89, 90))
        await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=2.0, reduce_only=True,
            )
        )
        assert exchange.realized_pnl == pytest.approx(20.0)

    async def test_position_flip_through_flat(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=1.0,
            )
        )
        await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.SELL,
                order_type=OrderType.MARKET, quantity=3.0,
            )
        )
        positions = await exchange.get_positions()
        assert len(positions) == 1
        assert positions[0].side is PositionSide.SHORT
        assert positions[0].quantity == pytest.approx(2.0)

    async def test_averaging_up_recomputes_entry(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=1.0,
            )
        )
        feed(exchange, candle(1, 100, 121, 99, 120))
        await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=1.0,
            )
        )
        position = (await exchange.get_positions())[0]
        assert position.quantity == pytest.approx(2.0)
        assert position.entry_price == pytest.approx(110.0)


# --------------------------------------------------------------------------- #
# Slippage
# --------------------------------------------------------------------------- #
class TestSlippage:
    async def test_buy_pays_above_mid(self, spec: InstrumentSpec) -> None:
        exchange = PaperExchange(
            PaperExchangeConfig(starting_balance=100_000.0, base_slippage_bps=10.0),
            instruments={spec.symbol: spec},
        )
        feed(exchange, candle(0, 100, 101, 99, 100))
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=1.0,
            )
        )
        assert order.average_fill_price > 100.0
        assert order.average_fill_price == pytest.approx(100.1, abs=0.05)

    async def test_sell_receives_below_mid(self, spec: InstrumentSpec) -> None:
        exchange = PaperExchange(
            PaperExchangeConfig(starting_balance=100_000.0, base_slippage_bps=10.0),
            instruments={spec.symbol: spec},
        )
        feed(exchange, candle(0, 100, 101, 99, 100))
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.SELL,
                order_type=OrderType.MARKET, quantity=1.0,
            )
        )
        assert order.average_fill_price < 100.0

    async def test_larger_orders_slip_more(self, spec: InstrumentSpec) -> None:
        def build() -> PaperExchange:
            ex = PaperExchange(
                PaperExchangeConfig(
                    starting_balance=10_000_000.0,
                    base_slippage_bps=1.0,
                    impact_coefficient_bps=500.0,
                ),
                instruments={spec.symbol: spec},
            )
            feed(ex, candle(0, 100, 101, 99, 100, volume=1_000))
            return ex

        small = build()
        big = build()
        small_order = await small.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=1.0,
            )
        )
        big_order = await big.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=200.0,
            )
        )
        assert big_order.average_fill_price > small_order.average_fill_price

    async def test_order_book_walk_is_used_when_available(
        self, exchange: PaperExchange
    ) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        exchange.set_order_book(
            OrderBook(
                symbol="BTCUSDT",
                timestamp=START + timedelta(hours=1),
                bids=(OrderBookLevel(price=99.9, quantity=10.0),),
                asks=(
                    OrderBookLevel(price=100.0, quantity=1.0),
                    OrderBookLevel(price=101.0, quantity=1.0),
                ),
            )
        )
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=2.0,
            )
        )
        # 1 @ 100 + 1 @ 101 -> average 100.5
        assert order.average_fill_price == pytest.approx(100.5)

    async def test_thin_book_gives_partial_fill(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        exchange.set_order_book(
            OrderBook(
                symbol="BTCUSDT",
                timestamp=START + timedelta(hours=1),
                bids=(OrderBookLevel(price=99.9, quantity=10.0),),
                asks=(OrderBookLevel(price=100.0, quantity=0.5),),
            )
        )
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=2.0,
            )
        )
        assert order.filled_quantity == pytest.approx(0.5)
        assert order.status is OrderStatus.CANCELLED


# --------------------------------------------------------------------------- #
# Limit orders
# --------------------------------------------------------------------------- #
class TestLimitOrders:
    async def test_resting_limit_does_not_fill_immediately(
        self, exchange: PaperExchange
    ) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.LIMIT, quantity=1.0, price=95.0,
            )
        )
        assert order.status is OrderStatus.OPEN
        assert order.filled_quantity == 0.0

    async def test_limit_fills_when_price_trades_through(
        self, exchange: PaperExchange
    ) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.LIMIT, quantity=1.0, price=95.0,
            )
        )
        feed(exchange, candle(1, 100, 100, 90, 92))
        assert order.status is OrderStatus.FILLED
        assert order.average_fill_price == pytest.approx(95.0)

    async def test_limit_partially_fills_when_only_touched(
        self, exchange: PaperExchange
    ) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.LIMIT, quantity=1.0, price=95.0,
            )
        )
        feed(exchange, candle(1, 100, 100, 95, 97))  # low exactly at the limit
        assert order.status is OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == pytest.approx(0.5)

    async def test_maker_fee_applied_to_resting_fill(
        self, exchange: PaperExchange
    ) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.LIMIT, quantity=1.0, price=95.0,
            )
        )
        feed(exchange, candle(1, 100, 100, 90, 92))
        assert exchange.fees_paid == pytest.approx(95.0 * 0.0002)

    async def test_crossing_limit_executes_as_taker(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.LIMIT, quantity=1.0, price=105.0,
            )
        )
        assert order.status is OrderStatus.FILLED
        assert exchange.fees_paid == pytest.approx(100.0 * 0.00055)

    async def test_post_only_rejects_instead_of_crossing(
        self, exchange: PaperExchange
    ) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.LIMIT, quantity=1.0, price=105.0,
                time_in_force=TimeInForce.POST_ONLY,
            )
        )
        assert order.status is OrderStatus.REJECTED
        assert "post_only" in (order.reject_reason or "")

    async def test_cancel_removes_from_open_orders(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.LIMIT, quantity=1.0, price=95.0,
            )
        )
        cancelled = await exchange.cancel_order(order.client_order_id)
        assert cancelled.status is OrderStatus.CANCELLED
        assert await exchange.get_open_orders() == []

    async def test_cancelling_terminal_order_is_not_an_error(
        self, exchange: PaperExchange
    ) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=1.0,
            )
        )
        again = await exchange.cancel_order(order.client_order_id)
        assert again.status is OrderStatus.FILLED


# --------------------------------------------------------------------------- #
# Stops and targets
# --------------------------------------------------------------------------- #
class TestStopsAndTargets:
    async def _open_long(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=1.0,
            )
        )

    async def test_stop_loss_closes_position(self, exchange: PaperExchange) -> None:
        await self._open_long(exchange)
        exchange.set_position_stops("BTCUSDT", stop_loss=95.0)
        feed(exchange, candle(1, 100, 100, 90, 92))
        assert await exchange.get_positions() == []
        assert exchange.realized_pnl < 0

    async def test_take_profit_closes_position(self, exchange: PaperExchange) -> None:
        await self._open_long(exchange)
        exchange.set_position_stops("BTCUSDT", take_profit=110.0)
        feed(exchange, candle(1, 100, 115, 99, 112))
        assert await exchange.get_positions() == []
        assert exchange.realized_pnl == pytest.approx(10.0)

    async def test_ambiguous_bar_resolves_to_the_stop(self, exchange: PaperExchange) -> None:
        """Both levels inside one bar: the stop must win, never the target."""
        await self._open_long(exchange)
        exchange.set_position_stops("BTCUSDT", stop_loss=95.0, take_profit=110.0)
        feed(exchange, candle(1, 100, 115, 90, 100))
        assert exchange.realized_pnl < 0

    async def test_optimistic_mode_can_be_selected(self, spec: InstrumentSpec) -> None:
        exchange = PaperExchange(
            PaperExchangeConfig(
                starting_balance=100_000.0,
                base_slippage_bps=0.0,
                pessimistic_intrabar=False,
            ),
            instruments={spec.symbol: spec},
        )
        await self._open_long(exchange)
        exchange.set_position_stops("BTCUSDT", stop_loss=95.0, take_profit=110.0)
        feed(exchange, candle(1, 100, 115, 90, 100))
        # Stops are still evaluated first in the ordering, so the stop fires; the flag only
        # controls whether the target is suppressed when both are hit.
        assert exchange.realized_pnl != 0

    async def test_trailing_stop_ratchets_up(self, exchange: PaperExchange) -> None:
        await self._open_long(exchange)
        exchange.set_position_stops("BTCUSDT", trail_offset=5.0)
        feed(exchange, candle(1, 100, 120, 100, 120))
        position = (await exchange.get_positions())[0]
        assert position.trailing_stop == pytest.approx(115.0)

        feed(exchange, candle(2, 120, 121, 110, 112))
        # Trailing stop stays at 115 and is hit by the 110 low.
        assert await exchange.get_positions() == []
        assert exchange.realized_pnl > 0

    async def test_stop_market_order_triggers(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.STOP_MARKET, quantity=1.0, trigger_price=105.0,
            )
        )
        assert order.status is OrderStatus.OPEN
        feed(exchange, candle(1, 100, 110, 99, 108))
        assert order.status is OrderStatus.FILLED

    async def test_invalid_stop_side_rejected(self, exchange: PaperExchange) -> None:
        await self._open_long(exchange)
        with pytest.raises(Exception, match="at or above"):
            exchange.set_position_stops("BTCUSDT", stop_loss=105.0)


# --------------------------------------------------------------------------- #
# Rejections and venue constraints
# --------------------------------------------------------------------------- #
class TestRejections:
    async def test_insufficient_balance_rejected(self, spec: InstrumentSpec) -> None:
        exchange = PaperExchange(
            PaperExchangeConfig(starting_balance=100.0), instruments={spec.symbol: spec}
        )
        feed(exchange, candle(0, 100, 101, 99, 100))
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=10.0,
            )
        )
        assert order.status is OrderStatus.REJECTED
        assert "Insufficient balance" in (order.reject_reason or "")
        assert exchange.cash == pytest.approx(100.0)

    async def test_below_min_notional_rejected(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=0.01,
            )
        )
        assert order.status is OrderStatus.REJECTED
        assert "notional" in (order.reject_reason or "").lower()

    async def test_sub_lot_quantity_rejected(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=0.0001,
            )
        )
        assert order.status is OrderStatus.REJECTED
        assert "rounds to zero" in (order.reject_reason or "")

    async def test_bad_tick_size_rejected(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.LIMIT, quantity=1.0, price=95.05,
            )
        )
        assert order.status is OrderStatus.REJECTED
        assert "tick size" in (order.reject_reason or "")

    async def test_excess_leverage_rejected(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=1.0, leverage=50.0,
            )
        )
        assert order.status is OrderStatus.REJECTED
        assert "Leverage" in (order.reject_reason or "")

    async def test_unknown_symbol_raises(self, exchange: PaperExchange) -> None:
        from app.core.exceptions import InstrumentNotSupportedError

        with pytest.raises(InstrumentNotSupportedError):
            await exchange.create_order(
                OrderRequest(
                    symbol="DOGEUSDT", side=OrderSide.BUY,
                    order_type=OrderType.MARKET, quantity=1.0,
                )
            )

    async def test_reduce_only_without_position_rejected(
        self, exchange: PaperExchange
    ) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.SELL,
                order_type=OrderType.MARKET, quantity=1.0, reduce_only=True,
            )
        )
        assert order.status is OrderStatus.REJECTED
        assert "no position" in (order.reject_reason or "")

    async def test_spot_shorting_rejected(self) -> None:
        spot = InstrumentSpec(
            symbol="ETHUSDT",
            base_asset="ETH",
            quote_asset="USDT",
            instrument_type=InstrumentType.SPOT,
            tick_size=0.01,
            lot_size=0.001,
            min_notional=10.0,
            max_leverage=1.0,
        )
        exchange = PaperExchange(
            PaperExchangeConfig(starting_balance=100_000.0), instruments={spot.symbol: spot}
        )
        feed(exchange, candle(0, 100, 101, 99, 100, symbol="ETHUSDT"))
        order = await exchange.create_order(
            OrderRequest(
                symbol="ETHUSDT", side=OrderSide.SELL,
                order_type=OrderType.MARKET, quantity=1.0,
            )
        )
        assert order.status is OrderStatus.REJECTED
        assert "short selling" in (order.reject_reason or "")

    async def test_rejections_are_recorded(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=0.01,
            )
        )
        assert len(exchange.rejections) == 1


# --------------------------------------------------------------------------- #
# Idempotency and accounting invariants
# --------------------------------------------------------------------------- #
class TestIdempotency:
    async def test_duplicate_client_order_id_does_not_double_execute(
        self, exchange: PaperExchange
    ) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        request = OrderRequest(
            symbol="BTCUSDT", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=1.0,
            client_order_id="fixed-id-1",
        )
        first = await exchange.create_order(request)
        second = await exchange.create_order(request)
        assert first is second
        positions = await exchange.get_positions()
        assert positions[0].quantity == pytest.approx(1.0)

    async def test_get_order_returns_none_for_unknown_id(
        self, exchange: PaperExchange
    ) -> None:
        assert await exchange.get_order("never-submitted") is None


class TestAccounting:
    async def test_margin_is_locked_and_released(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=10.0, leverage=5.0,
            )
        )
        # notional 1000 at 5x -> 200 margin
        assert exchange.locked_margin == pytest.approx(200.0)
        assert exchange.free_cash == pytest.approx(exchange.cash - 200.0)

        await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.SELL,
                order_type=OrderType.MARKET, quantity=10.0, reduce_only=True,
            )
        )
        assert exchange.locked_margin == pytest.approx(0.0, abs=1e-9)

    async def test_equity_tracks_unrealized_pnl(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=1.0,
            )
        )
        feed(exchange, candle(1, 100, 121, 99, 120))
        assert exchange.unrealized_pnl() == pytest.approx(20.0)
        assert exchange.equity() == pytest.approx(exchange.cash + 20.0)

    async def test_money_is_conserved_over_many_trades(
        self, exchange: PaperExchange
    ) -> None:
        """Cash must equal starting balance + realised PnL - fees, exactly."""
        price = 100.0
        for i in range(30):
            price = 100.0 + (i % 7) * 3.0
            feed(exchange, candle(i, price, price + 2, price - 2, price))
            side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
            await exchange.create_order(
                OrderRequest(
                    symbol="BTCUSDT", side=side,
                    order_type=OrderType.MARKET, quantity=1.0,
                )
            )
        expected = 100_000.0 + exchange.realized_pnl - exchange.fees_paid
        assert exchange.cash == pytest.approx(expected, abs=1e-6)

    async def test_locked_margin_never_negative(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        for i in range(10):
            await exchange.create_order(
                OrderRequest(
                    symbol="BTCUSDT",
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    order_type=OrderType.MARKET, quantity=1.0,
                )
            )
            assert exchange.locked_margin >= -1e-9

    async def test_balance_reports_free_and_locked(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=1.0, leverage=1.0,
            )
        )
        balance = await exchange.get_balance()
        usdt = balance.get("USDT")
        assert usdt.locked == pytest.approx(100.0)
        assert usdt.total == pytest.approx(exchange.cash)

    async def test_reset_restores_initial_state(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=1.0,
            )
        )
        exchange.reset()
        assert exchange.cash == pytest.approx(100_000.0)
        assert exchange.locked_margin == 0.0
        assert await exchange.get_positions() == []

    async def test_withdraw_respects_locked_margin(self, exchange: PaperExchange) -> None:
        from app.core.exceptions import InsufficientBalanceError

        feed(exchange, candle(0, 100, 101, 99, 100))
        await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=100.0, leverage=1.0,
            )
        )
        with pytest.raises(InsufficientBalanceError):
            await exchange.withdraw(95_000.0)


# --------------------------------------------------------------------------- #
# Safety surface
# --------------------------------------------------------------------------- #
class TestSafety:
    async def test_never_reports_itself_as_live(self, exchange: PaperExchange) -> None:
        assert exchange.is_live is False
        info = await exchange.get_info()
        assert info.testnet is True

    async def test_credentials_report_no_withdrawal(self, exchange: PaperExchange) -> None:
        permissions = await exchange.validate_credentials()
        assert permissions.can_withdraw is False
        assert permissions.is_safe_for_trading is True


# --------------------------------------------------------------------------- #
# Conditional-order trigger direction
# --------------------------------------------------------------------------- #
class TestConditionalTriggerDirection:
    """Regression tests for a bug that made every take-profit fire immediately.

    A sell stop-loss and a sell take-profit share a side but sit on opposite sides of the
    market. Deciding the trigger on side alone made every sell take-profit fire on the bar it
    was placed, which turned the simulator into a machine that only booked wins.
    """

    async def _open_long(self, exchange: PaperExchange) -> None:
        feed(exchange, candle(0, 100, 101, 99, 100))
        await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=1.0,
            )
        )

    async def test_sell_take_profit_does_not_fire_below_its_trigger(
        self, exchange: PaperExchange
    ) -> None:
        await self._open_long(exchange)
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.SELL,
                order_type=OrderType.TAKE_PROFIT_MARKET, quantity=1.0,
                trigger_price=120.0, reduce_only=True,
            )
        )
        # Price stays well below the target.
        feed(exchange, candle(1, 100, 105, 98, 102))
        assert order.status is OrderStatus.OPEN
        assert order.filled_quantity == 0.0
        assert len(await exchange.get_positions()) == 1

    async def test_sell_take_profit_fires_when_price_reaches_it(
        self, exchange: PaperExchange
    ) -> None:
        await self._open_long(exchange)
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.SELL,
                order_type=OrderType.TAKE_PROFIT_MARKET, quantity=1.0,
                trigger_price=120.0, reduce_only=True,
            )
        )
        feed(exchange, candle(1, 100, 125, 99, 122))
        assert order.status is OrderStatus.FILLED
        assert await exchange.get_positions() == []
        assert exchange.realized_pnl > 0

    async def test_sell_stop_loss_fires_when_price_falls(
        self, exchange: PaperExchange
    ) -> None:
        await self._open_long(exchange)
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.SELL,
                order_type=OrderType.STOP_MARKET, quantity=1.0,
                trigger_price=95.0, reduce_only=True,
            )
        )
        feed(exchange, candle(1, 100, 101, 90, 92))
        assert order.status is OrderStatus.FILLED
        assert exchange.realized_pnl < 0

    async def test_sell_stop_loss_does_not_fire_on_a_rally(
        self, exchange: PaperExchange
    ) -> None:
        await self._open_long(exchange)
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.SELL,
                order_type=OrderType.STOP_MARKET, quantity=1.0,
                trigger_price=95.0, reduce_only=True,
            )
        )
        feed(exchange, candle(1, 100, 130, 99, 128))
        assert order.status is OrderStatus.OPEN

    async def test_buy_take_profit_fires_when_price_falls(
        self, exchange: PaperExchange
    ) -> None:
        """Mirror image: covering a short profits when price drops."""
        feed(exchange, candle(0, 100, 101, 99, 100))
        await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.SELL,
                order_type=OrderType.MARKET, quantity=1.0,
            )
        )
        order = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY,
                order_type=OrderType.TAKE_PROFIT_MARKET, quantity=1.0,
                trigger_price=80.0, reduce_only=True,
            )
        )
        feed(exchange, candle(1, 100, 101, 78, 79))
        assert order.status is OrderStatus.FILLED
        assert exchange.realized_pnl > 0

    async def test_orphaned_protective_order_cannot_open_a_position(
        self, exchange: PaperExchange
    ) -> None:
        """The other half of the bracket must not become a new entry.

        After a stop-loss closes the position, the surviving take-profit is reduce-only with
        nothing to reduce. Filling it would open a short.
        """
        await self._open_long(exchange)
        stop = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.SELL,
                order_type=OrderType.STOP_MARKET, quantity=1.0,
                trigger_price=95.0, reduce_only=True,
            )
        )
        target = await exchange.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.SELL,
                order_type=OrderType.TAKE_PROFIT_MARKET, quantity=1.0,
                trigger_price=110.0, reduce_only=True,
            )
        )
        # A bar wide enough to touch both levels.
        feed(exchange, candle(1, 100, 115, 90, 100))

        assert stop.status is OrderStatus.FILLED
        assert target.status is OrderStatus.CANCELLED
        assert await exchange.get_positions() == [], "an orphaned bracket leg opened a position"
