"""Stateful paper-trading exchange simulator.

This is a real simulator, not a stub that returns ``{"status": "FILLED"}``. It maintains cash,
margin, positions and an order book of resting orders, and it advances deterministically as
market data arrives. The same object backs both paper trading and the backtester, which is what
makes a backtest a meaningful prediction of paper behaviour.

Modelled explicitly:

* market, limit, stop-market, stop-limit, take-profit and trailing-stop orders
* maker/taker fees, charged separately
* slippage — from the real book when one is supplied, otherwise from a size-vs-volume model
* partial fills when a limit order is only touched rather than traded through
* margin reservation, free/locked balances and rejection on insufficient funds
* venue constraints: tick size, lot size, minimum notional, maximum leverage
* the full order lifecycle including cancellation and rejection

Accounting model
----------------
One cash account denominated in the quote asset. Opening a position reserves
``notional / leverage`` as margin; closing releases it and settles realised PnL into cash. Spot
instruments are simply the ``leverage == 1`` case with shorting disabled, which makes spot and
perpetual accounting share one code path.

**Intrabar ordering.** Within a single candle the true path of price is unknown. This simulator
resolves ambiguity pessimistically: when both the stop and the take-profit of a position lie
inside one bar's range, the **stop** is filled. Optimistic resolution is the single most common
way a backtest manufactures returns that do not exist.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime

from app.core.clock import utcnow
from app.core.domain import (
    AccountBalance,
    Balance,
    Fill,
    InstrumentSpec,
    Order,
    OrderRequest,
    Position,
    new_id,
)
from app.core.enums import (
    InstrumentType,
    LiquidityRole,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    TimeInForce,
)
from app.core.exceptions import (
    InstrumentNotSupportedError,
    InsufficientBalanceError,
    OrderRejectedError,
)
from app.core.logging import get_logger
from app.core.numeric import EPSILON, bps, is_zero, safe_divide
from app.exchanges.base import (
    AccountPermissions,
    ExchangeAdapter,
    ExchangeInfo,
)
from app.market_data.models import Candle, OrderBook, PublicTrade, Ticker

logger = get_logger(__name__)


@dataclass(slots=True)
class PaperExchangeConfig:
    """Simulation realism knobs.

    Defaults are intentionally pessimistic. A simulator that flatters the strategy is worse than
    no simulator, because it produces confident wrong decisions.
    """

    starting_balance: float = 10_000.0
    quote_asset: str = "USDT"
    #: Baseline market-order slippage when no order book is available.
    base_slippage_bps: float = 3.0
    #: Extra slippage per unit of (order notional / bar quote volume). Models market impact.
    impact_coefficient_bps: float = 250.0
    #: Cap on modelled slippage, so a thin bar cannot produce an absurd fill.
    max_slippage_bps: float = 200.0
    #: Fraction of a resting limit order that fills when price merely touches its level.
    touch_fill_ratio: float = 0.5
    #: Reject market orders larger than this fraction of the bar's volume.
    max_volume_participation: float = 0.25
    #: Charge maker fees to limit orders that rest before filling.
    model_maker_fees: bool = True
    #: Simulated round-trip latency; recorded on fills for reporting.
    latency_ms: float = 50.0
    #: When true, a bar containing both stop and target fills the stop (pessimistic).
    pessimistic_intrabar: bool = True


@dataclass(slots=True)
class MarketState:
    """Latest known state for one symbol inside the simulator."""

    symbol: str
    last_price: float
    high: float
    low: float
    open: float
    volume: float
    quote_volume: float
    timestamp: datetime
    order_book: OrderBook | None = None
    bid: float | None = None
    ask: float | None = None

    @property
    def best_bid(self) -> float:
        if self.order_book is not None and self.order_book.best_bid is not None:
            return self.order_book.best_bid
        return self.bid if self.bid is not None else self.last_price

    @property
    def best_ask(self) -> float:
        if self.order_book is not None and self.order_book.best_ask is not None:
            return self.order_book.best_ask
        return self.ask if self.ask is not None else self.last_price


class PaperExchange(ExchangeAdapter):
    """In-process exchange simulator.

    Drive it with :meth:`process_candle` (backtests, bar-close paper trading) or
    :meth:`process_ticker` (tick-level paper trading). Both trigger resting-order evaluation.
    """

    name = "paper"
    is_live = False

    def __init__(
        self,
        config: PaperExchangeConfig | None = None,
        instruments: dict[str, InstrumentSpec] | None = None,
    ) -> None:
        self.config = config or PaperExchangeConfig()
        self._instruments: dict[str, InstrumentSpec] = dict(instruments or {})
        self._cash: float = self.config.starting_balance
        self._locked_margin: float = 0.0
        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}
        self._market: dict[str, MarketState] = {}
        self._fills: list[Fill] = []
        self._connected = False
        self._now: datetime = utcnow()
        # Whether `get_info()` reports wall-clock time instead of simulated time.
        #
        # `_now` tracks the last bar the simulator was fed, which is what a replay or a
        # backtest needs: simulated time must follow the data, not the wall. But a bot running
        # against a live feed uses the real clock, and the drift check compares the two. With
        # simulated time on one side and real time on the other it measures how old the last
        # bar is - up to a full interval - and reports normal bar lag as a clock fault that
        # blocks trading. The live feed sets this flag; replay leaves it alone.
        self.follow_wall_clock = False
        self._realized_pnl = 0.0
        self._fees_paid = 0.0
        self._rejections: list[tuple[datetime, str, str]] = []
        self._candle_queues: dict[str, asyncio.Queue[Candle]] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def connect(self) -> None:
        self._connected = True
        logger.info(
            "paper_exchange.connected",
            starting_balance=self.config.starting_balance,
            quote_asset=self.config.quote_asset,
            instruments=len(self._instruments),
        )

    async def close(self) -> None:
        self._connected = False

    async def get_info(self) -> ExchangeInfo:
        return ExchangeInfo(
            name=self.name,
            testnet=True,
            server_time=utcnow() if self.follow_wall_clock else self._now,
            supports_websocket=True,
            supports_leverage=True,
            rate_limit_per_minute=1_000_000,
        )

    async def validate_credentials(self) -> AccountPermissions:
        """The simulator has no credentials; it reports the safe permission set."""
        return AccountPermissions(can_read=True, can_trade=True, can_withdraw=False)

    # ------------------------------------------------------------------ #
    # Instruments
    # ------------------------------------------------------------------ #
    def register_instrument(self, spec: InstrumentSpec) -> None:
        self._instruments[spec.symbol] = spec

    async def get_instruments(self) -> dict[str, InstrumentSpec]:
        return dict(self._instruments)

    async def get_instrument(self, symbol: str) -> InstrumentSpec:
        return self._require_instrument(symbol)

    def _require_instrument(self, symbol: str) -> InstrumentSpec:
        spec = self._instruments.get(symbol)
        if spec is None:
            raise InstrumentNotSupportedError(
                f"{symbol} is not registered with the paper exchange",
                context={"symbol": symbol, "known": sorted(self._instruments)},
            )
        return spec

    # ------------------------------------------------------------------ #
    # Feeding market data
    # ------------------------------------------------------------------ #
    def process_candle(self, candle: Candle) -> list[Fill]:
        """Advance simulated time by one bar and settle any orders it triggers."""
        spec = self._instruments.get(candle.symbol)
        if spec is None:
            self.register_instrument(_infer_instrument(candle.symbol))

        self._now = candle.close_time
        self._market[candle.symbol] = MarketState(
            symbol=candle.symbol,
            last_price=candle.close,
            high=candle.high,
            low=candle.low,
            open=candle.open,
            volume=candle.volume,
            quote_volume=candle.quote_volume or candle.volume * candle.close,
            timestamp=candle.close_time,
        )
        fills = self._process_resting_orders(candle.symbol, bar=candle)
        self._mark_positions(candle.symbol, candle.close)
        queue = self._candle_queues.get(f"{candle.symbol}:{candle.interval}")
        if queue is not None:
            queue.put_nowait(candle)
        return fills

    def process_ticker(self, ticker: Ticker, book: OrderBook | None = None) -> list[Fill]:
        """Advance on a tick. Uses the tick price as both bar high and low."""
        if ticker.symbol not in self._instruments:
            self.register_instrument(_infer_instrument(ticker.symbol))
        self._now = ticker.timestamp
        previous = self._market.get(ticker.symbol)
        self._market[ticker.symbol] = MarketState(
            symbol=ticker.symbol,
            last_price=ticker.price,
            high=ticker.price,
            low=ticker.price,
            open=previous.last_price if previous else ticker.price,
            volume=previous.volume if previous else 0.0,
            quote_volume=previous.quote_volume if previous else 0.0,
            timestamp=ticker.timestamp,
            order_book=book,
            bid=ticker.bid,
            ask=ticker.ask,
        )
        fills = self._process_resting_orders(ticker.symbol, bar=None)
        self._mark_positions(ticker.symbol, ticker.price)
        return fills

    def set_order_book(self, book: OrderBook) -> None:
        """Attach a real book so market orders are filled by walking it."""
        state = self._market.get(book.symbol)
        if state is None:
            mid = book.mid or 0.0
            if mid <= 0:
                return
            self._market[book.symbol] = MarketState(
                symbol=book.symbol,
                last_price=mid,
                high=mid,
                low=mid,
                open=mid,
                volume=0.0,
                quote_volume=0.0,
                timestamp=book.timestamp,
                order_book=book,
            )
        else:
            state.order_book = book

    # ------------------------------------------------------------------ #
    # Market data reads
    # ------------------------------------------------------------------ #
    async def get_ticker(self, symbol: str) -> Ticker:
        state = self._require_market(symbol)
        return Ticker(
            symbol=symbol,
            price=state.last_price,
            timestamp=state.timestamp,
            bid=state.best_bid,
            ask=state.best_ask,
            volume_24h=state.volume,
        )

    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        state = self._require_market(symbol)
        if state.order_book is not None:
            return state.order_book
        from app.market_data.models import OrderBookLevel

        spread = state.last_price * bps(2.0)
        unit = max(state.volume, 1.0) / 10.0
        return OrderBook(
            symbol=symbol,
            timestamp=state.timestamp,
            bids=tuple(
                OrderBookLevel(price=state.last_price - spread * (i + 1), quantity=unit)
                for i in range(depth)
            ),
            asks=tuple(
                OrderBookLevel(price=state.last_price + spread * (i + 1), quantity=unit)
                for i in range(depth)
            ),
        )

    async def get_candles(
        self,
        symbol: str,
        interval: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[Candle]:
        """The simulator does not store history; supply it from a historical provider."""
        return []

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[PublicTrade]:
        return [
            PublicTrade(
                symbol=fill.symbol,
                price=fill.price,
                quantity=fill.quantity,
                side=fill.side.value,
                timestamp=fill.timestamp,
                trade_id=fill.fill_id,
            )
            for fill in self._fills
            if fill.symbol == symbol
        ][-limit:]

    def _require_market(self, symbol: str) -> MarketState:
        state = self._market.get(symbol)
        if state is None:
            raise InstrumentNotSupportedError(
                f"No market data has been fed to the paper exchange for {symbol}",
                context={"symbol": symbol},
            )
        return state

    # ------------------------------------------------------------------ #
    # Account
    # ------------------------------------------------------------------ #
    async def get_balance(self) -> AccountBalance:
        balances = {
            self.config.quote_asset: Balance(
                asset=self.config.quote_asset,
                free=self.free_cash,
                locked=self._locked_margin,
            )
        }
        for symbol, position in self._positions.items():
            if not position.is_open:
                continue
            spec = self._instruments.get(symbol)
            if spec and spec.instrument_type is InstrumentType.SPOT:
                balances[spec.base_asset] = Balance(
                    asset=spec.base_asset, free=position.quantity, locked=0.0
                )
        return AccountBalance(balances=balances, timestamp=self._now)

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        positions = [p for p in self._positions.values() if p.is_open]
        if symbol is not None:
            positions = [p for p in positions if p.symbol == symbol]
        return positions

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        orders = [o for o in self._orders.values() if o.is_active]
        if symbol is not None:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    async def get_order(
        self, client_order_id: str, *, symbol: str | None = None
    ) -> Order | None:
        return self._orders.get(client_order_id)

    def order_metadata(self, client_order_id: str) -> dict[str, object]:
        """Metadata attached to an order, including ``auto_exit`` on simulator-generated
        stop-loss and take-profit exits. Lets callers attribute a fill without reaching into
        private state."""
        order = self._orders.get(client_order_id)
        return dict(order.metadata) if order is not None else {}

    # ------------------------------------------------------------------ #
    # Derived account metrics
    # ------------------------------------------------------------------ #
    @property
    def cash(self) -> float:
        """Total quote-asset balance including margin currently locked."""
        return self._cash

    @property
    def free_cash(self) -> float:
        return max(0.0, self._cash - self._locked_margin)

    @property
    def locked_margin(self) -> float:
        return self._locked_margin

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def fees_paid(self) -> float:
        return self._fees_paid

    @property
    def rejections(self) -> list[tuple[datetime, str, str]]:
        """``(timestamp, client_order_id, reason)`` for every rejected order."""
        return list(self._rejections)

    def unrealized_pnl(self) -> float:
        return sum(
            position.unrealized_pnl()
            for position in self._positions.values()
            if position.is_open
        )

    def equity(self) -> float:
        """Cash plus mark-to-market PnL on open positions."""
        return self._cash + self.unrealized_pnl()

    def total_exposure(self) -> float:
        return sum(
            position.notional()
            for position in self._positions.values()
            if position.is_open
        )

    # ------------------------------------------------------------------ #
    # Order entry
    # ------------------------------------------------------------------ #
    async def create_order(self, request: OrderRequest) -> Order:
        """Place an order. Idempotent on ``client_order_id``."""
        existing = self._orders.get(request.client_order_id)
        if existing is not None:
            logger.info(
                "paper_exchange.duplicate_order_ignored",
                client_order_id=request.client_order_id,
                status=existing.status.value,
            )
            return existing

        spec = self._require_instrument(request.symbol)
        order = Order.from_request(request)
        self._orders[order.client_order_id] = order

        rejection = self._validate_request(request, spec)
        if rejection is not None:
            return self._reject(order, rejection)

        order.status = OrderStatus.SUBMITTED
        order.exchange_order_id = f"paper-{new_id()[:12]}"
        order.updated_at = self._now

        if request.order_type is OrderType.MARKET:
            self._execute_market_order(order, spec, request)
        elif request.order_type is OrderType.LIMIT:
            self._place_or_cross_limit(order, spec, request)
        else:
            order.status = OrderStatus.OPEN  # conditional: rests until triggered

        return order

    def _validate_request(
        self, request: OrderRequest, spec: InstrumentSpec
    ) -> str | None:
        state = self._market.get(request.symbol)
        if state is None:
            return f"No market data available for {request.symbol}"

        quantity = spec.round_quantity(request.quantity)
        if is_zero(quantity):
            return (
                f"Quantity {request.quantity:g} rounds to zero at lot size {spec.lot_size:g}"
            )

        reference = request.price or state.last_price
        problem = spec.validate_order(quantity, reference)
        if problem is not None:
            return problem

        if request.price is not None and not is_zero(
            request.price - spec.round_price(request.price)
        ):
            return (
                f"Price {request.price:g} does not respect tick size {spec.tick_size:g}"
            )

        leverage = request.leverage or 1.0
        if leverage > spec.max_leverage + EPSILON:
            return f"Leverage {leverage:g}x exceeds venue maximum {spec.max_leverage:g}x"

        if spec.instrument_type is InstrumentType.SPOT and request.side is OrderSide.SELL:
            position = self._positions.get(request.symbol)
            held = position.quantity if position and position.side is PositionSide.LONG else 0.0
            if quantity > held + EPSILON:
                return (
                    f"Spot short selling is not supported: holding {held:g}, "
                    f"tried to sell {quantity:g}"
                )

        if request.reduce_only:
            position = self._positions.get(request.symbol)
            if position is None or not position.is_open:
                return "reduce_only order has no position to reduce"
            if position.side.closing_side is not request.side:
                return "reduce_only order is on the same side as the position"

        return self._check_affordability(request, spec, quantity, reference, leverage)

    def _check_affordability(
        self,
        request: OrderRequest,
        spec: InstrumentSpec,
        quantity: float,
        reference: float,
        leverage: float,
    ) -> str | None:
        """Reject orders the account cannot fund, before anything mutates."""
        if request.reduce_only:
            return None
        position = self._positions.get(request.symbol)
        if (
            position is not None
            and position.is_open
            and position.side.closing_side is request.side
        ):
            # Closing or reducing frees margin rather than consuming it.
            return None

        notional = spec.notional(quantity, reference)
        required = notional / max(leverage, 1.0)
        fee = notional * spec.taker_fee
        if required + fee > self.free_cash + EPSILON:
            return (
                f"Insufficient balance: need {required + fee:.2f} "
                f"{self.config.quote_asset} (margin {required:.2f} + fee {fee:.2f}), "
                f"have {self.free_cash:.2f}"
            )
        return None

    def _reject(self, order: Order, reason: str) -> Order:
        order.status = OrderStatus.REJECTED
        order.reject_reason = reason
        order.updated_at = self._now
        self._rejections.append((self._now, order.client_order_id, reason))
        logger.info(
            "paper_exchange.order_rejected",
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            reason=reason,
        )
        return order

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def _execute_market_order(
        self, order: Order, spec: InstrumentSpec, request: OrderRequest
    ) -> None:
        state = self._require_market(order.symbol)
        quantity = spec.round_quantity(order.quantity)
        order.quantity = quantity

        price, filled = self._market_fill_price(state, spec, order.side, quantity)
        if filled <= EPSILON:
            self._reject(order, "Order book too thin to fill at any price")
            return
        if filled < quantity - EPSILON:
            logger.info(
                "paper_exchange.partial_market_fill",
                symbol=order.symbol,
                requested=quantity,
                filled=filled,
            )
        self._settle_fill(
            order, spec, price=price, quantity=filled, role=LiquidityRole.TAKER,
            leverage=request.leverage or 1.0,
        )
        if filled < quantity - EPSILON:
            # A market order's unfilled remainder never rests on the book: it is cancelled.
            order.status = OrderStatus.CANCELLED

    def _market_fill_price(
        self,
        state: MarketState,
        spec: InstrumentSpec,
        side: OrderSide,
        quantity: float,
    ) -> tuple[float, float]:
        """Determine execution price and fillable quantity for a taker order."""
        if state.order_book is not None:
            avg, filled = state.order_book.simulate_market_fill(
                "buy" if side is OrderSide.BUY else "sell", quantity
            )
            if filled > EPSILON:
                return avg, filled

        # No book: model slippage from participation in the bar's volume.
        reference = state.best_ask if side is OrderSide.BUY else state.best_bid
        notional = quantity * reference
        participation = safe_divide(notional, state.quote_volume, default=0.0)
        if (
            state.quote_volume > 0
            and participation > self.config.max_volume_participation
        ):
            logger.info(
                "paper_exchange.excessive_participation",
                symbol=state.symbol,
                participation=round(participation, 4),
                limit=self.config.max_volume_participation,
            )
        slippage = min(
            self.config.base_slippage_bps
            + participation * self.config.impact_coefficient_bps,
            self.config.max_slippage_bps,
        )
        direction = 1.0 if side is OrderSide.BUY else -1.0
        price = reference * (1.0 + direction * bps(slippage))
        return spec.round_price(price), quantity

    def _place_or_cross_limit(
        self, order: Order, spec: InstrumentSpec, request: OrderRequest
    ) -> None:
        """A limit order that is already through the market executes immediately as a taker."""
        state = self._require_market(order.symbol)
        order.quantity = spec.round_quantity(order.quantity)
        price = order.price
        assert price is not None  # guaranteed by OrderRequest validation

        crosses = (
            order.side is OrderSide.BUY and price >= state.best_ask
        ) or (order.side is OrderSide.SELL and price <= state.best_bid)

        if crosses and request.time_in_force is not TimeInForce.POST_ONLY:
            fill_price = state.best_ask if order.side is OrderSide.BUY else state.best_bid
            self._settle_fill(
                order, spec, price=fill_price, quantity=order.quantity,
                role=LiquidityRole.TAKER, leverage=request.leverage or 1.0,
            )
            return
        if crosses and request.time_in_force is TimeInForce.POST_ONLY:
            self._reject(order, "post_only order would have crossed the spread")
            return
        order.status = OrderStatus.OPEN

    def _settle_fill(
        self,
        order: Order,
        spec: InstrumentSpec,
        *,
        price: float,
        quantity: float,
        role: LiquidityRole,
        leverage: float = 1.0,
    ) -> Fill:
        """Apply a fill: charge fees, update the position, move cash and margin."""
        fee_rate = spec.maker_fee if role is LiquidityRole.MAKER else spec.taker_fee
        notional = spec.notional(quantity, price)
        fee = notional * fee_rate

        fill = Fill(
            fill_id=f"paperfill-{new_id()[:12]}",
            order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            fee=fee,
            fee_asset=self.config.quote_asset,
            role=role,
            timestamp=self._now,
            is_maker=role is LiquidityRole.MAKER,
        )
        order.apply_fill(fill)
        self._fills.append(fill)

        self._cash -= fee
        self._fees_paid += fee
        self._apply_to_position(order, spec, fill, leverage)

        logger.debug(
            "paper_exchange.filled",
            symbol=order.symbol,
            side=order.side.value,
            quantity=quantity,
            price=price,
            fee=round(fee, 6),
            role=role.value,
        )
        return fill

    def _apply_to_position(
        self, order: Order, spec: InstrumentSpec, fill: Fill, leverage: float
    ) -> None:
        position = self._positions.get(order.symbol)
        incoming_side = PositionSide.from_side(fill.side)

        if position is None or not position.is_open:
            margin = spec.notional(fill.quantity, fill.price) / max(leverage, 1.0)
            self._locked_margin += margin
            new_position = Position(
                symbol=order.symbol,
                side=incoming_side,
                quantity=fill.quantity,
                entry_price=fill.price,
                opened_at=self._now,
                updated_at=self._now,
                leverage=max(leverage, 1.0),
                mark_price=fill.price,
                fees_paid=fill.fee,
                metadata={"margin": margin},
            )
            self._positions[order.symbol] = new_position
            return

        if position.side is incoming_side:
            margin = spec.notional(fill.quantity, fill.price) / max(position.leverage, 1.0)
            self._locked_margin += margin
            position.metadata["margin"] = position.metadata.get("margin", 0.0) + margin
            position.add(fill.quantity, fill.price)
            position.fees_paid += fill.fee
            return

        # Opposite side: reduce, and possibly flip.
        closing = min(fill.quantity, position.quantity)
        original_quantity = position.quantity
        released_fraction = safe_divide(closing, original_quantity)
        released_margin = position.metadata.get("margin", 0.0) * released_fraction
        realized = position.reduce(closing, fill.price)

        self._locked_margin = max(0.0, self._locked_margin - released_margin)
        position.metadata["margin"] = max(
            0.0, position.metadata.get("margin", 0.0) - released_margin
        )
        self._cash += realized
        self._realized_pnl += realized
        position.fees_paid += fill.fee

        remainder = fill.quantity - closing
        if not position.is_open:
            self._positions.pop(order.symbol, None)
            if remainder > EPSILON:  # flipped through flat
                margin = spec.notional(remainder, fill.price) / max(leverage, 1.0)
                self._locked_margin += margin
                self._positions[order.symbol] = Position(
                    symbol=order.symbol,
                    side=incoming_side,
                    quantity=remainder,
                    entry_price=fill.price,
                    opened_at=self._now,
                    updated_at=self._now,
                    leverage=max(leverage, 1.0),
                    mark_price=fill.price,
                    metadata={"margin": margin},
                )

    # ------------------------------------------------------------------ #
    # Resting orders
    # ------------------------------------------------------------------ #
    def _process_resting_orders(self, symbol: str, *, bar: Candle | None) -> list[Fill]:
        """Evaluate every resting order against the new bar or tick."""
        state = self._require_market(symbol)
        high = bar.high if bar else state.last_price
        low = bar.low if bar else state.last_price
        fills: list[Fill] = []

        resting = [
            order
            for order in self._orders.values()
            if order.symbol == symbol and order.is_active
        ]
        # Deterministic ordering: stops before targets, then by creation time. This makes the
        # pessimistic intrabar resolution actually pessimistic.
        resting.sort(key=lambda o: (0 if _is_stop(o.order_type) else 1, o.created_at))

        for order in resting:
            spec = self._require_instrument(order.symbol)
            before = len(self._fills)
            if order.order_type is OrderType.LIMIT:
                self._try_fill_limit(order, spec, high=high, low=low, bar=bar)
            elif order.order_type.is_conditional:
                self._try_trigger_conditional(order, spec, high=high, low=low, state=state)
            fills.extend(self._fills[before:])

        fills.extend(self._apply_position_stops(symbol, high=high, low=low, bar=bar))
        return fills

    def _try_fill_limit(
        self,
        order: Order,
        spec: InstrumentSpec,
        *,
        high: float,
        low: float,
        bar: Candle | None,
    ) -> None:
        price = order.price
        if price is None:
            return
        touched = (order.side is OrderSide.BUY and low <= price) or (
            order.side is OrderSide.SELL and high >= price
        )
        if not touched:
            return

        # Traded fully through the level -> full fill. Only touched -> partial.
        through = (order.side is OrderSide.BUY and low < price) or (
            order.side is OrderSide.SELL and high > price
        )
        ratio = 1.0 if through else self.config.touch_fill_ratio
        quantity = spec.round_quantity(order.remaining_quantity * ratio)
        if is_zero(quantity):
            return

        role = LiquidityRole.MAKER if self.config.model_maker_fees else LiquidityRole.TAKER
        self._settle_fill(order, spec, price=price, quantity=quantity, role=role)

    def _try_trigger_conditional(
        self,
        order: Order,
        spec: InstrumentSpec,
        *,
        high: float,
        low: float,
        state: MarketState,
    ) -> None:
        trigger = order.trigger_price
        if trigger is None:
            return

        # The trigger direction depends on the order TYPE as well as its side. A sell stop and
        # a sell take-profit sit on opposite sides of the market and fire on opposite moves:
        #
        #   sell stop-loss     -> price falls to the trigger   (low  <= trigger)
        #   sell take-profit   -> price rises to the trigger   (high >= trigger)
        #   buy  stop-loss     -> price rises to the trigger   (high >= trigger)
        #   buy  take-profit   -> price falls to the trigger   (low  <= trigger)
        #
        # Deciding on side alone makes every take-profit fire on the bar it is placed, which
        # turns a backtest into a machine that only ever books wins.
        is_take_profit = order.order_type in {
            OrderType.TAKE_PROFIT_MARKET,
            OrderType.TAKE_PROFIT_LIMIT,
        }
        rises_to_trigger = (
            order.side is OrderSide.BUY if not is_take_profit else order.side is OrderSide.SELL
        )
        triggered = high >= trigger if rises_to_trigger else low <= trigger
        if not triggered:
            return

        # A reduce-only conditional order has nothing to reduce once its sibling has closed the
        # position. Filling it would open a new position in the opposite direction.
        if order.reduce_only:
            position = self._positions.get(order.symbol)
            if position is None or not position.is_open:
                order.status = OrderStatus.CANCELLED
                order.updated_at = self._now
                logger.debug(
                    "paper_exchange.protective_order_expired",
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                )
                return

        if order.order_type in {OrderType.STOP_MARKET, OrderType.TAKE_PROFIT_MARKET}:
            # Stops fill at the trigger plus slippage - they are market orders once hit.
            direction = 1.0 if order.side is OrderSide.BUY else -1.0
            price = spec.round_price(
                trigger * (1.0 + direction * bps(self.config.base_slippage_bps))
            )
            self._settle_fill(
                order, spec, price=price, quantity=order.remaining_quantity,
                role=LiquidityRole.TAKER,
            )
        else:  # stop-limit / take-profit-limit become resting limit orders
            order.order_type = OrderType.LIMIT
            order.price = order.price or trigger
            order.status = OrderStatus.OPEN

    def _apply_position_stops(
        self, symbol: str, *, high: float, low: float, bar: Candle | None
    ) -> list[Fill]:
        """Fill attached stop-loss / take-profit levels on an open position."""
        position = self._positions.get(symbol)
        if position is None or not position.is_open:
            return []
        spec = self._require_instrument(symbol)
        stop = position.effective_stop()
        target = position.take_profit

        stop_hit = stop is not None and (
            (position.side is PositionSide.LONG and low <= stop)
            or (position.side is PositionSide.SHORT and high >= stop)
        )
        target_hit = target is not None and (
            (position.side is PositionSide.LONG and high >= target)
            or (position.side is PositionSide.SHORT and low <= target)
        )

        if stop_hit and target_hit and self.config.pessimistic_intrabar:
            target_hit = False  # both inside one bar: assume the stop came first

        if stop_hit and stop is not None:
            return [self._close_at(position, spec, stop, "stop_loss", slip=True)]
        if target_hit and target is not None:
            return [self._close_at(position, spec, target, "take_profit", slip=False)]
        return []

    def _close_at(
        self,
        position: Position,
        spec: InstrumentSpec,
        price: float,
        reason: str,
        *,
        slip: bool,
    ) -> Fill:
        """Flatten a position at ``price``, recording it as a synthetic reduce-only order."""
        side = position.side.closing_side
        if slip:
            direction = 1.0 if side is OrderSide.BUY else -1.0
            price = spec.round_price(
                price * (1.0 + direction * bps(self.config.base_slippage_bps))
            )
        order = Order(
            client_order_id=f"auto-{reason}-{new_id()[:8]}",
            symbol=position.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=position.quantity,
            status=OrderStatus.SUBMITTED,
            reduce_only=True,
            created_at=self._now,
            metadata={"auto_exit": reason},
        )
        self._orders[order.client_order_id] = order
        fill = self._settle_fill(
            order, spec, price=price, quantity=position.quantity, role=LiquidityRole.TAKER
        )
        logger.info(
            "paper_exchange.auto_exit",
            symbol=position.symbol,
            reason=reason,
            price=price,
        )
        return fill

    def _mark_positions(self, symbol: str, price: float) -> None:
        position = self._positions.get(symbol)
        if position is not None and position.is_open:
            position.mark(price, self._now)

    # ------------------------------------------------------------------ #
    # Cancellation and position management
    # ------------------------------------------------------------------ #
    async def cancel_order(
        self, client_order_id: str, *, symbol: str | None = None
    ) -> Order:
        order = self._orders.get(client_order_id)
        if order is None:
            raise OrderRejectedError(
                f"Unknown order {client_order_id}",
                context={"client_order_id": client_order_id},
            )
        if order.status.is_terminal:
            # Cancelling an already-finished order is a no-op, not an error: the caller may be
            # retrying after a timeout and must not be punished for it.
            return order
        order.status = OrderStatus.CANCELLED
        order.updated_at = self._now
        return order

    def set_position_stops(
        self,
        symbol: str,
        *,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        trail_offset: float | None = None,
    ) -> Position:
        """Attach protective levels to an open position."""
        position = self._positions.get(symbol)
        if position is None or not position.is_open:
            raise OrderRejectedError(
                f"No open position on {symbol} to attach stops to",
                context={"symbol": symbol},
            )
        if stop_loss is not None:
            self._validate_stop_side(position, stop_loss)
            position.stop_loss = stop_loss
        if take_profit is not None:
            position.take_profit = take_profit
        if trail_offset is not None:
            if trail_offset <= 0:
                raise OrderRejectedError("trail_offset must be positive")
            position.trail_offset = trail_offset
        return position

    @staticmethod
    def _validate_stop_side(position: Position, stop_loss: float) -> None:
        reference = position.mark_price or position.entry_price
        if position.side is PositionSide.LONG and stop_loss >= reference:
            raise OrderRejectedError(
                f"Long stop {stop_loss:g} is at or above the current price {reference:g}"
            )
        if position.side is PositionSide.SHORT and stop_loss <= reference:
            raise OrderRejectedError(
                f"Short stop {stop_loss:g} is at or below the current price {reference:g}"
            )

    async def deposit(self, amount: float) -> float:
        """Add funds. Simulator-only; there is no such operation on a real venue."""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self._cash += amount
        return self._cash

    async def withdraw(self, amount: float) -> float:
        """Remove funds, respecting locked margin."""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.free_cash:
            raise InsufficientBalanceError(
                f"Cannot withdraw {amount:.2f}: only {self.free_cash:.2f} is free"
            )
        self._cash -= amount
        return self._cash

    def reset(self) -> None:
        """Return the simulator to its initial state, keeping registered instruments."""
        self._cash = self.config.starting_balance
        self._locked_margin = 0.0
        self._orders.clear()
        self._positions.clear()
        self._market.clear()
        self._fills.clear()
        self._rejections.clear()
        self._realized_pnl = 0.0
        self._fees_paid = 0.0

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    async def subscribe_candles(self, symbol: str, interval: str) -> AsyncIterator[Candle]:
        """Yield candles fed to :meth:`process_candle` after the subscription starts."""
        key = f"{symbol}:{interval}"
        queue: asyncio.Queue[Candle] = asyncio.Queue()
        self._candle_queues[key] = queue
        try:
            while True:
                yield await queue.get()
        finally:
            self._candle_queues.pop(key, None)

    async def subscribe_ticker(self, symbol: str) -> AsyncIterator[Ticker]:
        while True:
            state = self._market.get(symbol)
            if state is not None:
                yield Ticker(
                    symbol=symbol,
                    price=state.last_price,
                    timestamp=state.timestamp,
                    bid=state.best_bid,
                    ask=state.best_ask,
                )
            await asyncio.sleep(1.0)


def _is_stop(order_type: OrderType) -> bool:
    return order_type in {OrderType.STOP_MARKET, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP}


def _infer_instrument(symbol: str) -> InstrumentSpec:
    """Best-effort instrument spec for a symbol the caller never registered.

    Used so that tests and quick demos are not forced to declare metadata. Production paths
    always register real venue specs.
    """
    for quote in ("USDT", "USDC", "BUSD", "USD", "BTC", "ETH"):
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return InstrumentSpec(
                symbol=symbol,
                base_asset=symbol[: -len(quote)],
                quote_asset=quote,
                instrument_type=InstrumentType.SPOT,
                tick_size=0.01,
                lot_size=0.000001,
                min_notional=5.0,
                max_leverage=1.0,
            )
    return InstrumentSpec(
        symbol=symbol, base_asset=symbol, quote_asset="USDT", min_notional=5.0
    )
