"""Pure domain value objects shared by every layer above the database.

Nothing here imports SQLAlchemy, FastAPI or a network client. These types are what strategies,
the risk engine, the paper exchange and the backtester actually pass around; the ORM models in
``app.database.models`` are a persistence projection of them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from app.core.clock import ensure_utc, utcnow
from app.core.enums import (
    ExitReason,
    InstrumentType,
    LiquidityRole,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    TimeInForce,
    TradeStatus,
)
from app.core.numeric import EPSILON, is_zero, round_to_step, safe_divide


def new_id() -> str:
    """Fresh UUID4 string identifier."""
    return str(uuid.uuid4())


# --------------------------------------------------------------------------- #
# Instruments
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """Venue-normalised trading rules for one symbol.

    Every adapter must produce this so that position sizing never special-cases a venue.
    """

    symbol: str
    base_asset: str
    quote_asset: str
    instrument_type: InstrumentType = InstrumentType.SPOT
    tick_size: float = 0.01
    lot_size: float = 0.000001
    min_quantity: float = 0.0
    max_quantity: float = 1e12
    min_notional: float = 0.0
    max_leverage: float = 1.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.00055
    contract_size: float = 1.0
    is_active: bool = True

    def __post_init__(self) -> None:
        if self.tick_size <= 0:
            raise ValueError(f"{self.symbol}: tick_size must be positive")
        if self.lot_size <= 0:
            raise ValueError(f"{self.symbol}: lot_size must be positive")
        if self.min_quantity < 0 or self.max_quantity <= 0:
            raise ValueError(f"{self.symbol}: invalid quantity bounds")
        if self.max_leverage < 1:
            raise ValueError(f"{self.symbol}: max_leverage must be >= 1")

    @property
    def supports_leverage(self) -> bool:
        return self.instrument_type is not InstrumentType.SPOT and self.max_leverage > 1

    def round_price(self, price: float) -> float:
        return round_to_step(price, self.tick_size, mode="nearest")

    def round_quantity(self, quantity: float) -> float:
        """Round down: never size larger than the risk engine allowed."""
        return round_to_step(quantity, self.lot_size, mode="down")

    def notional(self, quantity: float, price: float) -> float:
        return abs(quantity) * price * self.contract_size

    def validate_order(self, quantity: float, price: float) -> str | None:
        """Return a human-readable reason the order is invalid, or ``None`` if it is fine."""
        if not self.is_active:
            return f"{self.symbol} is not tradable"
        qty = abs(quantity)
        if qty < self.min_quantity - EPSILON:
            return f"quantity {qty:g} below minimum {self.min_quantity:g}"
        if qty > self.max_quantity + EPSILON:
            return f"quantity {qty:g} above maximum {self.max_quantity:g}"
        if is_zero(qty):
            return "quantity rounds to zero at this lot size"
        notional = self.notional(qty, price)
        if notional < self.min_notional - EPSILON:
            return f"notional {notional:.2f} below minimum {self.min_notional:.2f}"
        return None


# --------------------------------------------------------------------------- #
# Balances
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Balance:
    """Balance for a single asset."""

    asset: str
    free: float
    locked: float = 0.0

    @property
    def total(self) -> float:
        return self.free + self.locked


@dataclass(frozen=True, slots=True)
class AccountBalance:
    """Snapshot of every asset held on a venue."""

    balances: dict[str, Balance] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utcnow)

    def get(self, asset: str) -> Balance:
        return self.balances.get(asset, Balance(asset=asset, free=0.0, locked=0.0))

    def free(self, asset: str) -> float:
        return self.get(asset).free

    def total(self, asset: str) -> float:
        return self.get(asset).total


# --------------------------------------------------------------------------- #
# Orders
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class OrderRequest:
    """An order the platform intends to place.

    ``client_order_id`` is the idempotency key. It is generated before the first submission
    attempt and reused for every state query, so a timed-out request can be resolved by asking
    the venue what happened rather than by sending a second order.
    """

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None
    trigger_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    close_position: bool = False
    leverage: float | None = None
    client_order_id: str = field(default_factory=new_id)
    trail_offset: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        if self.order_type.requires_limit_price and self.price is None:
            raise ValueError(f"{self.order_type} requires a limit price")
        if self.order_type.requires_trigger_price and self.trigger_price is None:
            raise ValueError(f"{self.order_type} requires a trigger price")
        if self.order_type is OrderType.TRAILING_STOP and self.trail_offset is None:
            raise ValueError("Trailing stop requires trail_offset")
        if self.price is not None and self.price <= 0:
            raise ValueError("Limit price must be positive")
        if self.trigger_price is not None and self.trigger_price <= 0:
            raise ValueError("Trigger price must be positive")


@dataclass(frozen=True, slots=True)
class Fill:
    """One execution against an order."""

    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    fee_asset: str
    role: LiquidityRole
    timestamp: datetime
    is_maker: bool = False

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("Fill quantity must be positive")
        if self.price <= 0:
            raise ValueError("Fill price must be positive")
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))

    @property
    def notional(self) -> float:
        return self.quantity * self.price


@dataclass(slots=True)
class Order:
    """Mutable order state as known locally.

    The venue is authoritative; this object is refreshed from adapter responses.
    """

    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    status: OrderStatus = OrderStatus.PENDING
    exchange_order_id: str | None = None
    price: float | None = None
    trigger_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    fees_paid: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    reject_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_quantity(self) -> float:
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def is_active(self) -> bool:
        return self.status.is_active

    @property
    def fill_ratio(self) -> float:
        return safe_divide(self.filled_quantity, self.quantity)

    def apply_fill(self, fill: Fill) -> None:
        """Record a fill and recompute the volume-weighted average price."""
        if fill.quantity <= 0:
            raise ValueError("Cannot apply a non-positive fill")
        new_filled = self.filled_quantity + fill.quantity
        if new_filled > self.quantity + EPSILON:
            raise ValueError(
                f"Fill would overfill order {self.client_order_id}: "
                f"{new_filled:g} > {self.quantity:g}"
            )
        total_notional = self.average_fill_price * self.filled_quantity + fill.notional
        self.filled_quantity = new_filled
        self.average_fill_price = safe_divide(total_notional, new_filled)
        self.fees_paid += fill.fee
        self.fills.append(fill)
        self.updated_at = fill.timestamp
        self.status = (
            OrderStatus.FILLED
            if abs(self.remaining_quantity) <= EPSILON
            else OrderStatus.PARTIALLY_FILLED
        )

    @classmethod
    def from_request(cls, request: OrderRequest) -> Order:
        return cls(
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            price=request.price,
            trigger_price=request.trigger_price,
            time_in_force=request.time_in_force,
            reduce_only=request.reduce_only,
            metadata=dict(request.metadata),
        )


# --------------------------------------------------------------------------- #
# Positions
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Position:
    """An open position with running PnL.

    Sign convention: :attr:`quantity` is always non-negative; direction lives in :attr:`side`.
    """

    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    opened_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    leverage: float = 1.0
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop: float | None = None
    trail_offset: float | None = None
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    mark_price: float | None = None
    position_id: str = field(default_factory=new_id)
    strategy_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quantity < 0:
            raise ValueError("Position quantity must be non-negative; use side for direction")
        if self.side is PositionSide.FLAT and not is_zero(self.quantity):
            raise ValueError("A flat position cannot hold quantity")

    @property
    def is_open(self) -> bool:
        return self.side is not PositionSide.FLAT and not is_zero(self.quantity)

    @property
    def signed_quantity(self) -> float:
        return self.quantity * self.side.sign

    def notional(self, price: float | None = None) -> float:
        reference = price if price is not None else (self.mark_price or self.entry_price)
        return self.quantity * reference

    def unrealized_pnl(self, price: float | None = None) -> float:
        """Mark-to-market PnL, excluding fees already paid."""
        if not self.is_open:
            return 0.0
        reference = price if price is not None else self.mark_price
        if reference is None:
            return 0.0
        return (reference - self.entry_price) * self.quantity * self.side.sign

    def unrealized_pnl_pct(self, price: float | None = None) -> float:
        cost = self.entry_price * self.quantity
        return safe_divide(self.unrealized_pnl(price), cost)

    def risk_per_unit(self) -> float | None:
        """Distance from entry to stop. ``None`` when no stop is set."""
        if self.stop_loss is None:
            return None
        return abs(self.entry_price - self.stop_loss)

    def mark(self, price: float, at: datetime | None = None) -> None:
        self.mark_price = price
        self.updated_at = ensure_utc(at) if at else utcnow()
        self._update_trailing_stop(price)

    def _update_trailing_stop(self, price: float) -> None:
        if self.trail_offset is None or not self.is_open:
            return
        if self.side is PositionSide.LONG:
            candidate = price - self.trail_offset
            if self.trailing_stop is None or candidate > self.trailing_stop:
                self.trailing_stop = candidate
        else:
            candidate = price + self.trail_offset
            if self.trailing_stop is None or candidate < self.trailing_stop:
                self.trailing_stop = candidate

    def effective_stop(self) -> float | None:
        """The tighter of the fixed stop and the trailing stop."""
        candidates = [s for s in (self.stop_loss, self.trailing_stop) if s is not None]
        if not candidates:
            return None
        return max(candidates) if self.side is PositionSide.LONG else min(candidates)

    def add(self, quantity: float, price: float) -> None:
        """Increase the position, recomputing the weighted average entry."""
        if quantity <= 0:
            raise ValueError("Added quantity must be positive")
        total_cost = self.entry_price * self.quantity + price * quantity
        self.quantity += quantity
        self.entry_price = safe_divide(total_cost, self.quantity, price)
        self.updated_at = utcnow()

    def reduce(self, quantity: float, price: float) -> float:
        """Reduce the position and return the realised PnL of the closed portion."""
        if quantity <= 0:
            raise ValueError("Reduced quantity must be positive")
        closing = min(quantity, self.quantity)
        pnl = (price - self.entry_price) * closing * self.side.sign
        self.quantity -= closing
        self.realized_pnl += pnl
        self.updated_at = utcnow()
        if is_zero(self.quantity):
            self.quantity = 0.0
            self.side = PositionSide.FLAT
        return pnl


# --------------------------------------------------------------------------- #
# Completed trades
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Trade:
    """A round-trip: entry through exit."""

    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    entry_time: datetime
    exit_price: float | None = None
    exit_time: datetime | None = None
    status: TradeStatus = TradeStatus.OPEN
    exit_reason: ExitReason | None = None
    gross_pnl: float = 0.0
    fees: float = 0.0
    slippage_cost: float = 0.0
    strategy_name: str | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    trade_id: str = field(default_factory=new_id)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.fees

    @property
    def return_pct(self) -> float:
        return safe_divide(self.net_pnl, self.entry_price * self.quantity)

    @property
    def duration_seconds(self) -> float | None:
        if self.exit_time is None:
            return None
        return (self.exit_time - self.entry_time).total_seconds()

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0

    @property
    def r_multiple(self) -> float | None:
        """PnL expressed in units of initial risk. The cleanest cross-symbol comparison."""
        if self.stop_loss is None:
            return None
        risk_per_unit = abs(self.entry_price - self.stop_loss)
        risk_total = risk_per_unit * self.quantity
        if is_zero(risk_total):
            return None
        return self.net_pnl / risk_total

    def close(
        self,
        *,
        price: float,
        at: datetime,
        reason: ExitReason,
        extra_fees: float = 0.0,
    ) -> None:
        self.exit_price = price
        self.exit_time = ensure_utc(at)
        self.exit_reason = reason
        self.fees += extra_fees
        self.gross_pnl = (price - self.entry_price) * self.quantity * self.side.sign
        self.status = TradeStatus.CLOSED


# --------------------------------------------------------------------------- #
# Portfolio snapshot
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """Point-in-time portfolio state. The unit of the equity curve."""

    timestamp: datetime
    cash: float
    equity: float
    unrealized_pnl: float
    realized_pnl: float
    total_exposure: float
    position_count: int
    fees_paid: float = 0.0
    peak_equity: float = 0.0
    drawdown: float = 0.0

    @property
    def leverage(self) -> float:
        return safe_divide(self.total_exposure, self.equity)

    def with_peak(self, peak: float) -> PortfolioSnapshot:
        """Return a copy with peak equity and drawdown recomputed."""
        new_peak = max(peak, self.equity)
        return replace(
            self,
            peak_equity=new_peak,
            drawdown=safe_divide(new_peak - self.equity, new_peak),
        )
