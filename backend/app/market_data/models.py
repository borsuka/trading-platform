"""Market data value objects.

All timestamps are timezone-aware UTC — enforced in ``__post_init__``, not documented and hoped
for. A :class:`Candle` also validates its own OHLC relationships, which catches malformed
exchange payloads at the boundary instead of three layers later inside an indicator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.core.clock import ensure_utc, interval_to_timedelta, utcnow
from app.core.numeric import safe_divide


@dataclass(frozen=True, slots=True)
class Candle:
    """One OHLCV bar.

    ``open_time`` is the bar's opening timestamp. A bar is only safe to make decisions on when
    :attr:`is_closed` is true; using a forming bar is a subtle form of lookahead because its
    close will still change.
    """

    symbol: str
    interval: str
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float | None = None
    trade_count: int | None = None
    is_closed: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "open_time", ensure_utc(self.open_time, field="open_time"))
        for name in ("open", "high", "low", "close"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{self.symbol} {self.open_time}: {name} is not finite")
            if value <= 0:
                raise ValueError(f"{self.symbol} {self.open_time}: {name} must be positive")
        if self.volume < 0 or not math.isfinite(self.volume):
            raise ValueError(f"{self.symbol} {self.open_time}: volume must be non-negative")
        if self.high < self.low:
            raise ValueError(f"{self.symbol} {self.open_time}: high {self.high} < low {self.low}")
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"{self.symbol} {self.open_time}: open outside [low, high]")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"{self.symbol} {self.open_time}: close outside [low, high]")

    @property
    def duration(self) -> timedelta:
        return interval_to_timedelta(self.interval)

    @property
    def close_time(self) -> datetime:
        return self.open_time + self.duration

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    def age(self, now: datetime | None = None) -> timedelta:
        """Time since the bar closed."""
        return (now or utcnow()) - self.close_time


@dataclass(frozen=True, slots=True)
class Ticker:
    """Latest price snapshot with top-of-book."""

    symbol: str
    price: float
    timestamp: datetime
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    volume_24h: float | None = None
    price_change_24h_pct: float | None = None
    funding_rate: float | None = None
    open_interest: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        if self.price <= 0 or not math.isfinite(self.price):
            raise ValueError(f"{self.symbol}: ticker price must be a positive finite number")
        if self.bid is not None and self.ask is not None and self.bid > self.ask:
            raise ValueError(f"{self.symbol}: crossed book bid {self.bid} > ask {self.ask}")

    @property
    def mid(self) -> float:
        if self.bid is None or self.ask is None:
            return self.price
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float | None:
        """Spread in basis points of mid. The number the risk engine actually gates on."""
        spread = self.spread
        if spread is None:
            return None
        return safe_divide(spread, self.mid) * 10_000.0

    def age(self, now: datetime | None = None) -> timedelta:
        return (now or utcnow()) - self.timestamp


@dataclass(frozen=True, slots=True)
class OrderBookLevel:
    price: float
    quantity: float

    @property
    def notional(self) -> float:
        return self.price * self.quantity


@dataclass(frozen=True, slots=True)
class OrderBook:
    """L2 depth snapshot, bids descending and asks ascending."""

    symbol: str
    timestamp: datetime
    bids: tuple[OrderBookLevel, ...] = ()
    asks: tuple[OrderBookLevel, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        object.__setattr__(self, "bids", tuple(sorted(self.bids, key=lambda x: -x.price)))
        object.__setattr__(self, "asks", tuple(sorted(self.asks, key=lambda x: x.price)))
        if self.bids and self.asks and self.bids[0].price > self.asks[0].price:
            raise ValueError(f"{self.symbol}: crossed order book")

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0].price + self.asks[0].price) / 2.0

    @property
    def spread_bps(self) -> float | None:
        mid = self.mid
        if mid is None:
            return None
        return safe_divide(self.asks[0].price - self.bids[0].price, mid) * 10_000.0

    def depth(self, side: str, levels: int | None = None) -> float:
        """Total quantity available on one side."""
        book = self.bids if side == "bid" else self.asks
        selected = book[:levels] if levels else book
        return sum(level.quantity for level in selected)

    def notional_depth(self, side: str, levels: int | None = None) -> float:
        book = self.bids if side == "bid" else self.asks
        selected = book[:levels] if levels else book
        return sum(level.notional for level in selected)

    def simulate_market_fill(self, side: str, quantity: float) -> tuple[float, float]:
        """Walk the book for a taker order.

        Returns ``(average_price, filled_quantity)``. When the book is too thin the filled
        quantity is less than requested, which is exactly the signal the risk engine needs to
        refuse the trade rather than discover the shortfall after sending it.
        """
        book = self.asks if side == "buy" else self.bids
        remaining = quantity
        cost = 0.0
        for level in book:
            if remaining <= 0:
                break
            take = min(remaining, level.quantity)
            cost += take * level.price
            remaining -= take
        filled = quantity - remaining
        if filled <= 0:
            return (0.0, 0.0)
        return (cost / filled, filled)

    def estimate_slippage_bps(self, side: str, quantity: float) -> float | None:
        """Expected slippage against mid, in basis points. ``None`` if the book cannot fill it."""
        mid = self.mid
        if mid is None:
            return None
        avg_price, filled = self.simulate_market_fill(side, quantity)
        if filled < quantity or avg_price <= 0:
            return None
        signed = (avg_price - mid) if side == "buy" else (mid - avg_price)
        return safe_divide(signed, mid) * 10_000.0


@dataclass(frozen=True, slots=True)
class PublicTrade:
    """A single executed trade printed on the public tape."""

    symbol: str
    price: float
    quantity: float
    side: str
    timestamp: datetime
    trade_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Everything the decision pipeline knows about one symbol at one instant.

    Passing a single immutable snapshot around, rather than letting each component fetch its own
    data, is what makes a backtest bar and a live bar structurally identical.
    """

    symbol: str
    timestamp: datetime
    candles: tuple[Candle, ...]
    ticker: Ticker | None = None
    order_book: OrderBook | None = None
    funding_rate: float | None = None
    open_interest: float | None = None
    metadata: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))

    @property
    def last_candle(self) -> Candle | None:
        return self.candles[-1] if self.candles else None

    @property
    def price(self) -> float | None:
        """Best available current price: ticker first, then the last close."""
        if self.ticker is not None:
            return self.ticker.price
        last = self.last_candle
        return last.close if last else None

    def closes(self) -> list[float]:
        return [c.close for c in self.candles]

    def highs(self) -> list[float]:
        return [c.high for c in self.candles]

    def lows(self) -> list[float]:
        return [c.low for c in self.candles]

    def volumes(self) -> list[float]:
        return [c.volume for c in self.candles]

    def has_history(self, minimum: int) -> bool:
        return len(self.candles) >= minimum
