"""Deterministic market-data replay for paper trading.

Feeds a fixed candle series to a bot one bar at a time, driving the paper exchange in step so
that resting orders, stops and targets are evaluated against each bar's real range.

Two uses, both important:

* **Demonstration.** A new user can watch a bot trade immediately, without waiting for live
  bars or configuring an exchange.
* **Testing.** The end-to-end acceptance scenario needs a bot whose every decision is
  reproducible; wall-clock polling of a live feed is neither.

The replay clock advances with the data, so the bot's staleness and drift checks see a
consistent world rather than comparing historical bars against the real current time.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from app.core.clock import ensure_utc
from app.core.exceptions import InsufficientDataError
from app.core.logging import get_logger
from app.exchanges.paper import PaperExchange
from app.market_data.models import Candle, MarketSnapshot, OrderBook, Ticker
from app.market_data.providers import LiveMarketDataProvider, build_snapshot

if TYPE_CHECKING:
    from app.paper_trading.runtime import TradingBot

logger = get_logger(__name__)


class ReplayClock:
    """Clock driven by replayed data rather than wall time."""

    def __init__(self, start: datetime) -> None:
        self._now = ensure_utc(start)
        self._monotonic = 0.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    def advance_to(self, moment: datetime) -> None:
        target = ensure_utc(moment)
        delta = (target - self._now).total_seconds()
        if delta > 0:
            self._monotonic += delta
        self._now = target


class ReplayMarketDataProvider(LiveMarketDataProvider):
    """Serves a fixed candle series, advancing one bar per step.

    The provider owns the cursor: :meth:`step` moves it forward and feeds the paper exchange,
    and :meth:`get_snapshot` returns only bars at or before the cursor. A strategy therefore
    cannot see the future even though the whole series is in memory.
    """

    def __init__(
        self,
        candles: Sequence[Candle],
        *,
        exchange: PaperExchange | None = None,
        warmup: int = 200,
        clock: ReplayClock | None = None,
    ) -> None:
        if not candles:
            raise InsufficientDataError("Replay requires at least one candle")
        self._by_symbol: dict[str, list[Candle]] = {}
        for candle in candles:
            self._by_symbol.setdefault(candle.symbol, []).append(candle)
        for series in self._by_symbol.values():
            series.sort(key=lambda c: c.open_time)

        self.exchange = exchange
        self._cursor = min(warmup, len(next(iter(self._by_symbol.values()))) - 1)
        self._warmup = warmup
        first = next(iter(self._by_symbol.values()))[0]
        self.clock = clock or ReplayClock(first.close_time)
        self._prime()

    # ------------------------------------------------------------------ #
    # Cursor
    # ------------------------------------------------------------------ #
    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def length(self) -> int:
        return max(len(series) for series in self._by_symbol.values())

    @property
    def exhausted(self) -> bool:
        return self._cursor >= self.length - 1

    def _prime(self) -> None:
        """Feed the warm-up bars to the exchange so it has prices before the first decision."""
        if self.exchange is None:
            return
        for series in self._by_symbol.values():
            for candle in series[: self._cursor + 1]:
                self.exchange.process_candle(candle)
        self.clock.advance_to(self._current_time())

    def _current_time(self) -> datetime:
        return max(
            series[min(self._cursor, len(series) - 1)].close_time
            for series in self._by_symbol.values()
        )

    def step(self, count: int = 1) -> int:
        """Advance the cursor, feeding each new bar to the exchange.

        Returns the number of bars actually advanced, which is less than ``count`` at the end
        of the series.
        """
        advanced = 0
        for _ in range(count):
            if self.exhausted:
                break
            self._cursor += 1
            advanced += 1
            if self.exchange is not None:
                for series in self._by_symbol.values():
                    if self._cursor < len(series):
                        self.exchange.process_candle(series[self._cursor])
        self.clock.advance_to(self._current_time())
        return advanced

    def reset(self) -> None:
        self._cursor = min(self._warmup, self.length - 1)
        if self.exchange is not None:
            self.exchange.reset()
        self._prime()

    # ------------------------------------------------------------------ #
    # LiveMarketDataProvider
    # ------------------------------------------------------------------ #
    def _visible(self, symbol: str, lookback: int) -> list[Candle]:
        series = self._by_symbol.get(symbol)
        if series is None:
            raise InsufficientDataError(f"No replay data for {symbol}")
        end = min(self._cursor + 1, len(series))
        return series[max(0, end - lookback) : end]

    async def get_snapshot(
        self, symbol: str, interval: str, lookback: int = 300
    ) -> MarketSnapshot:
        window = self._visible(symbol, lookback)
        if not window:
            raise InsufficientDataError(f"No replay data available yet for {symbol}")
        return build_snapshot(window, now=window[-1].close_time)

    async def get_ticker(self, symbol: str) -> Ticker:
        window = self._visible(symbol, 1)
        last = window[-1]
        half = last.close * 0.0001
        return Ticker(
            symbol=symbol,
            price=last.close,
            timestamp=last.close_time,
            bid=last.close - half,
            ask=last.close + half,
            volume_24h=last.volume,
        )

    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        snapshot = await self.get_snapshot(symbol, "", lookback=1)
        if snapshot.order_book is None:
            raise InsufficientDataError(f"No book available for {symbol}")
        return snapshot.order_book

    async def stream_candles(self, symbol: str, interval: str) -> AsyncIterator[Candle]:
        """Yield each remaining bar in order, advancing the cursor as it goes."""
        while not self.exhausted:
            self.step()
            window = self._visible(symbol, 1)
            if window:
                yield window[-1]


def build_replay_bot(
    candles: Sequence[Candle],
    *,
    bot_id: str,
    user_id: str,
    name: str,
    strategy_name: str,
    interval: str = "1h",
    starting_balance: float = 10_000.0,
    warmup: int = 250,
    **kwargs: Any,
) -> tuple[TradingBot, ReplayMarketDataProvider]:
    """Assemble a paper bot driven by replayed data.

    Returns ``(bot, provider)``. Call :meth:`ReplayMarketDataProvider.step` then
    :meth:`TradingBot.run_cycle` to advance the simulation one bar at a time.
    """
    from app.paper_trading.factory import build_paper_bot, build_paper_exchange

    symbols = sorted({c.symbol for c in candles})
    exchange = build_paper_exchange(symbols, starting_balance=starting_balance)
    provider = ReplayMarketDataProvider(candles, exchange=exchange, warmup=warmup)

    bot = build_paper_bot(
        bot_id=bot_id,
        user_id=user_id,
        name=name,
        strategy_name=strategy_name,
        symbols=symbols,
        interval=interval,
        starting_balance=starting_balance,
        market_data=provider,
        reconcile_on_start=False,
        **kwargs,
    )
    # Share the paper exchange and the replay clock so fills, stops and staleness checks all
    # see the same simulated moment.
    bot.exchange = exchange
    bot.order_manager.exchange = exchange
    bot.clock = provider.clock  # type: ignore[assignment]
    bot.reconciler.exchange = exchange
    return bot, provider


def replay_window(
    candles: Sequence[Candle], *, bars: int, offset: int = 0
) -> list[Candle]:
    """Slice a contiguous window out of a candle series."""
    if bars <= 0:
        raise ValueError("bars must be positive")
    start = max(0, offset)
    return list(candles[start : start + bars])


def default_interval(candles: Sequence[Candle]) -> str:
    return candles[0].interval if candles else "1h"


def estimated_duration(candles: Sequence[Candle]) -> timedelta:
    if len(candles) < 2:
        return timedelta(0)
    return candles[-1].close_time - candles[0].open_time
