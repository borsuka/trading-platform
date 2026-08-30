"""Real market prices for a paper-trading bot.

A paper bot needs prices from somewhere. The simulator does not invent them: it is a matching
engine, and until something feeds it a bar it correctly refuses to quote a market. The CLI's
replay bot feeds it synthetic history. A bot started from the dashboard is supposed to run
*now*, against what the market is actually doing, so it needs a live source.

This is that source. It reads a venue's **public** endpoints - candles, tickers, order books -
which need no API key and no account, and hands each closed bar to the paper exchange so that
orders can fill against real prices.

The distinction that matters: the venue here is a **data source, never an execution venue**.
The bot's exchange stays the paper simulator. Nothing in this module can place an order, and
no credentials are involved, so there is no key to misuse even if something tried.

This is what makes a paper run worth doing before live trading. Synthetic data tells you the
plumbing works; real prices, real spreads and real gaps tell you whether the strategy does.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import datetime

from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger
from app.exchanges.base import ExchangeAdapter, ExchangeCredentials
from app.exchanges.paper import PaperExchange
from app.market_data.models import Candle, MarketSnapshot, OrderBook, Ticker
from app.market_data.providers import ExchangeLiveProvider, LiveMarketDataProvider

logger = get_logger(__name__)

#: Venues whose public market data this can read. Public endpoints only - these names never
#: reach a signing path, and no credentials are stored for them.
PUBLIC_SOURCES = ("bybit", "binance")


def build_public_adapter(source: str) -> ExchangeAdapter:
    """Construct a read-only adapter for public market data.

    The credentials are deliberately empty. Public endpoints are unsigned, so there is nothing
    to authenticate with, and an adapter built here has no key it could accidentally use.
    """
    credentials = ExchangeCredentials(api_key="", api_secret="", testnet=False)
    if source == "bybit":
        from app.exchanges.bybit import BybitAdapter

        return BybitAdapter(credentials)
    if source == "binance":
        from app.exchanges.binance import BinanceAdapter

        return BinanceAdapter(credentials)
    raise ConfigurationError(
        f"Unknown market data source {source!r}; expected one of {', '.join(PUBLIC_SOURCES)}"
    )


class PublicMarketFeed(LiveMarketDataProvider):
    """Live public prices, mirrored into a paper exchange.

    Wraps :class:`~app.market_data.providers.ExchangeLiveProvider` and adds the one thing a
    simulated venue needs that a real one does not: every closed bar is also pushed into the
    paper exchange, so resting orders, stops and targets are evaluated against real movement.
    """

    def __init__(
        self,
        exchange: PaperExchange,
        *,
        source: str = "bybit",
        adapter: ExchangeAdapter | None = None,
        cache_size: int = 500,
    ) -> None:
        self.exchange = exchange
        # The bot runs on the real clock now, so the simulator has to answer clock questions
        # on the real clock too. Left as simulated time it reports the age of the last bar as
        # clock drift, which trips the drift guard and stops the bot trading.
        exchange.follow_wall_clock = True
        self.source = source
        self.adapter = adapter or build_public_adapter(source)
        self._inner = ExchangeLiveProvider(self.adapter, cache_size=cache_size)
        #: Last bar handed to the simulator per symbol, so a bar is never applied twice.
        self._last_fed: dict[str, datetime] = {}

    # ------------------------------------------------------------------ #
    # LiveMarketDataProvider
    # ------------------------------------------------------------------ #
    async def get_ticker(self, symbol: str) -> Ticker:
        return await self._inner.get_ticker(symbol)

    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        return await self._inner.get_order_book(symbol, depth=depth)

    async def get_snapshot(
        self, symbol: str, interval: str, lookback: int = 200
    ) -> MarketSnapshot:
        snapshot = await self._inner.get_snapshot(symbol, interval, lookback=lookback)
        self._feed(symbol, snapshot.candles)
        return snapshot

    async def stream_candles(self, symbol: str, interval: str) -> AsyncIterator[Candle]:
        async for candle in self._inner.stream_candles(symbol, interval):
            self._feed(symbol, (candle,))
            yield candle

    # ------------------------------------------------------------------ #
    # Feeding the simulator
    # ------------------------------------------------------------------ #
    def _feed(self, symbol: str, candles: Sequence[Candle]) -> None:
        """Apply bars the simulator has not seen yet, oldest first.

        Ordering and the high-water mark both matter. Replaying a bar the simulator has
        already processed would trigger the same stop twice; feeding them out of order would
        walk the price backwards and fire stops that never should have been touched.
        """
        if not candles:
            return
        last = self._last_fed.get(symbol)
        fresh = [c for c in candles if last is None or c.open_time > last]
        if not fresh:
            return
        for candle in sorted(fresh, key=lambda c: c.open_time):
            self.exchange.process_candle(candle)
        self._last_fed[symbol] = fresh[-1].open_time

    async def close(self) -> None:
        await self.adapter.close()

    def __repr__(self) -> str:
        return f"PublicMarketFeed(source={self.source!r}, symbols={list(self._last_fed)})"
