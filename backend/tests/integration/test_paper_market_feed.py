"""Where a paper bot gets its prices.

This exists because of a defect that made the dashboard's Start button useless: a paper bot
built without an explicit market-data source wrapped the paper exchange in a provider that
read *from that same exchange*. The simulator stores no history, so every cycle failed with
"no market data has been fed", the bot sat in `running` doing nothing, and the only sign was
an error buried in its event log.

The fix routes a paper bot's prices from a venue's public endpoints. The properties worth
protecting:

* a bot built with defaults has a source that can actually produce prices;
* the venue is a data source and never an execution venue - the bot still trades on the
  simulator, and no credentials are involved;
* bars reach the simulator once, in order, so a stop cannot be triggered twice or walked
  backwards.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.core.clock import utcnow
from app.core.exceptions import ConfigurationError
from app.market_data.models import Candle
from app.paper_trading.factory import (
    build_paper_bot,
    build_paper_exchange,
    default_paper_market_data,
)
from app.paper_trading.live_feed import PUBLIC_SOURCES, PublicMarketFeed, build_public_adapter


def bars(symbol: str = "BTCUSDT", count: int = 5, start_price: float = 100.0) -> list[Candle]:
    base = utcnow() - timedelta(minutes=15 * count)
    out = []
    for i in range(count):
        price = start_price + i
        out.append(
            Candle(
                symbol=symbol,
                interval="15m",
                open_time=base + timedelta(minutes=15 * i),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price + 0.5,
                volume=10.0,
            )
        )
    return out


class TestPublicAdapter:
    @pytest.mark.parametrize("source", PUBLIC_SOURCES)
    def test_known_sources_construct(self, source: str) -> None:
        adapter = build_public_adapter(source)
        assert adapter.name == source

    def test_no_credentials_are_carried(self) -> None:
        """A data-only adapter must have no key it could accidentally use."""
        adapter = build_public_adapter("bybit")
        assert adapter.credentials.api_key == ""
        assert adapter.credentials.api_secret == ""

    def test_unknown_source_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="Unknown market data source"):
            build_public_adapter("not-a-venue")


class TestFeeding:
    """The simulator has to receive each bar once, oldest first."""

    def test_bars_reach_the_simulator(self) -> None:
        exchange = build_paper_exchange(("BTCUSDT",), starting_balance=1000)
        feed = PublicMarketFeed(exchange, adapter=build_public_adapter("bybit"))
        feed._feed("BTCUSDT", bars())
        # The simulator can now quote a market it previously refused to.
        assert exchange._market["BTCUSDT"].last_price == pytest.approx(104.5)

    def test_a_bar_is_never_applied_twice(self) -> None:
        """Replaying a processed bar would trigger the same stop a second time."""
        exchange = build_paper_exchange(("BTCUSDT",), starting_balance=1000)
        feed = PublicMarketFeed(exchange, adapter=build_public_adapter("bybit"))
        series = bars()

        feed._feed("BTCUSDT", series)
        first = exchange._market["BTCUSDT"].last_price
        feed._feed("BTCUSDT", series)  # same bars again
        assert exchange._market["BTCUSDT"].last_price == first

    def test_only_new_bars_are_applied(self) -> None:
        exchange = build_paper_exchange(("BTCUSDT",), starting_balance=1000)
        feed = PublicMarketFeed(exchange, adapter=build_public_adapter("bybit"))
        series = bars(count=6)

        feed._feed("BTCUSDT", series[:3])
        feed._feed("BTCUSDT", series)  # overlapping window, as a rolling fetch produces
        assert exchange._market["BTCUSDT"].last_price == pytest.approx(105.5)

    def test_out_of_order_input_is_applied_in_order(self) -> None:
        """Walking the price backwards would fire stops that were never touched."""
        exchange = build_paper_exchange(("BTCUSDT",), starting_balance=1000)
        feed = PublicMarketFeed(exchange, adapter=build_public_adapter("bybit"))
        series = bars()

        feed._feed("BTCUSDT", list(reversed(series)))
        assert exchange._market["BTCUSDT"].last_price == pytest.approx(104.5)

    def test_empty_input_is_harmless(self) -> None:
        exchange = build_paper_exchange(("BTCUSDT",), starting_balance=1000)
        feed = PublicMarketFeed(exchange, adapter=build_public_adapter("bybit"))
        feed._feed("BTCUSDT", [])
        assert "BTCUSDT" not in exchange._market


class TestDefaultWiring:
    def test_default_source_can_actually_produce_prices(self, monkeypatch) -> None:
        """The regression: the default used to read from the simulator it was feeding."""
        from app.config import Settings

        settings = Settings(paper_market_data="bybit")
        monkeypatch.setattr("app.config.get_settings", lambda: settings)

        exchange = build_paper_exchange(("BTCUSDT",), starting_balance=1000)
        provider = default_paper_market_data(exchange, ("BTCUSDT",), "15m")
        assert isinstance(provider, PublicMarketFeed)
        # The point of the whole fix: the source is not the exchange being fed.
        assert provider.adapter is not exchange

    def test_synthetic_source_is_still_available(self, monkeypatch) -> None:
        """Running with no internet must remain possible, just clearly labelled."""
        from app.config import Settings
        from app.paper_trading.replay import ReplayMarketDataProvider

        settings = Settings(paper_market_data="synthetic")
        monkeypatch.setattr("app.config.get_settings", lambda: settings)

        exchange = build_paper_exchange(("BTCUSDT",), starting_balance=1000)
        provider = default_paper_market_data(exchange, ("BTCUSDT",), "15m")
        assert isinstance(provider, ReplayMarketDataProvider)

    def test_a_default_paper_bot_is_still_paper(self) -> None:
        """Real prices must not make the bot live. Only the data is real."""
        bot = build_paper_bot(
            bot_id="b1",
            user_id="u1",
            name="paper",
            strategy_name="trend_following",
            symbols=["BTCUSDT"],
        )
        assert bot.mode == "PAPER"
        assert bot.exchange.is_live is False

    def test_constructing_a_bot_makes_no_network_calls(self) -> None:
        """Construction must stay lazy, or every test that builds a bot hits the internet."""
        import httpx

        def explode(*args, **kwargs):
            raise AssertionError("a network call was made during construction")

        original = httpx.AsyncClient.request
        httpx.AsyncClient.request = explode  # type: ignore[method-assign]
        try:
            build_paper_bot(
                bot_id="b2",
                user_id="u1",
                name="paper",
                strategy_name="trend_following",
                symbols=["BTCUSDT"],
            )
        finally:
            httpx.AsyncClient.request = original  # type: ignore[method-assign]


class TestSimulatedClock:
    """Which clock the simulator answers with, and why it matters.

    `_now` follows the last bar fed in, which is what a replay needs: simulated time must
    track the data. But a bot on a live feed runs on the real clock, and the drift guard
    compares the two. With simulated time on one side and wall-clock on the other, the guard
    measures how old the last bar is - up to a full interval - and reports ordinary bar lag as
    a clock fault. That halts trading for a machine whose clock is perfectly correct.
    """

    @pytest.mark.asyncio
    async def test_replay_keeps_simulated_time(self) -> None:
        exchange = build_paper_exchange(("BTCUSDT",), starting_balance=1000)
        series = bars()
        for candle in series:
            exchange.process_candle(candle)

        info = await exchange.get_info()
        assert info.server_time == series[-1].close_time

    @pytest.mark.asyncio
    async def test_a_live_feed_switches_the_simulator_to_wall_clock(self) -> None:
        exchange = build_paper_exchange(("BTCUSDT",), starting_balance=1000)
        PublicMarketFeed(exchange, adapter=build_public_adapter("bybit"))
        assert exchange.follow_wall_clock is True

    @pytest.mark.asyncio
    async def test_bar_age_is_not_reported_as_clock_drift(self) -> None:
        """The regression: a 15-minute-old bar became '15 seconds of clock drift'."""
        from app.core.clock import measure_drift, utcnow

        exchange = build_paper_exchange(("BTCUSDT",), starting_balance=1000)
        feed = PublicMarketFeed(exchange, adapter=build_public_adapter("bybit"))
        # A bar that closed a while ago - entirely normal between closes.
        stale_bar = bars(count=1)[0]
        feed._feed("BTCUSDT", [stale_bar])

        info = await exchange.get_info()
        report = measure_drift(utcnow(), info.server_time, tolerance_seconds=2.0)
        assert report.within_tolerance, (
            f"an old bar was reported as {report.drift_seconds:.0f}s of clock drift"
        )
