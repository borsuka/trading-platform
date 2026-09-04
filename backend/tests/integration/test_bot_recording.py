"""A bot's activity has to outlive the process that produced it.

Everything a running bot did — its orders, its positions, its closed trades, its event log —
lived only in the runtime's memory. Nothing wrote it down. So a restart erased the history of
a run, and the Orders, Positions and Trades pages, which read from the database, were empty no
matter how long a bot had been trading.

That breaks the one thing the platform tells you to do before risking money: run paper trading
for days, then review how it behaved. A run that leaves no record cannot be reviewed.

The properties under test:

* a bot that trades leaves orders, positions and trades behind;
* recording twice does not duplicate anything — a duplicated trade silently corrupts every
  performance figure derived from the table, which is worse than a missing one because it
  still looks like an answer;
* a closed position is marked closed rather than left open forever;
* a recorder failure never propagates into the trading loop.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.core.enums import BotEventType, TradeStatus
from app.market_data.providers import generate_synthetic_candles
from app.paper_trading.recorder import BotStateRecorder
from app.paper_trading.replay import build_replay_bot

USER = "11111111-1111-4111-8111-111111111111"
BOT = "22222222-2222-4222-8222-222222222222"


@pytest.fixture
def database(tmp_path, monkeypatch) -> Iterator[None]:
    """A real database for the recorder to write into."""
    import asyncio

    from app.config import AppEnv, Settings

    url = f"sqlite+aiosqlite:///{tmp_path / 'record.db'}"
    settings = Settings(app_env=AppEnv.TEST, database_url=url, log_level="ERROR")
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    monkeypatch.setattr("app.config.settings.get_settings", lambda: settings)
    monkeypatch.setenv("DATABASE_URL", url)

    import app.database.session as session_module

    session_module._engine = None
    session_module._sessionmaker = None

    async def prepare() -> None:
        from app.database.models import Bot, Strategy, User
        from app.database.session import create_all, session_scope

        await create_all(settings)
        # The recorder writes rows referencing a user and a bot, so both have to exist.
        # Built through the ORM rather than hand-written SQL: the column list is then the
        # model's problem, not this fixture's.
        async with session_scope(settings) as session:
            session.add(User(id=USER, email="rec@example.com", password_hash="x"))
            await session.flush()
            strategy = Strategy(
                user_id=USER, name="rec", strategy_type="trend_following", parameters={}
            )
            session.add(strategy)
            await session.flush()
            session.add(
                Bot(id=BOT, user_id=USER, name="rec", strategy_id=strategy.id)
            )

    asyncio.run(prepare())
    yield
    session_module._engine = None
    session_module._sessionmaker = None


async def count(table: str) -> int:
    from sqlalchemy import text

    from app.database.session import session_scope

    async with session_scope() as session:
        result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
        return int(result.scalar_one())


async def run_until_trade(recorder: BotStateRecorder) -> tuple[int, int]:
    """Drive a replayed bot until it has closed at least one trade.

    Synthetic trending data is used precisely because it reliably produces trades: the point
    here is the recording, not the strategy.
    """
    candles = generate_synthetic_candles(
        "BTCUSDT", interval="1h", count=1200, seed=7, drift=0.0006
    )
    bot, provider = build_replay_bot(
        candles,
        bot_id=BOT,
        user_id=USER,
        name="rec",
        strategy_name="trend_following",
        interval="1h",
    )
    recorder.bind(bot)
    bot.event_handler = recorder
    await bot.start()

    closed = 0
    steps = 0
    while provider.step() and steps < 900:
        await bot.run_cycle()
        steps += 1
        closed = len(bot.portfolio.closed_trades())
        if closed >= 2:
            break
    await bot.stop()
    return closed, len(bot.portfolio.positions())


class TestRecording:
    @pytest.mark.asyncio
    async def test_a_trading_run_leaves_a_record(self, database: None) -> None:
        recorder = BotStateRecorder(bot_id=BOT, user_id=USER)
        closed, _ = await run_until_trade(recorder)

        assert closed >= 1, "the fixture should have produced at least one closed trade"
        assert await count("bot_events") > 0
        assert await count("orders") > 0
        assert await count("trades") == closed

    @pytest.mark.asyncio
    async def test_trades_are_not_duplicated_by_a_second_sync(
        self, database: None
    ) -> None:
        """The regression that would matter most: a duplicated trade corrupts every metric."""
        from app.database.session import session_scope

        recorder = BotStateRecorder(bot_id=BOT, user_id=USER)
        closed, _ = await run_until_trade(recorder)
        before = await count("trades")

        # Re-run the sync against the same state, as a reconnect or a replayed event would.
        assert recorder._bot is not None
        async with session_scope() as session:
            await recorder._sync_state(session, recorder._bot)

        assert await count("trades") == before == closed

    @pytest.mark.asyncio
    async def test_orders_are_upserted_not_appended(self, database: None) -> None:
        from app.database.session import session_scope

        recorder = BotStateRecorder(bot_id=BOT, user_id=USER)
        await run_until_trade(recorder)
        before = await count("orders")

        assert recorder._bot is not None
        async with session_scope() as session:
            await recorder._sync_state(session, recorder._bot)

        assert await count("orders") == before

    @pytest.mark.asyncio
    async def test_closed_positions_are_marked_closed(self, database: None) -> None:
        """A position left open forever would overstate exposure on every later reading."""
        from sqlalchemy import text

        from app.database.session import session_scope

        recorder = BotStateRecorder(bot_id=BOT, user_id=USER)
        _, still_open = await run_until_trade(recorder)

        async with session_scope() as session:
            open_rows = int(
                (
                    await session.execute(
                        text("SELECT COUNT(*) FROM positions WHERE is_open = 1")
                    )
                ).scalar_one()
            )
        assert open_rows == still_open

    @pytest.mark.asyncio
    async def test_stored_trades_carry_the_numbers_that_matter(
        self, database: None
    ) -> None:
        from sqlalchemy import text

        from app.database.session import session_scope

        recorder = BotStateRecorder(bot_id=BOT, user_id=USER)
        await run_until_trade(recorder)

        async with session_scope() as session:
            row = (
                await session.execute(
                    text(
                        "SELECT symbol, status, entry_price, exit_price, net_pnl, exit_time"
                        " FROM trades LIMIT 1"
                    )
                )
            ).one()
        symbol, status, entry, exit_price, net_pnl, exit_time = row
        assert symbol == "BTCUSDT"
        assert status == TradeStatus.CLOSED.value
        assert entry > 0 and exit_price > 0
        assert exit_time is not None
        assert net_pnl != 0.0


class TestFailureIsolation:
    @pytest.mark.asyncio
    async def test_a_broken_recorder_does_not_stop_the_bot(self, database: None) -> None:
        """Losing the record of a trade must not cost the bot its open position."""

        class Broken(BotStateRecorder):
            async def _sync_state(self, session, bot):  # type: ignore[no-untyped-def]
                raise RuntimeError("database on fire")

        recorder = Broken(bot_id=BOT, user_id=USER)
        closed, _ = await run_until_trade(recorder)

        # The bot kept trading despite every state sync failing.
        assert closed >= 1

    @pytest.mark.asyncio
    async def test_only_state_changing_events_trigger_a_sync(
        self, database: None
    ) -> None:
        """A signal that produced no trade cannot have changed any of the tables."""
        from app.paper_trading.recorder import STATE_CHANGING

        assert BotEventType.SIGNAL_GENERATED not in STATE_CHANGING
        assert BotEventType.HEARTBEAT not in STATE_CHANGING
        assert BotEventType.POSITION_OPENED in STATE_CHANGING
        assert BotEventType.POSITION_CLOSED in STATE_CHANGING
