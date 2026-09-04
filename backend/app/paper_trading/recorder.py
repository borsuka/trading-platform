"""Persisting a running bot's activity.

A bot's runtime lives in memory. Its orders, positions, closed trades and event log all lived
there too, and nothing wrote them down — so every restart erased the entire history of a run,
and the Orders, Positions and Trades pages, which read from the database, were permanently
empty no matter how long a bot had been trading.

That is a problem specifically for the thing this platform asks you to do before risking money:
run paper trading for days and review how it behaved. A run that leaves no record cannot be
reviewed.

This recorder closes that gap. It hangs off the bot's existing ``event_handler`` hook, so the
trading loop drives it and a failure here can never take that loop down.

Two properties matter more than completeness:

* **Never break trading.** Every write is wrapped; a database problem costs you the record of
  a trade, not the management of an open position. The exception is logged, never swallowed
  silently.
* **Idempotent.** Orders are keyed on ``client_order_id`` and trades on ``trade_id``, so a
  re-sync after a reconnect updates rows rather than duplicating them. A duplicated trade would
  corrupt every performance number computed from the table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from app.core.domain import Order, Position, Trade
from app.core.enums import BotEventType
from app.core.logging import get_logger
from app.database.models import BotEvent as BotEventRecord
from app.database.models import (
    OrderRecord,
    PositionRecord,
    TradeRecord,
)
from app.database.session import session_scope

if TYPE_CHECKING:
    from app.paper_trading.runtime import BotEvent as DomainBotEvent
    from app.paper_trading.runtime import TradingBot

logger = get_logger(__name__)

#: Events after which the structured state is worth re-reading. Every event is stored; only
#: these also trigger a sync of orders, positions and trades, because a signal that produced
#: no trade cannot have changed any of them.
STATE_CHANGING = frozenset(
    {
        BotEventType.ORDER_SUBMITTED,
        BotEventType.ORDER_FILLED,
        BotEventType.ORDER_REJECTED,
        BotEventType.POSITION_OPENED,
        BotEventType.POSITION_CLOSED,
        BotEventType.STOPPED,
        BotEventType.EMERGENCY_STOP,
    }
)


class BotStateRecorder:
    """Writes a bot's activity to the database as it happens."""

    def __init__(self, bot_id: str, user_id: str) -> None:
        self.bot_id = bot_id
        self.user_id = user_id
        self._bot: TradingBot | None = None

    def bind(self, bot: TradingBot) -> None:
        """Attach to the bot whose activity is being recorded."""
        self._bot = bot

    async def __call__(self, event: DomainBotEvent) -> None:
        """The bot's event handler.

        Exceptions are contained here as well as by the caller: losing a record must never
        cost the caller its ability to manage an open position.
        """
        try:
            async with session_scope() as session:
                session.add(self._event_row(event))
                if event.event_type in STATE_CHANGING and self._bot is not None:
                    await self._sync_state(session, self._bot)
        except Exception:
            logger.exception(
                "bot.recording_failed",
                bot_id=self.bot_id,
                failed_event=event.event_type.value,
            )

    # ------------------------------------------------------------------ #
    # Rows
    # ------------------------------------------------------------------ #
    def _event_row(self, event: DomainBotEvent) -> BotEventRecord:
        return BotEventRecord(
            bot_id=self.bot_id,
            user_id=self.user_id,
            event_type=event.event_type,
            message=event.message,
            severity=event.severity,
            payload=event.payload or {},
            occurred_at=event.occurred_at,
        )

    async def _sync_state(self, session: Any, bot: TradingBot) -> None:
        await self._sync_orders(session, bot.order_manager.known_orders())
        await self._sync_positions(session, bot.portfolio.positions())
        await self._sync_trades(session, bot.portfolio.closed_trades())

    async def _sync_orders(self, session: Any, orders: list[Order]) -> None:
        """Upsert on ``client_order_id`` — the same key that makes the order idempotent."""
        for order in orders:
            existing = (
                await session.execute(
                    select(OrderRecord).where(
                        OrderRecord.user_id == self.user_id,
                        OrderRecord.client_order_id == order.client_order_id,
                    )
                )
            ).scalar_one_or_none()

            values = {
                "exchange_order_id": order.exchange_order_id,
                "status": order.status,
                "filled_quantity": order.filled_quantity,
                "average_fill_price": order.average_fill_price,
                "fees_paid": order.fees_paid,
                "reject_reason": order.reject_reason,
                "updated_at": order.updated_at,
            }
            if existing is not None:
                for key, value in values.items():
                    setattr(existing, key, value)
                continue

            session.add(
                OrderRecord(
                    user_id=self.user_id,
                    bot_id=self.bot_id,
                    client_order_id=order.client_order_id,
                    exchange="paper",
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type,
                    time_in_force=order.time_in_force,
                    quantity=order.quantity,
                    price=order.price,
                    trigger_price=order.trigger_price,
                    reduce_only=order.reduce_only,
                    created_at=order.created_at,
                    **values,
                )
            )

    async def _sync_positions(self, session: Any, positions: list[Position]) -> None:
        """Keep one open row per symbol, and close the rows the bot no longer holds."""
        open_rows = (
            (
                await session.execute(
                    select(PositionRecord).where(
                        PositionRecord.bot_id == self.bot_id,
                        PositionRecord.is_open.is_(True),
                    )
                )
            )
            .scalars()
            .all()
        )
        by_symbol = {row.symbol: row for row in open_rows}

        for position in positions:
            row = by_symbol.pop(position.symbol, None)
            if row is None:
                session.add(
                    PositionRecord(
                        user_id=self.user_id,
                        bot_id=self.bot_id,
                        symbol=position.symbol,
                        side=position.side,
                        quantity=position.quantity,
                        entry_price=position.entry_price,
                        mark_price=position.mark_price,
                        leverage=position.leverage,
                        stop_loss=position.stop_loss,
                        take_profit=position.take_profit,
                        realized_pnl=position.realized_pnl,
                        unrealized_pnl=position.unrealized_pnl,
                        fees_paid=position.fees_paid,
                        is_open=True,
                        opened_at=position.opened_at,
                    )
                )
                continue
            row.quantity = position.quantity
            row.mark_price = position.mark_price
            row.stop_loss = position.stop_loss
            row.take_profit = position.take_profit
            row.realized_pnl = position.realized_pnl
            row.unrealized_pnl = position.unrealized_pnl
            row.fees_paid = position.fees_paid

        # Anything still here is a row the bot no longer holds: the position closed.
        for stale in by_symbol.values():
            stale.is_open = False
            stale.unrealized_pnl = 0.0

    async def _sync_trades(self, session: Any, trades: list[Trade]) -> None:
        """Insert closed trades that are not already stored.

        Keyed on ``trade_id``. A duplicate here would silently corrupt every performance
        figure derived from this table, which is worse than a missing row because it still
        looks like an answer.
        """
        if not trades:
            return
        # The domain trade id *is* the row's primary key, so "already stored" is a simple
        # key lookup and a re-sync cannot duplicate a trade.
        known = set(
            (
                await session.execute(
                    select(TradeRecord.id).where(TradeRecord.bot_id == self.bot_id)
                )
            )
            .scalars()
            .all()
        )
        for trade in trades:
            if trade.trade_id in known:
                continue
            session.add(
                TradeRecord(
                    id=trade.trade_id,
                    user_id=self.user_id,
                    bot_id=self.bot_id,
                    symbol=trade.symbol,
                    side=trade.side,
                    status=trade.status,
                    quantity=trade.quantity,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    entry_time=trade.entry_time,
                    exit_time=trade.exit_time,
                    gross_pnl=trade.gross_pnl,
                    net_pnl=trade.net_pnl,
                    fees=trade.fees,
                    slippage_cost=trade.slippage_cost,
                    exit_reason=trade.exit_reason,
                    strategy_name=trade.strategy_name,
                )
            )
