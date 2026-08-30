"""Orders, positions, trades, signals and the portfolio view.

Read-only apart from position closing. There is deliberately **no** endpoint that places an
arbitrary order: every order the platform sends originates from a strategy signal that has
passed the risk manager. A manual-order endpoint would be a second execution path that skips
sizing, exposure checks and the kill switch — exactly the bypass the architecture exists to
prevent.

Closing is the exception, and is always allowed: reducing exposure is never the risky direction.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query

from app.api.dependencies import CurrentUser, SessionDep
from app.api.schemas import (
    MessageResponse,
    OrderResponse,
    PortfolioResponse,
    PositionResponse,
    SignalResponse,
    TradeResponse,
)
from app.core.enums import ExitReason
from app.core.exceptions import NotFoundError
from app.database.repositories import (
    OrderRepository,
    PositionRepository,
    SignalRepository,
    TradeRepository,
)
from app.paper_trading.runtime import bot_registry

router = APIRouter(tags=["trading"])


# --------------------------------------------------------------------------- #
# Orders
# --------------------------------------------------------------------------- #
@router.get("/orders", response_model=list[OrderResponse])
async def list_orders(
    user: CurrentUser,
    session: SessionDep,
    bot_id: str | None = None,
    open_only: bool = False,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[OrderResponse]:
    repository = OrderRepository(session)
    records = (
        await repository.open_for_owner(user.id)
        if open_only
        else await repository.recent_for_owner(user.id, limit=limit, bot_id=bot_id)
    )
    return [OrderResponse.model_validate(r) for r in records]


@router.get("/orders/{client_order_id}", response_model=OrderResponse)
async def get_order(
    client_order_id: str, user: CurrentUser, session: SessionDep
) -> OrderResponse:
    """Look an order up by its idempotency key."""
    record = await OrderRepository(session).get_by_client_id(user.id, client_order_id)
    if record is None:
        raise NotFoundError(f"No order with client id {client_order_id}")
    return OrderResponse.model_validate(record)


# --------------------------------------------------------------------------- #
# Positions
# --------------------------------------------------------------------------- #
@router.get("/positions", response_model=list[dict])
async def list_positions(user: CurrentUser, session: SessionDep) -> list[dict[str, Any]]:
    """Open positions.

    Merges live runtime state with persisted records: a running bot holds the authoritative
    mark-to-market view, while the database covers bots that are not currently running.
    """
    live: dict[str, dict[str, Any]] = {}
    for bot in bot_registry.for_user(user.id):
        for position in bot.portfolio.positions():
            live[f"{bot.config.bot_id}:{position.symbol}"] = {
                "bot_id": bot.config.bot_id,
                "bot_name": bot.config.name,
                "symbol": position.symbol,
                "side": position.side.value,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "mark_price": position.mark_price,
                "unrealized_pnl": position.unrealized_pnl(),
                "unrealized_pnl_pct": position.unrealized_pnl_pct(),
                "realized_pnl": position.realized_pnl,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "trailing_stop": position.trailing_stop,
                "leverage": position.leverage,
                "opened_at": position.opened_at.isoformat(),
                "strategy_name": position.strategy_name,
                "source": "runtime",
            }

    for record in await PositionRepository(session).open_for_owner(user.id):
        key = f"{record.bot_id}:{record.symbol}"
        if key in live:
            continue
        live[key] = {
            **PositionResponse.model_validate(record).model_dump(mode="json"),
            "bot_id": record.bot_id,
            "source": "database",
        }
    return list(live.values())


@router.post("/positions/{bot_id}/{symbol}/close", response_model=MessageResponse)
async def close_position(
    bot_id: str, symbol: str, user: CurrentUser
) -> MessageResponse:
    """Close one position immediately.

    Not gated on risk approval: reducing exposure is always permitted, and blocking an exit
    because a limit is breached would be precisely backwards.
    """
    bot = bot_registry.get(bot_id)
    if bot is None or bot.config.user_id != user.id:
        raise NotFoundError(f"Bot {bot_id} is not currently running")

    closed = await bot.close_position(symbol, reason=ExitReason.MANUAL)
    if not closed:
        raise NotFoundError(f"No open position on {symbol.upper()}")
    return MessageResponse(message=f"Close order submitted for {symbol.upper()}")


# --------------------------------------------------------------------------- #
# Trades
# --------------------------------------------------------------------------- #
@router.get("/trades", response_model=list[dict])
async def list_trades(
    user: CurrentUser,
    session: SessionDep,
    bot_id: str | None = None,
    symbol: str | None = None,
    since: datetime | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    records = await TradeRepository(session).closed_for_owner(
        user.id, limit=limit, offset=offset, bot_id=bot_id, symbol=symbol, since=since
    )
    persisted = [
        TradeResponse.model_validate(r).model_dump(mode="json") for r in records
    ]

    # Include trades from running bots that have not yet been flushed to the database.
    runtime: list[dict[str, Any]] = []
    for bot in bot_registry.for_user(user.id):
        if bot_id and bot.config.bot_id != bot_id:
            continue
        for trade in bot.portfolio.closed_trades():
            if symbol and trade.symbol != symbol.upper():
                continue
            runtime.append(
                {
                    "id": trade.trade_id,
                    "symbol": trade.symbol,
                    "side": trade.side.value,
                    "quantity": trade.quantity,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "entry_time": trade.entry_time.isoformat(),
                    "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
                    "net_pnl": trade.net_pnl,
                    "fees": trade.fees,
                    "return_pct": trade.return_pct,
                    "r_multiple": trade.r_multiple,
                    "exit_reason": trade.exit_reason.value if trade.exit_reason else None,
                    "strategy_name": trade.strategy_name,
                    "bot_id": bot.config.bot_id,
                    "source": "runtime",
                }
            )
    combined = runtime + persisted
    combined.sort(key=lambda t: str(t.get("exit_time") or ""), reverse=True)
    return combined[:limit]


@router.get("/trades/summary", response_model=dict)
async def trade_summary(user: CurrentUser, session: SessionDep) -> dict[str, Any]:
    """Aggregate closed-trade statistics across every bot."""
    persisted = await TradeRepository(session).performance_summary(user.id)

    runtime_trades = [
        trade
        for bot in bot_registry.for_user(user.id)
        for trade in bot.portfolio.closed_trades()
    ]
    if runtime_trades:
        wins = sum(1 for t in runtime_trades if t.is_win)
        persisted["total_trades"] += len(runtime_trades)
        persisted["wins"] += wins
        persisted["losses"] += len(runtime_trades) - wins
        persisted["net_pnl"] += sum(t.net_pnl for t in runtime_trades)
        persisted["fees"] += sum(t.fees for t in runtime_trades)
        total = persisted["total_trades"]
        persisted["win_rate"] = (persisted["wins"] / total) if total else 0.0
    return persisted


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
@router.get("/signals", response_model=list[SignalResponse])
async def list_signals(
    user: CurrentUser,
    session: SessionDep,
    bot_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[SignalResponse]:
    """Recent signals, including those that produced no trade.

    Rejections are the interesting ones: they answer "why didn't the bot trade?".
    """
    records = await SignalRepository(session).recent_for_owner(
        user.id, limit=limit, bot_id=bot_id
    )
    return [SignalResponse.model_validate(r) for r in records]


# --------------------------------------------------------------------------- #
# Portfolio
# --------------------------------------------------------------------------- #
@router.get("/portfolio", response_model=dict)
async def portfolio(user: CurrentUser, bot_id: str | None = None) -> dict[str, Any]:
    """Aggregate portfolio across running bots."""
    bots = bot_registry.for_user(user.id)
    if bot_id:
        bots = [b for b in bots if b.config.bot_id == bot_id]
    if not bots:
        return {
            "bots": [],
            "aggregate": PortfolioResponse(
                equity=0.0, cash=0.0, realized_pnl=0.0, unrealized_pnl=0.0,
                fees_paid=0.0, total_return=0.0, drawdown=0.0, peak_equity=0.0,
                open_positions=0, total_exposure=0.0, closed_trades=0,
                wins=0, losses=0, win_rate=0.0, profit_factor=0.0,
            ).model_dump(),
        }

    per_bot = [
        {"bot_id": b.config.bot_id, "name": b.config.name, **b.portfolio.statistics()}
        for b in bots
    ]
    starting = sum(b.portfolio.starting_balance for b in bots)
    equity = sum(b.portfolio.equity() for b in bots)
    closed = [t for b in bots for t in b.portfolio.closed_trades()]
    wins = [t for t in closed if t.is_win]
    gross_profit = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in closed if not t.is_win))

    aggregate = PortfolioResponse(
        equity=equity,
        cash=sum(b.portfolio.cash for b in bots),
        realized_pnl=sum(b.portfolio.realized_pnl for b in bots),
        unrealized_pnl=sum(b.portfolio.unrealized_pnl() for b in bots),
        fees_paid=sum(b.portfolio.fees_paid for b in bots),
        total_return=((equity - starting) / starting) if starting else 0.0,
        drawdown=max((b.portfolio.drawdown() for b in bots), default=0.0),
        peak_equity=sum(b.portfolio.peak_equity for b in bots),
        open_positions=sum(len(b.portfolio.positions()) for b in bots),
        total_exposure=sum(b.portfolio.total_exposure() for b in bots),
        closed_trades=len(closed),
        wins=len(wins),
        losses=len(closed) - len(wins),
        win_rate=(len(wins) / len(closed)) if closed else 0.0,
        profit_factor=(gross_profit / gross_loss) if gross_loss > 0 else 0.0,
    )
    return {"bots": per_bot, "aggregate": aggregate.model_dump()}


@router.get("/portfolio/equity-curve", response_model=list[dict])
async def equity_curve(
    user: CurrentUser, bot_id: str | None = None, limit: int = Query(default=1000, le=5000)
) -> list[dict[str, Any]]:
    """Equity curve points from running bots."""
    bots = bot_registry.for_user(user.id)
    if bot_id:
        bots = [b for b in bots if b.config.bot_id == bot_id]

    points: list[dict[str, Any]] = []
    for bot in bots:
        points.extend(
            {
                "bot_id": bot.config.bot_id,
                "timestamp": s.timestamp.isoformat(),
                "equity": round(s.equity, 4),
                "drawdown": round(s.drawdown, 6),
                "cash": round(s.cash, 4),
                "exposure": round(s.total_exposure, 4),
            }
            for s in bot.portfolio.snapshots()
        )
    points.sort(key=lambda p: p["timestamp"])
    return points[-limit:]
