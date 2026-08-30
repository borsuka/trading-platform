"""Bot management and control.

Every control action here is deliberate about what it does *not* do. ``stop`` does not close
positions. ``emergency-stop`` does not close positions unless explicitly asked. The user makes
that call, not the API.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, status

from app.api.dependencies import CurrentUser, SessionDep
from app.api.schemas import (
    BotActionRequest,
    BotCreateRequest,
    BotEventResponse,
    BotResponse,
    BotRuntimeResponse,
    MessageResponse,
)
from app.core.enums import AuditAction, BotStatus
from app.core.exceptions import ConflictError, NotFoundError
from app.core.logging import get_logger
from app.database.models import Bot, Strategy, StrategyVersion
from app.database.repositories import (
    AuditLogRepository,
    BotRepository,
    StrategyRepository,
)
from app.paper_trading.factory import build_paper_bot
from app.paper_trading.runtime import TradingBot, bot_registry
from app.risk.limits import RiskLimits
from app.strategies.registry import create_strategy

logger = get_logger(__name__)
router = APIRouter(prefix="/bots", tags=["bots"])


def _to_response(record: Bot) -> BotResponse:
    return BotResponse(
        id=record.id,
        name=record.name,
        strategy_id=record.strategy_id,
        trading_mode=record.trading_mode,
        symbols=list(record.symbols.get("symbols", [])),
        interval=record.interval,
        status=record.status,
        kill_switch_active=record.kill_switch_active,
        last_error=record.last_error,
        created_at=record.created_at,
        started_at=record.started_at,
        stopped_at=record.stopped_at,
    )


def _require_runtime(bot_id: str, user_id: str) -> TradingBot:
    """Fetch a running bot, verifying ownership.

    Ownership is re-checked against the runtime object, not just the database record: the
    registry is process-global, so a bot id alone must never be enough to control it.
    """
    runtime = bot_registry.get(bot_id)
    if runtime is None or runtime.config.user_id != user_id:
        raise NotFoundError(f"Bot {bot_id} is not currently running")
    return runtime


@router.get("", response_model=list[BotResponse])
async def list_bots(user: CurrentUser, session: SessionDep) -> list[BotResponse]:
    records = await BotRepository(session).list_for_owner(user.id, limit=200)
    return [_to_response(r) for r in records]


@router.post("", response_model=BotResponse, status_code=status.HTTP_201_CREATED)
async def create_bot(
    payload: BotCreateRequest, user: CurrentUser, session: SessionDep
) -> BotResponse:
    """Create a bot. Always in paper mode.

    There is no way to create a live bot through this endpoint. Live requires the separate
    activation flow in ``/exchange-accounts/{id}/activate-live``, which runs the preflight.
    """
    repository = BotRepository(session)
    if await repository.get_by_name(user.id, payload.name) is not None:
        raise ConflictError(f"You already have a bot named {payload.name!r}")

    strategy_obj = create_strategy(payload.strategy_type, payload.strategy_parameters)
    limits = RiskLimits(**payload.risk.model_dump())

    strategies = StrategyRepository(session)
    strategy_record = await strategies.get_by_name(user.id, f"{payload.name} strategy")
    if strategy_record is None:
        strategy_record = Strategy(
            user_id=user.id,
            name=f"{payload.name} strategy",
            strategy_type=payload.strategy_type,
            description=strategy_obj.description,
            parameters=strategy_obj.params.model_dump(),
            current_version=strategy_obj.version,
        )
        await strategies.add(strategy_record)
        session.add(
            StrategyVersion(
                strategy_id=strategy_record.id,
                version=strategy_obj.version,
                parameters=strategy_record.parameters,
                changelog="created with bot",
                created_by=user.id,
            )
        )
        await session.flush()

    record = Bot(
        user_id=user.id,
        name=payload.name,
        strategy_id=strategy_record.id,
        trading_mode="paper",
        symbols={"symbols": payload.symbols},
        interval=payload.interval,
        status=BotStatus.CREATED,
        risk_config=limits.model_dump(),
        state={"starting_balance": payload.starting_balance},
    )
    await repository.add(record)
    logger.info("api.bot_created", bot_id=record.id, user_id=user.id)
    return _to_response(record)


@router.get("/{bot_id}", response_model=BotResponse)
async def get_bot(bot_id: str, user: CurrentUser, session: SessionDep) -> BotResponse:
    record = await BotRepository(session).require_for_owner(bot_id, user.id)
    return _to_response(record)


@router.get("/{bot_id}/runtime", response_model=BotRuntimeResponse)
async def runtime_state(bot_id: str, user: CurrentUser) -> BotRuntimeResponse:
    """Live runtime state. 404 when the bot is not running in this process."""
    runtime = _require_runtime(bot_id, user.id)
    return BotRuntimeResponse.model_validate(runtime.snapshot().to_dict())


@router.post("/{bot_id}/start", response_model=BotRuntimeResponse)
async def start_bot(
    bot_id: str, user: CurrentUser, session: SessionDep
) -> BotRuntimeResponse:
    """Start a bot in paper mode."""
    repository = BotRepository(session)
    record = await repository.require_for_owner(bot_id, user.id)

    existing = bot_registry.get(bot_id)
    if existing is not None and existing.is_running:
        raise ConflictError(f"Bot {record.name!r} is already running")

    strategy_record = await StrategyRepository(session).require_for_owner(
        record.strategy_id, user.id
    )
    limits = RiskLimits(**record.risk_config) if record.risk_config else RiskLimits.conservative()

    runtime = build_paper_bot(
        bot_id=record.id,
        user_id=user.id,
        name=record.name,
        strategy_name=strategy_record.strategy_type,
        strategy_parameters=strategy_record.parameters,
        symbols=record.symbols.get("symbols", []),
        interval=record.interval,
        starting_balance=float(record.state.get("starting_balance", 10_000.0)),
        risk_limits=limits,
        # A fresh paper exchange holds no positions, so there is nothing to reconcile
        # against and startup would otherwise fail on an empty comparison.
        reconcile_on_start=False,
    )
    if existing is not None:
        await bot_registry.unregister(bot_id)
    await bot_registry.register(runtime)
    await runtime.start()

    record.status = BotStatus.RUNNING
    record.started_at = runtime.snapshot().last_cycle_at or record.started_at
    record.last_error = None
    await session.flush()

    await AuditLogRepository(session).record(
        action=AuditAction.BOT_STARTED,
        user_id=user.id,
        resource_type="bot",
        resource_id=record.id,
        detail={"mode": runtime.mode},
    )
    return BotRuntimeResponse.model_validate(runtime.snapshot().to_dict())


@router.post("/{bot_id}/stop", response_model=MessageResponse)
async def stop_bot(
    bot_id: str,
    payload: BotActionRequest,
    user: CurrentUser,
    session: SessionDep,
) -> MessageResponse:
    """Stop a bot.

    Open positions stay open unless ``close_positions`` is set. Stopping a bot is an
    operational action; liquidating a position is a trading decision, and conflating them
    would force a sale at whatever price happens to be available.
    """
    runtime = _require_runtime(bot_id, user.id)
    await runtime.stop(close_positions=payload.close_positions)

    record = await BotRepository(session).require_for_owner(bot_id, user.id)
    record.status = BotStatus.STOPPED
    record.stopped_at = runtime.snapshot().last_cycle_at
    await session.flush()

    await AuditLogRepository(session).record(
        action=AuditAction.BOT_STOPPED,
        user_id=user.id,
        resource_type="bot",
        resource_id=bot_id,
        detail={"closed_positions": payload.close_positions},
    )
    return MessageResponse(
        message=(
            "Bot stopped and positions closed"
            if payload.close_positions
            else "Bot stopped. Open positions remain; close them from the positions page "
            "if you want to be flat."
        )
    )


@router.post("/{bot_id}/pause", response_model=MessageResponse)
async def pause_bot(bot_id: str, user: CurrentUser) -> MessageResponse:
    runtime = _require_runtime(bot_id, user.id)
    await runtime.pause()
    return MessageResponse(
        message="Bot paused. It continues to monitor positions but will not open new ones."
    )


@router.post("/{bot_id}/resume", response_model=MessageResponse)
async def resume_bot(bot_id: str, user: CurrentUser) -> MessageResponse:
    runtime = _require_runtime(bot_id, user.id)
    await runtime.resume()
    return MessageResponse(message="Bot resumed")


@router.post("/{bot_id}/emergency-stop", response_model=MessageResponse)
async def emergency_stop(
    bot_id: str,
    payload: BotActionRequest,
    user: CurrentUser,
    session: SessionDep,
) -> MessageResponse:
    """Trip the kill switch and cancel resting orders.

    Positions are closed only when explicitly requested: an emergency is frequently the worst
    moment to be a forced seller, and the operator is better placed than the software to judge
    whether exiting now beats holding through it.
    """
    runtime = _require_runtime(bot_id, user.id)
    await runtime.emergency_stop(
        actor=user.email, close_positions=payload.close_positions, note=payload.note
    )

    record = await BotRepository(session).require_for_owner(bot_id, user.id)
    record.status = BotStatus.HALTED
    record.kill_switch_active = True
    await session.flush()

    await AuditLogRepository(session).record(
        action=AuditAction.EMERGENCY_STOP,
        user_id=user.id,
        resource_type="bot",
        resource_id=bot_id,
        detail={"closed_positions": payload.close_positions, "note": payload.note},
    )
    return MessageResponse(
        message=(
            "Emergency stop engaged. Orders cancelled"
            + (" and positions closed." if payload.close_positions else "; positions remain open.")
        )
    )


@router.post("/{bot_id}/cycle", response_model=dict)
async def run_one_cycle(bot_id: str, user: CurrentUser) -> dict[str, Any]:
    """Run a single decision cycle immediately.

    Exists so paper trading can be driven deterministically from the UI and from tests,
    instead of waiting for the poll timer.
    """
    runtime = _require_runtime(bot_id, user.id)
    signals = await runtime.run_cycle()
    return {
        "cycles": runtime.snapshot().cycles,
        "signals": [s.to_dict() for s in signals],
        "runtime": runtime.snapshot().to_dict(),
    }


@router.get("/{bot_id}/events", response_model=list[BotEventResponse])
async def bot_events(
    bot_id: str, user: CurrentUser, limit: int = 100
) -> list[BotEventResponse]:
    runtime = _require_runtime(bot_id, user.id)
    events = runtime.events[-min(limit, 500) :]
    return [BotEventResponse.model_validate(e.to_dict()) for e in reversed(events)]


@router.delete("/{bot_id}", response_model=MessageResponse)
async def delete_bot(
    bot_id: str, user: CurrentUser, session: SessionDep
) -> MessageResponse:
    """Delete a bot. Refused while it is running."""
    repository = BotRepository(session)
    record = await repository.require_for_owner(bot_id, user.id)

    runtime = bot_registry.get(bot_id)
    if runtime is not None and runtime.is_running:
        raise ConflictError("Stop the bot before deleting it")

    await bot_registry.unregister(bot_id)
    await repository.delete(record)
    return MessageResponse(message="Bot deleted")
