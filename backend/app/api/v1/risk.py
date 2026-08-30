"""Risk status, limits and kill-switch control."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from app.api.dependencies import CurrentUser, SessionDep
from app.api.schemas import (
    KillSwitchResetRequest,
    MessageResponse,
    RiskLimitsPayload,
    RiskStatusResponse,
)
from app.core.enums import AuditAction
from app.core.exceptions import ConflictError, NotFoundError
from app.database.repositories import (
    AuditLogRepository,
    BotRepository,
    RiskEventRepository,
)
from app.paper_trading.runtime import bot_registry
from app.risk.limits import RiskLimits

router = APIRouter(prefix="/risk", tags=["risk"])


@router.get("/status/{bot_id}", response_model=RiskStatusResponse)
async def risk_status(bot_id: str, user: CurrentUser) -> RiskStatusResponse:
    """Current risk posture for a running bot, including remaining headroom."""
    bot = bot_registry.get(bot_id)
    if bot is None or bot.config.user_id != user.id:
        raise NotFoundError(f"Bot {bot_id} is not currently running")
    return RiskStatusResponse.model_validate(bot.risk_manager.describe())


@router.get("/events", response_model=list[dict])
async def risk_events(
    user: CurrentUser,
    session: SessionDep,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Risk events: every block, breach and limit trip."""
    persisted = [
        {
            "event_type": r.event_type.value,
            "severity": r.severity.value,
            "symbol": r.symbol,
            "message": r.message,
            "limit_value": r.limit_value,
            "observed_value": r.observed_value,
            "occurred_at": r.occurred_at.isoformat(),
            "source": "database",
        }
        for r in await RiskEventRepository(session).recent_for_owner(user.id, limit=limit)
    ]
    runtime = [
        {**event.to_dict(), "bot_id": bot.config.bot_id, "source": "runtime"}
        for bot in bot_registry.for_user(user.id)
        for event in bot.risk_manager.events
    ]
    combined = runtime + persisted
    combined.sort(key=lambda e: str(e.get("occurred_at") or ""), reverse=True)
    return combined[:limit]


@router.get("/limits/{bot_id}", response_model=dict)
async def get_limits(
    bot_id: str, user: CurrentUser, session: SessionDep
) -> dict[str, Any]:
    record = await BotRepository(session).require_for_owner(bot_id, user.id)
    limits = (
        RiskLimits(**record.risk_config)
        if record.risk_config
        else RiskLimits.conservative()
    )
    return {
        "limits": limits.model_dump(),
        "max_simultaneous_risk": round(limits.max_simultaneous_risk, 5),
        "explanation": (
            f"With {limits.risk_per_trade:.2%} risked per trade and at most "
            f"{limits.max_concurrent_positions} concurrent positions, "
            f"{limits.max_simultaneous_risk:.1%} of equity can be at risk at once. "
            f"Trading stops for the day at a {limits.max_daily_loss:.1%} loss and the kill "
            f"switch trips at a {limits.max_drawdown:.1%} drawdown."
        ),
    }


@router.put("/limits/{bot_id}", response_model=dict)
async def update_limits(
    bot_id: str,
    payload: RiskLimitsPayload,
    user: CurrentUser,
    session: SessionDep,
) -> dict[str, Any]:
    """Update a bot's risk limits.

    Refused while the bot is running: changing the risk budget under an open position means the
    position was sized against limits that no longer apply. Stop, adjust, restart.
    """
    runtime = bot_registry.get(bot_id)
    if runtime is not None and runtime.is_running:
        raise ConflictError(
            "Stop the bot before changing its risk limits. Changing them while positions "
            "are open would leave those positions sized against limits that no longer apply."
        )

    repository = BotRepository(session)
    record = await repository.require_for_owner(bot_id, user.id)
    limits = RiskLimits(**payload.model_dump())
    record.risk_config = limits.model_dump()
    await session.flush()

    await AuditLogRepository(session).record(
        action=AuditAction.RISK_CONFIG_CHANGED,
        user_id=user.id,
        resource_type="bot",
        resource_id=bot_id,
        detail={
            "risk_per_trade": limits.risk_per_trade,
            "max_drawdown": limits.max_drawdown,
        },
    )
    return {"limits": limits.model_dump(), "message": "Risk limits updated"}


@router.get("/kill-switch/{bot_id}", response_model=dict)
async def kill_switch_status(bot_id: str, user: CurrentUser) -> dict[str, Any]:
    bot = bot_registry.get(bot_id)
    if bot is None or bot.config.user_id != user.id:
        raise NotFoundError(f"Bot {bot_id} is not currently running")
    return bot.risk_manager.kill_switch.describe()


@router.post("/kill-switch/{bot_id}/reset", response_model=MessageResponse)
async def reset_kill_switch(
    bot_id: str,
    payload: KillSwitchResetRequest,
    user: CurrentUser,
    session: SessionDep,
) -> MessageResponse:
    """Clear the kill switch.

    Requires an identified actor and is recorded in the audit log. A kill switch that anything
    other than a person can clear is not a safety mechanism.
    """
    bot = bot_registry.get(bot_id)
    if bot is None or bot.config.user_id != user.id:
        raise NotFoundError(f"Bot {bot_id} is not currently running")

    cleared = bot.risk_manager.kill_switch.reset(reset_by=user.email, note=payload.note)
    if cleared is None:
        return MessageResponse(message="The kill switch was not engaged")

    record = await BotRepository(session).require_for_owner(bot_id, user.id)
    record.kill_switch_active = False
    record.kill_switch_reason = None
    await session.flush()

    await AuditLogRepository(session).record(
        action=AuditAction.KILL_SWITCH_RESET,
        user_id=user.id,
        resource_type="bot",
        resource_id=bot_id,
        detail={"previous_reason": cleared.reason.value, "note": payload.note},
    )
    return MessageResponse(
        message=(
            f"Kill switch cleared (was: {cleared.reason.value}). "
            "The bot must be resumed separately."
        )
    )


@router.get("/audit", response_model=list[dict])
async def audit_log(
    user: CurrentUser,
    session: SessionDep,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, Any]]:
    """This account's audit trail."""
    entries = await AuditLogRepository(session).for_user(user.id, limit=limit)
    return [
        {
            "action": e.action.value,
            "resource_type": e.resource_type,
            "resource_id": e.resource_id,
            "success": e.success,
            "ip_address": e.ip_address,
            "detail": e.detail,
            "occurred_at": e.occurred_at.isoformat(),
        }
        for e in entries
    ]
