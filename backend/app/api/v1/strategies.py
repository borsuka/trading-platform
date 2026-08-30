"""Strategy catalogue and per-user strategy configuration."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.dependencies import CurrentUser, SessionDep
from app.api.schemas import (
    MessageResponse,
    StrategyCatalogEntry,
    StrategyCreateRequest,
    StrategyResponse,
    StrategyUpdateRequest,
)
from app.core.enums import AuditAction
from app.core.exceptions import ConflictError
from app.database.models import Strategy, StrategyVersion
from app.database.repositories import AuditLogRepository, StrategyRepository
from app.strategies.registry import create_strategy, describe_all

router = APIRouter(prefix="/strategies", tags=["strategies"])


@router.get("/catalog", response_model=list[StrategyCatalogEntry])
async def catalog() -> list[StrategyCatalogEntry]:
    """Every strategy the platform ships, with its parameter schema.

    Unauthenticated: this is product documentation, not user data.
    """
    return [StrategyCatalogEntry.model_validate(entry) for entry in describe_all()]


@router.get("", response_model=list[StrategyResponse])
async def list_strategies(
    user: CurrentUser, session: SessionDep, limit: int = 100, offset: int = 0
) -> list[StrategyResponse]:
    records = await StrategyRepository(session).list_for_owner(
        user.id, limit=min(limit, 200), offset=offset
    )
    return [StrategyResponse.model_validate(r) for r in records]


@router.post("", response_model=StrategyResponse, status_code=status.HTTP_201_CREATED)
async def create(
    payload: StrategyCreateRequest, user: CurrentUser, session: SessionDep
) -> StrategyResponse:
    """Create a configured strategy instance.

    Parameters are validated by instantiating the strategy, so an invalid configuration is
    rejected here rather than when a bot first tries to trade with it.
    """
    strategy = create_strategy(payload.strategy_type, payload.parameters)

    repository = StrategyRepository(session)
    if await repository.get_by_name(user.id, payload.name) is not None:
        raise ConflictError(f"You already have a strategy named {payload.name!r}")

    record = Strategy(
        user_id=user.id,
        name=payload.name,
        strategy_type=payload.strategy_type,
        description=payload.description or strategy.description,
        parameters=strategy.params.model_dump(),
        current_version=strategy.version,
    )
    await repository.add(record)

    session.add(
        StrategyVersion(
            strategy_id=record.id,
            version=strategy.version,
            parameters=record.parameters,
            changelog="initial version",
            created_by=user.id,
        )
    )
    await AuditLogRepository(session).record(
        action=AuditAction.STRATEGY_CREATED,
        user_id=user.id,
        resource_type="strategy",
        resource_id=record.id,
        detail={"strategy_type": payload.strategy_type},
    )
    return StrategyResponse.model_validate(record)


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(
    strategy_id: str, user: CurrentUser, session: SessionDep
) -> StrategyResponse:
    record = await StrategyRepository(session).require_for_owner(strategy_id, user.id)
    return StrategyResponse.model_validate(record)


@router.patch("/{strategy_id}", response_model=StrategyResponse)
async def update(
    strategy_id: str,
    payload: StrategyUpdateRequest,
    user: CurrentUser,
    session: SessionDep,
) -> StrategyResponse:
    """Update a strategy.

    A parameter change creates a **new immutable version** rather than mutating the existing
    one. Bots pin a version, so an edit cannot silently change the behaviour of a bot that is
    already running with real money.
    """
    repository = StrategyRepository(session)
    record = await repository.require_for_owner(strategy_id, user.id)

    if payload.parameters is not None:
        strategy = create_strategy(record.strategy_type, payload.parameters)
        record.parameters = strategy.params.model_dump()

        major, minor, _patch = (int(p) for p in record.current_version.split("."))
        record.current_version = f"{major}.{minor + 1}.0"
        session.add(
            StrategyVersion(
                strategy_id=record.id,
                version=record.current_version,
                parameters=record.parameters,
                changelog=payload.changelog or "parameters updated",
                created_by=user.id,
            )
        )
    if payload.description is not None:
        record.description = payload.description
    if payload.is_active is not None:
        record.is_active = payload.is_active

    await session.flush()
    await AuditLogRepository(session).record(
        action=AuditAction.STRATEGY_UPDATED,
        user_id=user.id,
        resource_type="strategy",
        resource_id=record.id,
        detail={"version": record.current_version},
    )
    return StrategyResponse.model_validate(record)


@router.get("/{strategy_id}/versions", response_model=list[dict])
async def versions(
    strategy_id: str, user: CurrentUser, session: SessionDep
) -> list[dict]:
    """Version history, newest first."""
    from sqlalchemy import select

    record = await StrategyRepository(session).require_for_owner(strategy_id, user.id)
    result = await session.execute(
        select(StrategyVersion)
        .where(StrategyVersion.strategy_id == record.id)
        .order_by(StrategyVersion.created_at.desc())
    )
    return [
        {
            "id": v.id,
            "version": v.version,
            "parameters": v.parameters,
            "changelog": v.changelog,
            "created_at": v.created_at.isoformat(),
        }
        for v in result.scalars().all()
    ]


@router.delete("/{strategy_id}", response_model=MessageResponse)
async def delete(
    strategy_id: str, user: CurrentUser, session: SessionDep
) -> MessageResponse:
    """Delete a strategy. Refused while a bot still references it."""
    from sqlalchemy import select

    from app.database.models import Bot

    repository = StrategyRepository(session)
    record = await repository.require_for_owner(strategy_id, user.id)

    result = await session.execute(
        select(Bot).where(Bot.strategy_id == record.id, Bot.user_id == user.id)
    )
    bots = list(result.scalars().all())
    if bots:
        raise ConflictError(
            f"This strategy is used by {len(bots)} bot(s): "
            f"{', '.join(b.name for b in bots)}. Delete them first."
        )

    await repository.delete(record)
    return MessageResponse(message="Strategy deleted")
