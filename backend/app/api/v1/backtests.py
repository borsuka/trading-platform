"""Backtesting endpoints.

Every response carries an explicit disclaimer. That is not legal decoration: a backtest is a
simulation over one history with assumed costs, and presenting its numbers without that framing
is the single most misleading thing a trading platform can do.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, Query, status

from app.api.dependencies import CurrentUser, SessionDep
from app.api.schemas import (
    BacktestDetailResponse,
    BacktestRequest,
    BacktestResponse,
    MessageResponse,
)
from app.backtesting.engine import BacktestConfig, BacktestEngine
from app.core.clock import utcnow
from app.core.enums import BacktestStatus
from app.core.exceptions import ConflictError
from app.core.logging import get_logger
from app.database.models import Backtest
from app.database.repositories import BacktestRepository
from app.database.session import session_scope
from app.market_data.providers import generate_synthetic_candles
from app.risk.limits import RiskLimits
from app.strategies.registry import create_strategy

logger = get_logger(__name__)
router = APIRouter(prefix="/backtests", tags=["backtests"])

SYNTHETIC_NOTICE = (
    "This run used SYNTHETIC data generated from a random walk. It demonstrates the engine "
    "and the strategy's logic; it is not market data and its results are not a performance "
    "claim of any kind."
)


def _to_response(record: Backtest) -> BacktestResponse:
    return BacktestResponse(
        id=record.id,
        name=record.name,
        strategy_type=record.strategy_type,
        symbol=next(iter(record.symbols.get("symbols", ["?"]))),
        interval=record.interval,
        status=record.status,
        progress=record.progress,
        metrics=record.metrics,
        error_message=record.error_message,
        created_at=record.created_at,
        completed_at=record.completed_at,
    )


@router.get("", response_model=list[BacktestResponse])
async def list_backtests(
    user: CurrentUser, session: SessionDep, limit: int = Query(default=50, ge=1, le=200)
) -> list[BacktestResponse]:
    records = await BacktestRepository(session).recent_for_owner(user.id, limit=limit)
    return [_to_response(r) for r in records]


@router.post("", response_model=BacktestResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_backtest(
    payload: BacktestRequest,
    background: BackgroundTasks,
    user: CurrentUser,
    session: SessionDep,
) -> BacktestResponse:
    """Queue a backtest.

    Runs in the background because a multi-thousand-bar run takes seconds to minutes; poll the
    detail endpoint for progress.
    """
    # Validate the strategy configuration now, so a bad parameter fails immediately rather
    # than inside a background task the user has to go looking for.
    create_strategy(payload.strategy_type, payload.parameters)
    limits = RiskLimits(**payload.risk.model_dump())

    now = utcnow()
    record = Backtest(
        user_id=user.id,
        name=payload.name,
        strategy_type=payload.strategy_type,
        parameters=payload.parameters,
        risk_config=limits.model_dump(),
        symbols={"symbols": [payload.symbol.upper()]},
        interval=payload.interval,
        start_date=now,
        end_date=now,
        initial_balance=payload.initial_balance,
        status=BacktestStatus.PENDING,
    )
    await BacktestRepository(session).add(record)
    # Commit before scheduling. The background task opens its own session and would otherwise
    # race this transaction, finding no row for the id it was just handed - intermittently,
    # which is the worst way for it to fail.
    await session.commit()

    background.add_task(
        _run_backtest,
        backtest_id=record.id,
        user_id=user.id,
        payload=payload,
    )
    return _to_response(record)


async def _run_backtest(
    *, backtest_id: str, user_id: str, payload: BacktestRequest
) -> None:
    """Execute a backtest and persist the result.

    Opens its own session: the request's session is closed by the time this runs.
    """
    async with session_scope() as session:
        repository = BacktestRepository(session)
        record = await repository.get_for_owner(backtest_id, user_id)
        if record is None:
            logger.error("backtest.record_vanished", backtest_id=backtest_id)
            return
        record.status = BacktestStatus.RUNNING
        record.started_at = utcnow()
        await session.flush()

    try:
        candles = generate_synthetic_candles(
            payload.symbol.upper(),
            payload.interval,
            payload.bars,
            seed=payload.seed,
            volatility=0.01,
            start_price=30_000.0 if payload.symbol.upper().startswith("BTC") else 100.0,
        )
        engine = BacktestEngine(
            create_strategy(payload.strategy_type, payload.parameters),
            config=BacktestConfig(
                initial_balance=payload.initial_balance,
                warmup_bars=min(250, max(120, payload.bars // 8)),
            ),
            risk_limits=RiskLimits(**payload.risk.model_dump()),
        )
        result = await engine.run(candles)

        validation: dict[str, Any] | None = None
        if payload.run_validation:
            from app.backtesting.validation import monte_carlo, walk_forward

            wf = await walk_forward(
                payload.strategy_type,
                payload.parameters,
                candles,
                train_bars=max(500, payload.bars // 4),
                test_bars=max(200, payload.bars // 12),
                config=engine.config,
                risk_limits=engine.risk_limits,
            )
            mc = monte_carlo(result, simulations=500)
            validation = {"walk_forward": wf.to_dict(), "monte_carlo": mc.to_dict()}

    except Exception as exc:
        logger.exception("backtest.failed", backtest_id=backtest_id)
        async with session_scope() as session:
            record = await BacktestRepository(session).get_for_owner(backtest_id, user_id)
            if record is not None:
                record.status = BacktestStatus.FAILED
                record.error_message = f"{type(exc).__name__}: {exc}"
                record.completed_at = utcnow()
        return

    async with session_scope() as session:
        record = await BacktestRepository(session).get_for_owner(backtest_id, user_id)
        if record is None:
            return
        warnings = [*result.warnings, *result.metrics.reliability_warnings]
        if payload.data_source == "synthetic":
            warnings.insert(0, SYNTHETIC_NOTICE)

        record.status = BacktestStatus.COMPLETED
        record.progress = 1.0
        record.metrics = result.metrics.to_dict()
        record.equity_curve = {
            "points": result.equity_curve,
            "drawdown": result.drawdown_curve,
            "monthly": result.monthly_returns,
            "warnings": warnings,
            "trade_distribution": result.trade_distribution,
            "rejections": result.rejections[:100],
            "validation": validation,
        }
        record.start_date = result.start or record.start_date
        record.end_date = result.end or record.end_date
        record.completed_at = utcnow()
    logger.info(
        "backtest.completed",
        backtest_id=backtest_id,
        trades=result.metrics.total_trades,
        total_return=round(result.metrics.total_return, 4),
    )


@router.get("/{backtest_id}", response_model=BacktestDetailResponse)
async def get_backtest(
    backtest_id: str, user: CurrentUser, session: SessionDep
) -> BacktestDetailResponse:
    record = await BacktestRepository(session).require_for_owner(backtest_id, user.id)
    curve = record.equity_curve or {}
    base = _to_response(record)
    return BacktestDetailResponse(
        **base.model_dump(),
        parameters=record.parameters,
        equity_curve=curve.get("points", []),
        monthly_returns=curve.get("monthly", {}),
        warnings=curve.get("warnings", []),
    )


@router.get("/{backtest_id}/validation", response_model=dict)
async def backtest_validation(
    backtest_id: str, user: CurrentUser, session: SessionDep
) -> dict[str, Any]:
    """Robustness analysis, when it was requested for this run."""
    record = await BacktestRepository(session).require_for_owner(backtest_id, user.id)
    validation = (record.equity_curve or {}).get("validation")
    if validation is None:
        return {
            "available": False,
            "message": (
                "Robustness analysis was not requested for this run. Re-run with "
                "run_validation=true to see walk-forward and Monte Carlo results. "
                "An in-sample backtest on its own is weak evidence."
            ),
        }
    return {"available": True, **validation}


@router.delete("/{backtest_id}", response_model=MessageResponse)
async def delete_backtest(
    backtest_id: str, user: CurrentUser, session: SessionDep
) -> MessageResponse:
    repository = BacktestRepository(session)
    record = await repository.require_for_owner(backtest_id, user.id)
    if record.status is BacktestStatus.RUNNING:
        raise ConflictError("Cannot delete a backtest while it is running")
    await repository.delete(record)
    return MessageResponse(message="Backtest deleted")
