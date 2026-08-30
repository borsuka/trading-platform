"""Order manager.

This module owns the single most dangerous operation in the platform: sending an order. Its
whole design is organised around one failure mode.

The timeout problem
-------------------
When ``create_order`` times out, the order may have reached the venue or it may not. The
request failed; the *order* may not have. Retrying blindly is how a bot ends up with double the
intended position — and it happens exactly when conditions are worst, because that is when
venues time out.

The rule here is absolute: **a write that fails ambiguously is never retried. It is
resolved by querying.** Every order carries a ``client_order_id`` generated before the first
attempt, so the venue can be asked "do you have this order?" and the answer is authoritative.

* Venue says yes → adopt its state. Nothing more to do.
* Venue says no → the order never landed, and only then is a fresh submission safe.
* Venue cannot be reached → the order is marked ``UNKNOWN`` and the bot **halts**. An unknown
  position is a state-corruption event, not something to guess about.

Read operations are freely retried; they have no side effects.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime

from app.core.clock import utcnow
from app.core.domain import Order, OrderRequest, Position, new_id
from app.core.enums import (
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    TimeInForce,
)
from app.core.exceptions import (
    ExchangeConnectionError,
    ExchangeError,
    ExchangeRateLimitError,
    ExchangeTimeoutError,
    ExecutionError,
    OrderRejectedError,
)
from app.core.logging import get_logger
from app.exchanges.base import ExchangeAdapter
from app.risk.manager import RiskAssessment

logger = get_logger(__name__)


@dataclass(slots=True)
class RetryPolicy:
    """Backoff configuration.

    Applies to *reads* and to writes that failed unambiguously (an explicit rejection before
    the request reached the matching engine). It never applies to an ambiguous write.
    """

    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0
    backoff_multiplier: float = 2.0

    def delay_for(self, attempt: int) -> float:
        delay = self.base_delay_seconds * (self.backoff_multiplier ** max(0, attempt - 1))
        return min(delay, self.max_delay_seconds)


@dataclass(slots=True)
class ExecutionResult:
    """Outcome of an order submission."""

    order: Order | None
    submitted: bool
    reason: str = ""
    attempts: int = 1
    recovered_from_timeout: bool = False
    requires_halt: bool = False

    @property
    def succeeded(self) -> bool:
        return self.submitted and self.order is not None

    @property
    def filled(self) -> bool:
        return self.order is not None and self.order.status is OrderStatus.FILLED


@dataclass(slots=True)
class OrderManagerStats:
    submitted: int = 0
    filled: int = 0
    rejected: int = 0
    cancelled: int = 0
    timeouts: int = 0
    timeouts_recovered: int = 0
    duplicates_prevented: int = 0
    halts: int = 0
    retries: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "submitted": self.submitted,
            "filled": self.filled,
            "rejected": self.rejected,
            "cancelled": self.cancelled,
            "timeouts": self.timeouts,
            "timeouts_recovered": self.timeouts_recovered,
            "duplicates_prevented": self.duplicates_prevented,
            "halts": self.halts,
            "retries": self.retries,
        }


class OrderManager:
    """Submits, tracks and reconciles orders against an exchange adapter."""

    def __init__(
        self,
        exchange: ExchangeAdapter,
        *,
        retry_policy: RetryPolicy | None = None,
        state_query_attempts: int = 3,
        state_query_delay: float = 1.0,
    ) -> None:
        self.exchange = exchange
        self.retry_policy = retry_policy or RetryPolicy()
        self.state_query_attempts = state_query_attempts
        self.state_query_delay = state_query_delay
        self.stats = OrderManagerStats()
        self._orders: dict[str, Order] = {}
        self._in_flight: set[str] = set()
        self._halt_reason: str | None = None

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    @property
    def is_halted(self) -> bool:
        """True when an unresolvable order state has been encountered."""
        return self._halt_reason is not None

    @property
    def halt_reason(self) -> str | None:
        return self._halt_reason

    def clear_halt(self, *, cleared_by: str) -> None:
        """Clear a halt after a human has reconciled the ambiguous order."""
        if not cleared_by:
            raise ValueError("cleared_by is required: halt clearance must be attributable")
        logger.warning(
            "order_manager.halt_cleared", reason=self._halt_reason, cleared_by=cleared_by
        )
        self._halt_reason = None

    def known_orders(self) -> list[Order]:
        return list(self._orders.values())

    def get_tracked(self, client_order_id: str) -> Order | None:
        return self._orders.get(client_order_id)

    # ------------------------------------------------------------------ #
    # Submission
    # ------------------------------------------------------------------ #
    async def submit(
        self,
        request: OrderRequest,
        *,
        assessment: RiskAssessment | None = None,
    ) -> ExecutionResult:
        """Submit an order.

        ``assessment`` is not optional in practice: the bot runtime always supplies one, and
        passing a rejected assessment raises. It is typed as optional only so that recovery
        and reconciliation paths, which are not new risk decisions, can reuse this method.
        """
        if assessment is not None:
            assessment.require_approval()

        if self.is_halted:
            return ExecutionResult(
                order=None,
                submitted=False,
                reason=f"Order manager halted: {self._halt_reason}",
                requires_halt=True,
            )

        client_id = request.client_order_id

        # Local idempotency: the same id is never submitted twice from this process.
        existing = self._orders.get(client_id)
        if existing is not None:
            self.stats.duplicates_prevented += 1
            logger.info(
                "order_manager.duplicate_suppressed",
                client_order_id=client_id,
                status=existing.status.value,
            )
            return ExecutionResult(
                order=existing,
                submitted=False,
                reason="order with this client_order_id has already been submitted",
            )
        if client_id in self._in_flight:
            self.stats.duplicates_prevented += 1
            return ExecutionResult(
                order=None,
                submitted=False,
                reason="an order with this client_order_id is already in flight",
            )

        self._in_flight.add(client_id)
        try:
            return await self._submit_once(request)
        finally:
            self._in_flight.discard(client_id)

    async def _submit_once(self, request: OrderRequest) -> ExecutionResult:
        client_id = request.client_order_id
        try:
            order = await self.exchange.create_order(request)
        except ExchangeTimeoutError as exc:
            # THE dangerous case. Do not resend; find out what actually happened.
            self.stats.timeouts += 1
            logger.warning(
                "order_manager.submission_timed_out",
                client_order_id=client_id,
                symbol=request.symbol,
                error=str(exc),
            )
            return await self._resolve_ambiguous_submission(request)
        except ExchangeRateLimitError as exc:
            # Rate limits are refused *before* the engine sees the order, so no order exists.
            return ExecutionResult(
                order=None,
                submitted=False,
                reason=f"rate limited by the venue: {exc.message}",
            )
        except OrderRejectedError as exc:
            self.stats.rejected += 1
            logger.info(
                "order_manager.rejected_by_venue",
                client_order_id=client_id,
                symbol=request.symbol,
                reason=exc.message,
            )
            return ExecutionResult(
                order=None, submitted=False, reason=f"venue rejected the order: {exc.message}"
            )
        except ExchangeConnectionError as exc:
            # A connection failure before the request was written is safe; one after is not,
            # and the adapter cannot always tell them apart. Treat it as ambiguous.
            logger.warning(
                "order_manager.connection_failed",
                client_order_id=client_id,
                error=str(exc),
            )
            return await self._resolve_ambiguous_submission(request)
        except ExchangeError as exc:
            logger.error(
                "order_manager.submission_failed",
                client_order_id=client_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return ExecutionResult(
                order=None, submitted=False, reason=f"exchange error: {exc.message}"
            )

        return self._record(order)

    async def _resolve_ambiguous_submission(
        self, request: OrderRequest
    ) -> ExecutionResult:
        """Determine whether a timed-out order actually reached the venue.

        Queries by ``client_order_id``. Only a definitive "no such order" permits resubmission,
        and even then this method does not resubmit — it reports, and the caller decides.
        """
        client_id = request.client_order_id
        last_error: str = "unknown"

        for attempt in range(1, self.state_query_attempts + 1):
            try:
                found = await self.exchange.get_order(client_id, symbol=request.symbol)
            except ExchangeError as exc:
                last_error = exc.message
                logger.warning(
                    "order_manager.state_query_failed",
                    client_order_id=client_id,
                    attempt=attempt,
                    error=exc.message,
                )
                if attempt < self.state_query_attempts:
                    await asyncio.sleep(self.state_query_delay * attempt)
                continue

            if found is not None:
                self.stats.timeouts_recovered += 1
                logger.info(
                    "order_manager.timeout_recovered",
                    client_order_id=client_id,
                    status=found.status.value,
                    filled=found.filled_quantity,
                )
                result = self._record(found)
                result.recovered_from_timeout = True
                result.attempts = attempt + 1
                return result

            # A definitive negative: the venue has no record of this id.
            logger.info(
                "order_manager.timeout_resolved_not_placed", client_order_id=client_id
            )
            return ExecutionResult(
                order=None,
                submitted=False,
                reason=(
                    "submission timed out and the venue has no record of the order; "
                    "it is safe to submit a new order with a new client_order_id"
                ),
                attempts=attempt + 1,
                recovered_from_timeout=True,
            )

        # Could not determine the outcome. This is the halt condition.
        self.stats.halts += 1
        self._halt_reason = (
            f"Could not determine the state of order {client_id} on {request.symbol} after "
            f"{self.state_query_attempts} attempts (last error: {last_error}). "
            "The order may or may not exist on the venue. Trading is halted until this is "
            "reconciled manually."
        )
        placeholder = Order.from_request(request)
        placeholder.status = OrderStatus.UNKNOWN
        placeholder.reject_reason = "state could not be determined"
        self._orders[client_id] = placeholder
        logger.error(
            "order_manager.halt",
            client_order_id=client_id,
            symbol=request.symbol,
            reason=self._halt_reason,
        )
        return ExecutionResult(
            order=placeholder,
            submitted=False,
            reason=self._halt_reason,
            attempts=self.state_query_attempts,
            requires_halt=True,
        )

    def _record(self, order: Order) -> ExecutionResult:
        self._orders[order.client_order_id] = order
        self.stats.submitted += 1
        if order.status is OrderStatus.FILLED:
            self.stats.filled += 1
        elif order.status is OrderStatus.REJECTED:
            self.stats.rejected += 1
        logger.info(
            "order_manager.submitted",
            client_order_id=order.client_order_id,
            exchange_order_id=order.exchange_order_id,
            symbol=order.symbol,
            side=order.side.value,
            type=order.order_type.value,
            quantity=order.quantity,
            status=order.status.value,
        )
        if order.status is OrderStatus.REJECTED:
            return ExecutionResult(
                order=order,
                submitted=False,
                reason=order.reject_reason or "rejected by the venue",
            )
        return ExecutionResult(order=order, submitted=True, reason="submitted")

    # ------------------------------------------------------------------ #
    # Convenience builders
    # ------------------------------------------------------------------ #
    async def open_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        *,
        assessment: RiskAssessment | None = None,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        leverage: float | None = None,
        metadata: dict | None = None,
    ) -> ExecutionResult:
        """Open or add to a position with a fresh idempotency key."""
        return await self.submit(
            OrderRequest(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                leverage=leverage,
                client_order_id=make_client_order_id("open"),
                metadata=metadata or {},
            ),
            assessment=assessment,
        )

    async def close_position(
        self,
        position: Position,
        *,
        reason: str = "manual",
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
    ) -> ExecutionResult:
        """Flatten a position with a reduce-only order.

        Closing is not gated on a risk assessment: reducing exposure is always permitted, and
        blocking an exit because a limit is breached would be precisely backwards.
        """
        if not position.is_open:
            return ExecutionResult(order=None, submitted=False, reason="position is not open")
        return await self.submit(
            OrderRequest(
                symbol=position.symbol,
                side=position.side.closing_side,
                order_type=order_type,
                quantity=position.quantity,
                price=price,
                reduce_only=True,
                client_order_id=make_client_order_id("close"),
                metadata={"exit_reason": reason},
            )
        )

    async def place_protective_orders(
        self,
        position: Position,
        *,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> list[ExecutionResult]:
        """Place exchange-side stop-loss and take-profit orders.

        Venue-side protection matters: if the bot process dies, a local stop dies with it,
        while an exchange-side stop keeps working.
        """
        results: list[ExecutionResult] = []
        closing_side = position.side.closing_side

        if stop_loss is not None:
            results.append(
                await self.submit(
                    OrderRequest(
                        symbol=position.symbol,
                        side=closing_side,
                        order_type=OrderType.STOP_MARKET,
                        quantity=position.quantity,
                        trigger_price=stop_loss,
                        reduce_only=True,
                        time_in_force=TimeInForce.GTC,
                        client_order_id=make_client_order_id("sl"),
                        metadata={"protective": "stop_loss"},
                    )
                )
            )
        if take_profit is not None:
            results.append(
                await self.submit(
                    OrderRequest(
                        symbol=position.symbol,
                        side=closing_side,
                        order_type=OrderType.TAKE_PROFIT_MARKET,
                        quantity=position.quantity,
                        trigger_price=take_profit,
                        reduce_only=True,
                        time_in_force=TimeInForce.GTC,
                        client_order_id=make_client_order_id("tp"),
                        metadata={"protective": "take_profit"},
                    )
                )
            )
        return results

    # ------------------------------------------------------------------ #
    # Cancellation
    # ------------------------------------------------------------------ #
    async def cancel(self, client_order_id: str, *, symbol: str | None = None) -> Order | None:
        """Cancel an order. Safe to retry: cancellation is idempotent by nature."""
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            try:
                order = await self.exchange.cancel_order(client_order_id, symbol=symbol)
            except (ExchangeTimeoutError, ExchangeConnectionError) as exc:
                # Unlike submission, a duplicate cancel is harmless.
                if attempt >= self.retry_policy.max_attempts:
                    logger.error(
                        "order_manager.cancel_failed",
                        client_order_id=client_order_id,
                        error=str(exc),
                    )
                    return None
                self.stats.retries += 1
                await asyncio.sleep(self.retry_policy.delay_for(attempt))
                continue
            except ExchangeError as exc:
                logger.warning(
                    "order_manager.cancel_rejected",
                    client_order_id=client_order_id,
                    error=exc.message,
                )
                return None
            self._orders[client_order_id] = order
            if order.status is OrderStatus.CANCELLED:
                self.stats.cancelled += 1
            return order
        return None

    async def cancel_all(self, symbol: str | None = None) -> list[Order]:
        """Cancel every open order, tolerating individual failures."""
        cancelled: list[Order] = []
        try:
            open_orders = await self.exchange.get_open_orders(symbol)
        except ExchangeError as exc:
            logger.error("order_manager.cancel_all_failed", error=exc.message)
            return cancelled
        for order in open_orders:
            result = await self.cancel(order.client_order_id, symbol=order.symbol)
            if result is not None:
                cancelled.append(result)
        return cancelled

    # ------------------------------------------------------------------ #
    # Synchronisation
    # ------------------------------------------------------------------ #
    async def refresh(self, client_order_id: str, *, symbol: str | None = None) -> Order | None:
        """Re-read an order from the venue. Reads are safe to retry."""
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            try:
                order = await self.exchange.get_order(client_order_id, symbol=symbol)
            except ExchangeError:
                if attempt >= self.retry_policy.max_attempts:
                    raise
                self.stats.retries += 1
                await asyncio.sleep(self.retry_policy.delay_for(attempt))
                continue
            if order is not None:
                self._orders[client_order_id] = order
            return order
        return None

    async def sync_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Refresh local state from the venue's open-order list."""
        open_orders = await self.exchange.get_open_orders(symbol)
        venue_ids = set()
        for order in open_orders:
            self._orders[order.client_order_id] = order
            venue_ids.add(order.client_order_id)

        # Anything we think is active but the venue does not list has reached a terminal
        # state. Query each individually rather than assuming it was cancelled.
        for client_id, tracked in list(self._orders.items()):
            if tracked.is_active and client_id not in venue_ids:
                refreshed = await self.refresh(client_id, symbol=tracked.symbol)
                if refreshed is None:
                    logger.warning(
                        "order_manager.tracked_order_vanished", client_order_id=client_id
                    )
                    tracked.status = OrderStatus.UNKNOWN
        return open_orders

    def forget(self, client_order_id: str) -> None:
        """Stop tracking a terminal order, to bound memory in long-running bots."""
        order = self._orders.get(client_order_id)
        if order is not None and order.status.is_terminal:
            del self._orders[client_order_id]

    def prune(self, *, keep: int = 1000) -> int:
        """Drop the oldest terminal orders, keeping the most recent ``keep``."""
        terminal = [o for o in self._orders.values() if o.status.is_terminal]
        if len(terminal) <= keep:
            return 0
        terminal.sort(key=lambda o: o.updated_at)
        removed = 0
        for order in terminal[: len(terminal) - keep]:
            del self._orders[order.client_order_id]
            removed += 1
        return removed


def make_client_order_id(prefix: str = "ord") -> str:
    """Generate an idempotency key.

    Kept short because several venues cap client order IDs at 32-36 characters.
    """
    return f"{prefix}-{new_id().replace('-', '')[:20]}"


@dataclass(slots=True)
class ExecutionContext:
    """Bundle passed from the bot runtime into execution helpers."""

    order_manager: OrderManager
    now: datetime = field(default_factory=utcnow)

    async def enter(
        self,
        *,
        symbol: str,
        side: OrderSide,
        quantity: float,
        assessment: RiskAssessment,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        strategy_name: str = "unknown",
    ) -> ExecutionResult:
        """Open a position and attach venue-side protection.

        If the entry fills but the protective orders fail, the position is closed immediately.
        An unprotected position is not an acceptable resting state — it is the one situation
        where the platform's whole risk model stops applying.
        """
        result = await self.order_manager.open_position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            assessment=assessment,
            metadata={"strategy": strategy_name},
        )
        if not result.succeeded or result.order is None:
            return result
        if not result.filled:
            return result
        if stop_loss is None and take_profit is None:
            return result

        position = Position(
            symbol=symbol,
            side=PositionSide.from_side(side),
            quantity=result.order.filled_quantity,
            entry_price=result.order.average_fill_price,
            opened_at=self.now,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=strategy_name,
        )
        protective = await self.order_manager.place_protective_orders(
            position, stop_loss=stop_loss, take_profit=take_profit
        )
        if stop_loss is not None and not any(r.succeeded for r in protective):
            logger.error(
                "execution.protection_failed_closing_position",
                symbol=symbol,
                quantity=position.quantity,
            )
            await self.order_manager.close_position(position, reason="protection_failed")
            raise ExecutionError(
                f"Could not place a stop-loss for the new {symbol} position; the position "
                "was closed immediately rather than left unprotected.",
                context={"symbol": symbol},
            )
        return result
