"""Portfolio manager and reconciliation.

Tracks cash, positions, PnL and exposure, and — critically — verifies that its own view agrees
with the exchange's.

Why reconciliation is not optional
----------------------------------
The database is a *record* of what the platform believes. The exchange is the *truth*. They
diverge for entirely ordinary reasons: a fill arrives during a process restart, a user trades
manually in the venue's UI, a websocket drops a message, a liquidation happens.

Every one of those makes the local position wrong, and a bot acting on a wrong position size
will size its next trade wrong, compute PnL wrong, and place a stop for the wrong quantity. So
on startup and on a schedule the manager compares the two, and on mismatch the bot **halts new
orders** and surfaces the difference. It does not silently adopt either side: an unexplained
divergence is a symptom, and trading through it risks compounding whatever caused it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from app.core.clock import utcnow
from app.core.domain import (
    Fill,
    Order,
    PortfolioSnapshot,
    Position,
    Trade,
)
from app.core.enums import ExitReason, PositionSide, TradeStatus
from app.core.logging import get_logger
from app.core.numeric import EPSILON, is_zero, safe_divide
from app.exchanges.base import ExchangeAdapter
from app.risk.manager import PortfolioView

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PositionDiscrepancy:
    """One disagreement between local and venue state."""

    symbol: str
    local_quantity: float
    exchange_quantity: float
    local_side: PositionSide
    exchange_side: PositionSide
    kind: str  # "missing_locally" | "missing_on_exchange" | "quantity" | "side"

    @property
    def difference(self) -> float:
        return self.exchange_quantity - self.local_quantity

    def describe(self) -> str:
        return (
            f"{self.symbol}: local {self.local_side.value} {self.local_quantity:g} vs "
            f"exchange {self.exchange_side.value} {self.exchange_quantity:g} ({self.kind})"
        )


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    """Result of comparing local state against the venue."""

    timestamp: datetime
    matched: int = 0
    discrepancies: tuple[PositionDiscrepancy, ...] = ()
    balance_difference: float = 0.0
    balance_tolerance: float = 0.0
    orders_adopted: int = 0
    error: str | None = None

    @property
    def is_clean(self) -> bool:
        return (
            not self.discrepancies
            and self.error is None
            and abs(self.balance_difference) <= self.balance_tolerance
        )

    def summary(self) -> str:
        if self.error is not None:
            return f"reconciliation failed: {self.error}"
        if self.is_clean:
            return f"{self.matched} position(s) reconciled, balances agree"
        parts = [d.describe() for d in self.discrepancies]
        if abs(self.balance_difference) > self.balance_tolerance:
            parts.append(f"balance differs by {self.balance_difference:+.2f}")
        return "; ".join(parts)


class PortfolioManager:
    """Authoritative local view of cash, positions, trades and PnL."""

    def __init__(
        self,
        *,
        starting_balance: float,
        quote_asset: str = "USDT",
        balance_tolerance: float = 0.01,
    ) -> None:
        if starting_balance <= 0:
            raise ValueError("starting_balance must be positive")
        self.starting_balance = starting_balance
        self.quote_asset = quote_asset
        self.balance_tolerance = balance_tolerance

        self._cash = starting_balance
        self._locked_margin = 0.0
        self._positions: dict[str, Position] = {}
        self._closed_trades: list[Trade] = []
        self._open_trades: dict[str, Trade] = {}
        self._realized_pnl = 0.0
        self._fees_paid = 0.0
        self._peak_equity = starting_balance
        self._snapshots: list[PortfolioSnapshot] = []
        self._trading_enabled = True
        self._halt_reason: str | None = None
        self._last_reconciled: datetime | None = None

    # ------------------------------------------------------------------ #
    # Read-only state
    # ------------------------------------------------------------------ #
    @property
    def cash(self) -> float:
        return self._cash

    @property
    def free_margin(self) -> float:
        return max(0.0, self._cash - self._locked_margin)

    @property
    def locked_margin(self) -> float:
        return self._locked_margin

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def fees_paid(self) -> float:
        return self._fees_paid

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def trading_enabled(self) -> bool:
        return self._trading_enabled

    @property
    def halt_reason(self) -> str | None:
        return self._halt_reason

    @property
    def last_reconciled_at(self) -> datetime | None:
        return self._last_reconciled

    def positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.is_open]

    def position(self, symbol: str) -> Position | None:
        position = self._positions.get(symbol)
        return position if position is not None and position.is_open else None

    def closed_trades(self) -> list[Trade]:
        return list(self._closed_trades)

    def snapshots(self) -> list[PortfolioSnapshot]:
        return list(self._snapshots)

    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl() for p in self.positions())

    def equity(self) -> float:
        return self._cash + self.unrealized_pnl()

    def total_exposure(self) -> float:
        return sum(p.notional() for p in self.positions())

    def drawdown(self) -> float:
        return max(0.0, safe_divide(self._peak_equity - self.equity(), self._peak_equity))

    def to_view(self) -> PortfolioView:
        """Read-only projection handed to the risk manager."""
        return PortfolioView(
            equity=self.equity(),
            available_margin=self.free_margin,
            positions=tuple(self.positions()),
            total_exposure=self.total_exposure(),
        )

    # ------------------------------------------------------------------ #
    # Halt control
    # ------------------------------------------------------------------ #
    def halt(self, reason: str) -> None:
        """Block new orders. Existing positions and their stops are left untouched."""
        self._trading_enabled = False
        self._halt_reason = reason
        logger.error("portfolio.halted", reason=reason)

    def resume(self, *, resumed_by: str) -> None:
        if not resumed_by:
            raise ValueError("resumed_by is required: resuming must be attributable")
        logger.warning(
            "portfolio.resumed", previous_reason=self._halt_reason, resumed_by=resumed_by
        )
        self._trading_enabled = True
        self._halt_reason = None

    # ------------------------------------------------------------------ #
    # Applying fills
    # ------------------------------------------------------------------ #
    def apply_fill(
        self,
        fill: Fill,
        *,
        leverage: float = 1.0,
        strategy_name: str | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        exit_reason: ExitReason | None = None,
    ) -> Trade | None:
        """Update cash, margin and positions from a fill.

        Returns the completed :class:`Trade` when the fill closed a position, otherwise
        ``None``.
        """
        self._cash -= fill.fee
        self._fees_paid += fill.fee

        position = self._positions.get(fill.symbol)
        incoming_side = PositionSide.from_side(fill.side)

        if position is None or not position.is_open:
            self._open_position(fill, incoming_side, leverage, strategy_name,
                                stop_loss, take_profit)
            return None

        if position.side is incoming_side:
            self._increase_position(position, fill, leverage)
            return None

        return self._reduce_position(position, fill, exit_reason)

    def _open_position(
        self,
        fill: Fill,
        side: PositionSide,
        leverage: float,
        strategy_name: str | None,
        stop_loss: float | None,
        take_profit: float | None,
    ) -> None:
        margin = fill.notional / max(leverage, 1.0)
        self._locked_margin += margin
        position = Position(
            symbol=fill.symbol,
            side=side,
            quantity=fill.quantity,
            entry_price=fill.price,
            opened_at=fill.timestamp,
            updated_at=fill.timestamp,
            leverage=max(leverage, 1.0),
            mark_price=fill.price,
            fees_paid=fill.fee,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=strategy_name,
            metadata={"margin": margin},
        )
        self._positions[fill.symbol] = position
        self._open_trades[fill.symbol] = Trade(
            symbol=fill.symbol,
            side=side,
            quantity=fill.quantity,
            entry_price=fill.price,
            entry_time=fill.timestamp,
            status=TradeStatus.OPEN,
            fees=fill.fee,
            strategy_name=strategy_name,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        logger.info(
            "portfolio.position_opened",
            symbol=fill.symbol,
            side=side.value,
            quantity=fill.quantity,
            entry=fill.price,
            margin=round(margin, 2),
        )

    def _increase_position(self, position: Position, fill: Fill, leverage: float) -> None:
        margin = fill.notional / max(position.leverage, leverage, 1.0)
        self._locked_margin += margin
        position.metadata["margin"] = position.metadata.get("margin", 0.0) + margin
        position.add(fill.quantity, fill.price)
        position.fees_paid += fill.fee

        trade = self._open_trades.get(fill.symbol)
        if trade is not None:
            total_cost = trade.entry_price * trade.quantity + fill.notional
            trade.quantity += fill.quantity
            trade.entry_price = safe_divide(total_cost, trade.quantity, fill.price)
            trade.fees += fill.fee

    def _reduce_position(
        self, position: Position, fill: Fill, exit_reason: ExitReason | None
    ) -> Trade | None:
        closing = min(fill.quantity, position.quantity)
        original = position.quantity
        released = position.metadata.get("margin", 0.0) * safe_divide(closing, original)

        realized = position.reduce(closing, fill.price)
        self._locked_margin = max(0.0, self._locked_margin - released)
        position.metadata["margin"] = max(
            0.0, position.metadata.get("margin", 0.0) - released
        )
        self._cash += realized
        self._realized_pnl += realized
        position.fees_paid += fill.fee

        trade = self._open_trades.get(fill.symbol)
        completed: Trade | None = None

        if not position.is_open:
            self._positions.pop(fill.symbol, None)
            if trade is not None:
                trade.fees += fill.fee
                trade.close(
                    price=fill.price,
                    at=fill.timestamp,
                    reason=exit_reason or ExitReason.SIGNAL,
                )
                self._closed_trades.append(trade)
                self._open_trades.pop(fill.symbol, None)
                completed = trade
            logger.info(
                "portfolio.position_closed",
                symbol=fill.symbol,
                exit=fill.price,
                realized_pnl=round(realized, 4),
                reason=(exit_reason or ExitReason.SIGNAL).value,
            )
            # A fill larger than the position flips it through flat.
            remainder = fill.quantity - closing
            if remainder > EPSILON:
                flipped = Fill(
                    fill_id=f"{fill.fill_id}-flip",
                    order_id=fill.order_id,
                    symbol=fill.symbol,
                    side=fill.side,
                    quantity=remainder,
                    price=fill.price,
                    fee=0.0,
                    fee_asset=fill.fee_asset,
                    role=fill.role,
                    timestamp=fill.timestamp,
                )
                self._open_position(
                    flipped,
                    PositionSide.from_side(fill.side),
                    position.leverage,
                    position.strategy_name,
                    None,
                    None,
                )
        elif trade is not None:
            # Partial close: bank the realised part, keep the trade open.
            trade.quantity -= closing
            trade.gross_pnl += realized
            trade.fees += fill.fee

        return completed

    # ------------------------------------------------------------------ #
    # Marking
    # ------------------------------------------------------------------ #
    def mark(self, symbol: str, price: float, *, at: datetime | None = None) -> None:
        position = self._positions.get(symbol)
        if position is not None and position.is_open:
            position.mark(price, at)
        self._peak_equity = max(self._peak_equity, self.equity())

    def mark_all(self, prices: dict[str, float], *, at: datetime | None = None) -> None:
        for symbol, price in prices.items():
            self.mark(symbol, price, at=at)

    def snapshot(self, *, at: datetime | None = None) -> PortfolioSnapshot:
        """Record and return a point on the equity curve."""
        equity = self.equity()
        self._peak_equity = max(self._peak_equity, equity)
        snap = PortfolioSnapshot(
            timestamp=at or utcnow(),
            cash=self._cash,
            equity=equity,
            unrealized_pnl=self.unrealized_pnl(),
            realized_pnl=self._realized_pnl,
            total_exposure=self.total_exposure(),
            position_count=len(self.positions()),
            fees_paid=self._fees_paid,
            peak_equity=self._peak_equity,
            drawdown=self.drawdown(),
        )
        self._snapshots.append(snap)
        return snap

    def set_protective_levels(
        self,
        symbol: str,
        *,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        trail_offset: float | None = None,
    ) -> Position | None:
        position = self.position(symbol)
        if position is None:
            return None
        if stop_loss is not None:
            position.stop_loss = stop_loss
        if take_profit is not None:
            position.take_profit = take_profit
        if trail_offset is not None:
            position.trail_offset = trail_offset
        trade = self._open_trades.get(symbol)
        if trade is not None:
            trade.stop_loss = position.stop_loss
            trade.take_profit = position.take_profit
        return position

    # ------------------------------------------------------------------ #
    # Reconciliation
    # ------------------------------------------------------------------ #
    async def reconcile(
        self,
        exchange: ExchangeAdapter,
        *,
        quantity_tolerance: float = 1e-8,
        now: datetime | None = None,
        halt_on_mismatch: bool = True,
    ) -> ReconciliationReport:
        """Compare local state against the venue.

        On any mismatch the portfolio halts new orders. It deliberately does **not** silently
        adopt the exchange's view: the difference has a cause, and adopting it without
        understanding the cause hides the bug that produced it.
        """
        moment = now or utcnow()
        try:
            venue_positions = await exchange.get_positions()
            venue_balance = await exchange.get_balance()
        except Exception as exc:  # adapter errors of any kind
            report = ReconciliationReport(
                timestamp=moment, error=f"could not read venue state: {exc}"
            )
            if halt_on_mismatch:
                self.halt(report.summary())
            return report

        discrepancies = self._compare_positions(venue_positions, quantity_tolerance)

        venue_cash = venue_balance.total(self.quote_asset)
        balance_difference = venue_cash - self._cash if venue_cash > 0 else 0.0

        report = ReconciliationReport(
            timestamp=moment,
            matched=len(self.positions()) - len(
                [d for d in discrepancies if d.kind != "missing_locally"]
            ),
            discrepancies=tuple(discrepancies),
            balance_difference=balance_difference,
            balance_tolerance=max(self.balance_tolerance, abs(self._cash) * 1e-6),
        )
        self._last_reconciled = moment

        if report.is_clean:
            logger.info("portfolio.reconciled", summary=report.summary())
            return report

        logger.error(
            "portfolio.reconciliation_mismatch",
            summary=report.summary(),
            discrepancy_count=len(discrepancies),
            balance_difference=round(balance_difference, 6),
        )
        if halt_on_mismatch:
            self.halt(f"Reconciliation mismatch: {report.summary()}")
        return report

    def _compare_positions(
        self, venue_positions: list[Position], tolerance: float
    ) -> list[PositionDiscrepancy]:
        local = {p.symbol: p for p in self.positions()}
        venue = {p.symbol: p for p in venue_positions if p.is_open}
        discrepancies: list[PositionDiscrepancy] = []

        for symbol in sorted(set(local) | set(venue)):
            local_position = local.get(symbol)
            venue_position = venue.get(symbol)

            if local_position is None and venue_position is not None:
                discrepancies.append(
                    PositionDiscrepancy(
                        symbol=symbol,
                        local_quantity=0.0,
                        exchange_quantity=venue_position.quantity,
                        local_side=PositionSide.FLAT,
                        exchange_side=venue_position.side,
                        kind="missing_locally",
                    )
                )
            elif local_position is not None and venue_position is None:
                discrepancies.append(
                    PositionDiscrepancy(
                        symbol=symbol,
                        local_quantity=local_position.quantity,
                        exchange_quantity=0.0,
                        local_side=local_position.side,
                        exchange_side=PositionSide.FLAT,
                        kind="missing_on_exchange",
                    )
                )
            elif local_position is not None and venue_position is not None:
                if local_position.side is not venue_position.side:
                    kind = "side"
                elif abs(local_position.quantity - venue_position.quantity) > tolerance:
                    kind = "quantity"
                else:
                    continue
                discrepancies.append(
                    PositionDiscrepancy(
                        symbol=symbol,
                        local_quantity=local_position.quantity,
                        exchange_quantity=venue_position.quantity,
                        local_side=local_position.side,
                        exchange_side=venue_position.side,
                        kind=kind,
                    )
                )
        return discrepancies

    def adopt_exchange_state(
        self,
        venue_positions: list[Position],
        venue_cash: float | None = None,
        *,
        adopted_by: str,
    ) -> None:
        """Overwrite local state with the venue's, after a human has reviewed the difference.

        Separated from :meth:`reconcile` on purpose: adoption is a decision, not a repair that
        should happen automatically.
        """
        if not adopted_by:
            raise ValueError("adopted_by is required: state adoption must be attributable")
        logger.warning(
            "portfolio.adopting_exchange_state",
            adopted_by=adopted_by,
            venue_positions=len(venue_positions),
        )
        self._positions = {p.symbol: p for p in venue_positions if p.is_open}
        self._locked_margin = sum(
            p.notional() / max(p.leverage, 1.0) for p in self._positions.values()
        )
        if venue_cash is not None:
            self._cash = venue_cash
        self._open_trades = {
            symbol: Trade(
                symbol=symbol,
                side=position.side,
                quantity=position.quantity,
                entry_price=position.entry_price,
                entry_time=position.opened_at,
                status=TradeStatus.OPEN,
                strategy_name=position.strategy_name,
            )
            for symbol, position in self._positions.items()
        }

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    def validate_invariants(self) -> list[str]:
        """Check internal consistency. A non-empty result is a state-corruption event."""
        problems: list[str] = []
        if self._cash < -EPSILON:
            problems.append(f"cash is negative: {self._cash:.6f}")
        if self._locked_margin < -EPSILON:
            problems.append(f"locked margin is negative: {self._locked_margin:.6f}")
        if self._locked_margin > self._cash + EPSILON and self._cash > 0:
            problems.append(
                f"locked margin {self._locked_margin:.2f} exceeds cash {self._cash:.2f}"
            )
        for position in self._positions.values():
            if position.quantity < 0:
                problems.append(f"{position.symbol}: negative quantity")
            if position.is_open and position.entry_price <= 0:
                problems.append(f"{position.symbol}: non-positive entry price")
            if position.side is PositionSide.FLAT and not is_zero(position.quantity):
                problems.append(f"{position.symbol}: flat position holds quantity")
        expected_cash = (
            self.starting_balance + self._realized_pnl - self._fees_paid
        )
        if abs(self._cash - expected_cash) > max(0.01, abs(expected_cash) * 1e-6):
            problems.append(
                f"cash {self._cash:.4f} does not equal starting balance + realised PnL "
                f"- fees ({expected_cash:.4f})"
            )
        return problems

    def statistics(self) -> dict[str, float | int]:
        """Summary metrics for the dashboard."""
        closed = self._closed_trades
        wins = [t for t in closed if t.is_win]
        losses = [t for t in closed if not t.is_win]
        gross_profit = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in losses))
        return {
            "equity": round(self.equity(), 6),
            "cash": round(self._cash, 6),
            "realized_pnl": round(self._realized_pnl, 6),
            "unrealized_pnl": round(self.unrealized_pnl(), 6),
            "fees_paid": round(self._fees_paid, 6),
            "total_return": round(
                safe_divide(self.equity() - self.starting_balance, self.starting_balance), 6
            ),
            "drawdown": round(self.drawdown(), 6),
            "peak_equity": round(self._peak_equity, 6),
            "open_positions": len(self.positions()),
            "total_exposure": round(self.total_exposure(), 6),
            "closed_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(safe_divide(len(wins), len(closed)), 6),
            "profit_factor": round(safe_divide(gross_profit, gross_loss), 6),
        }

    def apply_order(self, order: Order, **kwargs: object) -> list[Trade]:
        """Apply every fill on an order. Convenience for adapters that return whole orders."""
        completed: list[Trade] = []
        for fill in order.fills:
            trade = self.apply_fill(fill, **kwargs)  # type: ignore[arg-type]
            if trade is not None:
                completed.append(trade)
        return completed


@dataclass(slots=True)
class ReconciliationScheduler:
    """Runs reconciliation on an interval during a live session."""

    portfolio: PortfolioManager
    exchange: ExchangeAdapter
    interval_seconds: float = 300.0
    _last_run: datetime | None = field(default=None, init=False)

    def is_due(self, *, now: datetime | None = None) -> bool:
        moment = now or utcnow()
        if self._last_run is None:
            return True
        return (moment - self._last_run).total_seconds() >= self.interval_seconds

    async def run_if_due(
        self, *, now: datetime | None = None
    ) -> ReconciliationReport | None:
        if not self.is_due(now=now):
            return None
        moment = now or utcnow()
        self._last_run = moment
        return await self.portfolio.reconcile(self.exchange, now=moment)
