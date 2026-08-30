"""Risk manager — the last gate before execution.

Every order the platform sends passes through :meth:`RiskManager.evaluate`. There is no second
path, no bypass flag and no "trusted strategy" shortcut. A strategy proposes; the risk manager
disposes.

Check order matters. Cheap, absolute vetoes run first (kill switch, data quality, loss limits)
so that an expensive sizing computation is never performed for a trade that was never going to
be allowed. Sizing runs last, and its result is verified against the risk budget afterwards —
the calculation and the constraint are checked independently, because a sizing bug that
silently oversizes is exactly the failure this system exists to prevent.

The output is always a :class:`RiskAssessment`. Rejections are data, not exceptions: the bot
records them, shows them to the user, and continues. An exception is raised only when a caller
tries to act on a rejected assessment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from app.core.clock import utcnow
from app.core.domain import InstrumentSpec, Position
from app.core.enums import (
    KillSwitchReason,
    OrderSide,
    RiskDecision,
    RiskEventType,
    RiskSeverity,
    SignalAction,
)
from app.core.logging import get_logger
from app.core.numeric import bps, safe_divide
from app.market_data.models import MarketSnapshot
from app.market_data.validation import MarketDataValidator, ValidationResult
from app.risk.kill_switch import KillSwitch
from app.risk.limits import RiskLimits
from app.risk.sizing import (
    SizingRequest,
    SizingResult,
    calculate_position_size,
    estimate_worst_case_loss,
    reward_risk_ratio,
)
from app.risk.state import RiskState

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RiskEvent:
    """A recorded risk observation. Persisted for the audit trail and the risk page."""

    event_type: RiskEventType
    severity: RiskSeverity
    message: str
    occurred_at: datetime
    symbol: str | None = None
    limit_value: float | None = None
    observed_value: float | None = None
    action_taken: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "occurred_at": self.occurred_at.isoformat(),
            "symbol": self.symbol,
            "limit_value": self.limit_value,
            "observed_value": self.observed_value,
            "action_taken": self.action_taken,
        }


@dataclass(frozen=True, slots=True)
class PortfolioView:
    """Everything the risk manager needs to know about current exposure.

    A read-only projection, so the risk manager cannot mutate portfolio state as a side effect
    of evaluating a trade.
    """

    equity: float
    available_margin: float
    positions: tuple[Position, ...] = ()
    total_exposure: float = 0.0

    @property
    def position_count(self) -> int:
        return sum(1 for p in self.positions if p.is_open)

    def exposure_for(self, symbol: str) -> float:
        return sum(p.notional() for p in self.positions if p.is_open and p.symbol == symbol)

    def exposure_for_asset(self, asset: str) -> float:
        """Exposure to a base asset across every symbol that references it."""
        return sum(
            p.notional()
            for p in self.positions
            if p.is_open and p.symbol.upper().startswith(asset.upper())
        )

    def position_for(self, symbol: str) -> Position | None:
        for position in self.positions:
            if position.is_open and position.symbol == symbol:
                return position
        return None

    @property
    def leverage(self) -> float:
        return safe_divide(self.total_exposure, self.equity)


@dataclass(frozen=True, slots=True)
class TradeProposal:
    """A trade the pipeline would like to make."""

    symbol: str
    action: SignalAction
    entry_price: float
    stop_loss: float
    instrument: InstrumentSpec
    take_profit: float | None = None
    confidence: float = 1.0
    leverage: float = 1.0
    strategy_name: str = "unknown"
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def side(self) -> OrderSide:
        return self.action.order_side


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """The risk manager's verdict."""

    decision: RiskDecision
    quantity: float = 0.0
    reason: str = ""
    sizing: SizingResult | None = None
    events: tuple[RiskEvent, ...] = ()
    warnings: tuple[str, ...] = ()
    blocked_by: RiskEventType | None = None

    @property
    def approved(self) -> bool:
        return self.decision in {RiskDecision.APPROVED, RiskDecision.REDUCED}

    def require_approval(self) -> None:
        """Raise unless this assessment permits trading."""
        from app.core.exceptions import RiskViolationError

        if not self.approved:
            raise RiskViolationError(
                self.reason or "Risk manager rejected the trade",
                context={
                    "decision": self.decision.value,
                    "blocked_by": self.blocked_by.value if self.blocked_by else None,
                },
            )

    @classmethod
    def reject(
        cls,
        reason: str,
        event_type: RiskEventType,
        *,
        events: tuple[RiskEvent, ...] = (),
    ) -> RiskAssessment:
        return cls(
            decision=RiskDecision.REJECTED,
            reason=reason,
            events=events,
            blocked_by=event_type,
        )


class RiskManager:
    """Enforces every risk limit. The only gate between a signal and an order."""

    def __init__(
        self,
        limits: RiskLimits,
        state: RiskState,
        *,
        kill_switch: KillSwitch | None = None,
        platform_ceiling: RiskLimits | None = None,
        validator: MarketDataValidator | None = None,
    ) -> None:
        # A per-bot configuration can only ever be tighter than the platform ceiling.
        self.limits = limits.clamped_to(platform_ceiling) if platform_ceiling else limits
        self.state = state
        self.kill_switch = kill_switch or KillSwitch()
        self.validator = validator or MarketDataValidator(
            max_staleness_seconds=self.limits.max_data_staleness_seconds,
            max_spread_bps=self.limits.max_spread_bps,
        )
        self._events: list[RiskEvent] = []

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        proposal: TradeProposal,
        portfolio: PortfolioView,
        snapshot: MarketSnapshot | None = None,
        *,
        now: datetime | None = None,
    ) -> RiskAssessment:
        """Decide whether, and at what size, a proposed trade may proceed."""
        moment = now or utcnow()
        events: list[RiskEvent] = []
        warnings: list[str] = []

        if not proposal.action.is_entry:
            return RiskAssessment(
                decision=RiskDecision.REJECTED,
                reason=f"{proposal.action.value} is not an entry; nothing to size",
                blocked_by=None,
            )

        self.state.mark_equity(portfolio.equity, now=moment)

        # A uniform signature lets every pre-trade veto run through one loop, so a new check
        # cannot be added and then forgotten at one of several call sites.
        for check in (
            self._check_kill_switch,
            self._check_loss_limits,
            self._check_drawdown,
            self._check_loss_streak,
            self._check_cooldown,
            self._check_trade_count,
            self._check_position_count,
        ):
            rejection = check(proposal, portfolio, moment, events)
            if rejection is not None:
                return rejection

        if snapshot is not None:
            rejection = self._check_market_quality(proposal, snapshot, moment, events, warnings)
            if rejection is not None:
                return rejection

        rejection = self._check_reward_risk(proposal, moment, events, warnings)
        if rejection is not None:
            return rejection

        rejection = self._check_exposure(proposal, portfolio, moment, events)
        if rejection is not None:
            return rejection

        return self._size(proposal, portfolio, snapshot, moment, events, warnings)

    # ------------------------------------------------------------------ #
    # Absolute vetoes
    # ------------------------------------------------------------------ #
    def _check_kill_switch(
        self,
        proposal: TradeProposal,
        _portfolio: PortfolioView,
        now: datetime,
        events: list[RiskEvent],
    ) -> RiskAssessment | None:
        if not self.kill_switch.is_active:
            return None
        trip = self.kill_switch.trip
        message = (
            f"Kill switch engaged ({trip.reason.value}): {trip.message}"
            if trip
            else "Kill switch engaged"
        )
        return self._reject(
            message, RiskEventType.KILL_SWITCH, RiskSeverity.CRITICAL,
            proposal.symbol, now, events,
        )

    def _check_loss_limits(
        self,
        proposal: TradeProposal,
        _portfolio: PortfolioView,
        now: datetime,
        events: list[RiskEvent],
    ) -> RiskAssessment | None:
        daily = self.state.daily_loss_fraction
        if daily >= self.limits.max_daily_loss:
            self.kill_switch.engage(
                KillSwitchReason.DAILY_LOSS,
                f"Daily loss {daily:.2%} reached the {self.limits.max_daily_loss:.2%} limit",
                detail={"daily_loss": round(daily, 5)},
                at=now,
            )
            return self._reject(
                f"Daily loss limit reached: {daily:.2%} of {self.limits.max_daily_loss:.2%}",
                RiskEventType.DAILY_LOSS_LIMIT, RiskSeverity.CRITICAL,
                proposal.symbol, now, events,
                limit_value=self.limits.max_daily_loss, observed_value=daily,
            )

        weekly = self.state.weekly_loss_fraction
        if weekly >= self.limits.max_weekly_loss:
            self.kill_switch.engage(
                KillSwitchReason.WEEKLY_LOSS,
                f"Weekly loss {weekly:.2%} reached the {self.limits.max_weekly_loss:.2%} limit",
                detail={"weekly_loss": round(weekly, 5)},
                at=now,
            )
            return self._reject(
                f"Weekly loss limit reached: {weekly:.2%} of {self.limits.max_weekly_loss:.2%}",
                RiskEventType.WEEKLY_LOSS_LIMIT, RiskSeverity.CRITICAL,
                proposal.symbol, now, events,
                limit_value=self.limits.max_weekly_loss, observed_value=weekly,
            )
        return None

    def _check_drawdown(
        self,
        proposal: TradeProposal,
        _portfolio: PortfolioView,
        now: datetime,
        events: list[RiskEvent],
    ) -> RiskAssessment | None:
        drawdown = self.state.drawdown
        if drawdown >= self.limits.max_drawdown:
            self.kill_switch.engage(
                KillSwitchReason.MAX_DRAWDOWN,
                f"Drawdown {drawdown:.2%} reached the {self.limits.max_drawdown:.2%} limit",
                detail={"drawdown": round(drawdown, 5)},
                at=now,
            )
            return self._reject(
                f"Maximum drawdown reached: {drawdown:.2%} from a peak of "
                f"{self.state.peak_equity:.2f}",
                RiskEventType.MAX_DRAWDOWN, RiskSeverity.CRITICAL,
                proposal.symbol, now, events,
                limit_value=self.limits.max_drawdown, observed_value=drawdown,
            )
        return None

    def _check_loss_streak(
        self,
        proposal: TradeProposal,
        _portfolio: PortfolioView,
        now: datetime,
        events: list[RiskEvent],
    ) -> RiskAssessment | None:
        streak = self.state.consecutive_losses
        if streak >= self.limits.max_loss_streak:
            return self._reject(
                f"{streak} consecutive losses reached the {self.limits.max_loss_streak} "
                "limit; the strategy is out of sync with current conditions",
                RiskEventType.LOSS_STREAK, RiskSeverity.WARNING,
                proposal.symbol, now, events,
                limit_value=float(self.limits.max_loss_streak), observed_value=float(streak),
            )
        return None

    def _check_cooldown(
        self,
        proposal: TradeProposal,
        _portfolio: PortfolioView,
        now: datetime,
        events: list[RiskEvent],
    ) -> RiskAssessment | None:
        active, remaining = self.state.in_cooldown(
            self.limits.cooldown_seconds,
            after_loss_only=self.limits.cooldown_after_loss_only,
            now=now,
        )
        if active:
            return self._reject(
                f"Cooldown active: {remaining:.0f}s remaining of "
                f"{self.limits.cooldown_seconds}s",
                RiskEventType.COOLDOWN, RiskSeverity.INFO,
                proposal.symbol, now, events,
                limit_value=float(self.limits.cooldown_seconds), observed_value=remaining,
            )
        return None

    def _check_trade_count(
        self,
        proposal: TradeProposal,
        _portfolio: PortfolioView,
        now: datetime,
        events: list[RiskEvent],
    ) -> RiskAssessment | None:
        if self.state.trades_today >= self.limits.max_daily_trades:
            return self._reject(
                f"Daily trade limit reached: {self.state.trades_today} of "
                f"{self.limits.max_daily_trades}",
                RiskEventType.TRADE_COUNT_LIMIT, RiskSeverity.WARNING,
                proposal.symbol, now, events,
                limit_value=float(self.limits.max_daily_trades),
                observed_value=float(self.state.trades_today),
            )
        return None

    def _check_position_count(
        self,
        proposal: TradeProposal,
        portfolio: PortfolioView,
        now: datetime,
        events: list[RiskEvent],
    ) -> RiskAssessment | None:
        # Adding to an existing position does not consume a new position slot.
        if portfolio.position_for(proposal.symbol) is not None:
            return None
        if portfolio.position_count >= self.limits.max_concurrent_positions:
            return self._reject(
                f"Position limit reached: {portfolio.position_count} of "
                f"{self.limits.max_concurrent_positions} concurrent positions",
                RiskEventType.POSITION_LIMIT, RiskSeverity.INFO,
                proposal.symbol, now, events,
                limit_value=float(self.limits.max_concurrent_positions),
                observed_value=float(portfolio.position_count),
            )
        return None

    # ------------------------------------------------------------------ #
    # Market quality
    # ------------------------------------------------------------------ #
    def _check_market_quality(
        self,
        proposal: TradeProposal,
        snapshot: MarketSnapshot,
        now: datetime,
        events: list[RiskEvent],
        warnings: list[str],
    ) -> RiskAssessment | None:
        result: ValidationResult = self.validator.validate_snapshot(snapshot, now=now)
        warnings.extend(str(issue) for issue in result.warnings)

        if not result.is_valid:
            stale = any(
                issue.code in {"stale_data", "stale_ticker", "stale_book"}
                for issue in result.fatal_issues
            )
            if stale:
                self.kill_switch.engage(
                    KillSwitchReason.STALE_MARKET_DATA,
                    f"Market data for {proposal.symbol} is stale: {result.reason()}",
                    at=now,
                )
            return self._reject(
                f"Market data is not safe to trade on: {result.reason()}",
                RiskEventType.STALE_DATA if stale else RiskEventType.LIMIT_BREACH,
                RiskSeverity.CRITICAL if stale else RiskSeverity.WARNING,
                proposal.symbol, now, events,
            )

        ticker = snapshot.ticker
        if ticker is not None:
            spread = ticker.spread_bps
            if spread is not None and spread > self.limits.max_spread_bps:
                return self._reject(
                    f"Spread {spread:.1f} bps exceeds the "
                    f"{self.limits.max_spread_bps:.1f} bps limit; execution would give away "
                    "more than the edge is worth",
                    RiskEventType.SPREAD, RiskSeverity.WARNING,
                    proposal.symbol, now, events,
                    limit_value=self.limits.max_spread_bps, observed_value=spread,
                )
        return None

    def _check_reward_risk(
        self,
        proposal: TradeProposal,
        now: datetime,
        events: list[RiskEvent],
        warnings: list[str],
    ) -> RiskAssessment | None:
        if self.limits.min_reward_risk <= 0:
            return None
        ratio = reward_risk_ratio(
            proposal.entry_price, proposal.stop_loss, proposal.take_profit
        )
        if ratio is None:
            warnings.append("no take-profit set; reward/risk could not be verified")
            return None
        if ratio < self.limits.min_reward_risk:
            return self._reject(
                f"Reward/risk {ratio:.2f} is below the required "
                f"{self.limits.min_reward_risk:.2f}",
                RiskEventType.LIMIT_BREACH, RiskSeverity.INFO,
                proposal.symbol, now, events,
                limit_value=self.limits.min_reward_risk, observed_value=ratio,
            )
        return None

    def _check_exposure(
        self,
        proposal: TradeProposal,
        portfolio: PortfolioView,
        now: datetime,
        events: list[RiskEvent],
    ) -> RiskAssessment | None:
        current_leverage = portfolio.leverage
        if current_leverage >= self.limits.max_portfolio_exposure:
            return self._reject(
                f"Portfolio exposure {current_leverage:.2f}x already at the "
                f"{self.limits.max_portfolio_exposure:.2f}x limit",
                RiskEventType.EXPOSURE_LIMIT, RiskSeverity.WARNING,
                proposal.symbol, now, events,
                limit_value=self.limits.max_portfolio_exposure,
                observed_value=current_leverage,
            )

        asset = proposal.instrument.base_asset
        asset_exposure = safe_divide(portfolio.exposure_for_asset(asset), portfolio.equity)
        if asset_exposure >= self.limits.max_asset_exposure:
            return self._reject(
                f"Exposure to {asset} is {asset_exposure:.1%}, at the "
                f"{self.limits.max_asset_exposure:.1%} limit",
                RiskEventType.EXPOSURE_LIMIT, RiskSeverity.WARNING,
                proposal.symbol, now, events,
                limit_value=self.limits.max_asset_exposure, observed_value=asset_exposure,
            )
        return None

    # ------------------------------------------------------------------ #
    # Sizing
    # ------------------------------------------------------------------ #
    def _size(
        self,
        proposal: TradeProposal,
        portfolio: PortfolioView,
        snapshot: MarketSnapshot | None,
        now: datetime,
        events: list[RiskEvent],
        warnings: list[str],
    ) -> RiskAssessment:
        estimated_slippage = self._estimate_slippage(proposal, snapshot, portfolio)
        # Size against a conservative slippage assumption, not the point estimate. A point
        # estimate is by construction exceeded about half the time, which would put half of
        # all fills over the risk budget.
        slippage_bps = max(
            (estimated_slippage or 0.0) * self.limits.slippage_safety_factor,
            self.limits.min_slippage_bps,
        )
        if slippage_bps > self.limits.max_slippage_bps:
            return self._reject(
                f"Estimated slippage {slippage_bps:.1f} bps exceeds the "
                f"{self.limits.max_slippage_bps:.1f} bps limit",
                RiskEventType.SLIPPAGE, RiskSeverity.WARNING,
                proposal.symbol, now, events,
                limit_value=self.limits.max_slippage_bps, observed_value=slippage_bps,
            )

        # Reject trades whose execution costs swamp the stop distance. A trade risking 0.5%
        # where 40% of that risk is fees needs a far higher hit rate than the strategy was
        # designed around, and the sizing model is most fragile exactly there.
        gross_risk_per_unit = abs(proposal.entry_price - proposal.stop_loss)
        cost_per_unit = (proposal.entry_price + proposal.stop_loss) * (
            proposal.instrument.taker_fee + bps(slippage_bps)
        )
        cost_ratio = safe_divide(cost_per_unit, gross_risk_per_unit, default=1.0)
        if cost_ratio > self.limits.max_cost_to_risk_ratio:
            return self._reject(
                f"Execution costs are {cost_ratio:.0%} of the stop distance, above the "
                f"{self.limits.max_cost_to_risk_ratio:.0%} limit; the stop is too tight to "
                "trade profitably at this fee and spread level",
                RiskEventType.SLIPPAGE, RiskSeverity.WARNING,
                proposal.symbol, now, events,
                limit_value=self.limits.max_cost_to_risk_ratio, observed_value=cost_ratio,
            )

        headroom = self._exposure_headroom(proposal, portfolio)
        sizing = calculate_position_size(
            SizingRequest(
                equity=portfolio.equity,
                available_margin=min(portfolio.available_margin, headroom),
                entry_price=proposal.entry_price,
                stop_price=proposal.stop_loss,
                side=proposal.side,
                instrument=proposal.instrument,
                risk_fraction=self.limits.risk_per_trade,
                max_position_fraction=self.limits.max_position_fraction,
                max_leverage=self.limits.max_leverage,
                leverage=proposal.leverage,
                expected_slippage_bps=slippage_bps,
                confidence=proposal.confidence,
                scale_by_confidence=self.limits.scale_size_by_confidence,
            )
        )

        if not sizing.is_tradable:
            return self._reject(
                f"Position sizing failed: {sizing.reason}",
                RiskEventType.SIZING_FAILED, RiskSeverity.INFO,
                proposal.symbol, now, events,
            )

        # Independent verification. The sizing routine and this check are separate on purpose:
        # a bug in one should not be able to silently authorise an oversized position.
        worst_case = estimate_worst_case_loss(
            sizing.quantity,
            proposal.entry_price,
            proposal.stop_loss,
            proposal.instrument,
            slippage_bps=slippage_bps,
        )
        worst_case_fraction = safe_divide(worst_case, portfolio.equity)
        tolerance = self.limits.risk_per_trade * 1.05  # allow rounding slack only
        if worst_case_fraction > tolerance:
            logger.error(
                "risk.sizing_verification_failed",
                symbol=proposal.symbol,
                worst_case_fraction=round(worst_case_fraction, 6),
                limit=self.limits.risk_per_trade,
                quantity=sizing.quantity,
            )
            return self._reject(
                f"Sizing verification failed: worst-case loss {worst_case_fraction:.3%} "
                f"exceeds the {self.limits.risk_per_trade:.3%} risk budget",
                RiskEventType.SIZING_FAILED, RiskSeverity.CRITICAL,
                proposal.symbol, now, events,
                limit_value=self.limits.risk_per_trade, observed_value=worst_case_fraction,
            )

        liquidity_rejection = self._check_liquidity(
            proposal, snapshot, sizing, now, events
        )
        if liquidity_rejection is not None:
            return liquidity_rejection

        decision = (
            RiskDecision.REDUCED
            if sizing.binding_constraint != "risk_per_trade" or sizing.constraints_applied
            else RiskDecision.APPROVED
        )
        logger.info(
            "risk.approved",
            symbol=proposal.symbol,
            side=proposal.side.value,
            quantity=sizing.quantity,
            notional=round(sizing.notional, 2),
            risk_fraction=round(sizing.effective_risk_fraction, 5),
            binding=sizing.binding_constraint,
            decision=decision.value,
        )
        return RiskAssessment(
            decision=decision,
            quantity=sizing.quantity,
            reason=sizing.reason,
            sizing=sizing,
            events=tuple(events),
            warnings=tuple(warnings),
        )

    def _exposure_headroom(
        self, proposal: TradeProposal, portfolio: PortfolioView
    ) -> float:
        """Margin still available before an exposure limit would bind."""
        portfolio_room = max(
            0.0,
            portfolio.equity * self.limits.max_portfolio_exposure - portfolio.total_exposure,
        )
        asset = proposal.instrument.base_asset
        asset_room = max(
            0.0,
            portfolio.equity * self.limits.max_asset_exposure
            - portfolio.exposure_for_asset(asset),
        )
        return min(portfolio_room, asset_room)

    def _estimate_slippage(
        self,
        proposal: TradeProposal,
        snapshot: MarketSnapshot | None,
        portfolio: PortfolioView,
    ) -> float | None:
        """Estimate execution slippage from the book, falling back to half the spread."""
        if snapshot is None:
            return None
        book = snapshot.order_book
        if book is not None:
            notional = portfolio.equity * self.limits.risk_per_trade * 20.0
            probe_quantity = safe_divide(notional, proposal.entry_price)
            if probe_quantity > 0:
                estimate = book.estimate_slippage_bps(
                    "buy" if proposal.side is OrderSide.BUY else "sell", probe_quantity
                )
                if estimate is not None:
                    return max(0.0, estimate)
        ticker = snapshot.ticker
        if ticker is not None and ticker.spread_bps is not None:
            return max(0.0, ticker.spread_bps / 2.0)
        return None

    def _check_liquidity(
        self,
        proposal: TradeProposal,
        snapshot: MarketSnapshot | None,
        sizing: SizingResult,
        now: datetime,
        events: list[RiskEvent],
    ) -> RiskAssessment | None:
        if snapshot is None or snapshot.order_book is None:
            return None
        book = snapshot.order_book
        side = "ask" if proposal.side is OrderSide.BUY else "bid"
        depth = book.depth(side)
        required = sizing.quantity * self.limits.min_liquidity_multiple
        if depth < required:
            return self._reject(
                f"Insufficient liquidity: book shows {depth:.6g} against a required "
                f"{required:.6g} ({self.limits.min_liquidity_multiple:g}x the order size)",
                RiskEventType.LIQUIDITY, RiskSeverity.WARNING,
                proposal.symbol, now, events,
                limit_value=required, observed_value=depth,
            )
        return None

    # ------------------------------------------------------------------ #
    # Automatic kill-switch triggers outside the order path
    # ------------------------------------------------------------------ #
    def on_api_failure(self, error: str, *, now: datetime | None = None) -> None:
        """Record an exchange API failure and trip the switch if they persist."""
        count = self.state.record_api_failure()
        if count >= self.limits.max_consecutive_api_failures:
            self.kill_switch.engage(
                KillSwitchReason.API_FAILURES,
                f"{count} consecutive exchange API failures; last error: {error}",
                detail={"consecutive_failures": count},
                at=now,
            )

    def on_api_success(self) -> None:
        self.state.record_api_success()

    def on_clock_drift(self, drift_seconds: float, *, now: datetime | None = None) -> None:
        """Trip the switch when local and exchange clocks disagree.

        Signed orders carry timestamps; a drifting clock causes silent rejections at best and
        orders applied at the wrong moment at worst.
        """
        if abs(drift_seconds) > self.limits.max_clock_drift_seconds:
            self.kill_switch.engage(
                KillSwitchReason.CLOCK_DRIFT,
                f"Clock drift {drift_seconds:+.2f}s exceeds the "
                f"{self.limits.max_clock_drift_seconds:.2f}s tolerance",
                detail={"drift_seconds": round(drift_seconds, 3)},
                at=now,
            )

    def on_reconciliation_failure(self, detail: str, *, now: datetime | None = None) -> None:
        """Trip the switch when local state and exchange state disagree."""
        self.kill_switch.engage(
            KillSwitchReason.EXCHANGE_DESYNC,
            f"Reconciliation failed: {detail}",
            at=now,
        )

    def on_state_corruption(self, detail: str, *, now: datetime | None = None) -> None:
        self.kill_switch.engage(
            KillSwitchReason.STATE_CORRUPTION,
            f"Invalid internal state: {detail}",
            at=now,
        )

    def emergency_stop(self, actor: str, note: str = "") -> None:
        """Manual kill switch."""
        self.kill_switch.engage(
            KillSwitchReason.MANUAL,
            f"Emergency stop requested by {actor}{f': {note}' if note else ''}",
            triggered_by=actor,
        )

    # ------------------------------------------------------------------ #
    # Bookkeeping
    # ------------------------------------------------------------------ #
    @property
    def events(self) -> list[RiskEvent]:
        """Every risk event recorded by this manager, for persistence."""
        return list(self._events)

    def drain_events(self) -> list[RiskEvent]:
        """Return and clear recorded events."""
        drained = list(self._events)
        self._events.clear()
        return drained

    def _reject(
        self,
        message: str,
        event_type: RiskEventType,
        severity: RiskSeverity,
        symbol: str | None,
        now: datetime,
        events: list[RiskEvent],
        *,
        limit_value: float | None = None,
        observed_value: float | None = None,
    ) -> RiskAssessment:
        event = RiskEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            occurred_at=now,
            symbol=symbol,
            limit_value=limit_value,
            observed_value=observed_value,
            action_taken="order_blocked",
        )
        events.append(event)
        self._events.append(event)
        logger.info(
            "risk.rejected",
            symbol=symbol,
            event_type=event_type.value,
            severity=severity.value,
            message=message,
        )
        return RiskAssessment.reject(message, event_type, events=tuple(events))

    def describe(self) -> dict[str, object]:
        """Current risk posture, for the dashboard's risk page."""
        return {
            "limits": self.limits.model_dump(),
            "state": self.state.to_dict(),
            "kill_switch": self.kill_switch.describe(),
            "headroom": {
                "daily_loss_remaining": round(
                    max(0.0, self.limits.max_daily_loss - self.state.daily_loss_fraction), 5
                ),
                "drawdown_remaining": round(
                    max(0.0, self.limits.max_drawdown - self.state.drawdown), 5
                ),
                "trades_remaining_today": max(
                    0, self.limits.max_daily_trades - self.state.trades_today
                ),
            },
        }
