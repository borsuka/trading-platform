"""Position sizing.

The core calculation is fixed-fractional risk::

    risk_amount   = equity * risk_per_trade
    risk_per_unit = |entry - stop|
    quantity      = risk_amount / risk_per_unit

Everything else in this module exists because that formula is not sufficient on its own:

* **Fees are part of the loss.** Stopping out costs the stop distance *plus* the entry and exit
  fees. Ignoring them makes the realised loss larger than the configured risk, systematically,
  on every trade.
* **Slippage widens the effective stop.** A stop is a trigger, not a fill price.
* **Venue constraints round the answer.** Lot size, minimum quantity and minimum notional can
  all make the ideal size unattainable; rounding is always *down* so the result never exceeds
  what risk allowed.
* **Leverage bounds it.** Margin available caps the size regardless of what the stop implies.

A sizing call that cannot produce a valid quantity returns a rejection with a reason, never a
"best effort" number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.domain import InstrumentSpec
from app.core.enums import OrderSide
from app.core.numeric import EPSILON, bps, is_zero, safe_divide


@dataclass(frozen=True, slots=True)
class SizingRequest:
    """Inputs to a position-sizing decision."""

    equity: float
    available_margin: float
    entry_price: float
    stop_price: float
    side: OrderSide
    instrument: InstrumentSpec
    risk_fraction: float
    max_position_fraction: float = 1.0
    max_leverage: float = 1.0
    leverage: float = 1.0
    expected_slippage_bps: float = 0.0
    confidence: float = 1.0
    scale_by_confidence: bool = False

    def __post_init__(self) -> None:
        if self.equity <= 0:
            raise ValueError("equity must be positive")
        if self.entry_price <= 0 or self.stop_price <= 0:
            raise ValueError("entry and stop prices must be positive")
        if self.risk_fraction <= 0:
            raise ValueError("risk_fraction must be positive")
        if self.side is OrderSide.BUY and self.stop_price >= self.entry_price:
            raise ValueError(
                f"Long stop {self.stop_price:g} must be below entry {self.entry_price:g}"
            )
        if self.side is OrderSide.SELL and self.stop_price <= self.entry_price:
            raise ValueError(
                f"Short stop {self.stop_price:g} must be above entry {self.entry_price:g}"
            )


@dataclass(frozen=True, slots=True)
class SizingResult:
    """The outcome of sizing, with every intermediate value exposed.

    The breakdown is not decoration: when a user asks why the bot bought 0.037 BTC, the answer
    has to be reconstructible without re-running the market.
    """

    quantity: float
    approved: bool
    reason: str = ""
    risk_amount: float = 0.0
    risk_per_unit: float = 0.0
    gross_risk_per_unit: float = 0.0
    notional: float = 0.0
    margin_required: float = 0.0
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    slippage_cost: float = 0.0
    #: The slippage the size was computed against, in bps. Realised slippage worse than this
    #: means the realised loss exceeds the risk budget - which is why the risk manager applies
    #: a safety factor to its estimate before sizing.
    assumed_slippage_bps: float = 0.0
    effective_risk_fraction: float = 0.0
    binding_constraint: str | None = None
    constraints_applied: list[str] = field(default_factory=list)

    @property
    def is_tradable(self) -> bool:
        return self.approved and self.quantity > 0

    @classmethod
    def rejected(
        cls,
        reason: str,
        *,
        risk_amount: float = 0.0,
        risk_per_unit: float = 0.0,
    ) -> SizingResult:
        return cls(
            quantity=0.0,
            approved=False,
            reason=reason,
            risk_amount=risk_amount,
            risk_per_unit=risk_per_unit,
        )


def calculate_position_size(request: SizingRequest) -> SizingResult:
    """Size a position from risk, cost and venue constraints.

    Returns a :class:`SizingResult` whose ``quantity`` is safe to submit as-is, or an
    unapproved result explaining why no valid size exists.
    """
    instrument = request.instrument
    constraints: list[str] = []

    # 1. Risk budget, optionally reduced by signal confidence.
    risk_fraction = request.risk_fraction
    if request.scale_by_confidence:
        # Confidence can only shrink the budget. A confident signal gets the configured risk,
        # never more; the alternative is a system whose worst-case loss is not knowable.
        scale = max(0.0, min(1.0, request.confidence))
        if scale < 1.0:
            constraints.append(f"confidence scaling x{scale:.2f}")
        risk_fraction *= scale
    if risk_fraction <= 0:
        return SizingResult.rejected("risk budget is zero after confidence scaling")

    risk_amount = request.equity * risk_fraction

    # 2. Risk per unit, widened by expected slippage on entry and exit.
    gross_risk_per_unit = abs(request.entry_price - request.stop_price)
    if is_zero(gross_risk_per_unit):
        return SizingResult.rejected("stop is at the entry price; risk per unit is zero")

    slip = bps(request.expected_slippage_bps)
    slippage_per_unit = (request.entry_price + request.stop_price) * slip
    fee_per_unit = (
        request.entry_price * instrument.taker_fee + request.stop_price * instrument.taker_fee
    )
    risk_per_unit = gross_risk_per_unit + slippage_per_unit + fee_per_unit
    if risk_per_unit <= 0:
        return SizingResult.rejected("computed risk per unit is not positive")

    quantity = risk_amount / risk_per_unit
    binding = "risk_per_trade"

    # 3. Position-size ceiling as a fraction of equity.
    max_notional = request.equity * request.max_position_fraction
    max_quantity_by_notional = safe_divide(max_notional, request.entry_price)
    if max_quantity_by_notional < quantity:
        quantity = max_quantity_by_notional
        binding = "max_position_fraction"
        constraints.append(
            f"capped at {request.max_position_fraction:.0%} of equity"
        )

    # 4. Margin ceiling.
    leverage = max(1.0, min(request.leverage, request.max_leverage, instrument.max_leverage))
    if leverage < request.leverage:
        constraints.append(f"leverage reduced to {leverage:g}x")
    max_quantity_by_margin = safe_divide(
        request.available_margin * leverage, request.entry_price
    )
    if max_quantity_by_margin < quantity:
        quantity = max_quantity_by_margin
        binding = "available_margin"
        constraints.append("capped by available margin")

    if quantity <= 0:
        return SizingResult.rejected(
            "no margin available for a new position",
            risk_amount=risk_amount,
            risk_per_unit=risk_per_unit,
        )

    # 5. Venue rounding. Always down: never exceed what risk allowed.
    rounded = instrument.round_quantity(quantity)
    if rounded < quantity:
        constraints.append(f"rounded down to lot size {instrument.lot_size:g}")
    quantity = rounded

    if is_zero(quantity):
        return SizingResult.rejected(
            f"position size rounds to zero at lot size {instrument.lot_size:g}: "
            f"equity {request.equity:.2f} with {risk_fraction:.3%} risk is too small "
            f"for this instrument",
            risk_amount=risk_amount,
            risk_per_unit=risk_per_unit,
        )

    problem = instrument.validate_order(quantity, request.entry_price)
    if problem is not None:
        return SizingResult.rejected(
            f"venue constraint: {problem}",
            risk_amount=risk_amount,
            risk_per_unit=risk_per_unit,
        )

    notional = instrument.notional(quantity, request.entry_price)
    margin_required = notional / leverage
    if margin_required > request.available_margin + EPSILON:
        return SizingResult.rejected(
            f"margin required {margin_required:.2f} exceeds available "
            f"{request.available_margin:.2f}",
            risk_amount=risk_amount,
            risk_per_unit=risk_per_unit,
        )

    entry_fee = notional * instrument.taker_fee
    exit_fee = quantity * request.stop_price * instrument.taker_fee
    slippage_cost = quantity * slippage_per_unit
    realised_risk = quantity * risk_per_unit

    return SizingResult(
        quantity=quantity,
        approved=True,
        reason=(
            f"risking {realised_risk:.2f} ({safe_divide(realised_risk, request.equity):.3%} "
            f"of {request.equity:.2f} equity) over a {gross_risk_per_unit:.6g} stop distance"
        ),
        risk_amount=risk_amount,
        risk_per_unit=risk_per_unit,
        gross_risk_per_unit=gross_risk_per_unit,
        notional=notional,
        margin_required=margin_required,
        entry_fee=entry_fee,
        exit_fee=exit_fee,
        slippage_cost=slippage_cost,
        assumed_slippage_bps=request.expected_slippage_bps,
        effective_risk_fraction=safe_divide(realised_risk, request.equity),
        binding_constraint=binding,
        constraints_applied=constraints,
    )


def estimate_worst_case_loss(
    quantity: float,
    entry_price: float,
    stop_price: float,
    instrument: InstrumentSpec,
    *,
    slippage_bps: float = 0.0,
) -> float:
    """Total loss if the stop is hit, including fees and slippage.

    Used by the risk manager to verify after sizing that the trade really does cost what the
    risk budget says it will.
    """
    distance = abs(entry_price - stop_price)
    slip = bps(slippage_bps) * (entry_price + stop_price)
    fees = (entry_price + stop_price) * instrument.taker_fee
    return quantity * (distance + slip + fees)


def reward_risk_ratio(
    entry: float, stop: float, target: float | None
) -> float | None:
    """Reward/risk of a proposed trade, or ``None`` when no target is set."""
    if target is None:
        return None
    risk = abs(entry - stop)
    if is_zero(risk):
        return None
    return abs(target - entry) / risk


def kelly_fraction(win_rate: float, reward_risk: float, *, cap: float = 0.25) -> float:
    """Kelly-optimal fraction, capped hard.

    Provided for analysis and reporting only. The platform does **not** size positions with
    Kelly: it is exquisitely sensitive to estimation error in ``win_rate``, and those estimates
    come from backtests, which are exactly where that error lives. Full Kelly on an
    overestimated edge is a reliable route to ruin. The cap makes accidental use survivable.
    """
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError("win_rate must be in [0, 1]")
    if reward_risk <= 0:
        raise ValueError("reward_risk must be positive")
    loss_rate = 1.0 - win_rate
    raw = win_rate - loss_rate / reward_risk
    return max(0.0, min(cap, raw))
