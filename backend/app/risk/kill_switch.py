"""Kill switch.

When the kill switch is engaged the bot places **no new orders**. That is the whole contract,
and it is deliberately narrow:

* It does **not** automatically liquidate. Force-closing positions into whatever conditions
  tripped the switch — a volatility spike, an exchange outage, a data feed failure — is
  frequently the worst available action, and it converts a recoverable situation into a
  realised loss. Liquidation is a separate, explicit emergency-stop decision.
* It does **not** cancel protective stops. Those are the position's remaining protection.
* It **cannot be cleared automatically**. Every trip requires a human to acknowledge and reset,
  because a switch that resets itself is not a safety mechanism.

The one exception is :attr:`KillSwitch.auto_reset_after`, which may be configured for
transient causes (stale data, API failures) where the condition demonstrably clears itself.
Loss-based trips never auto-reset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.core.clock import utcnow
from app.core.enums import KillSwitchReason
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Causes that may clear themselves once the underlying condition resolves.
TRANSIENT_REASONS: frozenset[KillSwitchReason] = frozenset(
    {
        KillSwitchReason.STALE_MARKET_DATA,
        KillSwitchReason.API_FAILURES,
        KillSwitchReason.CLOCK_DRIFT,
    }
)

#: Causes that always require a human reset.
PERMANENT_REASONS: frozenset[KillSwitchReason] = frozenset(
    {
        KillSwitchReason.MANUAL,
        KillSwitchReason.DAILY_LOSS,
        KillSwitchReason.WEEKLY_LOSS,
        KillSwitchReason.MAX_DRAWDOWN,
        KillSwitchReason.STATE_CORRUPTION,
        KillSwitchReason.EXCHANGE_DESYNC,
        KillSwitchReason.LICENSE_INVALID,
    }
)


@dataclass(frozen=True, slots=True)
class KillSwitchTrip:
    """A single activation record."""

    reason: KillSwitchReason
    message: str
    triggered_at: datetime
    triggered_by: str = "system"
    detail: dict[str, float | str] = field(default_factory=dict)

    @property
    def is_transient(self) -> bool:
        return self.reason in TRANSIENT_REASONS


class KillSwitch:
    """Blocks new order submission once engaged.

    Thread-confined to a single event loop; the bot runtime owns one instance per bot.
    """

    def __init__(self, *, auto_reset_after: timedelta | None = None) -> None:
        self._trip: KillSwitchTrip | None = None
        self._history: list[KillSwitchTrip] = []
        self.auto_reset_after = auto_reset_after

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    @property
    def is_active(self) -> bool:
        """True when new orders are blocked."""
        if self._trip is None:
            return False
        if self._can_auto_reset(self._trip):
            self._clear("auto-reset after the transient condition cleared")
            return False
        return True

    @property
    def trip(self) -> KillSwitchTrip | None:
        return self._trip

    @property
    def reason(self) -> KillSwitchReason | None:
        return self._trip.reason if self._trip else None

    @property
    def history(self) -> list[KillSwitchTrip]:
        return list(self._history)

    def _can_auto_reset(self, trip: KillSwitchTrip) -> bool:
        if self.auto_reset_after is None or not trip.is_transient:
            return False
        return utcnow() - trip.triggered_at >= self.auto_reset_after

    # ------------------------------------------------------------------ #
    # Control
    # ------------------------------------------------------------------ #
    def engage(
        self,
        reason: KillSwitchReason,
        message: str,
        *,
        triggered_by: str = "system",
        detail: dict[str, float | str] | None = None,
        at: datetime | None = None,
    ) -> KillSwitchTrip:
        """Engage the switch.

        Re-engaging while already active keeps the *original* trip, so the first cause is not
        obscured by a cascade of downstream failures — which is usually what the operator
        actually needs to see.
        """
        if self._trip is not None:
            logger.warning(
                "kill_switch.already_engaged",
                original_reason=self._trip.reason.value,
                new_reason=reason.value,
                new_message=message,
            )
            return self._trip

        trip = KillSwitchTrip(
            reason=reason,
            message=message,
            triggered_at=at or utcnow(),
            triggered_by=triggered_by,
            detail=detail or {},
        )
        self._trip = trip
        self._history.append(trip)
        logger.error(
            "kill_switch.engaged",
            reason=reason.value,
            message=message,
            triggered_by=triggered_by,
            transient=trip.is_transient,
            **{f"detail_{k}": v for k, v in trip.detail.items()},
        )
        return trip

    def reset(self, *, reset_by: str, note: str = "") -> KillSwitchTrip | None:
        """Clear the switch. Requires an identified actor; there is no anonymous reset."""
        if not reset_by:
            raise ValueError("reset_by is required: kill-switch resets must be attributable")
        if self._trip is None:
            return None
        cleared = self._trip
        self._clear(f"reset by {reset_by}{f': {note}' if note else ''}")
        return cleared

    def _clear(self, explanation: str) -> None:
        if self._trip is None:
            return
        logger.warning(
            "kill_switch.cleared",
            reason=self._trip.reason.value,
            explanation=explanation,
            active_for_seconds=round(
                (utcnow() - self._trip.triggered_at).total_seconds(), 1
            ),
        )
        self._trip = None

    def require_clear(self) -> None:
        """Raise if the switch is engaged. Called on every order submission path."""
        from app.core.exceptions import KillSwitchActiveError

        if self.is_active and self._trip is not None:
            raise KillSwitchActiveError(
                f"Kill switch engaged ({self._trip.reason.value}): {self._trip.message}. "
                "New orders are blocked until it is reset.",
                context={
                    "reason": self._trip.reason.value,
                    "triggered_at": self._trip.triggered_at.isoformat(),
                },
            )

    def describe(self) -> dict[str, object]:
        """Serialisable state for the API and the dashboard."""
        if self._trip is None:
            return {"active": False, "reason": None, "message": None}
        return {
            "active": True,
            "reason": self._trip.reason.value,
            "message": self._trip.message,
            "triggered_at": self._trip.triggered_at.isoformat(),
            "triggered_by": self._trip.triggered_by,
            "transient": self._trip.is_transient,
            "detail": self._trip.detail,
        }
