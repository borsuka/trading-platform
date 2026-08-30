"""Rolling risk state.

Tracks everything the risk manager needs to know about recent trading that is not derivable
from the current portfolio snapshot: realised PnL over rolling windows, consecutive losses,
cooldowns, trade counts and the equity high-water mark.

The window boundaries are UTC calendar boundaries, not rolling 24-hour windows. "Daily loss
limit" has to mean something an operator can reason about, and a limit that resets at a
different time every day is not that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from app.core.clock import utcnow
from app.core.numeric import safe_divide


def day_start(moment: datetime) -> datetime:
    """UTC midnight at the start of ``moment``'s day."""
    return moment.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def week_start(moment: datetime) -> datetime:
    """UTC midnight on the Monday of ``moment``'s week."""
    start = day_start(moment)
    return start - timedelta(days=start.weekday())


@dataclass(slots=True)
class TradeOutcome:
    """A closed trade, reduced to what the risk engine cares about."""

    symbol: str
    net_pnl: float
    closed_at: datetime
    strategy_name: str | None = None


@dataclass(slots=True)
class RiskState:
    """Rolling risk counters for one bot.

    Restored from the database on restart so limits survive a process restart. A bot that
    forgets it already lost 1.9% today and starts fresh is not risk-managed.
    """

    starting_equity: float
    peak_equity: float = 0.0
    current_equity: float = 0.0
    day_start_equity: float = 0.0
    week_start_equity: float = 0.0
    current_day: datetime | None = None
    current_week: datetime | None = None
    trades_today: int = 0
    consecutive_losses: int = 0
    last_trade_at: datetime | None = None
    last_loss_at: datetime | None = None
    consecutive_api_failures: int = 0
    recent_trades: list[TradeOutcome] = field(default_factory=list)
    max_recent_trades: int = 500

    def __post_init__(self) -> None:
        if self.starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        if self.peak_equity <= 0:
            self.peak_equity = self.starting_equity
        if self.current_equity <= 0:
            self.current_equity = self.starting_equity
        if self.day_start_equity <= 0:
            self.day_start_equity = self.starting_equity
        if self.week_start_equity <= 0:
            self.week_start_equity = self.starting_equity

    # ------------------------------------------------------------------ #
    # Updates
    # ------------------------------------------------------------------ #
    def mark_equity(self, equity: float, *, now: datetime | None = None) -> None:
        """Record the latest equity and roll the day/week windows if they have turned over."""
        moment = now or utcnow()
        self._roll_windows(moment, equity)
        self.current_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

    def _roll_windows(self, moment: datetime, equity: float) -> None:
        today = day_start(moment)
        if self.current_day is None:
            self.current_day = today
            self.day_start_equity = self.current_equity or equity
        elif today > self.current_day:
            self.current_day = today
            self.day_start_equity = self.current_equity or equity
            self.trades_today = 0

        this_week = week_start(moment)
        if self.current_week is None or this_week > self.current_week:
            self.current_week = this_week
            self.week_start_equity = self.current_equity or equity

    def record_trade(self, outcome: TradeOutcome) -> None:
        """Record a closed trade and update streak counters."""
        self._roll_windows(outcome.closed_at, self.current_equity)
        self.trades_today += 1
        self.last_trade_at = outcome.closed_at
        if outcome.net_pnl < 0:
            self.consecutive_losses += 1
            self.last_loss_at = outcome.closed_at
        else:
            self.consecutive_losses = 0
        self.recent_trades.append(outcome)
        if len(self.recent_trades) > self.max_recent_trades:
            del self.recent_trades[: len(self.recent_trades) - self.max_recent_trades]

    def record_order_submitted(self, *, now: datetime | None = None) -> None:
        """Count an order against the daily trade budget at submission time.

        Counting at submission rather than at close is deliberate: the limit exists to bound
        activity, and an open position is activity that has already consumed risk.
        """
        moment = now or utcnow()
        self._roll_windows(moment, self.current_equity)
        self.trades_today += 1
        self.last_trade_at = moment

    def record_api_failure(self) -> int:
        self.consecutive_api_failures += 1
        return self.consecutive_api_failures

    def record_api_success(self) -> None:
        self.consecutive_api_failures = 0

    # ------------------------------------------------------------------ #
    # Derived metrics
    # ------------------------------------------------------------------ #
    @property
    def daily_pnl(self) -> float:
        return self.current_equity - self.day_start_equity

    @property
    def daily_loss_fraction(self) -> float:
        """Loss today as a positive fraction of the day's opening equity. 0 when in profit."""
        return max(0.0, -safe_divide(self.daily_pnl, self.day_start_equity))

    @property
    def weekly_pnl(self) -> float:
        return self.current_equity - self.week_start_equity

    @property
    def weekly_loss_fraction(self) -> float:
        return max(0.0, -safe_divide(self.weekly_pnl, self.week_start_equity))

    @property
    def drawdown(self) -> float:
        """Current drawdown from the equity high-water mark, as a positive fraction."""
        return max(0.0, safe_divide(self.peak_equity - self.current_equity, self.peak_equity))

    @property
    def total_return(self) -> float:
        return safe_divide(
            self.current_equity - self.starting_equity, self.starting_equity
        )

    def seconds_since_last_trade(self, *, now: datetime | None = None) -> float | None:
        if self.last_trade_at is None:
            return None
        return ((now or utcnow()) - self.last_trade_at).total_seconds()

    def seconds_since_last_loss(self, *, now: datetime | None = None) -> float | None:
        if self.last_loss_at is None:
            return None
        return ((now or utcnow()) - self.last_loss_at).total_seconds()

    def in_cooldown(
        self, cooldown_seconds: int, *, after_loss_only: bool = True,
        now: datetime | None = None,
    ) -> tuple[bool, float]:
        """Whether a cooldown is active, and how many seconds remain."""
        if cooldown_seconds <= 0:
            return False, 0.0
        elapsed = (
            self.seconds_since_last_loss(now=now)
            if after_loss_only
            else self.seconds_since_last_trade(now=now)
        )
        if elapsed is None:
            return False, 0.0
        remaining = cooldown_seconds - elapsed
        return (remaining > 0, max(0.0, remaining))

    def to_dict(self) -> dict[str, float | int | str | None]:
        """Serialisable snapshot, persisted with the bot's state."""
        return {
            "starting_equity": self.starting_equity,
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            "day_start_equity": self.day_start_equity,
            "week_start_equity": self.week_start_equity,
            "current_day": self.current_day.isoformat() if self.current_day else None,
            "current_week": self.current_week.isoformat() if self.current_week else None,
            "trades_today": self.trades_today,
            "consecutive_losses": self.consecutive_losses,
            "last_trade_at": self.last_trade_at.isoformat() if self.last_trade_at else None,
            "last_loss_at": self.last_loss_at.isoformat() if self.last_loss_at else None,
            "daily_loss_fraction": round(self.daily_loss_fraction, 6),
            "weekly_loss_fraction": round(self.weekly_loss_fraction, 6),
            "drawdown": round(self.drawdown, 6),
        }

    @classmethod
    def from_dict(cls, data: dict) -> RiskState:
        """Restore from persisted state. Unknown or missing fields fall back to defaults."""

        def _dt(key: str) -> datetime | None:
            raw = data.get(key)
            return datetime.fromisoformat(raw) if isinstance(raw, str) else None

        state = cls(
            starting_equity=float(data.get("starting_equity", 0.0) or 1.0),
            peak_equity=float(data.get("peak_equity", 0.0) or 0.0),
            current_equity=float(data.get("current_equity", 0.0) or 0.0),
            day_start_equity=float(data.get("day_start_equity", 0.0) or 0.0),
            week_start_equity=float(data.get("week_start_equity", 0.0) or 0.0),
            trades_today=int(data.get("trades_today", 0) or 0),
            consecutive_losses=int(data.get("consecutive_losses", 0) or 0),
        )
        state.current_day = _dt("current_day")
        state.current_week = _dt("current_week")
        state.last_trade_at = _dt("last_trade_at")
        state.last_loss_at = _dt("last_loss_at")
        return state
