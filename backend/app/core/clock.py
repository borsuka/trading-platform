"""Time handling.

Every timestamp in the platform is timezone-aware UTC. Naive datetimes are rejected at module
boundaries rather than silently coerced, because a naive timestamp interpreted in local time is
a class of bug that produces plausible-looking but wrong backtests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol


def utcnow() -> datetime:
    """Current time as an aware UTC datetime."""
    return datetime.now(UTC)


def ensure_utc(value: datetime, *, field: str = "timestamp") -> datetime:
    """Return ``value`` as aware UTC, rejecting naive datetimes."""
    if value.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware; got naive datetime {value!r}")
    return value.astimezone(UTC)


def from_epoch_ms(milliseconds: int | float) -> datetime:
    """Convert exchange millisecond epochs to aware UTC datetimes."""
    return datetime.fromtimestamp(milliseconds / 1000.0, tz=UTC)


def to_epoch_ms(value: datetime) -> int:
    """Convert an aware datetime to a millisecond epoch."""
    return int(ensure_utc(value).timestamp() * 1000)


def floor_to_interval(value: datetime, interval: timedelta) -> datetime:
    """Round ``value`` down to the start of its interval bucket."""
    if interval <= timedelta(0):
        raise ValueError("interval must be positive")
    aware = ensure_utc(value)
    epoch_seconds = aware.timestamp()
    step = interval.total_seconds()
    return datetime.fromtimestamp(epoch_seconds - (epoch_seconds % step), tz=UTC)


class Clock(Protocol):
    """Time source. Injected so backtests can run on simulated time."""

    def now(self) -> datetime: ...

    def monotonic(self) -> float: ...


class SystemClock:
    """Wall-clock time."""

    def now(self) -> datetime:
        return utcnow()

    def monotonic(self) -> float:
        return time.monotonic()


class FixedClock:
    """Deterministic clock for tests and event-driven backtests."""

    def __init__(self, start: datetime) -> None:
        self._now = ensure_utc(start)
        self._monotonic = 0.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    def set(self, value: datetime) -> None:
        self._now = ensure_utc(value)

    def advance(self, delta: timedelta) -> datetime:
        self._now = self._now + delta
        self._monotonic += delta.total_seconds()
        return self._now


@dataclass(frozen=True, slots=True)
class ClockDriftReport:
    """Result of comparing local time against an exchange server time."""

    local_time: datetime
    reference_time: datetime
    drift_seconds: float
    tolerance_seconds: float

    @property
    def within_tolerance(self) -> bool:
        return abs(self.drift_seconds) <= self.tolerance_seconds

    def describe(self) -> str:
        state = "OK" if self.within_tolerance else "OUT OF TOLERANCE"
        return (
            f"clock drift {self.drift_seconds:+.3f}s "
            f"(tolerance {self.tolerance_seconds:.3f}s) - {state}"
        )


def measure_drift(
    local_time: datetime,
    reference_time: datetime,
    tolerance_seconds: float,
) -> ClockDriftReport:
    """Compare local time to a trusted reference (usually the exchange server clock)."""
    local = ensure_utc(local_time, field="local_time")
    reference = ensure_utc(reference_time, field="reference_time")
    return ClockDriftReport(
        local_time=local,
        reference_time=reference,
        drift_seconds=(local - reference).total_seconds(),
        tolerance_seconds=tolerance_seconds,
    )


INTERVAL_TO_TIMEDELTA: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "3m": timedelta(minutes=3),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
    "6h": timedelta(hours=6),
    "12h": timedelta(hours=12),
    "1d": timedelta(days=1),
    "1w": timedelta(weeks=1),
}


def interval_to_timedelta(interval: str) -> timedelta:
    """Map a canonical interval string (``"15m"``) to a :class:`timedelta`."""
    try:
        return INTERVAL_TO_TIMEDELTA[interval]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported interval {interval!r}; expected one of "
            f"{sorted(INTERVAL_TO_TIMEDELTA)}"
        ) from exc
