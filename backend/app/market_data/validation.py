"""Market data validation.

The validator answers one question: *is this data safe to trade on right now?* It is
deliberately conservative. A validator that says "probably fine" when the feed has been silent
for four minutes is how a bot ends up trading a stale price into a moving market.

The result is a structured :class:`ValidationResult` rather than an exception, because the
caller usually wants to record why trading was suppressed, not unwind a stack.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import pairwise

from app.core.clock import interval_to_timedelta, utcnow
from app.core.numeric import safe_divide
from app.market_data.models import Candle, MarketSnapshot, OrderBook, Ticker


@dataclass(slots=True)
class ValidationIssue:
    code: str
    message: str
    is_fatal: bool = True

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


@dataclass(slots=True)
class ValidationResult:
    """Outcome of validating a market data snapshot."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when nothing fatal was found. Warnings do not block trading."""
        return not any(issue.is_fatal for issue in self.issues)

    @property
    def fatal_issues(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.is_fatal]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if not issue.is_fatal]

    def add(self, code: str, message: str, *, fatal: bool = True) -> None:
        self.issues.append(ValidationIssue(code=code, message=message, is_fatal=fatal))

    def reason(self) -> str:
        if self.is_valid:
            return "ok"
        return "; ".join(str(issue) for issue in self.fatal_issues)


class MarketDataValidator:
    """Validates candles, tickers and books before they reach the decision pipeline.

    Parameters
    ----------
    max_staleness_seconds:
        How old the newest closed bar may be. Compared against bar *close* time plus one
        interval of tolerance, so a bar that has only just closed is never flagged.
    max_bar_move_pct:
        A single-bar move beyond this is treated as a probable bad print rather than a real
        move. Flagged as a warning, not fatal: real markets do gap.
    min_required_bars:
        Below this the snapshot cannot support the indicator warm-up windows.
    """

    def __init__(
        self,
        *,
        max_staleness_seconds: float = 120.0,
        max_bar_move_pct: float = 0.35,
        min_required_bars: int = 50,
        max_spread_bps: float = 100.0,
    ) -> None:
        self.max_staleness_seconds = max_staleness_seconds
        self.max_bar_move_pct = max_bar_move_pct
        self.min_required_bars = min_required_bars
        self.max_spread_bps = max_spread_bps

    # ------------------------------------------------------------------ #
    # Candles
    # ------------------------------------------------------------------ #
    def validate_candles(
        self,
        candles: Sequence[Candle],
        *,
        now: datetime | None = None,
        required_bars: int | None = None,
    ) -> ValidationResult:
        result = ValidationResult()
        minimum = required_bars if required_bars is not None else self.min_required_bars

        if not candles:
            result.add("no_data", "No candles available")
            return result

        if len(candles) < minimum:
            result.add(
                "insufficient_history",
                f"{len(candles)} candles available, {minimum} required",
            )

        self._check_ordering(candles, result)
        self._check_continuity(candles, result)
        self._check_values(candles, result)
        self._check_staleness(candles[-1], result, now=now)
        return result

    @staticmethod
    def _check_ordering(candles: Sequence[Candle], result: ValidationResult) -> None:
        for previous, current in pairwise(candles):
            if current.open_time <= previous.open_time:
                result.add(
                    "out_of_order",
                    f"Candles not strictly increasing at {current.open_time.isoformat()}",
                )
                return

    @staticmethod
    def _check_continuity(candles: Sequence[Candle], result: ValidationResult) -> None:
        if len(candles) < 2:
            return
        try:
            step = interval_to_timedelta(candles[-1].interval)
        except ValueError as exc:
            result.add("bad_interval", str(exc))
            return
        gaps = sum(
            1
            for previous, current in pairwise(candles)
            if current.open_time - previous.open_time > step
        )
        if gaps:
            ratio = gaps / max(1, len(candles) - 1)
            result.add(
                "gaps",
                f"{gaps} gap(s) in {len(candles)} candles ({ratio:.1%})",
                fatal=ratio > 0.10,
            )

    def _check_values(self, candles: Sequence[Candle], result: ValidationResult) -> None:
        for candle in candles:
            for name in ("open", "high", "low", "close"):
                value = getattr(candle, name)
                if not math.isfinite(value) or value <= 0:
                    result.add(
                        "invalid_price",
                        f"{candle.symbol} {candle.open_time.isoformat()}: {name}={value}",
                    )
                    return
        for previous, current in pairwise(candles):
            move = abs(safe_divide(current.close - previous.close, previous.close))
            if move > self.max_bar_move_pct:
                result.add(
                    "extreme_move",
                    f"{current.symbol} {current.open_time.isoformat()}: "
                    f"{move:.1%} single-bar move",
                    fatal=False,
                )

    def _check_staleness(
        self, latest: Candle, result: ValidationResult, *, now: datetime | None = None
    ) -> None:
        reference = now or utcnow()
        try:
            tolerance = interval_to_timedelta(latest.interval)
        except ValueError:
            tolerance = timedelta(0)
        age = reference - latest.close_time
        allowed = timedelta(seconds=self.max_staleness_seconds) + tolerance
        if age > allowed:
            result.add(
                "stale_data",
                f"Latest candle closed {age.total_seconds():.0f}s ago "
                f"(limit {allowed.total_seconds():.0f}s)",
            )
        elif age < -tolerance:
            result.add(
                "future_data",
                f"Latest candle closes {-age.total_seconds():.0f}s in the future; "
                "check clock synchronisation",
            )

    # ------------------------------------------------------------------ #
    # Ticker and book
    # ------------------------------------------------------------------ #
    def validate_ticker(
        self, ticker: Ticker, *, now: datetime | None = None
    ) -> ValidationResult:
        result = ValidationResult()
        age = ticker.age(now)
        if age > timedelta(seconds=self.max_staleness_seconds):
            result.add(
                "stale_ticker",
                f"Ticker is {age.total_seconds():.0f}s old "
                f"(limit {self.max_staleness_seconds:.0f}s)",
            )
        spread = ticker.spread_bps
        if spread is not None:
            if spread < 0:
                result.add("crossed_book", f"Negative spread {spread:.1f} bps")
            elif spread > self.max_spread_bps:
                result.add(
                    "wide_spread",
                    f"Spread {spread:.1f} bps exceeds {self.max_spread_bps:.1f} bps",
                )
        return result

    def validate_order_book(
        self, book: OrderBook, *, now: datetime | None = None
    ) -> ValidationResult:
        result = ValidationResult()
        if not book.bids or not book.asks:
            result.add("empty_book", f"{book.symbol}: one or both sides of the book are empty")
            return result
        age = (now or utcnow()) - book.timestamp
        if age > timedelta(seconds=self.max_staleness_seconds):
            result.add("stale_book", f"Order book is {age.total_seconds():.0f}s old")
        spread = book.spread_bps
        if spread is not None and spread > self.max_spread_bps:
            result.add("wide_spread", f"Book spread {spread:.1f} bps")
        return result

    # ------------------------------------------------------------------ #
    # Snapshot
    # ------------------------------------------------------------------ #
    def validate_snapshot(
        self,
        snapshot: MarketSnapshot,
        *,
        now: datetime | None = None,
        required_bars: int | None = None,
    ) -> ValidationResult:
        """Validate everything present in a snapshot in one pass."""
        result = self.validate_candles(
            snapshot.candles, now=now, required_bars=required_bars
        )
        if snapshot.ticker is not None:
            result.issues.extend(self.validate_ticker(snapshot.ticker, now=now).issues)
        if snapshot.order_book is not None:
            result.issues.extend(self.validate_order_book(snapshot.order_book, now=now).issues)

        # Cross-source sanity: a ticker far from the last close means one source is wrong.
        last = snapshot.last_candle
        if last is not None and snapshot.ticker is not None:
            divergence = abs(safe_divide(snapshot.ticker.price - last.close, last.close))
            if divergence > 0.10:
                result.add(
                    "source_divergence",
                    f"Ticker {snapshot.ticker.price:.4f} diverges "
                    f"{divergence:.1%} from last close {last.close:.4f}",
                )
        return result
