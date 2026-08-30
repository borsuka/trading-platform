"""Candle normalisation.

Exchange history is not clean. It arrives out of order, contains duplicate bars from
overlapping paginated requests, skips bars when the venue had no trades, and occasionally
carries a bar that has not closed yet. Every one of those produces a plausible-looking but wrong
backtest if it reaches an indicator, so normalisation happens once at the data boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import pairwise

from app.core.clock import interval_to_timedelta
from app.core.logging import get_logger
from app.market_data.models import Candle

logger = get_logger(__name__)


@dataclass(slots=True)
class NormalizationReport:
    """What the normaliser had to do. Surfaced in backtest reports for data-quality review."""

    input_count: int = 0
    output_count: int = 0
    duplicates_removed: int = 0
    reordered: bool = False
    gaps_detected: int = 0
    gaps_filled: int = 0
    unclosed_dropped: int = 0
    invalid_dropped: int = 0
    gap_ranges: list[tuple[datetime, datetime]] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return (
            self.duplicates_removed == 0
            and not self.reordered
            and self.gaps_detected == 0
            and self.invalid_dropped == 0
        )

    def summary(self) -> str:
        if self.is_clean:
            return f"{self.output_count} candles, clean"
        return (
            f"{self.output_count}/{self.input_count} candles: "
            f"{self.duplicates_removed} duplicates, {self.gaps_detected} gaps "
            f"({self.gaps_filled} filled), {self.invalid_dropped} invalid, "
            f"{self.unclosed_dropped} unclosed"
        )


class CandleNormalizer:
    """Turns a raw candle stream into a strictly increasing, gap-aware series.

    Parameters
    ----------
    fill_gaps:
        When true, missing bars are synthesised as zero-volume dojis at the previous close.
        This keeps indicator windows aligned in wall-clock time. When false, gaps are reported
        but the series stays sparse — appropriate when the strategy reasons in bar counts.
    drop_unclosed:
        Drop bars still forming. Enabled by default: a forming bar's close will still change,
        so acting on it is lookahead.
    max_gap_fill:
        Refuse to synthesise more than this many consecutive bars. A week-long outage should
        surface as a data problem, not as a week of flat synthetic prices.
    """

    def __init__(
        self,
        *,
        fill_gaps: bool = False,
        drop_unclosed: bool = True,
        max_gap_fill: int = 10,
    ) -> None:
        self.fill_gaps = fill_gaps
        self.drop_unclosed = drop_unclosed
        self.max_gap_fill = max_gap_fill

    def normalize(
        self, candles: Iterable[Candle], *, interval: str | None = None
    ) -> tuple[list[Candle], NormalizationReport]:
        raw = list(candles)
        report = NormalizationReport(input_count=len(raw))
        if not raw:
            return [], report

        resolved_interval = interval or raw[0].interval
        step = interval_to_timedelta(resolved_interval)

        kept = self._drop_unclosed(raw, report)
        if not kept:
            return [], report

        report.reordered = any(
            kept[i].open_time > kept[i + 1].open_time for i in range(len(kept) - 1)
        )
        kept.sort(key=lambda c: c.open_time)

        deduped = self._deduplicate(kept, report)
        aligned = self._align_to_grid(deduped, step, report)

        result = self._handle_gaps(aligned, step, report)
        report.output_count = len(result)
        if not report.is_clean:
            logger.info(
                "market_data.normalized",
                symbol=result[0].symbol if result else "?",
                interval=resolved_interval,
                summary=report.summary(),
            )
        return result, report

    def _drop_unclosed(self, candles: list[Candle], report: NormalizationReport) -> list[Candle]:
        if not self.drop_unclosed:
            return list(candles)
        kept = [candle for candle in candles if candle.is_closed]
        report.unclosed_dropped = len(candles) - len(kept)
        return kept

    def _deduplicate(
        self, candles: list[Candle], report: NormalizationReport
    ) -> list[Candle]:
        """Keep the last occurrence of each open_time.

        The last copy wins because paginated exchange responses append revisions, so a later
        duplicate is the corrected bar.
        """
        by_time: dict[datetime, Candle] = {}
        for candle in candles:
            by_time[candle.open_time] = candle
        report.duplicates_removed = len(candles) - len(by_time)
        return [by_time[key] for key in sorted(by_time)]

    def _align_to_grid(
        self, candles: list[Candle], step: timedelta, report: NormalizationReport
    ) -> list[Candle]:
        """Drop bars whose open_time is not on the interval grid."""
        if not candles:
            return []
        origin = candles[0].open_time
        aligned: list[Candle] = []
        for candle in candles:
            offset = (candle.open_time - origin) % step
            if offset != timedelta(0):
                report.invalid_dropped += 1
                logger.warning(
                    "market_data.candle_off_grid",
                    symbol=candle.symbol,
                    open_time=candle.open_time.isoformat(),
                    interval=candle.interval,
                )
                continue
            aligned.append(candle)
        return aligned

    def _handle_gaps(
        self, candles: list[Candle], step: timedelta, report: NormalizationReport
    ) -> list[Candle]:
        if len(candles) < 2:
            return candles
        result: list[Candle] = [candles[0]]
        for previous, current in pairwise(candles):
            expected = previous.open_time + step
            if current.open_time > expected:
                missing = int((current.open_time - expected) / step)
                report.gaps_detected += 1
                report.gap_ranges.append((expected, current.open_time))
                if self.fill_gaps and missing <= self.max_gap_fill:
                    result.extend(self._synthesize(previous, step, missing))
                    report.gaps_filled += missing
                elif self.fill_gaps:
                    logger.warning(
                        "market_data.gap_too_large_to_fill",
                        symbol=current.symbol,
                        missing_bars=missing,
                        limit=self.max_gap_fill,
                        gap_start=expected.isoformat(),
                    )
            result.append(current)
        return result

    @staticmethod
    def _synthesize(previous: Candle, step: timedelta, count: int) -> list[Candle]:
        """Zero-volume dojis at the last known close."""
        price = previous.close
        return [
            Candle(
                symbol=previous.symbol,
                interval=previous.interval,
                open_time=previous.open_time + step * (offset + 1),
                open=price,
                high=price,
                low=price,
                close=price,
                volume=0.0,
                quote_volume=0.0,
                trade_count=0,
                is_closed=True,
            )
            for offset in range(count)
        ]


def merge_candles(
    existing: Sequence[Candle], incoming: Sequence[Candle]
) -> list[Candle]:
    """Merge two candle series, preferring ``incoming`` where open times collide."""
    merged: dict[datetime, Candle] = {c.open_time: c for c in existing}
    merged.update({c.open_time: c for c in incoming})
    return [merged[key] for key in sorted(merged)]
