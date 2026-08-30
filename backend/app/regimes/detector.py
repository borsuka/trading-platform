"""Market regime detection.

The regime is the platform's answer to "what kind of market is this, right now?". It exists
because the same signal means different things in different conditions: a mean-reversion entry
that is excellent in a range is a catastrophe in a strong trend, and a breakout entry in a
choppy market is a fee-generating machine.

Classification is deliberately explainable — ADX for trend strength, directional indicators for
sign, ATR-relative-to-its-own-history for volatility, and price structure as a cross-check. Each
detection carries the component values so a user can see *why* the bot called it a range.

``UNKNOWN`` is a real, load-bearing outcome. When the inputs disagree or there is not enough
history, the detector says so, and every strategy refuses to trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from app.core.clock import utcnow
from app.core.enums import MarketRegime
from app.core.numeric import clamp, safe_divide
from app.indicators.core import adx, atr, ema, last_valid, rolling_volatility
from app.market_data.models import Candle


@dataclass(frozen=True, slots=True)
class RegimeDetection:
    """A regime classification with the evidence behind it."""

    regime: MarketRegime
    confidence: float
    timestamp: datetime
    adx_value: float | None = None
    plus_di: float | None = None
    minus_di: float | None = None
    atr_pct: float | None = None
    volatility_percentile: float | None = None
    trend_slope: float | None = None
    structure_score: float | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_tradable(self) -> bool:
        return self.regime.allows_trading

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
            "adx": None if self.adx_value is None else round(self.adx_value, 2),
            "plus_di": None if self.plus_di is None else round(self.plus_di, 2),
            "minus_di": None if self.minus_di is None else round(self.minus_di, 2),
            "atr_pct": None if self.atr_pct is None else round(self.atr_pct, 5),
            "volatility_percentile": (
                None
                if self.volatility_percentile is None
                else round(self.volatility_percentile, 3)
            ),
            "trend_slope": None if self.trend_slope is None else round(self.trend_slope, 6),
            "structure_score": self.structure_score,
            "reason": self.reason,
        }


@dataclass(slots=True)
class RegimeConfig:
    """Thresholds for regime classification.

    Defaults follow conventional ADX interpretation (below 20 is directionless, above 25 is a
    trend) with a deliberate dead band between them so the classification does not flicker on
    every bar.
    """

    adx_period: int = 14
    adx_trend_threshold: float = 25.0
    adx_range_threshold: float = 20.0
    atr_period: int = 14
    volatility_lookback: int = 100
    high_volatility_percentile: float = 0.80
    low_volatility_percentile: float = 0.20
    #: Volatility this extreme overrides the trend classification entirely.
    extreme_volatility_percentile: float = 0.95
    fast_ema: int = 21
    slow_ema: int = 55
    structure_lookback: int = 20
    min_history: int = 120

    def __post_init__(self) -> None:
        if self.adx_range_threshold >= self.adx_trend_threshold:
            raise ValueError("adx_range_threshold must be below adx_trend_threshold")
        if not 0.0 < self.low_volatility_percentile < self.high_volatility_percentile < 1.0:
            raise ValueError("Volatility percentiles must satisfy 0 < low < high < 1")
        if self.fast_ema >= self.slow_ema:
            raise ValueError("fast_ema must be shorter than slow_ema")


class MarketRegimeDetector:
    """Classifies a candle series into a :class:`MarketRegime`."""

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()

    def detect(
        self, candles: tuple[Candle, ...] | list[Candle], *, now: datetime | None = None
    ) -> RegimeDetection:
        timestamp = now or (candles[-1].close_time if candles else utcnow())
        config = self.config

        if len(candles) < config.min_history:
            return RegimeDetection(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                timestamp=timestamp,
                reason=(
                    f"insufficient history: {len(candles)}/{config.min_history} bars"
                ),
            )

        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)
        closes = np.array([c.close for c in candles], dtype=float)

        adx_series, plus_series, minus_series = adx(highs, lows, closes, config.adx_period)
        adx_value = last_valid(adx_series)
        plus_di = last_valid(plus_series)
        minus_di = last_valid(minus_series)

        atr_series = atr(highs, lows, closes, config.atr_period)
        atr_value = last_valid(atr_series)
        price = float(closes[-1])
        atr_pct = safe_divide(atr_value or 0.0, price)

        volatility_percentile = self._volatility_percentile(atr_series, closes)
        trend_slope = self._trend_slope(closes)
        structure = self._structure_score(candles)

        if adx_value is None or atr_value is None or volatility_percentile is None:
            return RegimeDetection(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                timestamp=timestamp,
                reason="indicators have not warmed up",
            )

        return self._classify(
            timestamp=timestamp,
            adx_value=adx_value,
            plus_di=plus_di or 0.0,
            minus_di=minus_di or 0.0,
            atr_pct=atr_pct,
            volatility_percentile=volatility_percentile,
            trend_slope=trend_slope,
            structure=structure,
        )

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #
    def _classify(
        self,
        *,
        timestamp: datetime,
        adx_value: float,
        plus_di: float,
        minus_di: float,
        atr_pct: float,
        volatility_percentile: float,
        trend_slope: float,
        structure: float,
    ) -> RegimeDetection:
        config = self.config

        def build(regime: MarketRegime, confidence: float, reason: str) -> RegimeDetection:
            return RegimeDetection(
                regime=regime,
                confidence=clamp(confidence, 0.0, 1.0),
                timestamp=timestamp,
                adx_value=adx_value,
                plus_di=plus_di,
                minus_di=minus_di,
                atr_pct=atr_pct,
                volatility_percentile=volatility_percentile,
                trend_slope=trend_slope,
                structure_score=structure,
                reason=reason,
            )

        # Extreme volatility dominates everything. Direction is unreliable here and stops get
        # run on noise, so it is classified as its own regime rather than as a strong trend.
        if volatility_percentile >= config.extreme_volatility_percentile:
            return build(
                MarketRegime.HIGH_VOLATILITY,
                0.9,
                f"volatility at the {volatility_percentile:.0%} percentile of recent history",
            )

        is_trending = adx_value >= config.adx_trend_threshold
        is_ranging = adx_value <= config.adx_range_threshold

        if is_trending:
            directional_gap = abs(plus_di - minus_di)
            agreement = self._trend_agreement(plus_di, minus_di, trend_slope, structure)
            strength = clamp((adx_value - config.adx_trend_threshold) / 30.0, 0.0, 1.0)
            confidence = clamp(0.45 + 0.35 * strength + 0.20 * agreement, 0.0, 1.0)

            if directional_gap < 3.0 or agreement < 0.25:
                # ADX says "trending" but nothing agrees on which way.
                return build(
                    MarketRegime.UNKNOWN,
                    0.3,
                    f"ADX {adx_value:.1f} indicates a trend but direction is ambiguous "
                    f"(+DI {plus_di:.1f} vs -DI {minus_di:.1f}, slope {trend_slope:+.4f})",
                )
            if plus_di > minus_di:
                return build(
                    MarketRegime.TRENDING_BULL,
                    confidence,
                    f"ADX {adx_value:.1f} with +DI {plus_di:.1f} > -DI {minus_di:.1f}, "
                    f"slope {trend_slope:+.4f}",
                )
            return build(
                MarketRegime.TRENDING_BEAR,
                confidence,
                f"ADX {adx_value:.1f} with -DI {minus_di:.1f} > +DI {plus_di:.1f}, "
                f"slope {trend_slope:+.4f}",
            )

        if volatility_percentile >= config.high_volatility_percentile:
            return build(
                MarketRegime.HIGH_VOLATILITY,
                0.6 + 0.3 * (volatility_percentile - config.high_volatility_percentile) / 0.2,
                f"no directional trend (ADX {adx_value:.1f}) with elevated volatility "
                f"({volatility_percentile:.0%} percentile)",
            )
        if volatility_percentile <= config.low_volatility_percentile:
            return build(
                MarketRegime.LOW_VOLATILITY,
                0.6,
                f"compressed volatility ({volatility_percentile:.0%} percentile), "
                f"ADX {adx_value:.1f}",
            )
        if is_ranging:
            return build(
                MarketRegime.RANGING,
                clamp(0.5 + (config.adx_range_threshold - adx_value) / 20.0, 0.0, 0.95),
                f"ADX {adx_value:.1f} below {config.adx_range_threshold:.0f}: no trend",
            )

        # ADX sits in the dead band between the range and trend thresholds.
        return build(
            MarketRegime.RANGING,
            0.35,
            f"ADX {adx_value:.1f} is in the transitional band "
            f"[{config.adx_range_threshold:.0f}, {config.adx_trend_threshold:.0f}]",
        )

    @staticmethod
    def _trend_agreement(
        plus_di: float, minus_di: float, slope: float, structure: float
    ) -> float:
        """How strongly the independent trend measures agree, in ``[0, 1]``."""
        direction = 1.0 if plus_di > minus_di else -1.0
        votes = [
            1.0 if (slope > 0) == (direction > 0) else 0.0,
            1.0 if (structure > 0) == (direction > 0) else 0.0 if structure != 0 else 0.5,
            clamp(abs(plus_di - minus_di) / 25.0, 0.0, 1.0),
        ]
        return float(np.mean(votes))

    def _volatility_percentile(
        self, atr_series: np.ndarray, closes: np.ndarray
    ) -> float | None:
        """Where current volatility sits within its own recent distribution.

        Comparing ATR to its own history, rather than to an absolute threshold, is what makes
        the detector work unchanged across symbols with wildly different price levels.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = atr_series / closes
        valid = normalized[~np.isnan(normalized)]
        if valid.size < 20:
            return None
        window = valid[-self.config.volatility_lookback :]
        current = float(valid[-1])
        return float(np.mean(window <= current))

    def _trend_slope(self, closes: np.ndarray) -> float:
        """Normalised slope of the fast EMA over the structure lookback."""
        fast = ema(closes, self.config.fast_ema)
        valid = fast[~np.isnan(fast)]
        window = self.config.structure_lookback
        if valid.size < window + 1:
            return 0.0
        start, end = float(valid[-window - 1]), float(valid[-1])
        return safe_divide(end - start, abs(start) * window)

    def _structure_score(self, candles: tuple[Candle, ...] | list[Candle]) -> float:
        """Higher-highs/higher-lows score in ``[-1, 1]``."""
        window = self.config.structure_lookback
        if len(candles) < window * 2:
            return 0.0
        recent = candles[-window:]
        prior = candles[-window * 2 : -window]
        score = 0.0
        score += 0.5 if max(c.high for c in recent) > max(c.high for c in prior) else -0.5
        score += 0.5 if min(c.low for c in recent) > min(c.low for c in prior) else -0.5
        return score


def annualization_for(interval: str) -> float:
    """Square-root-of-time factor for annualising volatility at a given bar interval.

    Crypto trades continuously, so the year has 365 days rather than 252 trading days.
    """
    from app.core.clock import interval_to_timedelta

    seconds = interval_to_timedelta(interval).total_seconds()
    bars_per_year = (365.0 * 24.0 * 3600.0) / seconds
    return float(np.sqrt(bars_per_year))


def realized_volatility(
    candles: tuple[Candle, ...] | list[Candle], period: int = 20, *, interval: str | None = None
) -> float | None:
    """Annualised realised volatility of the most recent window."""
    if len(candles) < period + 1:
        return None
    closes = np.array([c.close for c in candles], dtype=float)
    factor = annualization_for(interval or candles[-1].interval)
    series = rolling_volatility(closes, period, annualization_factor=factor)
    return last_valid(series)
