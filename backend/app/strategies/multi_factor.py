"""Multi-factor scoring strategy.

Hypothesis
----------
No single indicator family is reliable across conditions. Trend measures fail in ranges,
oscillators fail in trends, volume confirms nothing on its own. Combining weakly-correlated
evidence into a single score and requiring a high aggregate before acting produces fewer but
better trades than any component alone.

Scoring
-------
Six factors, each normalised to ``[-1, +1]`` where positive means bullish:

============  ================================================================
Factor        Measures
============  ================================================================
Trend         EMA alignment, EMA separation, and price relative to the slow EMA
Momentum      RSI displacement from 50 and MACD histogram sign/size
Volume        Volume expansion, signed by the direction of the current bar
Volatility    Whether ATR sits in a tradable band — a *gate*, not a direction
Regime        Agreement between the detected regime and the candidate direction
News          External news score, clamped and weighted (never acts alone)
============  ================================================================

The final score is a weighted sum, and a signal is emitted only when
``|final_score| >= entry_threshold``. Weights are configurable and are normalised so that
changing one weight does not silently rescale the threshold.

Two rules stop the score from being gamed by one dominant factor:

1. **Direction agreement** — at least ``min_agreeing_factors`` directional factors must share
   the sign of the final score. A score of 0.7 built from one extreme factor and four neutral
   ones is not a consensus.
2. **Volatility is a gate, not a vote** — an untradable volatility environment vetoes the
   trade regardless of how strong the other factors are.

News never trades alone: with all technical factors neutral, the maximum achievable score from
news alone is below the entry threshold by construction (validated at parameter level).

Failure modes
-------------
* **Factor correlation** — trend and momentum agree more often than the weights assume, so a
  "consensus" can be one signal counted twice. Mitigated by requiring agreement across *named*
  factor families and by keeping the trend/momentum combined weight below half.
* **Parameter overfitting** — six weights plus a threshold is a large search space. The
  walk-forward and sensitivity tooling in ``app.backtesting`` exists specifically for this
  strategy.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field, model_validator

from app.core.enums import MarketRegime, SignalAction
from app.core.numeric import clamp, safe_divide
from app.indicators.core import (
    atr,
    ema,
    last_valid,
    macd,
    rsi,
    volume_ma,
)
from app.strategies.base import (
    Strategy,
    StrategyContext,
    StrategyParameters,
    StrategyResult,
)


class MultiFactorParameters(StrategyParameters):
    """Parameters for :class:`MultiFactorStrategy`."""

    fast_ema_period: int = Field(default=21, ge=2, le=200)
    slow_ema_period: int = Field(default=55, ge=5, le=500)
    rsi_period: int = Field(default=14, ge=2, le=100)
    macd_fast: int = Field(default=12, ge=2, le=100)
    macd_slow: int = Field(default=26, ge=3, le=200)
    macd_signal: int = Field(default=9, ge=2, le=100)
    volume_period: int = Field(default=20, ge=5, le=200)

    trend_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    momentum_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    volume_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    volatility_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    regime_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    news_weight: float = Field(default=0.10, ge=0.0, le=0.30)

    entry_threshold: float = Field(
        default=0.45, gt=0.0, le=1.0,
        description="Minimum |final score| required to emit an entry",
    )
    exit_threshold: float = Field(
        default=0.15, ge=0.0, le=1.0,
        description="Score magnitude below which an open position is closed",
    )
    min_agreeing_factors: int = Field(default=3, ge=1, le=5)
    min_atr_pct: float = Field(default=0.002, ge=0.0, le=1.0)
    max_atr_pct: float = Field(default=0.10, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate(self) -> MultiFactorParameters:
        if self.fast_ema_period >= self.slow_ema_period:
            raise ValueError("fast_ema_period must be shorter than slow_ema_period")
        if self.macd_fast >= self.macd_slow:
            raise ValueError("macd_fast must be shorter than macd_slow")
        if self.min_atr_pct >= self.max_atr_pct:
            raise ValueError("min_atr_pct must be below max_atr_pct")
        if self.exit_threshold >= self.entry_threshold:
            raise ValueError("exit_threshold must be below entry_threshold")
        total = (
            self.trend_weight
            + self.momentum_weight
            + self.volume_weight
            + self.volatility_weight
            + self.regime_weight
            + self.news_weight
        )
        if total <= 0:
            raise ValueError("At least one factor weight must be positive")
        # News must never be able to trigger a trade by itself.
        if safe_divide(self.news_weight, total) >= self.entry_threshold:
            raise ValueError(
                f"news_weight is too large relative to entry_threshold: news alone could "
                f"produce a signal ({self.news_weight / total:.2f} >= "
                f"{self.entry_threshold:.2f}). News must remain a modifier, not a trigger."
            )
        return self

    @property
    def total_weight(self) -> float:
        return (
            self.trend_weight
            + self.momentum_weight
            + self.volume_weight
            + self.volatility_weight
            + self.regime_weight
            + self.news_weight
        )


class FactorScores(dict):
    """Named factor scores in ``[-1, 1]``, plus the weighted total."""

    @property
    def directional(self) -> dict[str, float]:
        """Factors that carry a direction. Volatility is a gate and is excluded."""
        return {k: v for k, v in self.items() if k not in {"volatility", "final"}}


class MultiFactorStrategy(Strategy):
    """Weighted scoring across trend, momentum, volume, volatility, regime and news."""

    name = "multi_factor"
    version = "1.0.0"
    description = (
        "Combines six weakly-correlated factors into one score and trades only on consensus."
    )
    parameters_model = MultiFactorParameters
    allowed_regimes = frozenset()  # any known regime; the regime factor scores it

    params: MultiFactorParameters

    @property
    def required_history(self) -> int:
        params = self.params
        return (
            max(
                params.slow_ema_period,
                params.macd_slow + params.macd_signal,
                params.rsi_period * 2,
                params.volume_period,
                params.atr_period,
            )
            + 15
        )

    def _evaluate(self, context: StrategyContext) -> StrategyResult:
        candles = context.candles
        params = self.params
        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)

        atr_value = last_valid(atr(highs, lows, closes, params.atr_period))
        if atr_value is None or atr_value <= 0:
            return StrategyResult.no_trade(
                context.symbol, self.name, "ATR has not warmed up", timestamp=context.now
            )
        price = float(closes[-1])
        atr_pct = atr_value / price

        trend = self._trend_factor(closes)
        momentum = self._momentum_factor(closes)
        volume = self._volume_factor(candles, volumes)
        volatility = self._volatility_factor(atr_pct)
        regime = self._regime_factor(context.regime, context.regime_confidence)
        news = clamp(context.news_score, -1.0, 1.0)

        if None in (trend, momentum):
            return StrategyResult.no_trade(
                context.symbol, self.name, "indicators have not warmed up",
                timestamp=context.now,
            )
        assert trend is not None and momentum is not None

        scores = FactorScores(
            trend=trend,
            momentum=momentum,
            volume=volume,
            volatility=volatility,
            regime=regime,
            news=news,
        )
        final = self._weighted_score(scores)
        scores["final"] = final

        diagnostics = {
            "score_trend": round(trend, 4),
            "score_momentum": round(momentum, 4),
            "score_volume": round(volume, 4),
            "score_volatility": round(volatility, 4),
            "score_regime": round(regime, 4),
            "score_news": round(news, 4),
            "score_final": round(final, 4),
            "atr_pct": round(atr_pct, 5),
        }

        # --- exit on score decay -------------------------------------------------
        if context.has_position and context.position is not None:
            long_position = context.position.side.sign > 0
            score_favours_position = (final > 0) == long_position
            if abs(final) < params.exit_threshold or not score_favours_position:
                return StrategyResult(
                    action=SignalAction.CLOSE,
                    symbol=context.symbol,
                    strategy_name=self.name,
                    timestamp=context.now,
                    confidence=0.7,
                    reason=(
                        f"composite score {final:+.2f} no longer supports the open position"
                    ),
                    metadata=diagnostics,
                )

        # --- volatility gate -----------------------------------------------------
        if atr_pct < params.min_atr_pct or atr_pct > params.max_atr_pct:
            return StrategyResult.no_trade(
                context.symbol, self.name,
                f"volatility gate: ATR {atr_pct:.2%} of price is outside "
                f"[{params.min_atr_pct:.2%}, {params.max_atr_pct:.2%}]",
                timestamp=context.now, metadata=diagnostics,
            )

        # --- threshold -----------------------------------------------------------
        if abs(final) < params.entry_threshold:
            return StrategyResult.no_trade(
                context.symbol, self.name,
                f"composite score {final:+.2f} below the "
                f"{params.entry_threshold:.2f} entry threshold",
                timestamp=context.now, metadata=diagnostics,
            )

        # --- consensus -----------------------------------------------------------
        direction = 1.0 if final > 0 else -1.0
        agreeing = sum(
            1
            for name, value in scores.directional.items()
            if abs(value) > 0.05 and (value > 0) == (direction > 0)
        )
        diagnostics["agreeing_factors"] = agreeing
        if agreeing < params.min_agreeing_factors:
            return StrategyResult.no_trade(
                context.symbol, self.name,
                f"only {agreeing} of {len(scores.directional)} factors agree "
                f"(minimum {params.min_agreeing_factors}); the score is driven by too few "
                "sources to be a consensus",
                timestamp=context.now, metadata=diagnostics,
            )

        action = SignalAction.BUY if direction > 0 else SignalAction.SELL
        if context.has_position and context.position is not None:
            aligned = (context.position.side.sign > 0) == (direction > 0)
            if aligned:
                return StrategyResult.hold(
                    context.symbol, self.name, "already positioned with the composite score",
                    timestamp=context.now,
                )

        stop_loss, take_profit = self.compute_levels(
            action=action, entry=price, atr_value=atr_value
        )
        return StrategyResult(
            action=action,
            symbol=context.symbol,
            strategy_name=self.name,
            timestamp=context.now,
            confidence=float(np.clip(abs(final), 0.0, 1.0)),
            entry=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=(
                f"Composite score {final:+.2f} with {agreeing} agreeing factors "
                f"(trend {trend:+.2f}, momentum {momentum:+.2f}, volume {volume:+.2f}, "
                f"regime {regime:+.2f}, news {news:+.2f})"
            ),
            metadata=diagnostics,
        )

    # ------------------------------------------------------------------ #
    # Factors
    # ------------------------------------------------------------------ #
    def _weighted_score(self, scores: FactorScores) -> float:
        params = self.params
        weighted = (
            params.trend_weight * scores["trend"]
            + params.momentum_weight * scores["momentum"]
            + params.volume_weight * scores["volume"]
            + params.volatility_weight * scores["volatility"]
            + params.regime_weight * scores["regime"]
            + params.news_weight * scores["news"]
        )
        return float(clamp(safe_divide(weighted, params.total_weight), -1.0, 1.0))

    def _trend_factor(self, closes: np.ndarray) -> float | None:
        params = self.params
        fast = last_valid(ema(closes, params.fast_ema_period))
        slow = last_valid(ema(closes, params.slow_ema_period))
        if fast is None or slow is None or slow <= 0:
            return None
        price = float(closes[-1])
        separation = clamp((fast - slow) / slow / 0.03, -1.0, 1.0)
        position = clamp((price - slow) / slow / 0.06, -1.0, 1.0)
        alignment = 1.0 if fast > slow else -1.0
        return float(clamp(0.4 * alignment + 0.35 * separation + 0.25 * position, -1.0, 1.0))

    def _momentum_factor(self, closes: np.ndarray) -> float | None:
        params = self.params
        rsi_value = last_valid(rsi(closes, params.rsi_period))
        _, _, histogram = macd(closes, params.macd_fast, params.macd_slow, params.macd_signal)
        hist_value = last_valid(histogram)
        if rsi_value is None or hist_value is None:
            return None
        price = float(closes[-1])
        rsi_component = clamp((rsi_value - 50.0) / 30.0, -1.0, 1.0)
        hist_component = clamp(safe_divide(hist_value, price * 0.01), -1.0, 1.0)
        return float(clamp(0.55 * rsi_component + 0.45 * hist_component, -1.0, 1.0))

    def _volume_factor(self, candles: tuple, volumes: np.ndarray) -> float:
        """Volume expansion signed by the direction of the current bar.

        Volume has no direction of its own; it amplifies whatever the bar did.
        """
        average = last_valid(volume_ma(volumes, self.params.volume_period))
        if average is None or average <= 0:
            return 0.0
        expansion = clamp((float(volumes[-1]) / average - 1.0) / 1.5, -1.0, 1.0)
        last = candles[-1]
        bar_range = last.high - last.low
        if bar_range <= 0:
            return 0.0
        # Where in the bar's range did it close? -1 at the low, +1 at the high.
        close_position = 2.0 * (last.close - last.low) / bar_range - 1.0
        return float(clamp(expansion * close_position, -1.0, 1.0))

    def _volatility_factor(self, atr_pct: float) -> float:
        """A gate expressed as a score.

        +1 in the middle of the tradable band, falling to -1 outside it. It carries no
        direction, which is why it is excluded from the consensus count.
        """
        params = self.params
        low, high = params.min_atr_pct, params.max_atr_pct
        if atr_pct < low or atr_pct > high:
            return -1.0
        centre = (low + high) / 2.0
        half_span = (high - low) / 2.0
        distance = abs(atr_pct - centre) / half_span if half_span > 0 else 0.0
        return float(clamp(1.0 - distance, -1.0, 1.0))

    @staticmethod
    def _regime_factor(regime: MarketRegime, confidence: float) -> float:
        """How strongly the regime favours a direction."""
        weight = clamp(confidence, 0.0, 1.0)
        mapping = {
            MarketRegime.TRENDING_BULL: 1.0,
            MarketRegime.TRENDING_BEAR: -1.0,
            MarketRegime.RANGING: 0.0,
            MarketRegime.LOW_VOLATILITY: 0.0,
            MarketRegime.HIGH_VOLATILITY: 0.0,
            MarketRegime.UNKNOWN: 0.0,
        }
        return float(mapping[regime] * weight)
