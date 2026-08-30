"""Trend-following strategy.

Hypothesis
----------
Markets that have moved persistently in one direction tend to continue for longer than a random
walk would predict, because trends are driven by slow information diffusion and positioning
that unwinds gradually. The edge is not in prediction accuracy — trend systems are wrong most
of the time — but in the payoff shape: many small losses funded by a few large wins.

That payoff shape only survives if losers are cut mechanically, which is why an ATR stop is
mandatory rather than optional, and why the take-profit defaults to a wide multiple.

Entry conditions (long; short is the mirror image)
--------------------------------------------------
1. **Direction** — fast EMA above slow EMA. The primary filter.
2. **Trend strength** — ADX above threshold with +DI above -DI. Distinguishes a real trend
   from two EMAs that happen to be ordered inside a range.
3. **Structure** — higher highs and higher lows over the lookback. An independent confirmation
   that does not share inputs with the EMAs, so it catches structural breaks the averages lag.
4. **Volume** — not abnormally thin. A trend advancing on collapsing volume is usually
   exhaustion.
5. **Volatility sanity** — ATR as a fraction of price within bounds. Too low and the stop is
   inside the noise; too high and position sizing collapses to nothing useful.
6. **Not over-extended** — price is not absurdly far above the fast EMA, which is where trend
   entries have the worst expectancy.

Exits
-----
* ATR stop, trailed by the strategy's exit signal.
* Take-profit at a configurable reward/risk multiple.
* Opposite EMA cross closes the position.

Failure modes
-------------
* **Whipsaw in a range.** Mitigated by the ADX filter and by refusing to trade the RANGING
  regime at all.
* **Volatility spikes.** A news-driven gap can jump the stop. Mitigated by the volatility
  ceiling and by the risk engine's exposure caps, not by this strategy.
* **Late entries.** Trend following is inherently late. The over-extension filter bounds how
  late.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field, model_validator

from app.core.enums import MarketRegime, SignalAction
from app.indicators.core import adx, atr, ema, last_valid
from app.strategies.base import (
    Strategy,
    StrategyContext,
    StrategyParameters,
    StrategyResult,
)


class TrendFollowingParameters(StrategyParameters):
    """Parameters for :class:`TrendFollowingStrategy`."""

    fast_ema_period: int = Field(default=21, ge=2, le=200)
    slow_ema_period: int = Field(default=55, ge=5, le=500)
    adx_period: int = Field(default=14, ge=2, le=100)
    adx_threshold: float = Field(
        default=25.0, ge=5.0, le=60.0, description="Minimum ADX for a tradable trend"
    )
    structure_lookback: int = Field(default=20, ge=5, le=200)
    min_volume_ratio: float = Field(
        default=0.6, ge=0.0, le=5.0, description="Latest volume / average, minimum"
    )
    min_atr_pct: float = Field(default=0.002, ge=0.0, le=1.0)
    max_atr_pct: float = Field(default=0.08, gt=0.0, le=1.0)
    max_extension_atr: float = Field(
        default=3.0, gt=0.0, le=20.0,
        description="Max distance from fast EMA, in ATRs, for a fresh entry",
    )
    require_structure: bool = True

    @model_validator(mode="after")
    def _validate(self) -> TrendFollowingParameters:
        if self.fast_ema_period >= self.slow_ema_period:
            raise ValueError(
                f"fast_ema_period ({self.fast_ema_period}) must be shorter than "
                f"slow_ema_period ({self.slow_ema_period})"
            )
        if self.min_atr_pct >= self.max_atr_pct:
            raise ValueError("min_atr_pct must be below max_atr_pct")
        return self


class TrendFollowingStrategy(Strategy):
    """EMA + ADX + structure trend follower with ATR-based risk levels."""

    name = "trend_following"
    version = "1.0.0"
    description = "Trades established directional trends confirmed by ADX and price structure."
    parameters_model = TrendFollowingParameters
    allowed_regimes = frozenset(
        {
            MarketRegime.TRENDING_BULL,
            MarketRegime.TRENDING_BEAR,
            MarketRegime.LOW_VOLATILITY,
        }
    )

    params: TrendFollowingParameters

    @property
    def required_history(self) -> int:
        return (
            max(
                self.params.slow_ema_period,
                self.params.adx_period * 3,
                self.params.structure_lookback * 2,
                self.params.atr_period,
            )
            + 10
        )

    def _evaluate(self, context: StrategyContext) -> StrategyResult:
        candles = context.candles
        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)

        params = self.params
        fast = last_valid(ema(closes, params.fast_ema_period))
        slow = last_valid(ema(closes, params.slow_ema_period))
        adx_series, plus_series, minus_series = adx(highs, lows, closes, params.adx_period)
        adx_value = last_valid(adx_series)
        plus_di = last_valid(plus_series)
        minus_di = last_valid(minus_series)
        atr_value = last_valid(atr(highs, lows, closes, params.atr_period))

        if None in (fast, slow, adx_value, plus_di, minus_di, atr_value):
            return StrategyResult.no_trade(
                context.symbol, self.name, "indicators have not warmed up",
                timestamp=context.now,
            )
        assert fast is not None and slow is not None and atr_value is not None
        assert adx_value is not None and plus_di is not None and minus_di is not None

        price = float(closes[-1])
        atr_pct = atr_value / price
        structure = self.trend_structure_score(candles, params.structure_lookback)
        volume_ratio = self.volume_ratio(candles, params.structure_lookback)

        diagnostics = {
            "fast_ema": round(fast, 6),
            "slow_ema": round(slow, 6),
            "adx": round(adx_value, 2),
            "plus_di": round(plus_di, 2),
            "minus_di": round(minus_di, 2),
            "atr": round(atr_value, 6),
            "atr_pct": round(atr_pct, 5),
            "structure": structure,
            "volume_ratio": round(volume_ratio, 3),
        }

        # --- universal filters -------------------------------------------------
        if adx_value < params.adx_threshold:
            return StrategyResult.no_trade(
                context.symbol, self.name,
                f"trend too weak: ADX {adx_value:.1f} < {params.adx_threshold:.1f}",
                timestamp=context.now, metadata=diagnostics,
            )
        if atr_pct < params.min_atr_pct:
            return StrategyResult.no_trade(
                context.symbol, self.name,
                f"volatility too low: ATR {atr_pct:.2%} of price; a stop here sits inside "
                "the noise",
                timestamp=context.now, metadata=diagnostics,
            )
        if atr_pct > params.max_atr_pct:
            return StrategyResult.no_trade(
                context.symbol, self.name,
                f"volatility too high: ATR {atr_pct:.2%} of price",
                timestamp=context.now, metadata=diagnostics,
            )
        if volume_ratio < params.min_volume_ratio:
            return StrategyResult.no_trade(
                context.symbol, self.name,
                f"volume {volume_ratio:.2f}x average is below the "
                f"{params.min_volume_ratio:.2f}x floor",
                timestamp=context.now, metadata=diagnostics,
            )

        bullish = fast > slow and plus_di > minus_di
        bearish = fast < slow and minus_di > plus_di

        # --- exits ------------------------------------------------------------
        if context.has_position:
            exit_result = self._maybe_exit(context, bullish=bullish, bearish=bearish,
                                           diagnostics=diagnostics)
            if exit_result is not None:
                return exit_result

        if not bullish and not bearish:
            return StrategyResult.no_trade(
                context.symbol, self.name,
                "EMA and directional indicators disagree on direction",
                timestamp=context.now, metadata=diagnostics,
            )

        # --- over-extension ----------------------------------------------------
        extension = abs(price - fast) / atr_value
        if extension > params.max_extension_atr:
            return StrategyResult.no_trade(
                context.symbol, self.name,
                f"price is {extension:.1f} ATR from the fast EMA; too extended to enter",
                timestamp=context.now, metadata={**diagnostics, "extension_atr": extension},
            )

        # --- structure ---------------------------------------------------------
        if params.require_structure:
            if bullish and structure < 0:
                return StrategyResult.no_trade(
                    context.symbol, self.name,
                    "EMAs are bullish but price structure is not making higher highs/lows",
                    timestamp=context.now, metadata=diagnostics,
                )
            if bearish and structure > 0:
                return StrategyResult.no_trade(
                    context.symbol, self.name,
                    "EMAs are bearish but price structure is not making lower highs/lows",
                    timestamp=context.now, metadata=diagnostics,
                )

        action = SignalAction.BUY if bullish else SignalAction.SELL
        if context.has_position and context.position is not None:
            # Already aligned with the trend: hold rather than pyramid. Adding to winners is a
            # position-sizing decision, and this strategy does not make those.
            aligned = (context.position.side.sign > 0) == (action is SignalAction.BUY)
            if aligned:
                return StrategyResult.hold(
                    context.symbol, self.name,
                    "already positioned with the trend", timestamp=context.now,
                )

        stop_loss, take_profit = self.compute_levels(
            action=action, entry=price, atr_value=atr_value
        )
        confidence = self._confidence(
            adx_value=adx_value,
            plus_di=plus_di,
            minus_di=minus_di,
            structure=structure,
            volume_ratio=volume_ratio,
            extension=extension,
            regime=context.regime,
            action=action,
        )
        return StrategyResult(
            action=action,
            symbol=context.symbol,
            strategy_name=self.name,
            timestamp=context.now,
            confidence=confidence,
            entry=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=(
                f"{'Bullish' if bullish else 'Bearish'} trend: "
                f"EMA{params.fast_ema_period} {'>' if bullish else '<'} "
                f"EMA{params.slow_ema_period}, ADX {adx_value:.1f}, "
                f"{'+' if bullish else '-'}DI dominant, structure {structure:+.1f}, "
                f"volume {volume_ratio:.2f}x"
            ),
            metadata={**diagnostics, "extension_atr": round(extension, 2)},
        )

    def _maybe_exit(
        self,
        context: StrategyContext,
        *,
        bullish: bool,
        bearish: bool,
        diagnostics: dict[str, float],
    ) -> StrategyResult | None:
        position = context.position
        assert position is not None
        long_position = position.side.sign > 0
        if (long_position and bearish) or (not long_position and bullish):
            return StrategyResult(
                action=SignalAction.CLOSE,
                symbol=context.symbol,
                strategy_name=self.name,
                timestamp=context.now,
                confidence=0.7,
                reason="trend has reversed against the open position",
                metadata=diagnostics,
            )
        return None

    def _confidence(
        self,
        *,
        adx_value: float,
        plus_di: float,
        minus_di: float,
        structure: float,
        volume_ratio: float,
        extension: float,
        regime: MarketRegime,
        action: SignalAction,
    ) -> float:
        """Blend the evidence into ``[0, 1]``.

        Confidence is used by the risk engine only to *reduce* size, never to increase it
        beyond the configured risk per trade.
        """
        params = self.params
        strength = np.clip((adx_value - params.adx_threshold) / 25.0, 0.0, 1.0)
        separation = np.clip(abs(plus_di - minus_di) / 25.0, 0.0, 1.0)
        structural = (structure + 1.0) / 2.0 if action is SignalAction.BUY else (
            1.0 - structure
        ) / 2.0
        volume_score = np.clip((volume_ratio - params.min_volume_ratio) / 1.5, 0.0, 1.0)
        freshness = np.clip(1.0 - extension / params.max_extension_atr, 0.0, 1.0)

        regime_bonus = 0.0
        if (regime is MarketRegime.TRENDING_BULL and action is SignalAction.BUY) or (
            regime is MarketRegime.TRENDING_BEAR and action is SignalAction.SELL
        ):
            regime_bonus = 0.10

        score = (
            0.30 * strength
            + 0.20 * separation
            + 0.20 * structural
            + 0.15 * freshness
            + 0.15 * volume_score
            + regime_bonus
        )
        return float(np.clip(score, 0.0, 1.0))
