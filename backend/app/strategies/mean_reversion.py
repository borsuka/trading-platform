"""Mean-reversion strategy.

Hypothesis
----------
Inside a range, price oscillates around a central value because liquidity providers lean
against moves and there is no information driving a sustained repricing. Buying statistically
cheap and selling statistically dear inside such a range has positive expectancy.

The critical qualifier is *inside a range*. The same logic applied during a trend is
catastrophic: it buys every step down of a decline and sells every step up of a rally,
producing a long run of small wins followed by one loss that erases them. This is the classic
mean-reversion blow-up, and it is why the regime filter here is not a refinement but the
strategy's central safety mechanism.

Guards against trading a trend
------------------------------
1. **Hard regime gate** — the strategy is only permitted to enter in ``RANGING`` and
   ``LOW_VOLATILITY`` regimes. Trending regimes are excluded at the framework level.
2. **ADX ceiling** — even within a "ranging" classification, an ADX above the ceiling vetoes
   entry. Two independent brakes, because this is the failure mode that kills the strategy.
3. **Slope check** — the mid-band must be roughly flat. A rising mean is a trend by another
   name.
4. **Band-walk detection** — if price has spent several consecutive bars beyond the band, it is
   not stretched, it is trending. Entry is refused.

Entry conditions (long; short is the mirror image)
--------------------------------------------------
* Close below the lower Bollinger band, **and**
* RSI below the oversold threshold, **and**
* z-score below the negative threshold.

Requiring all three, which are correlated but not identical, filters out the single-indicator
false positives that dominate this style.

Exits
-----
* Reversion to the mid-band (the trade's purpose is complete).
* ATR stop beyond the band — deliberately wider than the trend strategies, because mean
  reversion needs room, but hard-capped so a trend cannot run away with the position.
* RSI crossing back through neutral.

Failure modes
-------------
* **Trend disguised as a range** — the four guards above.
* **Volatility regime shift** — a compressing range that suddenly expands. Mitigated by the ATR
  stop and by the risk engine's drawdown controls.
* **Correlated entries** — many symbols oversold at once during a market-wide sell-off. That is
  handled by portfolio-level exposure limits in the risk manager, not here.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field, model_validator

from app.core.enums import MarketRegime, SignalAction
from app.core.numeric import safe_divide
from app.indicators.core import (
    adx,
    atr,
    bollinger_bands,
    last_valid,
    rsi,
    sma,
    zscore,
)
from app.strategies.base import (
    Strategy,
    StrategyContext,
    StrategyParameters,
    StrategyResult,
)


class MeanReversionParameters(StrategyParameters):
    """Parameters for :class:`MeanReversionStrategy`."""

    bollinger_period: int = Field(default=20, ge=5, le=200)
    bollinger_std: float = Field(default=2.0, gt=0.5, le=5.0)
    rsi_period: int = Field(default=14, ge=2, le=100)
    rsi_oversold: float = Field(default=30.0, ge=1.0, le=49.0)
    rsi_overbought: float = Field(default=70.0, ge=51.0, le=99.0)
    rsi_exit_long: float = Field(default=55.0, ge=40.0, le=80.0)
    rsi_exit_short: float = Field(default=45.0, ge=20.0, le=60.0)
    zscore_period: int = Field(default=20, ge=5, le=200)
    zscore_threshold: float = Field(default=2.0, gt=0.2, le=6.0)
    adx_period: int = Field(default=14, ge=2, le=100)
    max_adx: float = Field(
        default=22.0, ge=5.0, le=50.0,
        description="Trend-strength ceiling; above this the range premise is void",
    )
    max_mid_slope: float = Field(
        default=0.0015, ge=0.0, le=1.0,
        description="Max absolute normalised slope of the mid-band per bar",
    )
    band_walk_bars: int = Field(
        default=3, ge=1, le=20,
        description="Consecutive bars beyond the band that indicate a trend, not a stretch",
    )
    max_stop_atr: float = Field(
        default=4.0, gt=0.0, le=20.0, description="Hard ceiling on stop distance in ATRs"
    )
    exit_at_mid: bool = True

    @model_validator(mode="after")
    def _validate(self) -> MeanReversionParameters:
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError("rsi_oversold must be below rsi_overbought")
        if self.rsi_exit_short >= self.rsi_exit_long:
            raise ValueError("rsi_exit_short must be below rsi_exit_long")
        return self


class MeanReversionStrategy(Strategy):
    """Bollinger + RSI + z-score reversion, gated hard against trending markets."""

    name = "mean_reversion"
    version = "1.0.0"
    description = (
        "Fades statistical extremes inside a range. Refuses to trade when a trend is present."
    )
    parameters_model = MeanReversionParameters
    allowed_regimes = frozenset({MarketRegime.RANGING, MarketRegime.LOW_VOLATILITY})

    params: MeanReversionParameters

    @property
    def required_history(self) -> int:
        params = self.params
        return (
            max(
                params.bollinger_period,
                params.rsi_period * 2,
                params.zscore_period,
                params.adx_period * 3,
                params.atr_period,
            )
            + params.band_walk_bars
            + 10
        )

    def _evaluate(self, context: StrategyContext) -> StrategyResult:
        candles = context.candles
        params = self.params
        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)

        upper_band, mid_band, lower_band = bollinger_bands(
            closes, params.bollinger_period, params.bollinger_std
        )
        rsi_series = rsi(closes, params.rsi_period)
        z_series = zscore(closes, params.zscore_period)
        adx_series, _, _ = adx(highs, lows, closes, params.adx_period)
        atr_value = last_valid(atr(highs, lows, closes, params.atr_period))

        upper = float(upper_band[-1]) if not np.isnan(upper_band[-1]) else None
        mid = float(mid_band[-1]) if not np.isnan(mid_band[-1]) else None
        lower = float(lower_band[-1]) if not np.isnan(lower_band[-1]) else None
        rsi_value = last_valid(rsi_series)
        z_value = last_valid(z_series)
        adx_value = last_valid(adx_series)

        if None in (upper, mid, lower, rsi_value, z_value, adx_value, atr_value):
            return StrategyResult.no_trade(
                context.symbol, self.name, "indicators have not warmed up",
                timestamp=context.now,
            )
        assert upper is not None and mid is not None and lower is not None
        assert rsi_value is not None and z_value is not None
        assert adx_value is not None and atr_value is not None

        price = float(closes[-1])
        band_width_pct = safe_divide(upper - lower, mid)
        mid_slope = self._mid_slope(closes)

        diagnostics: dict[str, float] = {
            "upper_band": round(upper, 6),
            "mid_band": round(mid, 6),
            "lower_band": round(lower, 6),
            "rsi": round(rsi_value, 2),
            "zscore": round(z_value, 3),
            "adx": round(adx_value, 2),
            "atr": round(atr_value, 6),
            "band_width_pct": round(band_width_pct, 5),
            "mid_slope": round(mid_slope, 6),
        }

        # --- exits first --------------------------------------------------------
        if context.has_position and context.position is not None:
            exit_result = self._maybe_exit(
                context, mid=mid, rsi_value=rsi_value, diagnostics=diagnostics
            )
            if exit_result is not None:
                return exit_result

        # --- trend guards -------------------------------------------------------
        if adx_value > params.max_adx:
            return StrategyResult.no_trade(
                context.symbol, self.name,
                f"ADX {adx_value:.1f} exceeds the {params.max_adx:.1f} ceiling: this is a "
                "trend, and fading a trend is this strategy's worst failure mode",
                timestamp=context.now, metadata=diagnostics,
            )
        if abs(mid_slope) > params.max_mid_slope:
            return StrategyResult.no_trade(
                context.symbol, self.name,
                f"the mean itself is moving ({mid_slope:+.4f}/bar); there is no stable level "
                "to revert to",
                timestamp=context.now, metadata=diagnostics,
            )

        oversold = price < lower and rsi_value < params.rsi_oversold
        overbought = price > upper and rsi_value > params.rsi_overbought
        z_low = z_value < -params.zscore_threshold
        z_high = z_value > params.zscore_threshold

        if not ((oversold and z_low) or (overbought and z_high)):
            return StrategyResult.no_trade(
                context.symbol, self.name,
                f"no confirmed extreme: price {price:.4f} vs bands "
                f"[{lower:.4f}, {upper:.4f}], RSI {rsi_value:.1f}, z {z_value:+.2f}",
                timestamp=context.now, metadata=diagnostics,
            )

        going_long = oversold and z_low
        if self._is_walking_the_band(closes, upper_band, lower_band, going_long):
            return StrategyResult.no_trade(
                context.symbol, self.name,
                f"price has held beyond the band for {params.band_walk_bars}+ bars: it is "
                "trending, not stretched",
                timestamp=context.now, metadata=diagnostics,
            )

        action = SignalAction.BUY if going_long else SignalAction.SELL
        if context.has_position and context.position is not None:
            aligned = (context.position.side.sign > 0) == going_long
            if aligned:
                return StrategyResult.hold(
                    context.symbol, self.name, "already positioned for this reversion",
                    timestamp=context.now,
                )

        stop_loss, take_profit = self._levels(
            action=action, entry=price, atr_value=atr_value, mid=mid,
            upper=upper, lower=lower,
        )
        confidence = self._confidence(
            rsi_value=rsi_value,
            z_value=z_value,
            adx_value=adx_value,
            band_width_pct=band_width_pct,
            going_long=going_long,
            regime=context.regime,
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
                f"{'Oversold' if going_long else 'Overbought'} extreme in a range: "
                f"price {'below' if going_long else 'above'} the "
                f"{params.bollinger_std:g}-sigma band, RSI {rsi_value:.1f}, "
                f"z-score {z_value:+.2f}, ADX {adx_value:.1f}"
            ),
            metadata=diagnostics,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _levels(
        self,
        *,
        action: SignalAction,
        entry: float,
        atr_value: float,
        mid: float,
        upper: float,
        lower: float,
    ) -> tuple[float, float]:
        """Stop beyond the band, target at the mean.

        The stop is placed outside the band rather than at a fixed ATR multiple, because the
        band *is* the level being faded — but it is capped at ``max_stop_atr`` so a runaway
        move cannot produce an arbitrarily wide (and therefore arbitrarily small-sized) trade.
        """
        params = self.params
        cap = atr_value * params.max_stop_atr
        if action is SignalAction.BUY:
            raw_stop = min(lower, entry) - atr_value * params.atr_stop_multiplier
            stop = max(raw_stop, entry - cap)
            target = mid if params.exit_at_mid else entry + (entry - stop) * (
                params.risk_reward_ratio
            )
            # Guarantee the target sits above entry even if the mean has drifted below it.
            target = max(target, entry + (entry - stop) * 0.5)
            return stop, target

        raw_stop = max(upper, entry) + atr_value * params.atr_stop_multiplier
        stop = min(raw_stop, entry + cap)
        target = mid if params.exit_at_mid else entry - (stop - entry) * (
            params.risk_reward_ratio
        )
        target = min(target, entry - (stop - entry) * 0.5)
        return stop, target

    def _mid_slope(self, closes: np.ndarray) -> float:
        """Normalised per-bar slope of the mid-band over its own period."""
        mid = sma(closes, self.params.bollinger_period)
        valid = mid[~np.isnan(mid)]
        window = self.params.bollinger_period
        if valid.size < window + 1:
            return 0.0
        start, end = float(valid[-window - 1]), float(valid[-1])
        return safe_divide(end - start, abs(start) * window)

    def _is_walking_the_band(
        self,
        closes: np.ndarray,
        upper_band: np.ndarray,
        lower_band: np.ndarray,
        going_long: bool,
    ) -> bool:
        """True when price has closed beyond the band for several consecutive bars."""
        bars = self.params.band_walk_bars
        if closes.size < bars:
            return False
        for offset in range(1, bars + 1):
            index = -offset
            band = lower_band[index] if going_long else upper_band[index]
            if np.isnan(band):
                return False
            beyond = closes[index] < band if going_long else closes[index] > band
            if not beyond:
                return False
        return True

    def _maybe_exit(
        self,
        context: StrategyContext,
        *,
        mid: float,
        rsi_value: float,
        diagnostics: dict[str, float],
    ) -> StrategyResult | None:
        position = context.position
        assert position is not None
        price = context.candles[-1].close
        long_position = position.side.sign > 0
        params = self.params

        reverted = (long_position and price >= mid) or (not long_position and price <= mid)
        if reverted:
            return StrategyResult(
                action=SignalAction.CLOSE,
                symbol=context.symbol,
                strategy_name=self.name,
                timestamp=context.now,
                confidence=0.8,
                reason=f"price reverted to the mean at {mid:.4f}",
                metadata=diagnostics,
            )
        rsi_exit = (
            long_position and rsi_value >= params.rsi_exit_long
        ) or (not long_position and rsi_value <= params.rsi_exit_short)
        if rsi_exit:
            return StrategyResult(
                action=SignalAction.CLOSE,
                symbol=context.symbol,
                strategy_name=self.name,
                timestamp=context.now,
                confidence=0.65,
                reason=f"RSI {rsi_value:.1f} has returned to neutral",
                metadata=diagnostics,
            )
        return None

    def _confidence(
        self,
        *,
        rsi_value: float,
        z_value: float,
        adx_value: float,
        band_width_pct: float,
        going_long: bool,
        regime: MarketRegime,
    ) -> float:
        params = self.params
        if going_long:
            rsi_score = np.clip((params.rsi_oversold - rsi_value) / params.rsi_oversold, 0, 1)
        else:
            rsi_score = np.clip(
                (rsi_value - params.rsi_overbought) / (100.0 - params.rsi_overbought), 0, 1
            )
        z_score = np.clip(
            (abs(z_value) - params.zscore_threshold) / params.zscore_threshold, 0.0, 1.0
        )
        # A lower ADX is better for this strategy: more confidence the range holds.
        calmness = np.clip(1.0 - adx_value / params.max_adx, 0.0, 1.0)
        # A band that is too narrow means the "extreme" is not economically meaningful.
        width_score = np.clip(band_width_pct / 0.05, 0.0, 1.0)

        regime_bonus = 0.10 if regime is MarketRegime.RANGING else 0.0
        score = (
            0.30 * rsi_score
            + 0.25 * z_score
            + 0.25 * calmness
            + 0.20 * width_score
            + regime_bonus
        )
        return float(np.clip(score, 0.0, 1.0))
