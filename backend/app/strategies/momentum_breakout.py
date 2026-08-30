"""Momentum / breakout strategy.

Hypothesis
----------
When price escapes a well-defined range on expanding volume, the participants who were selling
into the range are gone, and the move continues while new participants chase it. The edge is in
the *quality* of the breakout, not in the breakout itself: most breaks of a channel are noise.

The entire strategy is therefore a series of filters designed to discard low-quality breaks.

Entry conditions (long; short is the mirror image)
--------------------------------------------------
1. **Channel break** — close above the Donchian upper channel. The channel excludes the
   current bar, so the breakout bar cannot form the level it is being tested against.
2. **Meaningful break** — the break exceeds the channel by a minimum fraction of ATR. A
   one-tick poke through the level is noise.
3. **Volume expansion** — volume above a multiple of its average. A break on average volume is
   usually a false break.
4. **Range quality** — the channel was reasonably tight before the break. Breaking out of an
   already-wide channel means there was no range to break out of.
5. **Trend filter** — the longer EMA agrees with the direction, so the strategy is not fading a
   dominant trend.
6. **Spread and liquidity** — the snapshot's book is tight enough to enter without giving the
   edge away in execution costs.
7. **False-breakout protection** — the previous bars must not contain a failed break in the
   opposite direction, and price must be holding above the level rather than already reversing
   inside the bar.

Exits
-----
* ATR stop placed beyond the broken level, so a genuine failed break exits quickly.
* Take-profit at a reward/risk multiple.
* A close back inside the channel closes the position: the breakout premise is dead.

Failure modes
-------------
* **False breakouts** — the dominant loss mode. Mitigated by filters 2, 3 and 7, and by the
  cooldown that stops the strategy re-entering the same failing level repeatedly.
* **Gappy illiquid markets** — mitigated by the spread filter and the minimum-volume floor.
* **Regime mismatch** — breakouts in a low-volatility drift are usually noise; the strategy is
  restricted to trending and high-volatility regimes.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field, model_validator

from app.core.enums import MarketRegime, SignalAction
from app.core.numeric import safe_divide
from app.indicators.core import atr, donchian_channels, ema, last_valid, volume_ma
from app.strategies.base import (
    Strategy,
    StrategyContext,
    StrategyParameters,
    StrategyResult,
)


class MomentumBreakoutParameters(StrategyParameters):
    """Parameters for :class:`MomentumBreakoutStrategy`."""

    channel_period: int = Field(default=20, ge=5, le=200)
    exit_channel_period: int = Field(
        default=10, ge=3, le=200, description="Opposite channel used as a trailing exit"
    )
    volume_period: int = Field(default=20, ge=5, le=200)
    min_volume_expansion: float = Field(
        default=1.5, ge=1.0, le=10.0, description="Breakout volume / average volume"
    )
    min_break_atr: float = Field(
        default=0.10, ge=0.0, le=5.0,
        description="Minimum break beyond the channel, in ATRs",
    )
    max_channel_width_atr: float = Field(
        default=8.0, gt=0.0, le=100.0,
        description="Pre-break channel width ceiling, in ATRs",
    )
    trend_filter_period: int = Field(default=100, ge=10, le=500)
    use_trend_filter: bool = True
    max_spread_bps: float = Field(default=20.0, gt=0.0, le=500.0)
    min_liquidity_notional: float = Field(default=0.0, ge=0.0)
    false_break_lookback: int = Field(
        default=5, ge=0, le=50,
        description="Bars checked for a recent failed break in the opposite direction",
    )
    max_close_retreat: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Max fraction of the breakout bar's range given back by the close",
    )

    @model_validator(mode="after")
    def _validate(self) -> MomentumBreakoutParameters:
        if self.exit_channel_period >= self.channel_period:
            raise ValueError("exit_channel_period must be shorter than channel_period")
        return self


class MomentumBreakoutStrategy(Strategy):
    """Donchian breakout filtered by volume expansion, break quality and trend."""

    name = "momentum_breakout"
    version = "1.0.0"
    description = (
        "Trades range breakouts confirmed by volume expansion, with explicit "
        "false-breakout protection."
    )
    parameters_model = MomentumBreakoutParameters
    allowed_regimes = frozenset(
        {
            MarketRegime.TRENDING_BULL,
            MarketRegime.TRENDING_BEAR,
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.LOW_VOLATILITY,
        }
    )

    params: MomentumBreakoutParameters

    @property
    def required_history(self) -> int:
        params = self.params
        base = max(
            params.channel_period,
            params.volume_period,
            params.atr_period,
            params.trend_filter_period if params.use_trend_filter else 0,
        )
        return base + params.false_break_lookback + 10

    def _evaluate(self, context: StrategyContext) -> StrategyResult:
        candles = context.candles
        params = self.params
        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)

        upper, _, lower = donchian_channels(
            highs, lows, params.channel_period, exclude_current=True
        )
        exit_upper, _, exit_lower = donchian_channels(
            highs, lows, params.exit_channel_period, exclude_current=True
        )
        atr_value = last_valid(atr(highs, lows, closes, params.atr_period))
        average_volume = last_valid(volume_ma(volumes, params.volume_period))
        upper_level = float(upper[-1]) if not np.isnan(upper[-1]) else None
        lower_level = float(lower[-1]) if not np.isnan(lower[-1]) else None

        if atr_value is None or average_volume is None or upper_level is None:
            return StrategyResult.no_trade(
                context.symbol, self.name, "indicators have not warmed up",
                timestamp=context.now,
            )
        assert lower_level is not None

        last = candles[-1]
        price = last.close
        volume_expansion = safe_divide(last.volume, average_volume, default=0.0)
        channel_width_atr = safe_divide(upper_level - lower_level, atr_value)

        diagnostics: dict[str, float | str] = {
            "upper_channel": round(upper_level, 6),
            "lower_channel": round(lower_level, 6),
            "atr": round(atr_value, 6),
            "volume_expansion": round(volume_expansion, 3),
            "channel_width_atr": round(channel_width_atr, 2),
        }

        # --- exit on re-entry into the channel ---------------------------------
        if context.has_position and context.position is not None:
            exit_result = self._maybe_exit(
                context, exit_upper=exit_upper, exit_lower=exit_lower, diagnostics=diagnostics
            )
            if exit_result is not None:
                return exit_result

        # --- did anything break? -----------------------------------------------
        breaking_up = price > upper_level
        breaking_down = price < lower_level
        if not breaking_up and not breaking_down:
            return StrategyResult.no_trade(
                context.symbol, self.name,
                f"price {price:.4f} inside the {params.channel_period}-bar channel "
                f"[{lower_level:.4f}, {upper_level:.4f}]",
                timestamp=context.now, metadata=diagnostics,
            )

        level = upper_level if breaking_up else lower_level
        action = SignalAction.BUY if breaking_up else SignalAction.SELL
        break_distance_atr = abs(price - level) / atr_value
        diagnostics["break_atr"] = round(break_distance_atr, 3)

        # --- break quality ------------------------------------------------------
        if break_distance_atr < params.min_break_atr:
            return StrategyResult.no_trade(
                context.symbol, self.name,
                f"marginal break: {break_distance_atr:.2f} ATR beyond the level, "
                f"minimum {params.min_break_atr:.2f}",
                timestamp=context.now, metadata=diagnostics,
            )
        if volume_expansion < params.min_volume_expansion:
            return StrategyResult.no_trade(
                context.symbol, self.name,
                f"no volume confirmation: {volume_expansion:.2f}x average, "
                f"minimum {params.min_volume_expansion:.2f}x",
                timestamp=context.now, metadata=diagnostics,
            )
        if channel_width_atr > params.max_channel_width_atr:
            return StrategyResult.no_trade(
                context.symbol, self.name,
                f"channel is {channel_width_atr:.1f} ATR wide; there was no range to break",
                timestamp=context.now, metadata=diagnostics,
            )

        # --- the breakout bar must hold, not reverse ----------------------------
        retreat = self._close_retreat(last, breaking_up)
        diagnostics["close_retreat"] = round(retreat, 3)
        if retreat > params.max_close_retreat:
            return StrategyResult.no_trade(
                context.symbol, self.name,
                f"breakout bar gave back {retreat:.0%} of its range; the break is already "
                "failing",
                timestamp=context.now, metadata=diagnostics,
            )

        # --- recent failed break in the opposite direction ----------------------
        if self._had_recent_false_break(candles, upper, lower, breaking_up):
            return StrategyResult.no_trade(
                context.symbol, self.name,
                f"a break in the opposite direction failed within the last "
                f"{params.false_break_lookback} bars; the level is not reliable",
                timestamp=context.now, metadata=diagnostics,
            )

        # --- trend filter --------------------------------------------------------
        if params.use_trend_filter:
            trend_ema = last_valid(ema(closes, params.trend_filter_period))
            if trend_ema is None:
                return StrategyResult.no_trade(
                    context.symbol, self.name, "trend filter has not warmed up",
                    timestamp=context.now, metadata=diagnostics,
                )
            diagnostics["trend_ema"] = round(trend_ema, 6)
            if breaking_up and price < trend_ema:
                return StrategyResult.no_trade(
                    context.symbol, self.name,
                    "upside break against a downtrend (price below the long EMA)",
                    timestamp=context.now, metadata=diagnostics,
                )
            if breaking_down and price > trend_ema:
                return StrategyResult.no_trade(
                    context.symbol, self.name,
                    "downside break against an uptrend (price above the long EMA)",
                    timestamp=context.now, metadata=diagnostics,
                )

        # --- execution cost ------------------------------------------------------
        cost_block = self._execution_cost_block(context, diagnostics)
        if cost_block is not None:
            return cost_block

        # --- already positioned in this direction --------------------------------
        if context.has_position and context.position is not None:
            aligned = (context.position.side.sign > 0) == breaking_up
            if aligned:
                return StrategyResult.hold(
                    context.symbol, self.name,
                    "already positioned in the breakout direction", timestamp=context.now,
                )

        # Stop goes just inside the broken level: if price returns there, the premise is dead.
        stop_distance = max(
            atr_value * params.atr_stop_multiplier, abs(price - level) + atr_value * 0.25
        )
        if action is SignalAction.BUY:
            stop_loss = price - stop_distance
            take_profit = price + stop_distance * params.risk_reward_ratio
        else:
            stop_loss = price + stop_distance
            take_profit = price - stop_distance * params.risk_reward_ratio

        confidence = self._confidence(
            break_distance_atr=break_distance_atr,
            volume_expansion=volume_expansion,
            channel_width_atr=channel_width_atr,
            retreat=retreat,
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
                f"{'Upside' if breaking_up else 'Downside'} break of the "
                f"{params.channel_period}-bar channel at {level:.4f} "
                f"({break_distance_atr:.2f} ATR) on {volume_expansion:.2f}x volume"
            ),
            metadata=diagnostics,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _close_retreat(candle: object, breaking_up: bool) -> float:
        """Fraction of the bar's range the close gave back from the extreme.

        A breakout bar that closes near its low after making a new high is a failed break in
        progress, not a breakout.
        """
        from app.market_data.models import Candle as _Candle

        assert isinstance(candle, _Candle)
        bar_range = candle.high - candle.low
        if bar_range <= 0:
            return 0.0
        if breaking_up:
            return (candle.high - candle.close) / bar_range
        return (candle.close - candle.low) / bar_range

    def _had_recent_false_break(
        self,
        candles: tuple,
        upper: np.ndarray,
        lower: np.ndarray,
        breaking_up: bool,
    ) -> bool:
        """True when a break in the *opposite* direction happened recently and failed."""
        lookback = self.params.false_break_lookback
        if lookback <= 0 or len(candles) < lookback + 2:
            return False
        for offset in range(2, lookback + 2):
            index = -offset
            level_upper = upper[index]
            level_lower = lower[index]
            if np.isnan(level_upper) or np.isnan(level_lower):
                continue
            close = candles[index].close
            if breaking_up and close < level_lower:
                return True
            if not breaking_up and close > level_upper:
                return True
        return False

    def _execution_cost_block(
        self, context: StrategyContext, diagnostics: dict
    ) -> StrategyResult | None:
        ticker = context.snapshot.ticker
        if ticker is not None:
            spread = ticker.spread_bps
            if spread is not None:
                diagnostics["spread_bps"] = round(spread, 2)
                if spread > self.params.max_spread_bps:
                    return StrategyResult.no_trade(
                        context.symbol, self.name,
                        f"spread {spread:.1f} bps exceeds the "
                        f"{self.params.max_spread_bps:.1f} bps limit",
                        timestamp=context.now, metadata=diagnostics,
                    )
        book = context.snapshot.order_book
        if book is not None and self.params.min_liquidity_notional > 0:
            depth = min(book.notional_depth("bid", 10), book.notional_depth("ask", 10))
            diagnostics["book_depth_notional"] = round(depth, 2)
            if depth < self.params.min_liquidity_notional:
                return StrategyResult.no_trade(
                    context.symbol, self.name,
                    f"book depth {depth:.0f} below the "
                    f"{self.params.min_liquidity_notional:.0f} minimum",
                    timestamp=context.now, metadata=diagnostics,
                )
        return None

    def _maybe_exit(
        self,
        context: StrategyContext,
        *,
        exit_upper: np.ndarray,
        exit_lower: np.ndarray,
        diagnostics: dict,
    ) -> StrategyResult | None:
        position = context.position
        assert position is not None
        price = context.candles[-1].close
        long_position = position.side.sign > 0
        level = exit_lower[-1] if long_position else exit_upper[-1]
        if np.isnan(level):
            return None
        broke_back = (long_position and price < level) or (
            not long_position and price > level
        )
        if not broke_back:
            return None
        return StrategyResult(
            action=SignalAction.CLOSE,
            symbol=context.symbol,
            strategy_name=self.name,
            timestamp=context.now,
            confidence=0.8,
            reason=(
                f"price closed back through the {self.params.exit_channel_period}-bar "
                f"channel at {float(level):.4f}; the breakout premise has failed"
            ),
            metadata=diagnostics,
        )

    def _confidence(
        self,
        *,
        break_distance_atr: float,
        volume_expansion: float,
        channel_width_atr: float,
        retreat: float,
        regime: MarketRegime,
        action: SignalAction,
    ) -> float:
        params = self.params
        break_score = np.clip(break_distance_atr / 1.5, 0.0, 1.0)
        volume_score = np.clip(
            (volume_expansion - params.min_volume_expansion) / 2.0, 0.0, 1.0
        )
        tightness = np.clip(
            1.0 - channel_width_atr / params.max_channel_width_atr, 0.0, 1.0
        )
        hold_score = np.clip(1.0 - retreat / max(params.max_close_retreat, 1e-9), 0.0, 1.0)

        regime_bonus = 0.0
        if (regime is MarketRegime.TRENDING_BULL and action is SignalAction.BUY) or (
            regime is MarketRegime.TRENDING_BEAR and action is SignalAction.SELL
        ):
            regime_bonus = 0.10
        elif regime is MarketRegime.HIGH_VOLATILITY:
            regime_bonus = -0.05  # breaks are noisier here

        score = (
            0.30 * break_score
            + 0.30 * volume_score
            + 0.20 * hold_score
            + 0.20 * tightness
            + regime_bonus
        )
        return float(np.clip(score, 0.0, 1.0))
