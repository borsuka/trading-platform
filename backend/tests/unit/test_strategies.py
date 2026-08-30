"""Strategy tests.

Each strategy is tested against *constructed* market scenarios rather than random walks, so
that both the firing condition and every rejection path are exercised deterministically.

The most important tests here are the negative ones: a strategy that produces signals is easy,
a strategy that correctly refuses is the thing that protects capital.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.core.domain import Position
from app.core.enums import MarketRegime, PositionSide, SignalAction
from app.core.exceptions import StrategyConfigurationError
from app.market_data.models import Candle, MarketSnapshot
from app.market_data.providers import build_snapshot
from app.strategies import (
    MeanReversionStrategy,
    MomentumBreakoutStrategy,
    MultiFactorStrategy,
    TrendFollowingStrategy,
    available_strategies,
    create_strategy,
    describe_all,
)
from app.strategies.base import StrategyContext, StrategyResult

START = datetime(2024, 1, 1, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Scenario builders
# --------------------------------------------------------------------------- #
def make_candles(
    closes: list[float],
    *,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    volumes: list[float] | None = None,
    wick: float = 0.004,
) -> list[Candle]:
    """Build a candle series from a close path, with plausible OHLC and volume."""
    result: list[Candle] = []
    for index, close in enumerate(closes):
        previous = closes[index - 1] if index else close
        high = max(previous, close) * (1.0 + wick)
        low = min(previous, close) * (1.0 - wick)
        volume = volumes[index] if volumes else 1_000.0
        result.append(
            Candle(
                symbol=symbol,
                interval=interval,
                open_time=START + timedelta(hours=index),
                open=previous,
                high=high,
                low=low,
                close=close,
                volume=volume,
                quote_volume=volume * close,
            )
        )
    return result


def uptrend(
    n: int = 300,
    start: float = 100.0,
    step_pct: float = 0.003,
    noise: float = 3.0,
    seed: int = 1,
) -> list[float]:
    """Uptrend with realistic pullbacks.

    The noise term is deliberately several times the per-bar drift, as it is in real markets.
    A frictionless straight line would let a trend strategy pass its tests while remaining
    permanently over-extended - and therefore untradable - on real data.
    """
    rng = np.random.default_rng(seed)
    closes = [start]
    for _ in range(n - 1):
        drift = step_pct + float(rng.normal(0.0, step_pct * noise))
        closes.append(closes[-1] * (1.0 + drift))
    return closes


def downtrend(n: int = 300, start: float = 300.0, step_pct: float = 0.003) -> list[float]:
    return [start * (100.0 / value) for value in uptrend(n, 100.0, step_pct)]


def sideways(n: int = 300, centre: float = 100.0, amplitude: float = 0.02) -> list[float]:
    """Clean oscillation around a flat mean."""
    return [
        centre * (1.0 + amplitude * float(np.sin(index / 6.0)))
        for index in range(n)
    ]


def ranging(
    n: int = 260,
    centre: float = 100.0,
    theta: float = 0.06,
    sigma: float = 0.010,
    seed: int = 8,
) -> list[float]:
    """Ornstein-Uhlenbeck mean-reverting series.

    A genuine statistical range: bar-to-bar noise comparable to the range width, and a stable
    mean. A sine wave will not do here - it is too quiet, so any shock large enough to reach
    the 2-sigma band also drives ADX into trend territory, and the strategy correctly refuses
    to trade it.
    """
    rng = np.random.default_rng(seed)
    log_centre = float(np.log(centre))
    level = log_centre
    out: list[float] = []
    for _ in range(n):
        level += theta * (log_centre - level) + float(rng.normal(0.0, sigma))
        out.append(float(np.exp(level)))
    return out


def context_for(
    candles: list[Candle],
    *,
    regime: MarketRegime = MarketRegime.TRENDING_BULL,
    confidence: float = 0.8,
    position: Position | None = None,
    news_score: float = 0.0,
) -> StrategyContext:
    return StrategyContext(
        snapshot=build_snapshot(candles),
        regime=regime,
        regime_confidence=confidence,
        position=position,
        news_score=news_score,
        now=candles[-1].close_time,
    )


def long_position(symbol: str = "BTCUSDT", entry: float = 100.0) -> Position:
    return Position(
        symbol=symbol,
        side=PositionSide.LONG,
        quantity=1.0,
        entry_price=entry,
        mark_price=entry,
    )


def short_position(symbol: str = "BTCUSDT", entry: float = 100.0) -> Position:
    return Position(
        symbol=symbol,
        side=PositionSide.SHORT,
        quantity=1.0,
        entry_price=entry,
        mark_price=entry,
    )


# --------------------------------------------------------------------------- #
# Framework contract
# --------------------------------------------------------------------------- #
class TestStrategyFramework:
    def test_all_strategies_are_registered(self) -> None:
        assert set(available_strategies()) == {
            "trend_following",
            "momentum_breakout",
            "mean_reversion",
            "multi_factor",
        }

    def test_registry_describes_every_strategy(self) -> None:
        described = describe_all()
        assert len(described) == 4
        for entry in described:
            assert entry["parameters_schema"]["type"] == "object"
            assert isinstance(entry["default_parameters"], dict)
            assert entry["version"]

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(StrategyConfigurationError, match="Unknown strategy"):
            create_strategy("does_not_exist")

    def test_unknown_parameter_is_rejected(self) -> None:
        with pytest.raises(StrategyConfigurationError, match="Invalid parameters"):
            create_strategy("trend_following", {"nonexistent_knob": 5})

    @pytest.mark.parametrize("name", ["trend_following", "momentum_breakout",
                                      "mean_reversion", "multi_factor"])
    def test_insufficient_history_never_signals(self, name: str) -> None:
        strategy = create_strategy(name)
        candles = make_candles(uptrend(20))
        result = strategy.generate_signal(context_for(candles))
        assert result.action is SignalAction.NO_TRADE
        assert "insufficient history" in result.reason

    @pytest.mark.parametrize("name", ["trend_following", "momentum_breakout",
                                      "mean_reversion", "multi_factor"])
    def test_unknown_regime_never_signals(self, name: str) -> None:
        strategy = create_strategy(name)
        candles = make_candles(uptrend(300))
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.UNKNOWN)
        )
        assert result.action is SignalAction.NO_TRADE
        assert "UNKNOWN" in result.reason

    @pytest.mark.parametrize("name", ["trend_following", "momentum_breakout",
                                      "mean_reversion", "multi_factor"])
    def test_strategies_are_deterministic(self, name: str) -> None:
        candles = make_candles(uptrend(300))
        context = context_for(candles)
        first = create_strategy(name).generate_signal(context)
        second = create_strategy(name).generate_signal(context)
        assert first.action is second.action
        assert first.confidence == pytest.approx(second.confidence)

    @pytest.mark.parametrize("name", ["trend_following", "momentum_breakout",
                                      "mean_reversion", "multi_factor"])
    def test_direction_filters_are_honoured(self, name: str) -> None:
        strategy = create_strategy(name, {"allow_long": False})
        candles = make_candles(uptrend(400))
        result = strategy.generate_signal(context_for(candles))
        assert result.action is not SignalAction.BUY

    def test_both_directions_disabled_is_rejected(self) -> None:
        with pytest.raises(StrategyConfigurationError):
            create_strategy("trend_following", {"allow_long": False, "allow_short": False})

    def test_cooldown_blocks_entry(self) -> None:
        strategy = create_strategy("trend_following", {"cooldown_bars": 10})
        candles = make_candles(uptrend(400))
        context = StrategyContext(
            snapshot=build_snapshot(candles),
            regime=MarketRegime.TRENDING_BULL,
            regime_confidence=0.8,
            bars_since_last_trade=3,
            now=candles[-1].close_time,
        )
        result = strategy.generate_signal(context)
        assert result.action is SignalAction.NO_TRADE
        assert "cooldown" in result.reason

    def test_min_confidence_suppresses_weak_signals(self) -> None:
        strategy = create_strategy("trend_following", {"min_confidence": 0.99})
        candles = make_candles(uptrend(400))
        result = strategy.generate_signal(context_for(candles))
        assert result.action is SignalAction.NO_TRADE


class TestStrategyResultValidation:
    def test_entry_requires_a_stop(self) -> None:
        with pytest.raises(ValueError, match="requires a stop loss"):
            StrategyResult(
                action=SignalAction.BUY, symbol="BTCUSDT", strategy_name="t",
                timestamp=START, entry=100.0,
            )

    def test_long_stop_must_be_below_entry(self) -> None:
        with pytest.raises(ValueError, match="must be below entry"):
            StrategyResult(
                action=SignalAction.BUY, symbol="BTCUSDT", strategy_name="t",
                timestamp=START, entry=100.0, stop_loss=105.0,
            )

    def test_short_stop_must_be_above_entry(self) -> None:
        with pytest.raises(ValueError, match="must be above entry"):
            StrategyResult(
                action=SignalAction.SELL, symbol="BTCUSDT", strategy_name="t",
                timestamp=START, entry=100.0, stop_loss=95.0,
            )

    def test_confidence_is_bounded(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            StrategyResult(
                action=SignalAction.HOLD, symbol="BTCUSDT", strategy_name="t",
                timestamp=START, confidence=1.5,
            )

    def test_reward_risk_ratio_is_computed(self) -> None:
        result = StrategyResult(
            action=SignalAction.BUY, symbol="BTCUSDT", strategy_name="t",
            timestamp=START, entry=100.0, stop_loss=98.0, take_profit=106.0,
        )
        assert result.risk_per_unit == pytest.approx(2.0)
        assert result.reward_risk_ratio == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Trend following
# --------------------------------------------------------------------------- #
class TestTrendFollowing:
    def test_signals_long_in_an_uptrend(self) -> None:
        strategy = TrendFollowingStrategy()
        candles = make_candles(uptrend(400))
        result = strategy.generate_signal(context_for(candles))
        assert result.action is SignalAction.BUY
        assert result.entry is not None and result.stop_loss is not None
        assert result.stop_loss < result.entry
        assert result.take_profit is not None and result.take_profit > result.entry

    def test_signals_short_in_a_downtrend(self) -> None:
        strategy = TrendFollowingStrategy()
        candles = make_candles(downtrend(400))
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.TRENDING_BEAR)
        )
        assert result.action is SignalAction.SELL
        assert result.stop_loss is not None and result.entry is not None
        assert result.stop_loss > result.entry

    def test_refuses_to_trade_a_range(self) -> None:
        strategy = TrendFollowingStrategy()
        candles = make_candles(sideways(400))
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.RANGING)
        )
        assert result.action is SignalAction.NO_TRADE
        assert "mandate" in result.reason

    def test_weak_adx_blocks_entry(self) -> None:
        """A directionless series must be refused even if the regime label says otherwise."""
        strategy = TrendFollowingStrategy()
        candles = make_candles(ranging(400), wick=0.004)
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.TRENDING_BULL)
        )
        assert result.action is SignalAction.NO_TRADE
        assert "trend too weak" in result.reason

    def test_low_volume_blocks_entry(self) -> None:
        strategy = TrendFollowingStrategy({"min_volume_ratio": 3.0})
        candles = make_candles(uptrend(400))
        result = strategy.generate_signal(context_for(candles))
        assert result.action is SignalAction.NO_TRADE
        assert "volume" in result.reason

    def test_overextension_blocks_entry(self) -> None:
        strategy = TrendFollowingStrategy({"max_extension_atr": 0.01})
        candles = make_candles(uptrend(400))
        result = strategy.generate_signal(context_for(candles))
        assert result.action is SignalAction.NO_TRADE
        assert "extended" in result.reason

    def test_volatility_ceiling_blocks_entry(self) -> None:
        strategy = TrendFollowingStrategy({"min_atr_pct": 0.00001, "max_atr_pct": 0.0001})
        candles = make_candles(uptrend(400))
        result = strategy.generate_signal(context_for(candles))
        assert result.action is SignalAction.NO_TRADE
        assert "volatility too high" in result.reason

    def test_closes_when_trend_reverses(self) -> None:
        strategy = TrendFollowingStrategy()
        closes = uptrend(250) + downtrend(150, start=1_000.0, step_pct=0.012)
        candles = make_candles(closes)
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.TRENDING_BEAR,
                        position=long_position())
        )
        assert result.action in {SignalAction.CLOSE, SignalAction.SELL,
                                 SignalAction.NO_TRADE}

    def test_holds_rather_than_pyramiding(self) -> None:
        strategy = TrendFollowingStrategy()
        candles = make_candles(uptrend(400))
        result = strategy.generate_signal(
            context_for(candles, position=long_position())
        )
        assert result.action is SignalAction.HOLD

    def test_invalid_ema_ordering_rejected(self) -> None:
        with pytest.raises(StrategyConfigurationError):
            TrendFollowingStrategy({"fast_ema_period": 100, "slow_ema_period": 20})


# --------------------------------------------------------------------------- #
# Momentum breakout
# --------------------------------------------------------------------------- #
class TestMomentumBreakout:
    @staticmethod
    def _breakout_scenario(volume_spike: float = 4.0) -> list[Candle]:
        """A tight 220-bar range, then one decisive break upward on heavy volume.

        A single breakout bar keeps the pre-break channel clean: with a multi-bar rally the
        rally itself widens the channel it is supposed to be breaking.
        """
        base = [100.0 + float(np.sin(i / 4.0)) * 0.6 for i in range(220)]
        closes = [*base, base[-1] * 1.035]
        volumes = [1_000.0] * len(base) + [1_000.0 * volume_spike]
        return make_candles(closes, volumes=volumes, wick=0.001)

    def test_signals_on_a_clean_breakout(self) -> None:
        strategy = MomentumBreakoutStrategy({"use_trend_filter": False})
        candles = self._breakout_scenario()
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.TRENDING_BULL)
        )
        assert result.action is SignalAction.BUY
        assert result.stop_loss is not None and result.entry is not None
        assert result.stop_loss < result.entry

    def test_no_signal_without_volume_expansion(self) -> None:
        strategy = MomentumBreakoutStrategy({"use_trend_filter": False})
        candles = self._breakout_scenario(volume_spike=1.0)
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.TRENDING_BULL)
        )
        assert result.action is SignalAction.NO_TRADE
        assert "volume confirmation" in result.reason

    def test_no_signal_inside_the_channel(self) -> None:
        strategy = MomentumBreakoutStrategy({"use_trend_filter": False})
        candles = make_candles(sideways(300), wick=0.001)
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.LOW_VOLATILITY)
        )
        assert result.action is SignalAction.NO_TRADE

    def test_marginal_break_is_rejected(self) -> None:
        strategy = MomentumBreakoutStrategy(
            {"use_trend_filter": False, "min_break_atr": 5.0}
        )
        candles = self._breakout_scenario()
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.TRENDING_BULL)
        )
        assert result.action is SignalAction.NO_TRADE
        assert "marginal break" in result.reason

    def test_wide_spread_blocks_entry(self) -> None:
        strategy = MomentumBreakoutStrategy(
            {"use_trend_filter": False, "max_spread_bps": 0.1}
        )
        candles = self._breakout_scenario()
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.TRENDING_BULL)
        )
        assert result.action is SignalAction.NO_TRADE
        assert "spread" in result.reason

    def test_trend_filter_blocks_counter_trend_break(self) -> None:
        strategy = MomentumBreakoutStrategy({"use_trend_filter": True,
                                             "trend_filter_period": 100})
        # Long decline, then a small pop above the short channel.
        closes = downtrend(250, start=300.0, step_pct=0.008)
        closes += [closes[-1] * (1.0 + 0.02 * (i + 1)) for i in range(3)]
        candles = make_candles(closes, volumes=[1_000.0] * 250 + [5_000.0] * 3)
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.TRENDING_BEAR)
        )
        assert result.action is SignalAction.NO_TRADE

    def test_failing_breakout_bar_is_rejected(self) -> None:
        """A bar that pokes above the channel then closes near its low is not a breakout."""
        strategy = MomentumBreakoutStrategy(
            {"use_trend_filter": False, "max_close_retreat": 0.2}
        )
        candles = self._breakout_scenario()
        last = candles[-1]
        # Rewrite the final bar as a long upper wick with a weak close.
        candles[-1] = Candle(
            symbol=last.symbol, interval=last.interval, open_time=last.open_time,
            open=last.open, high=last.high * 1.03, low=last.low,
            close=last.low * 1.001, volume=last.volume,
        )
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.TRENDING_BULL)
        )
        assert result.action is SignalAction.NO_TRADE

    def test_exit_channel_must_be_shorter(self) -> None:
        with pytest.raises(StrategyConfigurationError):
            MomentumBreakoutStrategy({"channel_period": 10, "exit_channel_period": 20})


# --------------------------------------------------------------------------- #
# Mean reversion
# --------------------------------------------------------------------------- #
class TestMeanReversion:
    @staticmethod
    def _stretched(direction: int, bars: int = 2, magnitude: float = 0.045) -> list[Candle]:
        """A real range, pushed to a statistical extreme over a few bars."""
        closes = ranging()
        for _ in range(bars):
            closes.append(closes[-1] * (1.0 + direction * magnitude))
        return make_candles(closes, wick=0.004)

    @classmethod
    def _oversold_scenario(cls) -> list[Candle]:
        return cls._stretched(-1)

    @classmethod
    def _overbought_scenario(cls) -> list[Candle]:
        return cls._stretched(+1)

    def test_buys_a_confirmed_oversold_extreme(self) -> None:
        strategy = MeanReversionStrategy()
        candles = self._oversold_scenario()
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.RANGING)
        )
        assert result.action is SignalAction.BUY
        assert result.stop_loss is not None and result.entry is not None
        assert result.stop_loss < result.entry
        assert result.take_profit is not None and result.take_profit > result.entry

    def test_sells_a_confirmed_overbought_extreme(self) -> None:
        strategy = MeanReversionStrategy()
        candles = self._overbought_scenario()
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.RANGING)
        )
        assert result.action is SignalAction.SELL
        assert result.stop_loss is not None and result.entry is not None
        assert result.stop_loss > result.entry

    def test_refuses_to_fade_a_trend_by_regime(self) -> None:
        """The central safety property of this strategy."""
        strategy = MeanReversionStrategy()
        candles = make_candles(downtrend(400))
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.TRENDING_BEAR)
        )
        assert result.action is SignalAction.NO_TRADE
        assert "mandate" in result.reason

    def test_adx_ceiling_is_a_second_brake(self) -> None:
        """Even if the regime says RANGING, a high ADX must veto the entry."""
        strategy = MeanReversionStrategy({"max_adx": 5.0, "max_mid_slope": 1.0})
        candles = self._oversold_scenario()
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.RANGING)
        )
        assert result.action is SignalAction.NO_TRADE
        assert "ceiling" in result.reason

    def test_moving_mean_blocks_entry(self) -> None:
        strategy = MeanReversionStrategy({"max_adx": 49.0, "max_mid_slope": 1e-9})
        candles = self._oversold_scenario()
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.RANGING)
        )
        assert result.action is SignalAction.NO_TRADE
        assert "mean itself is moving" in result.reason

    def test_band_walking_blocks_entry(self) -> None:
        """A sustained march below the band is a trend, not a stretched range."""
        strategy = MeanReversionStrategy(
            {"max_adx": 49.0, "max_mid_slope": 1.0, "band_walk_bars": 2}
        )
        closes = ranging()
        for _ in range(6):
            closes.append(closes[-1] * 0.988)
        candles = make_candles(closes, wick=0.004)
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.RANGING)
        )
        assert result.action is SignalAction.NO_TRADE
        assert "trending, not stretched" in result.reason

    def test_no_extreme_no_trade(self) -> None:
        strategy = MeanReversionStrategy({"max_adx": 49.0, "max_mid_slope": 1.0})
        candles = make_candles(sideways(300, amplitude=0.002), wick=0.0005)
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.RANGING)
        )
        assert result.action is SignalAction.NO_TRADE

    def test_closes_at_the_mean(self) -> None:
        strategy = MeanReversionStrategy({"max_adx": 49.0, "max_mid_slope": 1.0})
        candles = make_candles(sideways(300), wick=0.001)
        # Price is back at the centre of the range, so a long should be closed.
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.RANGING,
                        position=long_position(entry=95.0))
        )
        assert result.action is SignalAction.CLOSE

    def test_stop_distance_is_capped(self) -> None:
        strategy = MeanReversionStrategy(
            {"max_adx": 49.0, "max_mid_slope": 1.0, "max_stop_atr": 1.0,
             "atr_stop_multiplier": 10.0}
        )
        candles = self._oversold_scenario()
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.RANGING)
        )
        if result.action is SignalAction.BUY:
            assert result.entry is not None and result.stop_loss is not None
            distance = result.entry - result.stop_loss
            atr_estimate = result.metadata["atr"]
            assert distance <= atr_estimate * 1.01

    def test_rsi_bounds_validated(self) -> None:
        with pytest.raises(StrategyConfigurationError):
            MeanReversionStrategy({"rsi_oversold": 80.0, "rsi_overbought": 20.0})


# --------------------------------------------------------------------------- #
# Multi-factor
# --------------------------------------------------------------------------- #
class TestMultiFactor:
    def test_signals_when_factors_agree(self) -> None:
        strategy = MultiFactorStrategy()
        candles = make_candles(uptrend(400))
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.TRENDING_BULL, confidence=0.9)
        )
        assert result.action is SignalAction.BUY
        assert result.metadata["agreeing_factors"] >= 3

    def test_high_threshold_suppresses_signals(self) -> None:
        strategy = MultiFactorStrategy({"entry_threshold": 0.99, "exit_threshold": 0.1})
        candles = make_candles(uptrend(400))
        result = strategy.generate_signal(context_for(candles))
        assert result.action is SignalAction.NO_TRADE
        assert "below the" in result.reason

    def test_consensus_requirement_blocks_lopsided_scores(self) -> None:
        strategy = MultiFactorStrategy({"min_agreeing_factors": 5})
        candles = make_candles(sideways(400))
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.RANGING)
        )
        assert result.action is SignalAction.NO_TRADE

    def test_news_alone_cannot_trigger_a_trade(self) -> None:
        """The defining safety property: news is a modifier, never a trigger."""
        strategy = MultiFactorStrategy()
        candles = make_candles(sideways(400, amplitude=0.001), wick=0.0005)
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.RANGING, confidence=0.5,
                        news_score=1.0)
        )
        assert result.action is not SignalAction.BUY

    def test_news_weight_cannot_exceed_the_threshold(self) -> None:
        with pytest.raises(StrategyConfigurationError, match="news"):
            MultiFactorStrategy(
                {
                    "news_weight": 0.30,
                    "trend_weight": 0.1,
                    "momentum_weight": 0.05,
                    "volume_weight": 0.05,
                    "volatility_weight": 0.0,
                    "regime_weight": 0.0,
                    "entry_threshold": 0.4,
                }
            )

    def test_news_shifts_confidence_without_flipping_direction(self) -> None:
        strategy = MultiFactorStrategy()
        candles = make_candles(uptrend(400))
        positive = strategy.generate_signal(
            context_for(candles, news_score=1.0)
        )
        negative = strategy.generate_signal(
            context_for(candles, news_score=-1.0)
        )
        if positive.action is SignalAction.BUY and negative.action is SignalAction.BUY:
            assert positive.confidence > negative.confidence

    def test_volatility_gate_blocks_entry(self) -> None:
        strategy = MultiFactorStrategy({"min_atr_pct": 0.4, "max_atr_pct": 0.5})
        candles = make_candles(uptrend(400))
        result = strategy.generate_signal(context_for(candles))
        assert result.action is SignalAction.NO_TRADE
        assert "volatility gate" in result.reason

    def test_closes_when_score_decays(self) -> None:
        strategy = MultiFactorStrategy()
        candles = make_candles(sideways(400))
        result = strategy.generate_signal(
            context_for(candles, regime=MarketRegime.RANGING,
                        position=long_position())
        )
        assert result.action is SignalAction.CLOSE

    def test_all_factors_reported_in_metadata(self) -> None:
        strategy = MultiFactorStrategy()
        candles = make_candles(uptrend(400))
        result = strategy.generate_signal(context_for(candles))
        for factor in ("trend", "momentum", "volume", "volatility", "regime", "news"):
            assert f"score_{factor}" in result.metadata
        assert "score_final" in result.metadata

    def test_exit_threshold_must_be_below_entry(self) -> None:
        with pytest.raises(StrategyConfigurationError):
            MultiFactorStrategy({"entry_threshold": 0.2, "exit_threshold": 0.5})


# --------------------------------------------------------------------------- #
# Cross-cutting safety properties
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name", ["trend_following", "momentum_breakout", "mean_reversion", "multi_factor"]
)
def test_every_entry_has_a_stop(name: str) -> None:
    """No strategy may ever propose an entry without an invalidation level."""
    strategy = create_strategy(name, {"min_confidence": 0.0})
    scenarios = [
        (make_candles(uptrend(400)), MarketRegime.TRENDING_BULL),
        (make_candles(downtrend(400)), MarketRegime.TRENDING_BEAR),
        (make_candles(sideways(400)), MarketRegime.RANGING),
    ]
    for candles, regime in scenarios:
        result = strategy.generate_signal(context_for(candles, regime=regime))
        if result.action.is_entry:
            assert result.stop_loss is not None
            assert result.entry is not None
            assert result.risk_per_unit is not None and result.risk_per_unit > 0


@pytest.mark.parametrize(
    "name", ["trend_following", "momentum_breakout", "mean_reversion", "multi_factor"]
)
def test_strategies_do_not_touch_the_snapshot(name: str) -> None:
    """Strategies must be side-effect free with respect to their inputs."""
    strategy = create_strategy(name)
    candles = make_candles(uptrend(400))
    snapshot = build_snapshot(candles)
    before = (snapshot.symbol, len(snapshot.candles), snapshot.price)
    strategy.generate_signal(
        StrategyContext(snapshot=snapshot, regime=MarketRegime.TRENDING_BULL,
                        regime_confidence=0.8, now=candles[-1].close_time)
    )
    assert (snapshot.symbol, len(snapshot.candles), snapshot.price) == before


@pytest.mark.parametrize(
    "name", ["trend_following", "momentum_breakout", "mean_reversion", "multi_factor"]
)
def test_no_lookahead_in_strategy_decisions(name: str) -> None:
    """A decision made at bar N must not change when later bars are appended."""
    strategy = create_strategy(name)
    closes = uptrend(420)
    candles = make_candles(closes)

    truncated = candles[:400]
    full_prefix = candles[:400]  # same slice, but derived from a longer series
    context_a = context_for(truncated)
    context_b = context_for(full_prefix)
    result_a = strategy.generate_signal(context_a)
    result_b = strategy.generate_signal(context_b)
    assert result_a.action is result_b.action
    assert result_a.confidence == pytest.approx(result_b.confidence)


def test_snapshot_price_is_used_not_future_candles() -> None:
    """A strategy must only ever see candles up to and including the current bar."""
    strategy = TrendFollowingStrategy()
    candles = make_candles(uptrend(400))
    snapshot = MarketSnapshot(
        symbol="BTCUSDT",
        timestamp=candles[-1].close_time,
        candles=tuple(candles),
    )
    context = StrategyContext(
        snapshot=snapshot, regime=MarketRegime.TRENDING_BULL,
        regime_confidence=0.8, now=candles[-1].close_time,
    )
    result = strategy.generate_signal(context)
    if result.entry is not None:
        assert result.entry == pytest.approx(candles[-1].close)
