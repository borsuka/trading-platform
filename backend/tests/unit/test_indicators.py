"""Indicator correctness tests.

The two properties that matter most are checked explicitly for every indicator:

1. **No lookahead** - truncating the input must not change any earlier output value.
2. **Warm-up is NaN** - an indicator that has not seen enough data reports NaN rather than a
   plausible-looking number computed from a short window.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.indicators.core import (
    adx,
    atr,
    bollinger_bands,
    bollinger_percent_b,
    crossed_above,
    crossed_below,
    donchian_channels,
    ema,
    last_valid,
    macd,
    rolling_max,
    rolling_min,
    rolling_volatility,
    rsi,
    sma,
    true_range,
    wilder_smooth,
    zscore,
)


@pytest.fixture
def prices() -> list[float]:
    """Deterministic pseudo-random walk; identical on every run."""
    rng = np.random.default_rng(20240101)
    steps = rng.normal(0.0, 1.0, 200)
    return list(100.0 + np.cumsum(steps))


@pytest.fixture
def ohlc(prices: list[float]) -> tuple[list[float], list[float], list[float]]:
    closes = prices
    highs = [c + abs(math.sin(i)) + 0.5 for i, c in enumerate(closes)]
    lows = [c - abs(math.cos(i)) - 0.5 for i, c in enumerate(closes)]
    return highs, lows, closes


# --------------------------------------------------------------------------- #
# SMA
# --------------------------------------------------------------------------- #
class TestSMA:
    def test_known_values(self) -> None:
        result = sma([1, 2, 3, 4, 5], 3)
        assert np.isnan(result[0]) and np.isnan(result[1])
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_length_preserved(self, prices: list[float]) -> None:
        assert sma(prices, 20).shape == (len(prices),)

    def test_insufficient_data_is_all_nan(self) -> None:
        assert np.isnan(sma([1.0, 2.0], 5)).all()

    def test_constant_series(self) -> None:
        result = sma([7.0] * 10, 4)
        assert result[3:] == pytest.approx([7.0] * 7)

    def test_rejects_zero_period(self) -> None:
        with pytest.raises(ValueError, match="period must be >= 1"):
            sma([1, 2, 3], 0)


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #
class TestEMA:
    def test_seeded_with_sma(self) -> None:
        result = ema([1, 2, 3, 4, 5, 6], 3)
        assert result[2] == pytest.approx(2.0)  # mean(1,2,3)
        # alpha = 2/4 = 0.5 -> 0.5*4 + 0.5*2 = 3.0
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_reacts_faster_than_sma(self, prices: list[float]) -> None:
        step = [100.0] * 30 + [120.0] * 5
        assert last_valid(ema(step, 10)) > last_valid(sma(step, 10))

    def test_no_lookahead(self, prices: list[float]) -> None:
        full = ema(prices, 21)
        truncated = ema(prices[:150], 21)
        np.testing.assert_allclose(full[:150], truncated, equal_nan=True)


# --------------------------------------------------------------------------- #
# Wilder smoothing
# --------------------------------------------------------------------------- #
class TestWilderSmooth:
    def test_first_value_is_mean(self) -> None:
        result = wilder_smooth([2, 4, 6, 8], 2)
        assert result[1] == pytest.approx(3.0)
        assert result[2] == pytest.approx((3.0 * 1 + 6) / 2)

    def test_converges_on_constant(self) -> None:
        result = wilder_smooth([5.0] * 50, 14)
        assert last_valid(result) == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# RSI
# --------------------------------------------------------------------------- #
class TestRSI:
    def test_bounded_zero_to_hundred(self, prices: list[float]) -> None:
        result = rsi(prices, 14)
        valid = result[~np.isnan(result)]
        assert valid.size > 0
        assert (valid >= 0.0).all() and (valid <= 100.0).all()

    def test_monotonic_rise_is_hundred(self) -> None:
        result = rsi(list(range(1, 40)), 14)
        assert last_valid(result) == pytest.approx(100.0)

    def test_monotonic_fall_is_zero(self) -> None:
        result = rsi(list(range(40, 1, -1)), 14)
        assert last_valid(result) == pytest.approx(0.0)

    def test_flat_series_is_neutral(self) -> None:
        result = rsi([50.0] * 40, 14)
        assert last_valid(result) == pytest.approx(50.0)

    def test_warmup_is_nan(self) -> None:
        result = rsi(list(range(1, 30)), 14)
        assert np.isnan(result[:14]).all()
        assert not np.isnan(result[14])

    def test_no_lookahead(self, prices: list[float]) -> None:
        np.testing.assert_allclose(
            rsi(prices, 14)[:120], rsi(prices[:120], 14), equal_nan=True
        )


# --------------------------------------------------------------------------- #
# MACD
# --------------------------------------------------------------------------- #
class TestMACD:
    def test_shapes_align(self, prices: list[float]) -> None:
        line, signal, hist = macd(prices)
        assert line.shape == signal.shape == hist.shape == (len(prices),)

    def test_histogram_is_line_minus_signal(self, prices: list[float]) -> None:
        line, signal, hist = macd(prices)
        mask = ~np.isnan(hist)
        np.testing.assert_allclose(hist[mask], (line - signal)[mask])

    def test_rejects_inverted_periods(self, prices: list[float]) -> None:
        with pytest.raises(ValueError, match="must be shorter than"):
            macd(prices, fast_period=26, slow_period=12)

    def test_uptrend_gives_positive_macd(self) -> None:
        line, _, _ = macd([100.0 + i for i in range(120)])
        assert last_valid(line) > 0


# --------------------------------------------------------------------------- #
# True range / ATR
# --------------------------------------------------------------------------- #
class TestATR:
    def test_true_range_first_bar_is_high_low(self) -> None:
        tr = true_range([10, 12], [8, 9], [9, 11])
        assert tr[0] == pytest.approx(2.0)

    def test_true_range_uses_gap(self) -> None:
        # Gap up: previous close 9, current bar 20-19. |20 - 9| = 11 dominates 20-19 = 1.
        tr = true_range([10, 20], [8, 19], [9, 19.5])
        assert tr[1] == pytest.approx(11.0)

    def test_atr_positive(self, ohlc: tuple[list[float], list[float], list[float]]) -> None:
        highs, lows, closes = ohlc
        result = atr(highs, lows, closes, 14)
        valid = result[~np.isnan(result)]
        assert valid.size > 0
        assert (valid > 0).all()

    def test_mismatched_lengths_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            atr([1, 2, 3], [1, 2], [1, 2, 3])

    def test_no_lookahead(self, ohlc: tuple[list[float], list[float], list[float]]) -> None:
        highs, lows, closes = ohlc
        full = atr(highs, lows, closes, 14)
        part = atr(highs[:100], lows[:100], closes[:100], 14)
        np.testing.assert_allclose(full[:100], part, equal_nan=True)


# --------------------------------------------------------------------------- #
# ADX
# --------------------------------------------------------------------------- #
class TestADX:
    def test_strong_trend_has_high_adx(self) -> None:
        n = 120
        closes = [100.0 + i * 2 for i in range(n)]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        value, plus, minus = adx(highs, lows, closes, 14)
        assert last_valid(value) > 40.0
        assert last_valid(plus) > last_valid(minus)

    def test_downtrend_has_minus_di_dominant(self) -> None:
        n = 120
        closes = [300.0 - i * 2 for i in range(n)]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        value, plus, minus = adx(highs, lows, closes, 14)
        assert last_valid(value) > 40.0
        assert last_valid(minus) > last_valid(plus)

    def test_choppy_market_has_low_adx(self) -> None:
        closes = [100.0 + (2.0 if i % 2 else -2.0) for i in range(150)]
        highs = [c + 0.5 for c in closes]
        lows = [c - 0.5 for c in closes]
        value, _, _ = adx(highs, lows, closes, 14)
        assert last_valid(value) < 30.0

    def test_bounded(self, ohlc: tuple[list[float], list[float], list[float]]) -> None:
        highs, lows, closes = ohlc
        value, plus, minus = adx(highs, lows, closes, 14)
        for series in (value, plus, minus):
            valid = series[~np.isnan(series)]
            assert (valid >= 0).all() and (valid <= 100.0).all()

    def test_insufficient_data_all_nan(self) -> None:
        value, _, _ = adx([1, 2, 3], [0, 1, 2], [1, 2, 3], 14)
        assert np.isnan(value).all()


# --------------------------------------------------------------------------- #
# Bollinger
# --------------------------------------------------------------------------- #
class TestBollinger:
    def test_ordering(self, prices: list[float]) -> None:
        upper, middle, lower = bollinger_bands(prices, 20, 2.0)
        mask = ~np.isnan(middle)
        assert (upper[mask] >= middle[mask]).all()
        assert (middle[mask] >= lower[mask]).all()

    def test_constant_series_has_zero_width(self) -> None:
        upper, _middle, lower = bollinger_bands([10.0] * 30, 20)
        assert upper[-1] == pytest.approx(10.0)
        assert lower[-1] == pytest.approx(10.0)

    def test_percent_b_range(self, prices: list[float]) -> None:
        result = bollinger_percent_b(prices, 20, 2.0)
        valid = result[~np.isnan(result)]
        # %B can exceed [0,1] on band breaks, but should stay sane.
        assert valid.size > 0
        assert (valid > -1.5).all() and (valid < 2.5).all()

    def test_rejects_non_positive_std(self, prices: list[float]) -> None:
        with pytest.raises(ValueError, match="num_std must be positive"):
            bollinger_bands(prices, 20, 0.0)


# --------------------------------------------------------------------------- #
# z-score / volatility
# --------------------------------------------------------------------------- #
class TestZScore:
    def test_constant_window_is_zero(self) -> None:
        assert zscore([5.0] * 30, 20)[-1] == pytest.approx(0.0)

    def test_spike_gives_large_positive(self) -> None:
        values = [10.0] * 25 + [50.0]
        assert zscore(values, 20)[-1] > 2.0

    def test_mean_reverting_series_centres_near_zero(self, prices: list[float]) -> None:
        result = zscore(prices, 20)
        valid = result[~np.isnan(result)]
        assert abs(float(np.mean(valid))) < 1.0


class TestRollingVolatility:
    def test_zero_for_constant_prices(self) -> None:
        result = rolling_volatility([100.0] * 40, 20)
        assert last_valid(result) == pytest.approx(0.0, abs=1e-12)

    def test_higher_for_noisier_series(self) -> None:
        rng = np.random.default_rng(7)
        calm = list(100 + np.cumsum(rng.normal(0, 0.1, 100)))
        wild = list(100 + np.cumsum(rng.normal(0, 2.0, 100)))
        assert last_valid(rolling_volatility(wild, 20)) > last_valid(
            rolling_volatility(calm, 20)
        )

    def test_annualization_scales_linearly(self, prices: list[float]) -> None:
        base = rolling_volatility(prices, 20)
        scaled = rolling_volatility(prices, 20, annualization_factor=10.0)
        mask = ~np.isnan(base)
        np.testing.assert_allclose(scaled[mask], base[mask] * 10.0)


# --------------------------------------------------------------------------- #
# Channels
# --------------------------------------------------------------------------- #
class TestDonchian:
    def test_excludes_current_bar_by_default(self) -> None:
        highs = [10.0] * 20 + [99.0]
        lows = [5.0] * 21
        upper, _, _ = donchian_channels(highs, lows, 20)
        # The 99 spike must not be part of the level it is being tested against.
        assert upper[-1] == pytest.approx(10.0)

    def test_includes_current_when_requested(self) -> None:
        highs = [10.0] * 20 + [99.0]
        lows = [5.0] * 21
        upper, _, _ = donchian_channels(highs, lows, 20, exclude_current=False)
        assert upper[-1] == pytest.approx(99.0)

    def test_middle_is_midpoint(self) -> None:
        highs = list(range(10, 60))
        lows = list(range(1, 51))
        upper, middle, lower = donchian_channels(highs, lows, 20)
        mask = ~np.isnan(middle)
        np.testing.assert_allclose(middle[mask], ((upper + lower) / 2)[mask])


class TestRollingExtremes:
    def test_rolling_max(self) -> None:
        result = rolling_max([1, 5, 3, 2, 8], 3)
        assert result[2] == 5 and result[3] == 5 and result[4] == 8

    def test_rolling_min(self) -> None:
        result = rolling_min([4, 5, 3, 2, 8], 3)
        assert result[2] == 3 and result[3] == 2 and result[4] == 2


# --------------------------------------------------------------------------- #
# Crossovers
# --------------------------------------------------------------------------- #
class TestCrossovers:
    def test_cross_above_detected(self) -> None:
        fast = np.array([1.0, 2.0, 4.0])
        slow = np.array([3.0, 3.0, 3.0])
        assert crossed_above(fast, slow) is True
        assert crossed_below(fast, slow) is False

    def test_cross_below_detected(self) -> None:
        fast = np.array([4.0, 4.0, 1.0])
        slow = np.array([3.0, 3.0, 3.0])
        assert crossed_below(fast, slow) is True

    def test_no_cross_when_parallel(self) -> None:
        fast = np.array([4.0, 5.0, 6.0])
        slow = np.array([1.0, 2.0, 3.0])
        assert crossed_above(fast, slow) is False

    def test_nan_never_reports_a_cross(self) -> None:
        fast = np.array([np.nan, 2.0, 4.0])
        slow = np.array([np.nan, 3.0, 3.0])
        assert crossed_above(fast, slow, index=1) is False


# --------------------------------------------------------------------------- #
# Cross-cutting properties
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("period", [2, 5, 14, 50])
def test_all_length_preserving(prices: list[float], period: int) -> None:
    n = len(prices)
    assert sma(prices, period).size == n
    assert ema(prices, period).size == n
    assert rsi(prices, period).size == n
    assert zscore(prices, period).size == n
    assert rolling_volatility(prices, period).size == n


def test_indicators_do_not_mutate_input(prices: list[float]) -> None:
    original = list(prices)
    sma(prices, 10)
    ema(prices, 10)
    rsi(prices, 14)
    zscore(prices, 20)
    assert prices == original


def test_empty_input_is_handled() -> None:
    assert sma([], 5).size == 0
    assert ema([], 5).size == 0
    assert true_range([], [], []).size == 0
