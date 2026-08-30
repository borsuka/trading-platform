"""Technical indicators.

Design rules, all of them load-bearing:

* **Pure functions.** No global state, no caching keyed on symbol, no hidden mutation of inputs.
* **No lookahead.** ``result[i]`` depends only on ``values[:i + 1]``. The warm-up region is
  ``NaN`` rather than back-filled, so a strategy that reads an unwarmed indicator gets ``NaN``
  and fails loudly instead of silently trading on a value derived from too little data.
* **Length preserving.** Every function returns an array the same length as its input, so
  indices always line up with candle indices.

All functions accept any sequence of floats and return ``numpy`` float arrays.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

__all__ = [
    "adx",
    "atr",
    "bollinger_bands",
    "donchian_channels",
    "ema",
    "macd",
    "rolling_max",
    "rolling_min",
    "rolling_volatility",
    "rsi",
    "sma",
    "true_range",
    "volume_ma",
    "wilder_smooth",
    "zscore",
]


def _as_array(values: Sequence[float] | FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1-D sequence, got shape {array.shape}")
    return array


def _validate_period(period: int, name: str = "period") -> None:
    if period < 1:
        raise ValueError(f"{name} must be >= 1, got {period}")


def _empty_like(array: FloatArray) -> FloatArray:
    return np.full(array.shape, np.nan, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Moving averages
# --------------------------------------------------------------------------- #
def sma(values: Sequence[float] | FloatArray, period: int) -> FloatArray:
    """Simple moving average."""
    _validate_period(period)
    array = _as_array(values)
    result = _empty_like(array)
    if array.size < period:
        return result
    cumulative = np.cumsum(np.insert(array, 0, 0.0))
    windows = (cumulative[period:] - cumulative[:-period]) / period
    result[period - 1 :] = windows
    return result


def ema(values: Sequence[float] | FloatArray, period: int) -> FloatArray:
    """Exponential moving average.

    Seeded with the SMA of the first ``period`` values so the series does not depend on how much
    history happens to have been loaded — a common source of backtest/live divergence.
    """
    _validate_period(period)
    array = _as_array(values)
    result = _empty_like(array)
    if array.size < period:
        return result
    alpha = 2.0 / (period + 1.0)
    current = float(np.mean(array[:period]))
    result[period - 1] = current
    for index in range(period, array.size):
        current = alpha * array[index] + (1.0 - alpha) * current
        result[index] = current
    return result


def wilder_smooth(values: Sequence[float] | FloatArray, period: int) -> FloatArray:
    """Wilder's smoothing (RMA), the basis of RSI, ATR and ADX.

    Equivalent to an EMA with ``alpha = 1 / period``.
    """
    _validate_period(period)
    array = _as_array(values)
    result = _empty_like(array)
    if array.size < period:
        return result
    current = float(np.mean(array[:period]))
    result[period - 1] = current
    for index in range(period, array.size):
        current = (current * (period - 1) + array[index]) / period
        result[index] = current
    return result


def volume_ma(volumes: Sequence[float] | FloatArray, period: int = 20) -> FloatArray:
    """Volume moving average. A thin alias for :func:`sma` that documents intent."""
    return sma(volumes, period)


# --------------------------------------------------------------------------- #
# Oscillators
# --------------------------------------------------------------------------- #
def rsi(values: Sequence[float] | FloatArray, period: int = 14) -> FloatArray:
    """Relative Strength Index (Wilder).

    Returns values in ``[0, 100]``. A period of all-gains yields 100 rather than dividing by
    zero.
    """
    _validate_period(period)
    array = _as_array(values)
    result = _empty_like(array)
    if array.size <= period:
        return result

    deltas = np.diff(array)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    result[period] = _rsi_from_averages(avg_gain, avg_loss)

    for index in range(period, deltas.size):
        avg_gain = (avg_gain * (period - 1) + gains[index]) / period
        avg_loss = (avg_loss * (period - 1) + losses[index]) / period
        result[index + 1] = _rsi_from_averages(avg_gain, avg_loss)
    return result


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(
    values: Sequence[float] | FloatArray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """MACD line, signal line and histogram."""
    _validate_period(fast_period, "fast_period")
    _validate_period(slow_period, "slow_period")
    _validate_period(signal_period, "signal_period")
    if fast_period >= slow_period:
        raise ValueError(
            f"fast_period ({fast_period}) must be shorter than slow_period ({slow_period})"
        )
    array = _as_array(values)
    macd_line = ema(array, fast_period) - ema(array, slow_period)

    signal_line = _empty_like(array)
    valid = ~np.isnan(macd_line)
    if valid.any():
        first_valid = int(np.argmax(valid))
        tail = macd_line[first_valid:]
        if tail.size >= signal_period:
            signal_line[first_valid:] = ema(tail, signal_period)
    return macd_line, signal_line, macd_line - signal_line


def zscore(values: Sequence[float] | FloatArray, period: int = 20) -> FloatArray:
    """Rolling z-score: how many standard deviations from the rolling mean.

    Constant windows (zero standard deviation) yield 0.0, not ``inf``.
    """
    _validate_period(period)
    array = _as_array(values)
    result = _empty_like(array)
    if array.size < period:
        return result
    means = sma(array, period)
    for index in range(period - 1, array.size):
        window = array[index - period + 1 : index + 1]
        std = float(np.std(window))
        result[index] = 0.0 if std == 0.0 else (array[index] - means[index]) / std
    return result


# --------------------------------------------------------------------------- #
# Volatility and range
# --------------------------------------------------------------------------- #
def true_range(
    highs: Sequence[float] | FloatArray,
    lows: Sequence[float] | FloatArray,
    closes: Sequence[float] | FloatArray,
) -> FloatArray:
    """True range. The first element is simply ``high - low``: there is no prior close."""
    high = _as_array(highs)
    low = _as_array(lows)
    close = _as_array(closes)
    if not (high.size == low.size == close.size):
        raise ValueError("highs, lows and closes must be the same length")
    if high.size == 0:
        return np.array([], dtype=np.float64)
    result = np.empty(high.size, dtype=np.float64)
    result[0] = high[0] - low[0]
    if high.size > 1:
        prev_close = close[:-1]
        result[1:] = np.maximum.reduce(
            [
                high[1:] - low[1:],
                np.abs(high[1:] - prev_close),
                np.abs(low[1:] - prev_close),
            ]
        )
    return result


def atr(
    highs: Sequence[float] | FloatArray,
    lows: Sequence[float] | FloatArray,
    closes: Sequence[float] | FloatArray,
    period: int = 14,
) -> FloatArray:
    """Average True Range (Wilder). The platform's canonical volatility unit for stops."""
    return wilder_smooth(true_range(highs, lows, closes), period)


def rolling_volatility(
    values: Sequence[float] | FloatArray,
    period: int = 20,
    *,
    annualization_factor: float | None = None,
) -> FloatArray:
    """Standard deviation of log returns over a rolling window.

    Pass ``annualization_factor`` (e.g. ``sqrt(365 * 24)`` for hourly crypto bars) to express
    the result annualised.
    """
    _validate_period(period)
    array = _as_array(values)
    result = _empty_like(array)
    if array.size < period + 1:
        return result
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(np.log(array))
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    for index in range(period - 1, returns.size):
        window = returns[index - period + 1 : index + 1]
        result[index + 1] = float(np.std(window, ddof=1)) if period > 1 else 0.0
    if annualization_factor is not None:
        result *= annualization_factor
    return result


def bollinger_bands(
    values: Sequence[float] | FloatArray,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Bollinger Bands. Returns ``(upper, middle, lower)``."""
    _validate_period(period)
    if num_std <= 0:
        raise ValueError("num_std must be positive")
    array = _as_array(values)
    middle = sma(array, period)
    deviation = _empty_like(array)
    for index in range(period - 1, array.size):
        deviation[index] = float(np.std(array[index - period + 1 : index + 1]))
    return middle + num_std * deviation, middle, middle - num_std * deviation


def bollinger_percent_b(
    values: Sequence[float] | FloatArray,
    period: int = 20,
    num_std: float = 2.0,
) -> FloatArray:
    """Position within the bands: 0.0 at the lower band, 1.0 at the upper."""
    array = _as_array(values)
    upper, _, lower = bollinger_bands(array, period, num_std)
    width = upper - lower
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(width > 0, (array - lower) / width, 0.5)
    return np.where(np.isnan(upper), np.nan, result).astype(np.float64)


# --------------------------------------------------------------------------- #
# Trend strength
# --------------------------------------------------------------------------- #
def adx(
    highs: Sequence[float] | FloatArray,
    lows: Sequence[float] | FloatArray,
    closes: Sequence[float] | FloatArray,
    period: int = 14,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Average Directional Index.

    Returns ``(adx, plus_di, minus_di)``. ADX measures trend *strength* without direction;
    the DI pair supplies direction.
    """
    _validate_period(period)
    high = _as_array(highs)
    low = _as_array(lows)
    close = _as_array(closes)
    if not (high.size == low.size == close.size):
        raise ValueError("highs, lows and closes must be the same length")

    size = high.size
    adx_out = np.full(size, np.nan, dtype=np.float64)
    plus_di = np.full(size, np.nan, dtype=np.float64)
    minus_di = np.full(size, np.nan, dtype=np.float64)
    if size < period * 2:
        return adx_out, plus_di, minus_di

    up_move = np.zeros(size, dtype=np.float64)
    down_move = np.zeros(size, dtype=np.float64)
    up = high[1:] - high[:-1]
    down = low[:-1] - low[1:]
    up_move[1:] = np.where((up > down) & (up > 0), up, 0.0)
    down_move[1:] = np.where((down > up) & (down > 0), down, 0.0)

    tr = true_range(high, low, close)
    # Smoothing starts at index 1 because index 0 has no directional movement.
    atr_smoothed = wilder_smooth(tr[1:], period)
    plus_smoothed = wilder_smooth(up_move[1:], period)
    minus_smoothed = wilder_smooth(down_move[1:], period)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di[1:] = np.where(atr_smoothed > 0, 100.0 * plus_smoothed / atr_smoothed, 0.0)
        minus_di[1:] = np.where(atr_smoothed > 0, 100.0 * minus_smoothed / atr_smoothed, 0.0)

    di_sum = plus_di + minus_di
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = np.where(di_sum > 0, 100.0 * np.abs(plus_di - minus_di) / di_sum, 0.0)

    valid = ~np.isnan(plus_di) & ~np.isnan(minus_di)
    if valid.any():
        first_valid = int(np.argmax(valid))
        dx_tail = dx[first_valid:]
        if dx_tail.size >= period:
            adx_out[first_valid:] = wilder_smooth(dx_tail, period)
    return adx_out, plus_di, minus_di


# --------------------------------------------------------------------------- #
# Channels
# --------------------------------------------------------------------------- #
def rolling_max(values: Sequence[float] | FloatArray, period: int) -> FloatArray:
    """Rolling maximum over the trailing ``period`` values, inclusive of the current one."""
    _validate_period(period)
    array = _as_array(values)
    result = _empty_like(array)
    for index in range(period - 1, array.size):
        result[index] = float(np.max(array[index - period + 1 : index + 1]))
    return result


def rolling_min(values: Sequence[float] | FloatArray, period: int) -> FloatArray:
    """Rolling minimum over the trailing ``period`` values, inclusive of the current one."""
    _validate_period(period)
    array = _as_array(values)
    result = _empty_like(array)
    for index in range(period - 1, array.size):
        result[index] = float(np.min(array[index - period + 1 : index + 1]))
    return result


def donchian_channels(
    highs: Sequence[float] | FloatArray,
    lows: Sequence[float] | FloatArray,
    period: int = 20,
    *,
    exclude_current: bool = True,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Donchian channels. Returns ``(upper, middle, lower)``.

    ``exclude_current=True`` shifts the channel by one bar so the current bar's own high cannot
    form the level it is being tested against. Without this, "price broke above the channel" is
    tautological — the breakout bar always sets its own channel high — and the resulting
    backtest looks far better than the strategy is.
    """
    _validate_period(period)
    high = _as_array(highs)
    low = _as_array(lows)
    if high.size != low.size:
        raise ValueError("highs and lows must be the same length")

    upper = rolling_max(high, period)
    lower = rolling_min(low, period)
    if exclude_current:
        upper = _shift(upper, 1)
        lower = _shift(lower, 1)
    return upper, (upper + lower) / 2.0, lower


def _shift(array: FloatArray, periods: int) -> FloatArray:
    """Shift forward in time, padding the head with NaN."""
    result = np.full(array.shape, np.nan, dtype=np.float64)
    if periods <= 0 or periods >= array.size:
        return result if periods >= array.size else array.copy()
    result[periods:] = array[:-periods]
    return result


def last_valid(array: FloatArray) -> float | None:
    """Most recent non-NaN value, or ``None`` if the series never warmed up."""
    valid = ~np.isnan(array)
    if not valid.any():
        return None
    return float(array[np.max(np.flatnonzero(valid))])


def crossed_above(fast: FloatArray, slow: FloatArray, index: int = -1) -> bool:
    """True when ``fast`` crossed from at-or-below to above ``slow`` on bar ``index``."""
    return _crossed(fast, slow, index, direction=1)


def crossed_below(fast: FloatArray, slow: FloatArray, index: int = -1) -> bool:
    """True when ``fast`` crossed from at-or-above to below ``slow`` on bar ``index``."""
    return _crossed(fast, slow, index, direction=-1)


def _crossed(fast: FloatArray, slow: FloatArray, index: int, direction: int) -> bool:
    if fast.size != slow.size or fast.size < 2:
        return False
    position = index if index >= 0 else fast.size + index
    if position < 1 or position >= fast.size:
        return False
    previous = fast[position - 1] - slow[position - 1]
    current = fast[position] - slow[position]
    if np.isnan(previous) or np.isnan(current):
        return False
    if direction > 0:
        return bool(previous <= 0 < current)
    return bool(previous >= 0 > current)
