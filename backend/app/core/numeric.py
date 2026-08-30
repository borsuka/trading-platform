"""Numeric helpers for money and instrument quantities.

Order sizes and prices must respect venue tick/lot constraints exactly. Floating point is used
throughout the analytics layer for speed, but anything that becomes an order field is rounded
through :class:`decimal.Decimal` so that ``0.1 + 0.2`` never becomes an order the exchange
rejects.
"""

from __future__ import annotations

import math
from decimal import ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, Decimal, InvalidOperation

#: Values below this are treated as zero when comparing quantities/notionals.
EPSILON = 1e-12


def to_decimal(value: float | int | str | Decimal) -> Decimal:
    """Convert to :class:`Decimal` without inheriting binary float noise."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:  # pragma: no cover - defensive
        raise ValueError(f"Cannot convert {value!r} to Decimal") from exc


def round_to_step(
    value: float | Decimal,
    step: float | Decimal,
    *,
    mode: str = "down",
) -> float:
    """Round ``value`` to a multiple of ``step``.

    ``mode`` is ``"down"`` for quantities (never exceed what risk allowed), ``"up"`` for
    minimums, and ``"nearest"`` for prices.
    """
    step_dec = to_decimal(step)
    if step_dec <= 0:
        return float(value)
    rounding = {"down": ROUND_DOWN, "up": ROUND_UP, "nearest": ROUND_HALF_UP}.get(mode)
    if rounding is None:
        raise ValueError(f"Unknown rounding mode {mode!r}")
    quotient = (to_decimal(value) / step_dec).quantize(Decimal(1), rounding=rounding)
    return float(quotient * step_dec)


def decimals_for_step(step: float | Decimal) -> int:
    """Number of decimal places implied by a tick/lot size."""
    exponent = to_decimal(step).normalize().as_tuple().exponent
    if not isinstance(exponent, int):  # 'n', 'N', 'F' for special values
        return 0
    return max(0, -exponent)


def format_quantity(value: float, step: float) -> str:
    """Render a quantity with exactly the precision the venue expects."""
    return f"{value:.{decimals_for_step(step)}f}"


def is_zero(value: float, tolerance: float = EPSILON) -> bool:
    return abs(value) <= tolerance


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns ``default`` instead of raising or producing inf/nan."""
    if is_zero(denominator):
        return default
    result = numerator / denominator
    if not math.isfinite(result):
        return default
    return result


def clamp(value: float, low: float, high: float) -> float:
    """Constrain ``value`` to ``[low, high]``."""
    if low > high:
        raise ValueError(f"clamp bounds inverted: low={low} high={high}")
    return max(low, min(high, value))


def bps(value: float) -> float:
    """Convert basis points to a fraction (25 bps -> 0.0025)."""
    return value / 10_000.0


def to_bps(fraction: float) -> float:
    """Convert a fraction to basis points (0.0025 -> 25)."""
    return fraction * 10_000.0


def pct_change(new: float, old: float) -> float:
    """Fractional change from ``old`` to ``new``; 0.0 when ``old`` is zero."""
    return safe_divide(new - old, abs(old))
