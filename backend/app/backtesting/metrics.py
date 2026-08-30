"""Backtest performance metrics.

Every metric here is computed from the equity curve and the trade list — never from assumed
returns — and each is annotated with the assumption that makes it meaningful, because a Sharpe
ratio quoted without its periodisation is a number, not information.

Two deliberate choices worth stating:

* **No metric is annualised silently.** Annualisation requires knowing the bar interval, so it
  is an explicit parameter. A 6.2 Sharpe computed by annualising 5-minute bars is the most
  common way backtest results are inflated.
* **Sample-size warnings are part of the output.** A profit factor computed from nine trades is
  noise. :attr:`PerformanceMetrics.reliability_warnings` says so, in the report, rather than
  leaving the reader to notice.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from app.core.domain import PortfolioSnapshot, Trade
from app.core.numeric import safe_divide

#: Below this many trades, ratio metrics are not statistically meaningful.
MIN_TRADES_FOR_CONFIDENCE = 30

#: Seconds per year, using a 365-day year (crypto trades continuously).
SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


@dataclass(frozen=True, slots=True)
class DrawdownInfo:
    """The worst peak-to-trough decline and where it happened."""

    max_drawdown: float = 0.0
    max_drawdown_value: float = 0.0
    peak_equity: float = 0.0
    trough_equity: float = 0.0
    peak_at: datetime | None = None
    trough_at: datetime | None = None
    recovered_at: datetime | None = None
    duration_seconds: float | None = None
    recovery_seconds: float | None = None
    longest_underwater_seconds: float = 0.0

    @property
    def recovered(self) -> bool:
        return self.recovered_at is not None


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Complete performance summary for a backtest or a live session."""

    # --- returns ---
    initial_equity: float = 0.0
    final_equity: float = 0.0
    total_return: float = 0.0
    cagr: float = 0.0
    # --- risk-adjusted ---
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0
    downside_volatility: float = 0.0
    # --- drawdown ---
    max_drawdown: float = 0.0
    drawdown: DrawdownInfo = field(default_factory=DrawdownInfo)
    # --- trades ---
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    expectancy_r: float | None = None
    average_trade: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    average_holding_seconds: float = 0.0
    # --- costs ---
    total_fees: float = 0.0
    total_slippage: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    turnover: float = 0.0
    fees_as_pct_of_gross_profit: float = 0.0
    # --- meta ---
    start: datetime | None = None
    end: datetime | None = None
    duration_days: float = 0.0
    bars: int = 0
    periods_per_year: float = 0.0
    reliability_warnings: tuple[str, ...] = ()

    @property
    def is_statistically_meaningful(self) -> bool:
        return self.total_trades >= MIN_TRADES_FOR_CONFIDENCE

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_equity": round(self.initial_equity, 2),
            "final_equity": round(self.final_equity, 2),
            "total_return": round(self.total_return, 6),
            "cagr": round(self.cagr, 6),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "volatility": round(self.volatility, 6),
            "max_drawdown": round(self.max_drawdown, 6),
            "max_drawdown_duration_days": (
                round(self.drawdown.duration_seconds / 86400.0, 2)
                if self.drawdown.duration_seconds
                else None
            ),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "expectancy": round(self.expectancy, 4),
            "expectancy_r": (
                round(self.expectancy_r, 4) if self.expectancy_r is not None else None
            ),
            "average_trade": round(self.average_trade, 4),
            "average_win": round(self.average_win, 4),
            "average_loss": round(self.average_loss, 4),
            "largest_win": round(self.largest_win, 4),
            "largest_loss": round(self.largest_loss, 4),
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "average_holding_hours": round(self.average_holding_seconds / 3600.0, 2),
            "total_fees": round(self.total_fees, 4),
            "total_slippage": round(self.total_slippage, 4),
            "gross_profit": round(self.gross_profit, 4),
            "gross_loss": round(self.gross_loss, 4),
            "turnover": round(self.turnover, 2),
            "fees_as_pct_of_gross_profit": round(self.fees_as_pct_of_gross_profit, 4),
            "duration_days": round(self.duration_days, 2),
            "bars": self.bars,
            "statistically_meaningful": self.is_statistically_meaningful,
            "reliability_warnings": list(self.reliability_warnings),
        }

    def summary(self) -> str:
        """One-paragraph human summary, warnings included."""
        lines = [
            f"Return {self.total_return:+.2%} over {self.duration_days:.0f} days "
            f"(CAGR {self.cagr:+.2%})",
            f"Max drawdown {self.max_drawdown:.2%}, Sharpe {self.sharpe_ratio:.2f}, "
            f"Sortino {self.sortino_ratio:.2f}, Calmar {self.calmar_ratio:.2f}",
            f"{self.total_trades} trades, win rate {self.win_rate:.1%}, "
            f"profit factor {self.profit_factor:.2f}, expectancy {self.expectancy:+.2f}",
            f"Fees {self.total_fees:.2f} ({self.fees_as_pct_of_gross_profit:.1%} of gross "
            f"profit), slippage {self.total_slippage:.2f}",
        ]
        if self.reliability_warnings:
            lines.append("Warnings: " + "; ".join(self.reliability_warnings))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Equity-curve metrics
# --------------------------------------------------------------------------- #
def compute_drawdown(equity: Sequence[float], timestamps: Sequence[datetime] | None = None
                     ) -> DrawdownInfo:
    """Maximum peak-to-trough decline, plus its duration and recovery."""
    if len(equity) < 2:
        return DrawdownInfo()

    values = np.asarray(equity, dtype=float)
    running_peak = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(running_peak > 0, (running_peak - values) / running_peak, 0.0)

    trough_index = int(np.argmax(drawdowns))
    max_drawdown = float(drawdowns[trough_index])
    if max_drawdown <= 0:
        return DrawdownInfo(peak_equity=float(values[-1]), trough_equity=float(values[-1]))

    peak_index = int(np.argmax(values[: trough_index + 1]))
    peak_value = float(values[peak_index])
    trough_value = float(values[trough_index])

    recovered_index: int | None = None
    after = np.flatnonzero(values[trough_index:] >= peak_value)
    if after.size:
        recovered_index = trough_index + int(after[0])

    peak_at = trough_at = recovered_at = None
    duration = recovery = None
    longest_underwater = 0.0
    if timestamps is not None and len(timestamps) == len(equity):
        peak_at = timestamps[peak_index]
        trough_at = timestamps[trough_index]
        duration = (trough_at - peak_at).total_seconds()
        if recovered_index is not None:
            recovered_at = timestamps[recovered_index]
            recovery = (recovered_at - trough_at).total_seconds()
        longest_underwater = _longest_underwater(values, timestamps)

    return DrawdownInfo(
        max_drawdown=max_drawdown,
        max_drawdown_value=peak_value - trough_value,
        peak_equity=peak_value,
        trough_equity=trough_value,
        peak_at=peak_at,
        trough_at=trough_at,
        recovered_at=recovered_at,
        duration_seconds=duration,
        recovery_seconds=recovery,
        longest_underwater_seconds=longest_underwater,
    )


def _longest_underwater(values: np.ndarray, timestamps: Sequence[datetime]) -> float:
    """Longest stretch spent below a previous high-water mark, in seconds.

    Often more informative than the drawdown depth: traders abandon a system because of how
    long it stayed underwater, not how deep it went.
    """
    peak = values[0]
    underwater_start: datetime | None = None
    longest = 0.0
    for index, value in enumerate(values):
        if value >= peak:
            peak = value
            if underwater_start is not None:
                longest = max(
                    longest, (timestamps[index] - underwater_start).total_seconds()
                )
                underwater_start = None
        elif underwater_start is None:
            underwater_start = timestamps[index]
    if underwater_start is not None:
        longest = max(longest, (timestamps[-1] - underwater_start).total_seconds())
    return longest


def periodic_returns(equity: Sequence[float]) -> np.ndarray:
    """Simple period-over-period returns from an equity curve."""
    values = np.asarray(equity, dtype=float)
    if values.size < 2:
        return np.array([], dtype=float)
    previous = values[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.where(previous != 0, (values[1:] - previous) / previous, 0.0)
    return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)


def _degenerate_tolerance(mean: float) -> float:
    """Dispersion at or below this is floating-point noise, not real variance."""
    return max(1e-12, abs(mean) * 1e-9)


def sharpe_ratio(
    returns: Sequence[float] | np.ndarray,
    *,
    periods_per_year: float,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualised Sharpe ratio.

    ``periods_per_year`` must match the sampling frequency of ``returns``. There is no default:
    guessing it is how a 5-minute strategy reports an impossible Sharpe.
    """
    values = np.asarray(returns, dtype=float)
    if values.size < 2:
        return 0.0
    excess = values - risk_free_rate / periods_per_year
    mean = float(np.mean(excess))
    std = float(np.std(excess, ddof=1))
    # An exact-zero comparison is not safe here: the standard deviation of a constant series
    # computed in floating point is ~1e-18 rather than 0, which would divide out to a Sharpe
    # of ~1e17. The tolerance scales with the returns so it works at any magnitude.
    if std <= _degenerate_tolerance(mean):
        return 0.0
    return float(mean / std * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: Sequence[float] | np.ndarray,
    *,
    periods_per_year: float,
    risk_free_rate: float = 0.0,
    target: float = 0.0,
) -> float:
    """Annualised Sortino ratio: like Sharpe, but penalising only downside deviation."""
    values = np.asarray(returns, dtype=float)
    if values.size < 2:
        return 0.0
    excess = values - risk_free_rate / periods_per_year
    downside = excess[excess < target]
    if downside.size == 0:
        # No losing periods at all. Reporting infinity would be worse than useless; report a
        # capped value and let the reliability warnings explain the sample size.
        return 0.0 if float(np.mean(excess)) <= 0 else float("inf")
    mean = float(np.mean(excess))
    downside_deviation = float(np.sqrt(np.mean(np.square(downside - target))))
    if downside_deviation <= _degenerate_tolerance(mean):
        return 0.0
    return float(mean / downside_deviation * math.sqrt(periods_per_year))


def calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """CAGR divided by maximum drawdown. Zero drawdown yields 0.0, not infinity."""
    if max_drawdown <= 0:
        return 0.0
    return cagr / max_drawdown


def compound_annual_growth_rate(
    initial: float, final: float, duration_days: float
) -> float:
    """CAGR. Returns 0.0 for degenerate inputs rather than a complex or infinite number."""
    if initial <= 0 or final <= 0 or duration_days <= 0:
        return 0.0
    years = duration_days / 365.0
    if years < 1e-9:
        return 0.0
    return float((final / initial) ** (1.0 / years) - 1.0)


def periods_per_year_for(interval_seconds: float) -> float:
    """Number of bars of ``interval_seconds`` in a 365-day year."""
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    return SECONDS_PER_YEAR / interval_seconds


# --------------------------------------------------------------------------- #
# Trade metrics
# --------------------------------------------------------------------------- #
def _streaks(trades: Sequence[Trade]) -> tuple[int, int]:
    """Longest winning and losing streaks."""
    best_win = best_loss = current_win = current_loss = 0
    for trade in trades:
        if trade.is_win:
            current_win += 1
            current_loss = 0
        else:
            current_loss += 1
            current_win = 0
        best_win = max(best_win, current_win)
        best_loss = max(best_loss, current_loss)
    return best_win, best_loss


def compute_metrics(
    snapshots: Sequence[PortfolioSnapshot],
    trades: Sequence[Trade],
    *,
    initial_equity: float,
    periods_per_year: float,
    risk_free_rate: float = 0.0,
    slippage_cost: float = 0.0,
) -> PerformanceMetrics:
    """Compute the full metric suite."""
    if not snapshots:
        return PerformanceMetrics(initial_equity=initial_equity, final_equity=initial_equity)

    equity = [s.equity for s in snapshots]
    timestamps = [s.timestamp for s in snapshots]
    final_equity = equity[-1]
    start, end = timestamps[0], timestamps[-1]
    duration_days = max((end - start).total_seconds() / 86400.0, 0.0)

    returns = periodic_returns(equity)
    drawdown_info = compute_drawdown(equity, timestamps)
    cagr = compound_annual_growth_rate(initial_equity, final_equity, duration_days)

    closed = [t for t in trades if t.exit_time is not None]
    wins = [t for t in closed if t.is_win]
    losses = [t for t in closed if not t.is_win]
    gross_profit = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))
    total_fees = sum(t.fees for t in closed)
    max_wins, max_losses = _streaks(closed)

    r_multiples = [t.r_multiple for t in closed if t.r_multiple is not None]
    holding = [t.duration_seconds for t in closed if t.duration_seconds is not None]
    turnover = sum(t.entry_price * t.quantity for t in closed)

    volatility = (
        float(np.std(returns, ddof=1)) * math.sqrt(periods_per_year)
        if returns.size > 1
        else 0.0
    )
    downside = returns[returns < 0]
    downside_volatility = (
        float(np.std(downside, ddof=1)) * math.sqrt(periods_per_year)
        if downside.size > 1
        else 0.0
    )

    metrics = PerformanceMetrics(
        initial_equity=initial_equity,
        final_equity=final_equity,
        total_return=safe_divide(final_equity - initial_equity, initial_equity),
        cagr=cagr,
        sharpe_ratio=sharpe_ratio(
            returns, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate
        ),
        sortino_ratio=sortino_ratio(
            returns, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate
        ),
        calmar_ratio=calmar_ratio(cagr, drawdown_info.max_drawdown),
        volatility=volatility,
        downside_volatility=downside_volatility,
        max_drawdown=drawdown_info.max_drawdown,
        drawdown=drawdown_info,
        total_trades=len(closed),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=safe_divide(len(wins), len(closed)),
        profit_factor=safe_divide(gross_profit, gross_loss),
        expectancy=safe_divide(sum(t.net_pnl for t in closed), len(closed)),
        expectancy_r=(
            float(np.mean(r_multiples)) if r_multiples else None
        ),
        average_trade=safe_divide(sum(t.net_pnl for t in closed), len(closed)),
        average_win=safe_divide(gross_profit, len(wins)),
        average_loss=-safe_divide(gross_loss, len(losses)),
        largest_win=max((t.net_pnl for t in wins), default=0.0),
        largest_loss=min((t.net_pnl for t in losses), default=0.0),
        max_consecutive_wins=max_wins,
        max_consecutive_losses=max_losses,
        average_holding_seconds=float(np.mean(holding)) if holding else 0.0,
        total_fees=total_fees,
        total_slippage=slippage_cost or sum(t.slippage_cost for t in closed),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        turnover=turnover,
        fees_as_pct_of_gross_profit=safe_divide(total_fees, gross_profit),
        start=start,
        end=end,
        duration_days=duration_days,
        bars=len(snapshots),
        periods_per_year=periods_per_year,
    )
    return _with_warnings(metrics)


def _with_warnings(metrics: PerformanceMetrics) -> PerformanceMetrics:
    """Attach honesty warnings about sample size and cost structure."""
    from dataclasses import replace

    warnings: list[str] = []
    if metrics.total_trades == 0:
        warnings.append("no trades were executed; performance metrics are not meaningful")
    elif metrics.total_trades < MIN_TRADES_FOR_CONFIDENCE:
        warnings.append(
            f"only {metrics.total_trades} trades: ratio metrics such as win rate and "
            f"profit factor are not statistically reliable below "
            f"{MIN_TRADES_FOR_CONFIDENCE}"
        )
    if metrics.duration_days < 90 and metrics.total_trades > 0:
        warnings.append(
            f"test period is only {metrics.duration_days:.0f} days; results are unlikely "
            "to cover more than one market regime"
        )
    if metrics.fees_as_pct_of_gross_profit > 0.5:
        warnings.append(
            f"fees consume {metrics.fees_as_pct_of_gross_profit:.0%} of gross profit; "
            "the result is highly sensitive to the assumed fee tier"
        )
    if metrics.total_trades > 0 and metrics.win_rate > 0.75:
        warnings.append(
            f"win rate of {metrics.win_rate:.0%} is unusually high; check for lookahead "
            "or an over-optimistic fill model"
        )
    if math.isinf(metrics.sortino_ratio):
        warnings.append("no losing periods occurred, so Sortino is undefined")
    if metrics.max_drawdown < 0.005 and metrics.total_trades > 5:
        warnings.append(
            "maximum drawdown is implausibly small; verify that losses are being recorded"
        )
    return replace(metrics, reliability_warnings=tuple(warnings))


# --------------------------------------------------------------------------- #
# Report tables
# --------------------------------------------------------------------------- #
def monthly_returns(snapshots: Sequence[PortfolioSnapshot]) -> dict[str, float]:
    """Return by calendar month, keyed ``"YYYY-MM"``."""
    if len(snapshots) < 2:
        return {}
    buckets: dict[str, list[PortfolioSnapshot]] = {}
    for snapshot in snapshots:
        key = snapshot.timestamp.strftime("%Y-%m")
        buckets.setdefault(key, []).append(snapshot)

    result: dict[str, float] = {}
    previous_close: float | None = None
    for key in sorted(buckets):
        month = buckets[key]
        opening = previous_close if previous_close is not None else month[0].equity
        closing = month[-1].equity
        result[key] = safe_divide(closing - opening, opening)
        previous_close = closing
    return result


def equity_curve_points(
    snapshots: Sequence[PortfolioSnapshot], *, max_points: int = 2000
) -> list[dict[str, float | str]]:
    """Downsample the equity curve for charting."""
    if not snapshots:
        return []
    step = max(1, len(snapshots) // max_points)
    sampled = list(snapshots[::step])
    if sampled[-1] is not snapshots[-1]:
        sampled.append(snapshots[-1])
    return [
        {
            "timestamp": s.timestamp.isoformat(),
            "equity": round(s.equity, 4),
            "drawdown": round(s.drawdown, 6),
            "cash": round(s.cash, 4),
            "exposure": round(s.total_exposure, 4),
        }
        for s in sampled
    ]


def trade_distribution(trades: Sequence[Trade], bins: int = 20) -> dict[str, list[float]]:
    """Histogram of trade PnL, for the distribution chart."""
    closed = [t.net_pnl for t in trades if t.exit_time is not None]
    if not closed:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(np.asarray(closed, dtype=float), bins=bins)
    return {
        "edges": [round(float(e), 4) for e in edges],
        "counts": [int(c) for c in counts],
    }


def drawdown_curve(snapshots: Sequence[PortfolioSnapshot]) -> list[dict[str, float | str]]:
    """Drawdown over time, for the underwater chart."""
    if not snapshots:
        return []
    peak = snapshots[0].equity
    points: list[dict[str, float | str]] = []
    for snapshot in snapshots:
        peak = max(peak, snapshot.equity)
        points.append(
            {
                "timestamp": snapshot.timestamp.isoformat(),
                "drawdown": round(safe_divide(peak - snapshot.equity, peak), 6),
            }
        )
    return points
