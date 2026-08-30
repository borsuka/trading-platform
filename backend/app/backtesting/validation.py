"""Robustness analysis: the tools that decide whether a backtest result is real.

A single backtest number is nearly worthless. Given enough parameter combinations, some setting
will look excellent on any history — that is a property of searching, not of the strategy. The
functions here exist to distinguish a genuine edge from a well-fitted curve.

Four independent checks, each attacking a different failure mode:

============================  =====================================================
Check                         Question it answers
============================  =====================================================
Train/test split              Does it work on data that was not used to choose it?
Walk-forward                  Does it keep working as the market changes?
Parameter sensitivity         Is it a plateau, or a spike that vanishes if a knob moves?
Monte Carlo                   Was the equity curve luck in trade *ordering*?
============================  =====================================================

A strategy that passes only the in-sample test has demonstrated nothing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from typing import Any

import numpy as np

from app.backtesting.engine import BacktestConfig, BacktestEngine, BacktestResult
from app.backtesting.metrics import compute_drawdown
from app.core.exceptions import BacktestError, InsufficientDataError
from app.core.logging import get_logger
from app.core.numeric import safe_divide
from app.market_data.models import Candle
from app.risk.limits import RiskLimits
from app.strategies.registry import create_strategy

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """One in-sample/out-of-sample pair."""

    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_return: float
    test_return: float
    train_sharpe: float
    test_sharpe: float
    test_trades: int
    test_max_drawdown: float

    #: In-sample returns below this make the ratio meaningless (a 0.01% in-sample return
    #: with a 2% out-of-sample return is not "20,000% retention").
    MIN_TRAIN_RETURN_FOR_RATIO = 0.01

    @property
    def degradation(self) -> float | None:
        """How much of the in-sample return survived out of sample.

        1.0 means the out-of-sample result matched; 0.0 means it vanished; negative means the
        strategy lost money on data it had not seen. ``None`` when the in-sample return was
        too small for the ratio to carry information.
        """
        if self.train_return < self.MIN_TRAIN_RETURN_FOR_RATIO:
            return None
        return self.test_return / self.train_return


@dataclass(frozen=True, slots=True)
class WalkForwardReport:
    """Aggregate walk-forward result."""

    windows: tuple[WalkForwardWindow, ...] = ()
    error: str | None = None

    @property
    def window_count(self) -> int:
        return len(self.windows)

    @property
    def profitable_windows(self) -> int:
        return sum(1 for w in self.windows if w.test_return > 0)

    @property
    def consistency(self) -> float:
        """Fraction of out-of-sample windows that were profitable."""
        return safe_divide(self.profitable_windows, self.window_count)

    @property
    def mean_test_return(self) -> float:
        return float(np.mean([w.test_return for w in self.windows])) if self.windows else 0.0

    @property
    def mean_degradation(self) -> float | None:
        """Average retention across windows where the ratio is meaningful."""
        values = [
            w.degradation for w in self.windows if w.degradation is not None
        ]
        return float(np.mean(values)) if values else None

    @property
    def total_test_trades(self) -> int:
        return sum(w.test_trades for w in self.windows)

    @property
    def is_robust(self) -> bool:
        """A deliberately demanding bar.

        Requires a majority of out-of-sample windows profitable, a positive average
        out-of-sample return, and enough trades for the result to mean anything.
        """
        return (
            self.window_count >= 3
            and self.consistency >= 0.6
            and self.mean_test_return > 0
            and self.total_test_trades >= 30
        )

    def verdict(self) -> str:
        if self.error:
            return f"walk-forward could not run: {self.error}"
        if not self.windows:
            return "no walk-forward windows were produced"
        parts = [
            f"{self.profitable_windows}/{self.window_count} out-of-sample windows profitable "
            f"({self.consistency:.0%})",
            f"mean out-of-sample return {self.mean_test_return:+.2%}",
            (
                f"retained {self.mean_degradation:.0%} of in-sample performance"
                if self.mean_degradation is not None
                else "in-sample returns too small for a retention ratio"
            ),
            f"{self.total_test_trades} out-of-sample trades",
        ]
        judgement = (
            "PASSES the robustness bar"
            if self.is_robust
            else "FAILS the robustness bar - treat the in-sample result as curve fitting"
        )
        return "; ".join(parts) + f". {judgement}."

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_count": self.window_count,
            "profitable_windows": self.profitable_windows,
            "consistency": round(self.consistency, 4),
            "mean_test_return": round(self.mean_test_return, 6),
            "mean_degradation": (
                round(self.mean_degradation, 4)
                if self.mean_degradation is not None
                else None
            ),
            "total_test_trades": self.total_test_trades,
            "is_robust": self.is_robust,
            "verdict": self.verdict(),
            "windows": [
                {
                    "index": w.index,
                    "train_start": w.train_start.isoformat(),
                    "test_start": w.test_start.isoformat(),
                    "test_end": w.test_end.isoformat(),
                    "train_return": round(w.train_return, 6),
                    "test_return": round(w.test_return, 6),
                    "test_trades": w.test_trades,
                    "degradation": (
                        round(w.degradation, 4) if w.degradation is not None else None
                    ),
                }
                for w in self.windows
            ],
            "error": self.error,
        }


async def walk_forward(
    strategy_name: str,
    parameters: dict[str, Any],
    candles: Sequence[Candle],
    *,
    train_bars: int = 2000,
    test_bars: int = 500,
    step_bars: int | None = None,
    config: BacktestConfig | None = None,
    risk_limits: RiskLimits | None = None,
) -> WalkForwardReport:
    """Rolling out-of-sample evaluation.

    Each window trains on ``train_bars`` and evaluates on the ``test_bars`` that immediately
    follow, then rolls forward. Windows never overlap in their test portions, so out-of-sample
    results are independent of each other.

    Note that this implementation evaluates *fixed* parameters on each window rather than
    re-optimising per window. Re-optimisation is a valid variant but it multiplies the
    search space, and the search itself is what overfits; measuring one parameter set across
    many regimes is the more honest question for a shipped strategy.
    """
    stride = step_bars or test_bars
    if train_bars < 100 or test_bars < 50:
        raise BacktestError("train_bars must be >= 100 and test_bars >= 50")
    if len(candles) < train_bars + test_bars:
        return WalkForwardReport(
            error=(
                f"{len(candles)} bars is not enough for a {train_bars}+{test_bars} "
                "walk-forward window"
            )
        )

    windows: list[WalkForwardWindow] = []
    index = 0
    offset = 0
    while offset + train_bars + test_bars <= len(candles):
        train = list(candles[offset : offset + train_bars])
        test = list(candles[offset + train_bars : offset + train_bars + test_bars])

        try:
            train_result = await _run_one(
                strategy_name, parameters, train, config, risk_limits
            )
            test_result = await _run_one(
                strategy_name, parameters, test, config, risk_limits
            )
        except InsufficientDataError as exc:
            logger.info("walk_forward.window_skipped", index=index, reason=str(exc))
            offset += stride
            index += 1
            continue

        windows.append(
            WalkForwardWindow(
                index=index,
                train_start=train[0].open_time,
                train_end=train[-1].close_time,
                test_start=test[0].open_time,
                test_end=test[-1].close_time,
                train_return=train_result.metrics.total_return,
                test_return=test_result.metrics.total_return,
                train_sharpe=train_result.metrics.sharpe_ratio,
                test_sharpe=test_result.metrics.sharpe_ratio,
                test_trades=test_result.metrics.total_trades,
                test_max_drawdown=test_result.metrics.max_drawdown,
            )
        )
        offset += stride
        index += 1

    report = WalkForwardReport(windows=tuple(windows))
    logger.info("walk_forward.complete", verdict=report.verdict())
    return report


# --------------------------------------------------------------------------- #
# Parameter sensitivity
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SensitivityPoint:
    parameters: dict[str, Any]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    trades: int


@dataclass(frozen=True, slots=True)
class SensitivityReport:
    """How performance varies across a parameter grid."""

    points: tuple[SensitivityPoint, ...] = ()
    baseline: SensitivityPoint | None = None

    @property
    def returns(self) -> np.ndarray:
        return np.array([p.total_return for p in self.points], dtype=float)

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.returns)) if self.points else 0.0

    @property
    def return_std(self) -> float:
        return float(np.std(self.returns)) if len(self.points) > 1 else 0.0

    @property
    def profitable_fraction(self) -> float:
        return safe_divide(int(np.sum(self.returns > 0)), len(self.points))

    @property
    def best(self) -> SensitivityPoint | None:
        return max(self.points, key=lambda p: p.total_return) if self.points else None

    @property
    def worst(self) -> SensitivityPoint | None:
        return min(self.points, key=lambda p: p.total_return) if self.points else None

    @property
    def is_plateau(self) -> bool:
        """True when performance is broadly positive rather than concentrated in a spike.

        A strategy whose profitability depends on an exact parameter value has found an
        artefact of this particular history. A plateau — most nearby settings also work —
        is weak evidence of something real.
        """
        return self.profitable_fraction >= 0.6 and self.mean_return > 0

    @property
    def peak_ratio(self) -> float:
        """Best result divided by the mean. Very high values indicate a lone spike."""
        if not self.points or self.mean_return <= 0:
            return float("inf") if self.points else 0.0
        best = self.best
        return safe_divide(best.total_return, self.mean_return) if best else 0.0

    def verdict(self) -> str:
        if not self.points:
            return "no parameter combinations were evaluated"
        shape = "plateau" if self.is_plateau else "spike"
        return (
            f"{len(self.points)} combinations: {self.profitable_fraction:.0%} profitable, "
            f"mean {self.mean_return:+.2%} (sd {self.return_std:.2%}), "
            f"best/mean ratio {self.peak_ratio:.1f}. "
            f"Surface looks like a {shape}"
            + (
                "."
                if self.is_plateau
                else " - the result likely depends on exact parameter values."
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "combinations": len(self.points),
            "profitable_fraction": round(self.profitable_fraction, 4),
            "mean_return": round(self.mean_return, 6),
            "return_std": round(self.return_std, 6),
            "peak_ratio": round(self.peak_ratio, 3),
            "is_plateau": self.is_plateau,
            "verdict": self.verdict(),
            "best": (
                {"parameters": self.best.parameters,
                 "total_return": round(self.best.total_return, 6)}
                if self.best
                else None
            ),
            "worst": (
                {"parameters": self.worst.parameters,
                 "total_return": round(self.worst.total_return, 6)}
                if self.worst
                else None
            ),
        }


async def parameter_sensitivity(
    strategy_name: str,
    base_parameters: dict[str, Any],
    grid: dict[str, Sequence[Any]],
    candles: Sequence[Candle],
    *,
    config: BacktestConfig | None = None,
    risk_limits: RiskLimits | None = None,
    max_combinations: int = 200,
) -> SensitivityReport:
    """Evaluate a parameter grid and report the shape of the performance surface."""
    if not grid:
        raise BacktestError("parameter_sensitivity requires at least one parameter to vary")

    names = sorted(grid)
    combinations = list(product(*(grid[name] for name in names)))
    if len(combinations) > max_combinations:
        raise BacktestError(
            f"{len(combinations)} combinations exceeds the {max_combinations} limit; "
            "narrow the grid. A very large search is itself a source of overfitting."
        )

    points: list[SensitivityPoint] = []
    for values in combinations:
        parameters = {**base_parameters, **dict(zip(names, values, strict=True))}
        try:
            result = await _run_one(
                strategy_name, parameters, list(candles), config, risk_limits
            )
        except (BacktestError, InsufficientDataError, ValueError) as exc:
            logger.info(
                "sensitivity.combination_skipped", parameters=parameters, reason=str(exc)
            )
            continue
        points.append(
            SensitivityPoint(
                parameters=dict(zip(names, values, strict=True)),
                total_return=result.metrics.total_return,
                sharpe_ratio=result.metrics.sharpe_ratio,
                max_drawdown=result.metrics.max_drawdown,
                trades=result.metrics.total_trades,
            )
        )

    baseline: SensitivityPoint | None = None
    try:
        base_result = await _run_one(
            strategy_name, base_parameters, list(candles), config, risk_limits
        )
        baseline = SensitivityPoint(
            parameters=base_parameters,
            total_return=base_result.metrics.total_return,
            sharpe_ratio=base_result.metrics.sharpe_ratio,
            max_drawdown=base_result.metrics.max_drawdown,
            trades=base_result.metrics.total_trades,
        )
    except (BacktestError, InsufficientDataError, ValueError):
        baseline = None

    report = SensitivityReport(points=tuple(points), baseline=baseline)
    logger.info("sensitivity.complete", verdict=report.verdict())
    return report


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class MonteCarloReport:
    """Distribution of outcomes under trade reshuffling and resampling.

    Two distinct analyses, because they answer different questions and one of them is
    degenerate on its own:

    **Permutation** shuffles the realised trades into a different order. The final equity is
    *identical* for every permutation — a sum does not care about order — so the total return
    carries no information here. What does vary, sometimes dramatically, is the **drawdown**:
    the same trades with the losers clustered produce a far deeper hole. That is the risk this
    analysis exposes.

    **Bootstrap** resamples trades with replacement, producing a different trade set drawn
    from the same distribution. This does vary the total return, and answers "how much of the
    result was the particular draw of trades we happened to get?"

    Both assume trades are independent and identically distributed. Neither assumption is quite
    true — losses cluster by regime — so both understate tail risk.
    """

    simulations: int = 0
    observed_return: float = 0.0
    observed_max_drawdown: float = 0.0
    #: Drawdown distribution from permutation (ordering risk).
    drawdown_percentiles: dict[str, float] = field(default_factory=dict)
    worst_case_drawdown: float = 0.0
    observed_drawdown_percentile: float = 0.0
    #: Return distribution from bootstrap resampling (sampling risk).
    return_percentiles: dict[str, float] = field(default_factory=dict)
    probability_of_loss: float = 0.0
    worst_case_return: float = 0.0

    @property
    def ordering_was_lucky(self) -> bool:
        """True when the realised trade order produced an unusually shallow drawdown.

        Below the 10th percentile of reshuffles means the equity curve looked smooth largely
        because the losses happened to arrive spread out. That will not repeat.
        """
        return self.simulations > 0 and self.observed_drawdown_percentile < 0.10

    @property
    def was_lucky(self) -> bool:
        """Backwards-compatible alias for :attr:`ordering_was_lucky`."""
        return self.ordering_was_lucky

    def verdict(self) -> str:
        if self.simulations == 0:
            return "not enough trades for a Monte Carlo analysis"
        parts = [
            f"{self.simulations} simulations",
            f"observed drawdown {self.observed_max_drawdown:.2%} sits at the "
            f"{self.observed_drawdown_percentile:.0%} percentile of reorderings",
            f"95th percentile drawdown {self.drawdown_percentiles.get('p95', 0.0):.2%}",
            f"bootstrap probability of a losing outcome {self.probability_of_loss:.0%}",
            f"bootstrap 5th percentile return "
            f"{self.return_percentiles.get('p5', 0.0):+.2%}",
        ]
        if self.ordering_was_lucky:
            parts.append(
                "the realised trade order was unusually kind; a worse ordering of the same "
                "trades would have produced a materially deeper drawdown"
            )
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "simulations": self.simulations,
            "observed_return": round(self.observed_return, 6),
            "observed_max_drawdown": round(self.observed_max_drawdown, 6),
            "drawdown_percentiles": {
                k: round(v, 6) for k, v in self.drawdown_percentiles.items()
            },
            "worst_case_drawdown": round(self.worst_case_drawdown, 6),
            "observed_drawdown_percentile": round(self.observed_drawdown_percentile, 4),
            "return_percentiles": {
                k: round(v, 6) for k, v in self.return_percentiles.items()
            },
            "probability_of_loss": round(self.probability_of_loss, 4),
            "worst_case_return": round(self.worst_case_return, 6),
            "ordering_was_lucky": self.ordering_was_lucky,
            "verdict": self.verdict(),
        }


def monte_carlo(
    result: BacktestResult,
    *,
    simulations: int = 1000,
    seed: int = 42,
    min_trades: int = 20,
) -> MonteCarloReport:
    """Run permutation and bootstrap analyses over the realised trades.

    Trades are handled as **absolute PnL amounts** and equity paths are built additively. The
    tempting alternative — treating each trade's return on its own notional as a portfolio
    return and compounding those — is wrong by orders of magnitude: a 5% move on a position
    worth a fifth of the account is a 1% portfolio move, not a 5% one.
    """
    pnls = [trade.net_pnl for trade in result.trades if trade.exit_time is not None]
    if len(pnls) < min_trades:
        return MonteCarloReport(observed_return=result.metrics.total_return)

    rng = np.random.default_rng(seed)
    trade_pnls = np.array(pnls, dtype=float)
    initial = result.metrics.initial_equity or 1.0
    count = trade_pnls.size

    permutation_drawdowns = np.empty(simulations, dtype=float)
    bootstrap_returns = np.empty(simulations, dtype=float)
    bootstrap_drawdowns = np.empty(simulations, dtype=float)

    for index in range(simulations):
        shuffled = rng.permutation(trade_pnls)
        permutation_drawdowns[index] = compute_drawdown(
            _equity_path(initial, shuffled)
        ).max_drawdown

        resampled = rng.choice(trade_pnls, size=count, replace=True)
        curve = _equity_path(initial, resampled)
        bootstrap_returns[index] = safe_divide(curve[-1] - initial, initial)
        bootstrap_drawdowns[index] = compute_drawdown(curve).max_drawdown

    percentiles = (5, 25, 50, 75, 95)
    observed_drawdown = result.metrics.max_drawdown
    return MonteCarloReport(
        simulations=simulations,
        observed_return=result.metrics.total_return,
        observed_max_drawdown=observed_drawdown,
        drawdown_percentiles={
            f"p{p}": float(np.percentile(permutation_drawdowns, p)) for p in percentiles
        },
        worst_case_drawdown=float(np.max(permutation_drawdowns)),
        observed_drawdown_percentile=float(
            np.mean(permutation_drawdowns <= observed_drawdown)
        ),
        return_percentiles={
            f"p{p}": float(np.percentile(bootstrap_returns, p)) for p in percentiles
        },
        probability_of_loss=float(np.mean(bootstrap_returns < 0)),
        worst_case_return=float(np.min(bootstrap_returns)),
    )


def _equity_path(initial: float, pnls: np.ndarray) -> list[float]:
    """Additive equity path starting at ``initial``."""
    path: list[float] = np.concatenate(
        ([initial], initial + np.cumsum(pnls))
    ).tolist()
    return path


# --------------------------------------------------------------------------- #
# Combined report
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RobustnessReport:
    """Everything the validation suite concluded."""

    in_sample: BacktestResult
    out_of_sample: BacktestResult | None = None
    walk_forward_report: WalkForwardReport | None = None
    sensitivity: SensitivityReport | None = None
    monte_carlo_report: MonteCarloReport | None = None

    @property
    def passed(self) -> bool:
        """Every applicable check must pass. Silence is not a pass."""
        checks: list[bool] = []
        if self.out_of_sample is not None:
            checks.append(self.out_of_sample.metrics.total_return > 0)
        if self.walk_forward_report is not None:
            checks.append(self.walk_forward_report.is_robust)
        if self.sensitivity is not None:
            checks.append(self.sensitivity.is_plateau)
        if self.monte_carlo_report is not None and self.monte_carlo_report.simulations:
            checks.append(not self.monte_carlo_report.ordering_was_lucky)
        return bool(checks) and all(checks)

    def summary(self) -> str:
        lines = [
            "ROBUSTNESS REPORT",
            "=" * 60,
            f"In-sample:      {self.in_sample.metrics.total_return:+.2%} "
            f"({self.in_sample.metrics.total_trades} trades)",
        ]
        if self.out_of_sample is not None:
            lines.append(
                f"Out-of-sample:  {self.out_of_sample.metrics.total_return:+.2%} "
                f"({self.out_of_sample.metrics.total_trades} trades)"
            )
        if self.walk_forward_report is not None:
            lines.append(f"Walk-forward:   {self.walk_forward_report.verdict()}")
        if self.sensitivity is not None:
            lines.append(f"Sensitivity:    {self.sensitivity.verdict()}")
        if self.monte_carlo_report is not None:
            lines.append(f"Monte Carlo:    {self.monte_carlo_report.verdict()}")
        lines.append("=" * 60)
        lines.append(
            "VERDICT: " + ("PASSED" if self.passed else "FAILED")
            + " - "
            + (
                "the edge survived out-of-sample testing"
                if self.passed
                else "this configuration does not survive validation and should not be traded"
            )
        )
        lines.append(
            "\nPast performance and backtest results do not guarantee future performance."
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_sample": self.in_sample.to_dict(),
            "out_of_sample": (
                self.out_of_sample.to_dict() if self.out_of_sample else None
            ),
            "walk_forward": (
                self.walk_forward_report.to_dict() if self.walk_forward_report else None
            ),
            "sensitivity": self.sensitivity.to_dict() if self.sensitivity else None,
            "monte_carlo": (
                self.monte_carlo_report.to_dict() if self.monte_carlo_report else None
            ),
            "passed": self.passed,
            "summary": self.summary(),
        }


async def validate_strategy(
    strategy_name: str,
    parameters: dict[str, Any],
    candles: Sequence[Candle],
    *,
    train_fraction: float = 0.7,
    config: BacktestConfig | None = None,
    risk_limits: RiskLimits | None = None,
    run_walk_forward: bool = True,
    sensitivity_grid: dict[str, Sequence[Any]] | None = None,
    monte_carlo_simulations: int = 1000,
) -> RobustnessReport:
    """Run the full validation suite on one strategy configuration."""
    from app.backtesting.engine import split_candles

    train, test = split_candles(candles, train_fraction)

    in_sample = await _run_one(strategy_name, parameters, train, config, risk_limits)
    out_of_sample: BacktestResult | None = None
    try:
        out_of_sample = await _run_one(strategy_name, parameters, test, config, risk_limits)
    except InsufficientDataError as exc:
        logger.warning("validation.out_of_sample_skipped", reason=str(exc))

    walk_forward_report: WalkForwardReport | None = None
    if run_walk_forward:
        train_bars = max(500, len(candles) // 4)
        walk_forward_report = await walk_forward(
            strategy_name,
            parameters,
            candles,
            train_bars=train_bars,
            test_bars=max(200, train_bars // 3),
            config=config,
            risk_limits=risk_limits,
        )

    sensitivity: SensitivityReport | None = None
    if sensitivity_grid:
        sensitivity = await parameter_sensitivity(
            strategy_name, parameters, sensitivity_grid, train,
            config=config, risk_limits=risk_limits,
        )

    monte_carlo_report = monte_carlo(in_sample, simulations=monte_carlo_simulations)

    return RobustnessReport(
        in_sample=in_sample,
        out_of_sample=out_of_sample,
        walk_forward_report=walk_forward_report,
        sensitivity=sensitivity,
        monte_carlo_report=monte_carlo_report,
    )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
async def _run_one(
    strategy_name: str,
    parameters: dict[str, Any],
    candles: list[Candle],
    config: BacktestConfig | None,
    risk_limits: RiskLimits | None,
) -> BacktestResult:
    engine = BacktestEngine(
        create_strategy(strategy_name, parameters),
        config=config or BacktestConfig(),
        risk_limits=risk_limits,
    )
    return await engine.run(candles)


def validate_strategy_sync(
    strategy_name: str,
    parameters: dict[str, Any],
    candles: Sequence[Candle],
    **kwargs: Any,
) -> RobustnessReport:
    """Synchronous wrapper for CLI use."""
    return asyncio.run(validate_strategy(strategy_name, parameters, candles, **kwargs))
