"""Robustness-analysis tests.

The properties that matter: a strategy fitted to one period must fail walk-forward; a flat
performance surface must be reported as a plateau and a lone spike as a spike; and the Monte
Carlo maths must not manufacture returns.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from app.backtesting.engine import BacktestConfig, BacktestResult
from app.backtesting.metrics import PerformanceMetrics
from app.backtesting.validation import (
    MonteCarloReport,
    SensitivityPoint,
    SensitivityReport,
    WalkForwardReport,
    WalkForwardWindow,
    monte_carlo,
    parameter_sensitivity,
    validate_strategy,
    walk_forward,
)
from app.core.domain import Trade
from app.core.enums import ExitReason, PositionSide, TradeStatus
from app.core.exceptions import BacktestError
from app.market_data.providers import generate_synthetic_candles
from app.risk.limits import RiskLimits

START = datetime(2024, 1, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def market() -> list:
    return generate_synthetic_candles(
        "BTCUSDT", "1h", 2600, seed=21, start_price=30_000.0,
        volatility=0.01, start=START,
    )


@pytest.fixture
def limits() -> RiskLimits:
    return RiskLimits(
        risk_per_trade=0.01, max_concurrent_positions=3, max_daily_loss=0.05,
        max_weekly_loss=0.15, max_drawdown=0.30, max_loss_streak=20,
        cooldown_seconds=0, min_reward_risk=0.0,
    )


@pytest.fixture
def config() -> BacktestConfig:
    return BacktestConfig(initial_balance=10_000.0, warmup_bars=250)


def make_trade(pnl: float, index: int) -> Trade:
    return Trade(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        quantity=1.0,
        entry_price=100.0,
        entry_time=START,
        exit_price=100.0 + pnl,
        exit_time=START,
        status=TradeStatus.CLOSED,
        exit_reason=ExitReason.SIGNAL,
        gross_pnl=pnl,
        fees=0.0,
    )


def result_with(pnls: list[float], *, initial: float = 10_000.0) -> BacktestResult:
    trades = [make_trade(pnl, i) for i, pnl in enumerate(pnls)]
    total = sum(pnls)
    return BacktestResult(
        metrics=PerformanceMetrics(
            initial_equity=initial,
            final_equity=initial + total,
            total_return=total / initial,
            max_drawdown=0.05,
            total_trades=len(trades),
        ),
        trades=trades,
    )


# =========================================================================== #
# Walk-forward
# =========================================================================== #
class TestWalkForward:
    async def test_produces_windows(self, market: list, config, limits) -> None:
        report = await walk_forward(
            "trend_following", {}, market,
            train_bars=800, test_bars=400, config=config, risk_limits=limits,
        )
        assert report.window_count >= 2
        assert report.error is None
        for window in report.windows:
            assert window.test_start >= window.train_end - (
                window.train_end - window.train_start
            )

    async def test_test_windows_do_not_overlap(self, market: list, config, limits) -> None:
        report = await walk_forward(
            "trend_following", {}, market,
            train_bars=800, test_bars=400, config=config, risk_limits=limits,
        )
        for previous, current in zip(report.windows, report.windows[1:], strict=False):
            assert current.test_start >= previous.test_end

    async def test_insufficient_data_reports_an_error(self, market: list) -> None:
        report = await walk_forward(
            "trend_following", {}, market[:300], train_bars=2000, test_bars=500
        )
        assert report.error is not None
        assert not report.is_robust

    async def test_invalid_window_sizes_rejected(self, market: list) -> None:
        with pytest.raises(BacktestError, match="train_bars"):
            await walk_forward("trend_following", {}, market, train_bars=10, test_bars=5)

    def test_robustness_bar_requires_all_conditions(self) -> None:
        def window(test_return: float, trades: int) -> WalkForwardWindow:
            return WalkForwardWindow(
                index=0, train_start=START, train_end=START, test_start=START,
                test_end=START, train_return=0.1, test_return=test_return,
                train_sharpe=1.0, test_sharpe=1.0, test_trades=trades,
                test_max_drawdown=0.05,
            )

        strong = WalkForwardReport(windows=tuple(window(0.05, 15) for _ in range(4)))
        assert strong.is_robust

        # Same returns, too few trades.
        thin = WalkForwardReport(windows=tuple(window(0.05, 2) for _ in range(4)))
        assert not thin.is_robust

        # Enough trades, but mostly losing windows.
        inconsistent = WalkForwardReport(
            windows=(window(0.05, 20), window(-0.05, 20), window(-0.05, 20))
        )
        assert not inconsistent.is_robust

    def test_degradation_is_none_for_tiny_in_sample_returns(self) -> None:
        window = WalkForwardWindow(
            index=0, train_start=START, train_end=START, test_start=START,
            test_end=START, train_return=0.0001, test_return=0.02,
            train_sharpe=0.0, test_sharpe=0.0, test_trades=5, test_max_drawdown=0.0,
        )
        assert window.degradation is None

    def test_degradation_ratio(self) -> None:
        window = WalkForwardWindow(
            index=0, train_start=START, train_end=START, test_start=START,
            test_end=START, train_return=0.20, test_return=0.10,
            train_sharpe=0.0, test_sharpe=0.0, test_trades=5, test_max_drawdown=0.0,
        )
        assert window.degradation == pytest.approx(0.5)

    def test_report_serialises(self) -> None:
        report = WalkForwardReport(error="not enough data")
        payload = report.to_dict()
        assert payload["error"] == "not enough data"
        assert payload["is_robust"] is False


# =========================================================================== #
# Sensitivity
# =========================================================================== #
class TestSensitivity:
    async def test_grid_is_evaluated(self, market: list, config, limits) -> None:
        report = await parameter_sensitivity(
            "trend_following", {},
            {"adx_threshold": [20.0, 25.0, 30.0]},
            market[:1200], config=config, risk_limits=limits,
        )
        assert len(report.points) == 3
        assert report.best is not None and report.worst is not None

    async def test_oversized_grid_rejected(self, market: list) -> None:
        with pytest.raises(BacktestError, match="exceeds"):
            await parameter_sensitivity(
                "trend_following", {},
                {"adx_threshold": list(np.linspace(10, 50, 30)),
                 "atr_stop_multiplier": list(np.linspace(1, 5, 20))},
                market, max_combinations=50,
            )

    async def test_empty_grid_rejected(self, market: list) -> None:
        with pytest.raises(BacktestError, match="at least one parameter"):
            await parameter_sensitivity("trend_following", {}, {}, market)

    def test_plateau_detected(self) -> None:
        points = tuple(
            SensitivityPoint({"x": i}, 0.10 + i * 0.001, 1.0, 0.05, 40) for i in range(10)
        )
        report = SensitivityReport(points=points)
        assert report.is_plateau
        assert report.peak_ratio < 1.2
        assert "plateau" in report.verdict()

    def test_spike_detected(self) -> None:
        """One brilliant setting surrounded by losers is curve fitting, not an edge."""
        points = (
            *(SensitivityPoint({"x": i}, -0.02, 0.0, 0.2, 10) for i in range(9)),
            SensitivityPoint({"x": 9}, 0.80, 3.0, 0.05, 12),
        )
        report = SensitivityReport(points=points)
        assert not report.is_plateau
        assert "spike" in report.verdict()

    def test_empty_report(self) -> None:
        report = SensitivityReport()
        assert not report.is_plateau
        assert "no parameter combinations" in report.verdict()


# =========================================================================== #
# Monte Carlo
# =========================================================================== #
class TestMonteCarlo:
    def test_too_few_trades_reports_nothing(self) -> None:
        report = monte_carlo(result_with([10.0] * 5), simulations=100)
        assert report.simulations == 0
        assert "not enough trades" in report.verdict()

    def test_returns_are_not_manufactured(self) -> None:
        """The bug this guards against inflated a 4% result into 390%.

        Trades are absolute PnL amounts; the bootstrap distribution must stay in the same
        neighbourhood as the observed result, not orders of magnitude above it.
        """
        pnls = [50.0] * 20 + [-30.0] * 20  # net +400 on 10,000 -> +4%
        report = monte_carlo(result_with(pnls), simulations=300, seed=1)
        assert report.observed_return == pytest.approx(0.04)
        median = report.return_percentiles["p50"]
        assert 0.0 < median < 0.15, f"bootstrap median {median:.2%} is implausible"
        assert report.return_percentiles["p95"] < 0.25

    def test_drawdown_distribution_is_informative(self) -> None:
        """Permutation must produce a *spread* of drawdowns, not a single value."""
        pnls = [100.0] * 15 + [-80.0] * 15
        report = monte_carlo(result_with(pnls), simulations=400, seed=2)
        assert report.drawdown_percentiles["p95"] > report.drawdown_percentiles["p5"]
        assert report.worst_case_drawdown >= report.drawdown_percentiles["p95"]

    def test_clustered_losses_produce_deep_drawdowns(self) -> None:
        pnls = [20.0] * 30 + [-25.0] * 10
        report = monte_carlo(result_with(pnls), simulations=400, seed=3)
        assert report.worst_case_drawdown > 0.0

    def test_all_winners_have_no_drawdown(self) -> None:
        report = monte_carlo(result_with([10.0] * 30), simulations=100, seed=4)
        assert report.worst_case_drawdown == pytest.approx(0.0)
        assert report.probability_of_loss == pytest.approx(0.0)

    def test_all_losers_always_lose(self) -> None:
        report = monte_carlo(result_with([-10.0] * 30), simulations=100, seed=5)
        assert report.probability_of_loss == pytest.approx(1.0)
        assert report.return_percentiles["p95"] < 0

    def test_deterministic_with_seed(self) -> None:
        pnls = [50.0, -30.0] * 15
        first = monte_carlo(result_with(pnls), simulations=200, seed=9)
        second = monte_carlo(result_with(pnls), simulations=200, seed=9)
        assert first.return_percentiles == second.return_percentiles
        assert first.drawdown_percentiles == second.drawdown_percentiles

    def test_serialises(self) -> None:
        report = monte_carlo(result_with([10.0, -5.0] * 15), simulations=100)
        payload = report.to_dict()
        assert "drawdown_percentiles" in payload
        assert "return_percentiles" in payload
        assert isinstance(payload["ordering_was_lucky"], bool)

    def test_empty_report_verdict(self) -> None:
        assert "not enough trades" in MonteCarloReport().verdict()


# =========================================================================== #
# Combined
# =========================================================================== #
class TestValidateStrategy:
    async def test_full_suite_runs(self, market: list, config, limits) -> None:
        report = await validate_strategy(
            "trend_following", {}, market,
            config=config, risk_limits=limits,
            run_walk_forward=True,
            sensitivity_grid={"adx_threshold": [22.0, 25.0, 28.0]},
            monte_carlo_simulations=100,
        )
        assert report.in_sample is not None
        assert report.walk_forward_report is not None
        assert report.sensitivity is not None
        assert report.monte_carlo_report is not None
        text = report.summary()
        assert "ROBUSTNESS REPORT" in text
        assert "do not guarantee future performance" in text

    async def test_verdict_requires_every_check(self, market: list, config, limits) -> None:
        report = await validate_strategy(
            "trend_following", {}, market,
            config=config, risk_limits=limits,
            run_walk_forward=True, monte_carlo_simulations=100,
        )
        # `passed` must be the conjunction of the checks that actually ran.
        expected = (
            (report.out_of_sample is None or report.out_of_sample.metrics.total_return > 0)
            and report.walk_forward_report is not None
            and report.walk_forward_report.is_robust
            and not report.monte_carlo_report.ordering_was_lucky
        )
        assert report.passed == bool(expected)

    async def test_serialises(self, market: list, config, limits) -> None:
        report = await validate_strategy(
            "trend_following", {}, market,
            config=config, risk_limits=limits,
            run_walk_forward=False, monte_carlo_simulations=50,
        )
        payload = report.to_dict()
        assert "in_sample" in payload
        assert isinstance(payload["passed"], bool)
