"""Backtester and metrics tests.

Two groups matter most:

* **Metric correctness** — hand-computable cases with known answers.
* **Bias detection** — the tests that decide whether the backtester can be trusted at all. A
  coin-flip strategy must lose money at the rate fees imply; enabling the same-bar-fill
  lookahead cheat must visibly improve results; and a decision must never change when future
  bars are appended.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.backtesting.engine import (
    BacktestConfig,
    BacktestEngine,
    date_range_slice,
    split_candles,
)
from app.backtesting.metrics import (
    MIN_TRADES_FOR_CONFIDENCE,
    calmar_ratio,
    compound_annual_growth_rate,
    compute_drawdown,
    compute_metrics,
    monthly_returns,
    periodic_returns,
    periods_per_year_for,
    sharpe_ratio,
    sortino_ratio,
    trade_distribution,
)
from app.core.domain import PortfolioSnapshot, Trade
from app.core.enums import ExitReason, PositionSide, SignalAction, TradeStatus
from app.core.exceptions import BacktestError, InsufficientDataError
from app.market_data.models import Candle
from app.market_data.providers import generate_synthetic_candles
from app.risk.limits import RiskLimits
from app.strategies import create_strategy
from app.strategies.base import (
    Strategy,
    StrategyContext,
    StrategyParameters,
    StrategyResult,
)

START = datetime(2024, 1, 1, tzinfo=UTC)


def snapshot(equity: float, index: int, *, cash: float | None = None) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        timestamp=START + timedelta(hours=index),
        cash=cash if cash is not None else equity,
        equity=equity,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        total_exposure=0.0,
        position_count=0,
    )


def trade(pnl: float, index: int = 0, *, entry: float = 100.0, stop: float | None = None) -> Trade:
    return Trade(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        quantity=1.0,
        entry_price=entry,
        entry_time=START + timedelta(hours=index),
        exit_price=entry + pnl,
        exit_time=START + timedelta(hours=index + 1),
        status=TradeStatus.CLOSED,
        exit_reason=ExitReason.SIGNAL,
        gross_pnl=pnl,
        fees=0.0,
        stop_loss=stop,
    )


# =========================================================================== #
# Metrics
# =========================================================================== #
class TestDrawdown:
    def test_no_drawdown_on_monotonic_growth(self) -> None:
        info = compute_drawdown([100, 110, 120, 130])
        assert info.max_drawdown == pytest.approx(0.0)

    def test_known_drawdown(self) -> None:
        info = compute_drawdown([100, 200, 100, 150])
        assert info.max_drawdown == pytest.approx(0.5)
        assert info.peak_equity == pytest.approx(200.0)
        assert info.trough_equity == pytest.approx(100.0)

    def test_recovery_detected(self) -> None:
        timestamps = [START + timedelta(hours=i) for i in range(5)]
        info = compute_drawdown([100, 200, 100, 180, 220], timestamps)
        assert info.recovered
        assert info.recovered_at == timestamps[4]

    def test_unrecovered_drawdown(self) -> None:
        timestamps = [START + timedelta(hours=i) for i in range(4)]
        info = compute_drawdown([100, 200, 100, 150], timestamps)
        assert not info.recovered

    def test_longest_underwater_period(self) -> None:
        timestamps = [START + timedelta(hours=i) for i in range(6)]
        info = compute_drawdown([100, 90, 85, 88, 95, 105], timestamps)
        assert info.longest_underwater_seconds >= 4 * 3600

    def test_empty_input(self) -> None:
        assert compute_drawdown([]).max_drawdown == 0.0


class TestRatios:
    def test_sharpe_of_constant_returns_is_zero(self) -> None:
        assert sharpe_ratio([0.01] * 20, periods_per_year=365) == 0.0

    def test_sharpe_positive_for_positive_drift(self) -> None:
        rng = np.random.default_rng(0)
        returns = rng.normal(0.002, 0.005, 500)
        assert sharpe_ratio(returns, periods_per_year=365) > 0

    def test_sharpe_scales_with_periodisation(self) -> None:
        """Annualisation is explicit precisely because it changes the number so much."""
        rng = np.random.default_rng(1)
        returns = rng.normal(0.001, 0.01, 500)
        daily = sharpe_ratio(returns, periods_per_year=365)
        hourly = sharpe_ratio(returns, periods_per_year=365 * 24)
        assert hourly > daily * 4

    def test_sortino_ignores_upside_volatility(self) -> None:
        mixed = [0.05, -0.01, 0.05, -0.01, 0.05, -0.01] * 10
        assert sortino_ratio(mixed, periods_per_year=365) > sharpe_ratio(
            mixed, periods_per_year=365
        )

    def test_sortino_infinite_when_no_losses(self) -> None:
        assert sortino_ratio([0.01] * 30, periods_per_year=365) == float("inf")

    def test_calmar(self) -> None:
        assert calmar_ratio(0.30, 0.15) == pytest.approx(2.0)
        assert calmar_ratio(0.30, 0.0) == 0.0

    def test_cagr(self) -> None:
        assert compound_annual_growth_rate(100.0, 200.0, 365.0) == pytest.approx(1.0)
        assert compound_annual_growth_rate(100.0, 100.0, 365.0) == pytest.approx(0.0)
        assert compound_annual_growth_rate(0.0, 100.0, 365.0) == 0.0

    def test_periods_per_year(self) -> None:
        assert periods_per_year_for(3600) == pytest.approx(8760.0)
        assert periods_per_year_for(86400) == pytest.approx(365.0)
        with pytest.raises(ValueError):
            periods_per_year_for(0)

    def test_periodic_returns(self) -> None:
        returns = periodic_returns([100, 110, 99])
        assert returns[0] == pytest.approx(0.10)
        assert returns[1] == pytest.approx(-0.10)


class TestComputeMetrics:
    def test_empty_run(self) -> None:
        metrics = compute_metrics([], [], initial_equity=1000.0, periods_per_year=365)
        assert metrics.total_trades == 0
        assert metrics.final_equity == 1000.0

    def test_win_rate_and_profit_factor(self) -> None:
        trades = [trade(10.0, 0), trade(10.0, 2), trade(-5.0, 4), trade(-5.0, 6)]
        snapshots = [snapshot(1000.0 + i * 2.5, i) for i in range(5)]
        metrics = compute_metrics(
            snapshots, trades, initial_equity=1000.0, periods_per_year=8760
        )
        assert metrics.total_trades == 4
        assert metrics.win_rate == pytest.approx(0.5)
        assert metrics.profit_factor == pytest.approx(2.0)
        assert metrics.expectancy == pytest.approx(2.5)

    def test_streaks(self) -> None:
        trades = [
            trade(1.0, 0), trade(1.0, 1), trade(1.0, 2),
            trade(-1.0, 3), trade(-1.0, 4),
            trade(1.0, 5),
        ]
        metrics = compute_metrics(
            [snapshot(1000.0, i) for i in range(3)], trades,
            initial_equity=1000.0, periods_per_year=8760,
        )
        assert metrics.max_consecutive_wins == 3
        assert metrics.max_consecutive_losses == 2

    def test_r_multiple_expectancy(self) -> None:
        trades = [trade(20.0, 0, stop=90.0), trade(-10.0, 1, stop=90.0)]
        metrics = compute_metrics(
            [snapshot(1000.0, i) for i in range(3)], trades,
            initial_equity=1000.0, periods_per_year=8760,
        )
        # +2R and -1R -> average +0.5R
        assert metrics.expectancy_r == pytest.approx(0.5)

    def test_small_sample_is_flagged(self) -> None:
        metrics = compute_metrics(
            [snapshot(1000.0 + i, i) for i in range(50)],
            [trade(10.0, i) for i in range(5)],
            initial_equity=1000.0, periods_per_year=8760,
        )
        assert not metrics.is_statistically_meaningful
        assert any("statistically reliable" in w for w in metrics.reliability_warnings)

    def test_large_sample_not_flagged_for_size(self) -> None:
        metrics = compute_metrics(
            [snapshot(1000.0 + i, i) for i in range(3000)],
            [trade(1.0 if i % 2 else -1.0, i) for i in range(MIN_TRADES_FOR_CONFIDENCE + 5)],
            initial_equity=1000.0, periods_per_year=8760,
        )
        assert metrics.is_statistically_meaningful
        assert not any("statistically reliable" in w for w in metrics.reliability_warnings)

    def test_high_win_rate_is_flagged(self) -> None:
        metrics = compute_metrics(
            [snapshot(1000.0 + i, i) for i in range(200)],
            [trade(10.0, i) for i in range(40)],
            initial_equity=1000.0, periods_per_year=8760,
        )
        assert any("unusually high" in w for w in metrics.reliability_warnings)

    def test_fee_dominance_is_flagged(self) -> None:
        trades = [trade(10.0, 0), trade(-5.0, 1)]
        trades[0].fees = 9.0
        metrics = compute_metrics(
            [snapshot(1000.0, i) for i in range(30)], trades,
            initial_equity=1000.0, periods_per_year=8760,
        )
        assert any("fees consume" in w for w in metrics.reliability_warnings)

    def test_summary_is_readable(self) -> None:
        metrics = compute_metrics(
            [snapshot(1000.0 + i, i) for i in range(100)],
            [trade(5.0, i) for i in range(10)],
            initial_equity=1000.0, periods_per_year=8760,
        )
        text = metrics.summary()
        assert "Return" in text and "drawdown" in text

    def test_monthly_returns(self) -> None:
        snapshots = [
            PortfolioSnapshot(
                timestamp=datetime(2024, month, 15, tzinfo=UTC),
                cash=0.0, equity=1000.0 * (1.0 + 0.01 * month),
                unrealized_pnl=0.0, realized_pnl=0.0,
                total_exposure=0.0, position_count=0,
            )
            for month in range(1, 5)
        ]
        result = monthly_returns(snapshots)
        assert set(result) == {"2024-01", "2024-02", "2024-03", "2024-04"}

    def test_trade_distribution(self) -> None:
        result = trade_distribution([trade(v, i) for i, v in enumerate([1, 2, -1, -2, 3])])
        assert sum(result["counts"]) == 5


# =========================================================================== #
# Test strategies for bias detection
# =========================================================================== #
class CoinFlipParameters(StrategyParameters):
    """Parameters for the deliberately edgeless strategy."""


class CoinFlipStrategy(Strategy):
    """Random entries with a fixed stop and target.

    Exists purely as a bias detector. A backtester with an optimistic fill model, lookahead,
    or unpriced costs will show this strategy making money. It must lose, at roughly the rate
    fees and spread imply.
    """

    name = "coin_flip"
    version = "1.0.0"
    description = "Random entries. Used to detect backtester bias; never for trading."
    parameters_model = CoinFlipParameters

    def __init__(self, parameters: object = None, *, seed: int = 7) -> None:
        super().__init__(parameters)  # type: ignore[arg-type]
        self._rng = np.random.default_rng(seed)

    @property
    def required_history(self) -> int:
        return 60

    def _evaluate(self, context: StrategyContext) -> StrategyResult:
        # Trade on ~10% of bars so the sample is large enough to be conclusive.
        if self._rng.random() > 0.10:
            return StrategyResult.no_trade(
                context.symbol, self.name, "no flip", timestamp=context.now
            )
        price = context.candles[-1].close
        going_long = bool(self._rng.random() > 0.5)
        distance = price * 0.02
        if going_long:
            return StrategyResult(
                action=SignalAction.BUY, symbol=context.symbol, strategy_name=self.name,
                timestamp=context.now, confidence=1.0, entry=price,
                stop_loss=price - distance, take_profit=price + distance,
                reason="coin flip: long",
            )
        return StrategyResult(
            action=SignalAction.SELL, symbol=context.symbol, strategy_name=self.name,
            timestamp=context.now, confidence=1.0, entry=price,
            stop_loss=price + distance, take_profit=price - distance,
            reason="coin flip: short",
        )


class AlwaysBuyStrategy(Strategy):
    """Enters long on the first opportunity and holds. Used for deterministic assertions."""

    name = "always_buy"
    version = "1.0.0"
    parameters_model = StrategyParameters

    @property
    def required_history(self) -> int:
        return 60

    def _evaluate(self, context: StrategyContext) -> StrategyResult:
        if context.has_position:
            return StrategyResult.hold(context.symbol, self.name, timestamp=context.now)
        price = context.candles[-1].close
        return StrategyResult(
            action=SignalAction.BUY, symbol=context.symbol, strategy_name=self.name,
            timestamp=context.now, confidence=1.0, entry=price,
            stop_loss=price * 0.9, take_profit=price * 1.2, reason="always buy",
        )


@pytest.fixture(scope="module")
def flat_market() -> list[Candle]:
    """A driftless random walk. Any consistent profit here is a backtester artefact."""
    return generate_synthetic_candles(
        "BTCUSDT", "1h", 1500, seed=99, start_price=30_000.0,
        volatility=0.008, drift=0.0, regime_shifts=False, start=START,
    )


@pytest.fixture(scope="module")
def gapped_market() -> list[Candle]:
    """Driftless walk that gaps between bars, as real markets do."""
    return generate_synthetic_candles(
        "BTCUSDT", "1h", 1500, seed=99, start_price=30_000.0,
        volatility=0.008, drift=0.0, regime_shifts=False, start=START,
        gap_volatility=0.003,
    )


@pytest.fixture
def permissive_limits() -> RiskLimits:
    return RiskLimits(
        risk_per_trade=0.01,
        max_concurrent_positions=3,
        max_daily_loss=0.20,
        max_weekly_loss=0.40,
        max_drawdown=0.60,
        max_daily_trades=1000,
        max_loss_streak=50,
        cooldown_seconds=0,
        min_reward_risk=0.0,
        max_position_fraction=1.0,
        max_portfolio_exposure=2.0,
        max_asset_exposure=1.0,
        scale_size_by_confidence=False,
    )


# =========================================================================== #
# Bias detection - the tests that decide whether backtests mean anything
# =========================================================================== #
class TestBacktesterBias:
    async def test_coin_flip_does_not_make_money(
        self, flat_market: list[Candle], permissive_limits: RiskLimits
    ) -> None:
        """The single most important backtester test.

        Random entries in a driftless market, paying real fees and spread, must lose. A
        backtester that shows this strategy profitable is not measuring the strategy.
        """
        engine = BacktestEngine(
            CoinFlipStrategy(),
            config=BacktestConfig(initial_balance=100_000.0, warmup_bars=120),
            risk_limits=permissive_limits,
        )
        result = await engine.run(flat_market)

        assert result.metrics.total_trades >= 20, "not enough trades to be conclusive"
        assert result.metrics.total_fees > 0, "fees were not charged"
        assert result.metrics.total_return <= 0.02, (
            f"a coin-flip strategy returned {result.metrics.total_return:+.2%}; "
            "the fill model is optimistic or there is lookahead"
        )

    async def test_costs_are_actually_charged(
        self, flat_market: list[Candle], permissive_limits: RiskLimits
    ) -> None:
        """Gross PnL minus fees must equal the equity change, exactly."""
        engine = BacktestEngine(
            CoinFlipStrategy(),
            config=BacktestConfig(initial_balance=100_000.0, warmup_bars=120),
            risk_limits=permissive_limits,
        )
        result = await engine.run(flat_market)
        gross = sum(t.gross_pnl for t in result.trades)
        fees = sum(t.fees for t in result.trades)
        equity_change = result.metrics.final_equity - result.metrics.initial_equity
        assert equity_change == pytest.approx(gross - fees, abs=0.5)

    async def test_lookahead_cheat_changes_results(
        self, gapped_market: list[Candle], permissive_limits: RiskLimits
    ) -> None:
        """Same-bar fills must produce different results than next-bar fills.

        This needs *gapped* data. In a continuous series the next bar's open equals the
        previous close, so both fill modes land on the same price and the test would pass
        vacuously - which is exactly what it did before the gap was added.
        """
        honest = BacktestEngine(
            CoinFlipStrategy(),
            config=BacktestConfig(
                initial_balance=100_000.0, warmup_bars=120, execute_next_bar=True
            ),
            risk_limits=permissive_limits,
        )
        cheating = BacktestEngine(
            CoinFlipStrategy(),
            config=BacktestConfig(
                initial_balance=100_000.0, warmup_bars=120, execute_next_bar=False
            ),
            risk_limits=permissive_limits,
        )
        honest_result = await honest.run(gapped_market)
        cheating_result = await cheating.run(gapped_market)

        assert honest_result.metrics.final_equity != pytest.approx(
            cheating_result.metrics.final_equity
        )
        assert any("lookahead" in w for w in cheating_result.warnings)
        assert not any("lookahead" in w for w in honest_result.warnings)

    async def test_results_are_deterministic(
        self, flat_market: list[Candle], permissive_limits: RiskLimits
    ) -> None:
        def build() -> BacktestEngine:
            return BacktestEngine(
                CoinFlipStrategy(seed=3),
                config=BacktestConfig(initial_balance=100_000.0, warmup_bars=120),
                risk_limits=permissive_limits,
            )

        first = await build().run(flat_market)
        second = await build().run(flat_market)
        assert first.metrics.final_equity == pytest.approx(second.metrics.final_equity)
        assert first.metrics.total_trades == second.metrics.total_trades

    async def test_appending_future_bars_does_not_change_the_past(
        self, flat_market: list[Candle], permissive_limits: RiskLimits
    ) -> None:
        """The definitive lookahead test."""
        def build() -> BacktestEngine:
            return BacktestEngine(
                CoinFlipStrategy(seed=5),
                config=BacktestConfig(
                    initial_balance=100_000.0, warmup_bars=120, close_at_end=False
                ),
                risk_limits=permissive_limits,
            )

        short_run = await build().run(flat_market[:800])
        long_run = await build().run(flat_market)

        # Signals produced over the shared prefix must be identical.
        shared = min(len(short_run.signals), 800 - 120)
        for index in range(shared):
            assert short_run.signals[index].action is long_run.signals[index].action
            assert short_run.signals[index].timestamp == long_run.signals[index].timestamp

    async def test_portfolio_invariants_hold_throughout(
        self, flat_market: list[Candle], permissive_limits: RiskLimits
    ) -> None:
        engine = BacktestEngine(
            CoinFlipStrategy(),
            config=BacktestConfig(initial_balance=100_000.0, warmup_bars=120),
            risk_limits=permissive_limits,
        )
        result = await engine.run(flat_market)
        assert not any("invariants were violated" in w for w in result.warnings)


# =========================================================================== #
# Engine behaviour
# =========================================================================== #
class TestBacktestEngine:
    async def test_produces_a_complete_result(
        self, flat_market: list[Candle], permissive_limits: RiskLimits
    ) -> None:
        engine = BacktestEngine(
            AlwaysBuyStrategy(),
            config=BacktestConfig(initial_balance=10_000.0, warmup_bars=120),
            risk_limits=permissive_limits,
        )
        result = await engine.run(flat_market)
        assert result.bars_processed == len(flat_market)
        assert result.equity_curve
        assert result.drawdown_curve
        assert result.strategy_name == "always_buy"
        assert result.symbol == "BTCUSDT"
        assert "do not guarantee future performance" in result.summary()

    async def test_all_positions_closed_at_the_end(
        self, flat_market: list[Candle], permissive_limits: RiskLimits
    ) -> None:
        engine = BacktestEngine(
            AlwaysBuyStrategy(),
            config=BacktestConfig(
                initial_balance=10_000.0, warmup_bars=120, close_at_end=True
            ),
            risk_limits=permissive_limits,
        )
        result = await engine.run(flat_market)
        assert all(t.exit_time is not None for t in result.trades)

    async def test_risk_rejections_are_recorded(
        self, flat_market: list[Candle]
    ) -> None:
        strict = RiskLimits(
            risk_per_trade=0.001, max_concurrent_positions=1, max_daily_trades=2,
            max_daily_loss=0.02, max_weekly_loss=0.06, max_drawdown=0.1,
            cooldown_seconds=86400,
        )
        engine = BacktestEngine(
            CoinFlipStrategy(),
            config=BacktestConfig(initial_balance=10_000.0, warmup_bars=120),
            risk_limits=strict,
        )
        result = await engine.run(flat_market)
        assert result.rejections
        assert any(r["blocked_by"] for r in result.rejections)

    async def test_empty_candles_rejected(self) -> None:
        engine = BacktestEngine(create_strategy("trend_following"))
        with pytest.raises(InsufficientDataError):
            await engine.run([])

    async def test_insufficient_history_rejected(
        self, flat_market: list[Candle]
    ) -> None:
        engine = BacktestEngine(create_strategy("trend_following"))
        with pytest.raises(InsufficientDataError, match="not enough"):
            await engine.run(flat_market[:50])

    async def test_mixed_symbols_rejected(self, flat_market: list[Candle]) -> None:
        other = generate_synthetic_candles("ETHUSDT", "1h", 400, seed=1, start=START)
        engine = BacktestEngine(CoinFlipStrategy())
        with pytest.raises(BacktestError, match="one symbol"):
            await engine.run([*flat_market[:400], *other])

    async def test_out_of_order_candles_rejected(
        self, flat_market: list[Candle]
    ) -> None:
        scrambled = [*flat_market[:400], flat_market[100]]
        engine = BacktestEngine(CoinFlipStrategy())
        with pytest.raises(BacktestError, match="strictly increasing"):
            await engine.run(scrambled)

    async def test_progress_callback_fires(
        self, flat_market: list[Candle], permissive_limits: RiskLimits
    ) -> None:
        seen: list[float] = []
        engine = BacktestEngine(
            CoinFlipStrategy(),
            config=BacktestConfig(
                initial_balance=10_000.0, warmup_bars=120,
                progress_callback=seen.append,
            ),
            risk_limits=permissive_limits,
        )
        await engine.run(flat_market)
        assert seen
        assert all(0.0 <= value <= 1.0 for value in seen)

    async def test_result_serialises(
        self, flat_market: list[Candle], permissive_limits: RiskLimits
    ) -> None:
        engine = BacktestEngine(
            CoinFlipStrategy(),
            config=BacktestConfig(initial_balance=10_000.0, warmup_bars=120),
            risk_limits=permissive_limits,
        )
        payload = (await engine.run(flat_market)).to_dict()
        assert payload["strategy"] == "coin_flip"
        assert "metrics" in payload
        assert isinstance(payload["metrics"]["total_return"], float)


# =========================================================================== #
# Data splitting
# =========================================================================== #
class TestSplitting:
    def test_chronological_split(self, flat_market: list[Candle]) -> None:
        train, test = split_candles(flat_market, 0.7)
        assert len(train) + len(test) == len(flat_market)
        assert train[-1].open_time < test[0].open_time

    def test_split_fraction_validated(self, flat_market: list[Candle]) -> None:
        with pytest.raises(ValueError, match="strictly between"):
            split_candles(flat_market, 1.0)

    def test_split_needs_data(self) -> None:
        with pytest.raises(InsufficientDataError):
            split_candles([], 0.7)

    def test_date_range_slice(self, flat_market: list[Candle]) -> None:
        window = date_range_slice(
            flat_market, START + timedelta(hours=10), START + timedelta(hours=20)
        )
        assert len(window) == 10
        assert window[0].open_time == START + timedelta(hours=10)


# =========================================================================== #
# News replay
# =========================================================================== #
class TestNewsReplay:
    async def test_news_cannot_be_read_before_publication(
        self, flat_market: list[Candle], permissive_limits: RiskLimits
    ) -> None:
        """An article published later must not influence an earlier decision."""
        from app.news.providers import InMemoryNewsProvider, synthetic_news

        # synthetic_news walks backwards from `start`, so anchor it ahead of the cutoff to
        # guarantee every article is published after it.
        late = flat_market[int(len(flat_market) * 0.75)].close_time
        articles = synthetic_news(
            "BTC", count=5, start=late + timedelta(hours=4), positive=True
        )
        earliest = min(a.published_at for a in articles)
        assert earliest >= late

        engine = BacktestEngine(
            CoinFlipStrategy(seed=11),
            config=BacktestConfig(initial_balance=100_000.0, warmup_bars=120),
            risk_limits=permissive_limits,
            news_provider=InMemoryNewsProvider(articles),
        )
        result = await engine.run(flat_market)

        early_signals = [s for s in result.signals if s.timestamp < earliest]
        assert early_signals
        assert all(s.news_score == 0.0 for s in early_signals), (
            "news influenced a decision made before it was published"
        )
        later_signals = [s for s in result.signals if s.timestamp > earliest]
        assert any(s.news_score != 0.0 for s in later_signals), (
            "news was published but never reached the pipeline"
        )
