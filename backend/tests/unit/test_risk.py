"""Risk engine tests.

The property under test throughout is: **no configuration, signal or market condition may
produce a trade that risks more than the configured fraction of equity, and no strategy may
bypass the manager.**
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.core.domain import InstrumentSpec, Position
from app.core.enums import (
    InstrumentType,
    KillSwitchReason,
    OrderSide,
    PositionSide,
    RiskDecision,
    RiskEventType,
    SignalAction,
)
from app.core.exceptions import KillSwitchActiveError, RiskViolationError
from app.market_data.models import (
    Candle,
    MarketSnapshot,
    OrderBook,
    OrderBookLevel,
    Ticker,
)
from app.risk.kill_switch import KillSwitch
from app.risk.limits import RiskLimits
from app.risk.manager import PortfolioView, RiskManager, TradeProposal
from app.risk.sizing import (
    SizingRequest,
    calculate_position_size,
    estimate_worst_case_loss,
    kelly_fraction,
    reward_risk_ratio,
)
from app.risk.state import RiskState, TradeOutcome, day_start, week_start

NOW = datetime(2024, 6, 12, 12, 0, tzinfo=UTC)  # a Wednesday


@pytest.fixture
def instrument() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        instrument_type=InstrumentType.LINEAR_PERPETUAL,
        tick_size=0.1,
        lot_size=0.001,
        min_quantity=0.001,
        min_notional=10.0,
        max_leverage=10.0,
        maker_fee=0.0002,
        taker_fee=0.00055,
    )


@pytest.fixture
def limits() -> RiskLimits:
    return RiskLimits(
        risk_per_trade=0.01,
        max_position_fraction=0.5,
        max_leverage=3.0,
        max_portfolio_exposure=2.0,
        max_asset_exposure=1.0,
        max_concurrent_positions=5,
        max_daily_loss=0.02,
        max_weekly_loss=0.06,
        max_drawdown=0.10,
        max_loss_streak=4,
        max_daily_trades=30,
        cooldown_seconds=0,
        min_reward_risk=0.0,
        scale_size_by_confidence=False,
    )


@pytest.fixture
def state() -> RiskState:
    return RiskState(starting_equity=100_000.0)


@pytest.fixture
def manager(limits: RiskLimits, state: RiskState) -> RiskManager:
    return RiskManager(limits, state)


@pytest.fixture
def portfolio() -> PortfolioView:
    return PortfolioView(equity=100_000.0, available_margin=100_000.0)


def proposal(
    instrument: InstrumentSpec,
    *,
    entry: float = 100.0,
    stop: float = 98.0,
    target: float | None = 106.0,
    action: SignalAction = SignalAction.BUY,
    confidence: float = 1.0,
    leverage: float = 1.0,
) -> TradeProposal:
    return TradeProposal(
        symbol=instrument.symbol,
        action=action,
        entry_price=entry,
        stop_loss=stop,
        take_profit=target,
        instrument=instrument,
        confidence=confidence,
        leverage=leverage,
        strategy_name="test",
    )


def snapshot_with(
    *,
    price: float = 100.0,
    spread_bps: float = 2.0,
    depth: float = 10_000.0,
    age_seconds: float = 0.0,
    now: datetime = NOW,
) -> MarketSnapshot:
    timestamp = now - timedelta(seconds=age_seconds)
    half = price * spread_bps / 2 / 10_000
    candles = tuple(
        Candle(
            symbol="BTCUSDT",
            interval="1h",
            open_time=timestamp - timedelta(hours=60 - i),
            open=price, high=price * 1.001, low=price * 0.999, close=price, volume=1_000.0,
        )
        for i in range(60)
    )
    return MarketSnapshot(
        symbol="BTCUSDT",
        timestamp=timestamp,
        candles=candles,
        ticker=Ticker(
            symbol="BTCUSDT", price=price, timestamp=timestamp,
            bid=price - half, ask=price + half,
        ),
        order_book=OrderBook(
            symbol="BTCUSDT",
            timestamp=timestamp,
            bids=tuple(
                OrderBookLevel(price=price - half * (i + 1), quantity=depth / 10)
                for i in range(10)
            ),
            asks=tuple(
                OrderBookLevel(price=price + half * (i + 1), quantity=depth / 10)
                for i in range(10)
            ),
        ),
    )


# =========================================================================== #
# Position sizing
# =========================================================================== #
class TestPositionSizing:
    def test_textbook_case(self, instrument: InstrumentSpec) -> None:
        """$10,000 equity, 0.5% risk, $2 stop distance -> ~25 units before costs."""
        spec = InstrumentSpec(
            symbol="X", base_asset="X", quote_asset="USDT",
            tick_size=0.01, lot_size=0.0001, min_notional=0.0,
            maker_fee=0.0, taker_fee=0.0,
        )
        result = calculate_position_size(
            SizingRequest(
                equity=10_000.0, available_margin=10_000.0,
                entry_price=100.0, stop_price=98.0, side=OrderSide.BUY,
                instrument=spec, risk_fraction=0.005, max_position_fraction=1.0,
            )
        )
        assert result.approved
        assert result.quantity == pytest.approx(25.0)
        assert result.risk_amount == pytest.approx(50.0)

    def test_fees_reduce_size(self, instrument: InstrumentSpec) -> None:
        """With fees included, the same risk budget buys strictly less."""
        free = InstrumentSpec(
            symbol="X", base_asset="X", quote_asset="USDT",
            tick_size=0.01, lot_size=0.0001, maker_fee=0.0, taker_fee=0.0,
        )
        costly = InstrumentSpec(
            symbol="X", base_asset="X", quote_asset="USDT",
            tick_size=0.01, lot_size=0.0001, maker_fee=0.001, taker_fee=0.002,
        )
        def size(spec: InstrumentSpec) -> float:
            return calculate_position_size(
                SizingRequest(
                    equity=10_000.0, available_margin=10_000.0,
                    entry_price=100.0, stop_price=98.0, side=OrderSide.BUY,
                    instrument=spec, risk_fraction=0.005, max_position_fraction=1.0,
                )
            ).quantity
        assert size(costly) < size(free)

    def test_slippage_reduces_size(self, instrument: InstrumentSpec) -> None:
        def size(slippage: float) -> float:
            return calculate_position_size(
                SizingRequest(
                    equity=100_000.0, available_margin=100_000.0,
                    entry_price=100.0, stop_price=98.0, side=OrderSide.BUY,
                    instrument=instrument, risk_fraction=0.01,
                    max_position_fraction=1.0, expected_slippage_bps=slippage,
                )
            ).quantity
        assert size(50.0) < size(0.0)

    def test_wider_stop_gives_smaller_size(self, instrument: InstrumentSpec) -> None:
        """The defining property of risk-based sizing."""
        def size(stop: float) -> float:
            return calculate_position_size(
                SizingRequest(
                    equity=100_000.0, available_margin=100_000.0,
                    entry_price=100.0, stop_price=stop, side=OrderSide.BUY,
                    instrument=instrument, risk_fraction=0.01, max_position_fraction=1.0,
                )
            ).quantity
        assert size(90.0) < size(98.0) < size(99.5)

    def test_position_fraction_caps_size(self, instrument: InstrumentSpec) -> None:
        result = calculate_position_size(
            SizingRequest(
                equity=100_000.0, available_margin=100_000.0,
                entry_price=100.0, stop_price=99.99, side=OrderSide.BUY,
                instrument=instrument, risk_fraction=0.01, max_position_fraction=0.1,
            )
        )
        assert result.approved
        assert result.notional <= 100_000.0 * 0.1 + 1e-6
        assert result.binding_constraint == "max_position_fraction"

    def test_margin_caps_size(self, instrument: InstrumentSpec) -> None:
        result = calculate_position_size(
            SizingRequest(
                equity=100_000.0, available_margin=500.0,
                entry_price=100.0, stop_price=99.9, side=OrderSide.BUY,
                instrument=instrument, risk_fraction=0.01,
                max_position_fraction=1.0, leverage=1.0, max_leverage=1.0,
            )
        )
        assert result.approved
        assert result.margin_required <= 500.0 + 1e-6
        assert result.binding_constraint == "available_margin"

    def test_rounds_down_never_up(self, instrument: InstrumentSpec) -> None:
        result = calculate_position_size(
            SizingRequest(
                equity=100_000.0, available_margin=100_000.0,
                entry_price=100.0, stop_price=98.0, side=OrderSide.BUY,
                instrument=instrument, risk_fraction=0.01, max_position_fraction=1.0,
            )
        )
        ideal = result.risk_amount / result.risk_per_unit
        assert result.quantity <= ideal
        assert result.quantity == pytest.approx(
            instrument.round_quantity(result.quantity)
        )

    def test_tiny_account_is_rejected_not_rounded_to_zero(
        self, instrument: InstrumentSpec
    ) -> None:
        result = calculate_position_size(
            SizingRequest(
                equity=5.0, available_margin=5.0,
                entry_price=100.0, stop_price=98.0, side=OrderSide.BUY,
                instrument=instrument, risk_fraction=0.005, max_position_fraction=1.0,
            )
        )
        assert not result.approved
        assert result.quantity == 0.0
        assert "rounds to zero" in result.reason or "minimum" in result.reason

    def test_min_notional_enforced(self, instrument: InstrumentSpec) -> None:
        result = calculate_position_size(
            SizingRequest(
                equity=500.0, available_margin=500.0,
                entry_price=100.0, stop_price=50.0, side=OrderSide.BUY,
                instrument=instrument, risk_fraction=0.005, max_position_fraction=1.0,
            )
        )
        assert not result.approved
        assert "notional" in result.reason.lower() or "zero" in result.reason

    def test_confidence_only_shrinks(self, instrument: InstrumentSpec) -> None:
        def size(confidence: float) -> float:
            return calculate_position_size(
                SizingRequest(
                    equity=100_000.0, available_margin=100_000.0,
                    entry_price=100.0, stop_price=98.0, side=OrderSide.BUY,
                    instrument=instrument, risk_fraction=0.01, max_position_fraction=1.0,
                    confidence=confidence, scale_by_confidence=True,
                )
            ).quantity
        full = size(1.0)
        assert size(0.5) < full
        assert size(5.0) == pytest.approx(full)  # cannot exceed the configured risk

    def test_short_side_sizing(self, instrument: InstrumentSpec) -> None:
        result = calculate_position_size(
            SizingRequest(
                equity=100_000.0, available_margin=100_000.0,
                entry_price=100.0, stop_price=102.0, side=OrderSide.SELL,
                instrument=instrument, risk_fraction=0.01, max_position_fraction=1.0,
            )
        )
        assert result.approved
        assert result.quantity > 0

    def test_inverted_stop_rejected(self, instrument: InstrumentSpec) -> None:
        with pytest.raises(ValueError, match="must be below entry"):
            SizingRequest(
                equity=100_000.0, available_margin=100_000.0,
                entry_price=100.0, stop_price=102.0, side=OrderSide.BUY,
                instrument=instrument, risk_fraction=0.01,
            )

    def test_worst_case_loss_matches_budget(self, instrument: InstrumentSpec) -> None:
        result = calculate_position_size(
            SizingRequest(
                equity=100_000.0, available_margin=100_000.0,
                entry_price=100.0, stop_price=98.0, side=OrderSide.BUY,
                instrument=instrument, risk_fraction=0.01, max_position_fraction=1.0,
                expected_slippage_bps=10.0,
            )
        )
        loss = estimate_worst_case_loss(
            result.quantity, 100.0, 98.0, instrument, slippage_bps=10.0
        )
        assert loss <= 100_000.0 * 0.01 + 1e-6

    def test_reward_risk_ratio(self) -> None:
        assert reward_risk_ratio(100.0, 98.0, 106.0) == pytest.approx(3.0)
        assert reward_risk_ratio(100.0, 98.0, None) is None
        assert reward_risk_ratio(100.0, 100.0, 106.0) is None

    def test_kelly_is_capped(self) -> None:
        assert kelly_fraction(0.99, 10.0) == pytest.approx(0.25)
        assert kelly_fraction(0.3, 1.0) == 0.0
        with pytest.raises(ValueError):
            kelly_fraction(1.5, 2.0)


# =========================================================================== #
# Risk state
# =========================================================================== #
class TestRiskState:
    def test_daily_loss_fraction(self) -> None:
        state = RiskState(starting_equity=10_000.0)
        state.mark_equity(10_000.0, now=NOW)
        state.mark_equity(9_800.0, now=NOW + timedelta(hours=1))
        assert state.daily_loss_fraction == pytest.approx(0.02)

    def test_profit_is_not_a_loss(self) -> None:
        state = RiskState(starting_equity=10_000.0)
        state.mark_equity(10_000.0, now=NOW)
        state.mark_equity(11_000.0, now=NOW + timedelta(hours=1))
        assert state.daily_loss_fraction == 0.0

    def test_drawdown_from_peak(self) -> None:
        state = RiskState(starting_equity=10_000.0)
        state.mark_equity(12_000.0, now=NOW)
        state.mark_equity(10_800.0, now=NOW + timedelta(hours=1))
        assert state.peak_equity == pytest.approx(12_000.0)
        assert state.drawdown == pytest.approx(0.10)

    def test_day_rollover_resets_counters(self) -> None:
        state = RiskState(starting_equity=10_000.0)
        state.mark_equity(10_000.0, now=NOW)
        state.record_order_submitted(now=NOW)
        assert state.trades_today == 1
        state.mark_equity(9_500.0, now=NOW + timedelta(days=1))
        assert state.trades_today == 0
        assert state.day_start_equity == pytest.approx(10_000.0)

    def test_week_boundary_is_monday(self) -> None:
        assert week_start(NOW).weekday() == 0
        assert day_start(NOW).hour == 0

    def test_loss_streak_counting(self) -> None:
        state = RiskState(starting_equity=10_000.0)
        for i in range(3):
            state.record_trade(
                TradeOutcome("BTCUSDT", -100.0, NOW + timedelta(minutes=i))
            )
        assert state.consecutive_losses == 3
        state.record_trade(TradeOutcome("BTCUSDT", 50.0, NOW + timedelta(minutes=10)))
        assert state.consecutive_losses == 0

    def test_cooldown_after_loss(self) -> None:
        state = RiskState(starting_equity=10_000.0)
        state.record_trade(TradeOutcome("BTCUSDT", -100.0, NOW))
        active, remaining = state.in_cooldown(300, now=NOW + timedelta(seconds=60))
        assert active and remaining == pytest.approx(240.0)
        expired, _ = state.in_cooldown(300, now=NOW + timedelta(seconds=400))
        assert not expired

    def test_cooldown_ignores_wins_when_configured(self) -> None:
        state = RiskState(starting_equity=10_000.0)
        state.record_trade(TradeOutcome("BTCUSDT", 100.0, NOW))
        active, _ = state.in_cooldown(300, after_loss_only=True, now=NOW)
        assert not active
        active_any, _ = state.in_cooldown(300, after_loss_only=False, now=NOW)
        assert active_any

    def test_round_trips_through_dict(self) -> None:
        state = RiskState(starting_equity=10_000.0)
        state.mark_equity(9_500.0, now=NOW)
        state.record_trade(TradeOutcome("BTCUSDT", -100.0, NOW))
        restored = RiskState.from_dict(state.to_dict())
        assert restored.peak_equity == pytest.approx(state.peak_equity)
        assert restored.consecutive_losses == state.consecutive_losses
        assert restored.trades_today == state.trades_today


# =========================================================================== #
# Kill switch
# =========================================================================== #
class TestKillSwitch:
    def test_starts_clear(self) -> None:
        assert KillSwitch().is_active is False

    def test_engages_and_blocks(self) -> None:
        switch = KillSwitch()
        switch.engage(KillSwitchReason.MANUAL, "operator stopped the bot")
        assert switch.is_active
        with pytest.raises(KillSwitchActiveError):
            switch.require_clear()

    def test_first_cause_is_preserved(self) -> None:
        switch = KillSwitch()
        switch.engage(KillSwitchReason.MAX_DRAWDOWN, "drawdown breached")
        switch.engage(KillSwitchReason.API_FAILURES, "downstream noise")
        assert switch.reason is KillSwitchReason.MAX_DRAWDOWN

    def test_reset_requires_an_actor(self) -> None:
        switch = KillSwitch()
        switch.engage(KillSwitchReason.MANUAL, "stopped")
        with pytest.raises(ValueError, match="reset_by is required"):
            switch.reset(reset_by="")

    def test_reset_clears(self) -> None:
        switch = KillSwitch()
        switch.engage(KillSwitchReason.MANUAL, "stopped")
        switch.reset(reset_by="operator@example.com", note="investigated")
        assert not switch.is_active
        switch.require_clear()

    def test_loss_trips_never_auto_reset(self) -> None:
        switch = KillSwitch(auto_reset_after=timedelta(seconds=0))
        switch.engage(KillSwitchReason.DAILY_LOSS, "daily loss limit")
        assert switch.is_active

    def test_transient_trips_can_auto_reset(self) -> None:
        switch = KillSwitch(auto_reset_after=timedelta(seconds=0))
        switch.engage(KillSwitchReason.STALE_MARKET_DATA, "feed went quiet")
        assert switch.is_active is False

    def test_history_is_retained(self) -> None:
        switch = KillSwitch()
        switch.engage(KillSwitchReason.MANUAL, "one")
        switch.reset(reset_by="op")
        switch.engage(KillSwitchReason.DAILY_LOSS, "two")
        assert len(switch.history) == 2


# =========================================================================== #
# Risk limits
# =========================================================================== #
class TestRiskLimits:
    def test_bot_limits_cannot_loosen_the_ceiling(self) -> None:
        ceiling = RiskLimits(risk_per_trade=0.005, max_leverage=2.0, max_daily_loss=0.02)
        greedy = RiskLimits(
            risk_per_trade=0.05, max_leverage=20.0, max_daily_loss=0.5,
            max_weekly_loss=0.8, max_drawdown=0.9, max_concurrent_positions=5,
        )
        clamped = greedy.clamped_to(ceiling)
        assert clamped.risk_per_trade == pytest.approx(0.005)
        assert clamped.max_leverage == pytest.approx(2.0)
        assert clamped.max_daily_loss == pytest.approx(0.02)

    def test_tighter_bot_limits_are_preserved(self) -> None:
        ceiling = RiskLimits(risk_per_trade=0.02, max_leverage=5.0)
        careful = RiskLimits(risk_per_trade=0.001, max_leverage=1.0)
        clamped = careful.clamped_to(ceiling)
        assert clamped.risk_per_trade == pytest.approx(0.001)
        assert clamped.max_leverage == pytest.approx(1.0)

    def test_minimums_clamp_upward(self) -> None:
        ceiling = RiskLimits(min_liquidity_multiple=20.0, cooldown_seconds=600)
        loose = RiskLimits(min_liquidity_multiple=1.0, cooldown_seconds=0)
        clamped = loose.clamped_to(ceiling)
        assert clamped.min_liquidity_multiple == pytest.approx(20.0)
        assert clamped.cooldown_seconds == 600

    def test_daily_cannot_exceed_weekly(self) -> None:
        with pytest.raises(ValueError, match="max_daily_loss"):
            RiskLimits(max_daily_loss=0.1, max_weekly_loss=0.05)

    def test_weekly_cannot_exceed_drawdown(self) -> None:
        with pytest.raises(ValueError, match="max_weekly_loss"):
            RiskLimits(max_daily_loss=0.01, max_weekly_loss=0.5, max_drawdown=0.1)

    def test_incoherent_aggregate_risk_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_concurrent_positions"):
            RiskLimits(
                risk_per_trade=0.05, max_concurrent_positions=20,
                max_daily_loss=0.02, max_weekly_loss=0.06, max_drawdown=0.1,
            )

    def test_conservative_preset_is_coherent(self) -> None:
        preset = RiskLimits.conservative()
        assert preset.max_simultaneous_risk <= preset.max_drawdown * 2


# =========================================================================== #
# Risk manager
# =========================================================================== #
class TestRiskManager:
    def test_approves_a_clean_trade(
        self, manager: RiskManager, portfolio: PortfolioView, instrument: InstrumentSpec
    ) -> None:
        assessment = manager.evaluate(
            proposal(instrument), portfolio, snapshot_with(), now=NOW
        )
        assert assessment.approved
        assert assessment.quantity > 0
        assert assessment.sizing is not None

    def test_approved_size_respects_the_risk_budget(
        self, manager: RiskManager, portfolio: PortfolioView, instrument: InstrumentSpec
    ) -> None:
        assessment = manager.evaluate(
            proposal(instrument), portfolio, snapshot_with(), now=NOW
        )
        assert assessment.sizing is not None
        loss = estimate_worst_case_loss(
            assessment.quantity, 100.0, 98.0, instrument,
            slippage_bps=assessment.sizing.assumed_slippage_bps,
        )
        assert loss <= portfolio.equity * manager.limits.risk_per_trade * 1.05

    def test_sizing_assumes_worse_slippage_than_it_measures(
        self, manager: RiskManager, portfolio: PortfolioView, instrument: InstrumentSpec
    ) -> None:
        """The safety factor must actually be applied, not merely configured."""
        assessment = manager.evaluate(
            proposal(instrument), portfolio, snapshot_with(spread_bps=4.0), now=NOW
        )
        assert assessment.sizing is not None
        # The book implies ~2 bps; the size must be computed against strictly more.
        assert assessment.sizing.assumed_slippage_bps >= manager.limits.min_slippage_bps
        assert assessment.sizing.assumed_slippage_bps > 2.0

    def test_costly_tight_stop_is_rejected(
        self, manager: RiskManager, portfolio: PortfolioView, instrument: InstrumentSpec
    ) -> None:
        """A stop so tight that fees and spread dominate it has no realistic edge."""
        assessment = manager.evaluate(
            proposal(instrument, entry=100.0, stop=99.9, target=None),
            portfolio, snapshot_with(spread_bps=10.0), now=NOW,
        )
        assert not assessment.approved
        assert "Execution costs" in assessment.reason

    def test_non_entry_actions_are_rejected(
        self, manager: RiskManager, portfolio: PortfolioView, instrument: InstrumentSpec
    ) -> None:
        assessment = manager.evaluate(
            TradeProposal(
                symbol="BTCUSDT", action=SignalAction.HOLD, entry_price=100.0,
                stop_loss=98.0, instrument=instrument,
            ),
            portfolio, snapshot_with(), now=NOW,
        )
        assert not assessment.approved

    def test_kill_switch_blocks_everything(
        self, manager: RiskManager, portfolio: PortfolioView, instrument: InstrumentSpec
    ) -> None:
        manager.emergency_stop("operator@example.com", "market conditions")
        assessment = manager.evaluate(
            proposal(instrument), portfolio, snapshot_with(), now=NOW
        )
        assert assessment.decision is RiskDecision.REJECTED
        assert assessment.blocked_by is RiskEventType.KILL_SWITCH

    def test_daily_loss_limit_blocks_and_trips_switch(
        self, manager: RiskManager, instrument: InstrumentSpec
    ) -> None:
        manager.state.mark_equity(100_000.0, now=NOW)
        low = PortfolioView(equity=97_500.0, available_margin=97_500.0)
        assessment = manager.evaluate(proposal(instrument), low, snapshot_with(), now=NOW)
        assert assessment.blocked_by is RiskEventType.DAILY_LOSS_LIMIT
        assert manager.kill_switch.reason is KillSwitchReason.DAILY_LOSS

    def test_drawdown_limit_blocks(
        self, manager: RiskManager, instrument: InstrumentSpec
    ) -> None:
        """A slow bleed below the high-water mark, with today's loss still inside its limit."""
        # The peak is weeks old, so the daily and weekly windows are both quiet; only the
        # distance from the high-water mark breaches.
        manager.state.mark_equity(120_000.0, now=NOW - timedelta(days=20))
        manager.state.mark_equity(108_000.0, now=NOW - timedelta(days=8))
        manager.state.mark_equity(108_000.0, now=NOW - timedelta(days=2))
        manager.state.mark_equity(107_500.0, now=NOW)
        low = PortfolioView(equity=107_500.0, available_margin=107_500.0)
        assessment = manager.evaluate(proposal(instrument), low, snapshot_with(), now=NOW)
        assert manager.state.daily_loss_fraction < manager.limits.max_daily_loss
        assert manager.state.weekly_loss_fraction < manager.limits.max_weekly_loss
        assert assessment.blocked_by is RiskEventType.MAX_DRAWDOWN

    def test_loss_streak_blocks(
        self, manager: RiskManager, portfolio: PortfolioView, instrument: InstrumentSpec
    ) -> None:
        for i in range(4):
            manager.state.record_trade(
                TradeOutcome("BTCUSDT", -10.0, NOW - timedelta(minutes=10 - i))
            )
        assessment = manager.evaluate(
            proposal(instrument), portfolio, snapshot_with(), now=NOW
        )
        assert assessment.blocked_by is RiskEventType.LOSS_STREAK

    def test_cooldown_blocks(
        self, limits: RiskLimits, state: RiskState, portfolio: PortfolioView,
        instrument: InstrumentSpec,
    ) -> None:
        cooled = limits.model_copy(update={"cooldown_seconds": 600})
        state.record_trade(TradeOutcome("BTCUSDT", -10.0, NOW - timedelta(seconds=60)))
        manager = RiskManager(cooled, state)
        assessment = manager.evaluate(
            proposal(instrument), portfolio, snapshot_with(), now=NOW
        )
        assert assessment.blocked_by is RiskEventType.COOLDOWN

    def test_daily_trade_limit_blocks(
        self, manager: RiskManager, portfolio: PortfolioView, instrument: InstrumentSpec
    ) -> None:
        manager.state.mark_equity(100_000.0, now=NOW)
        for _ in range(30):
            manager.state.record_order_submitted(now=NOW)
        assessment = manager.evaluate(
            proposal(instrument), portfolio, snapshot_with(), now=NOW
        )
        assert assessment.blocked_by is RiskEventType.TRADE_COUNT_LIMIT

    def test_position_count_limit_blocks(
        self, manager: RiskManager, instrument: InstrumentSpec
    ) -> None:
        positions = tuple(
            Position(
                symbol=f"SYM{i}USDT", side=PositionSide.LONG, quantity=1.0,
                entry_price=100.0, mark_price=100.0,
            )
            for i in range(5)
        )
        crowded = PortfolioView(
            equity=100_000.0, available_margin=90_000.0,
            positions=positions, total_exposure=500.0,
        )
        assessment = manager.evaluate(
            proposal(instrument), crowded, snapshot_with(), now=NOW
        )
        assert assessment.blocked_by is RiskEventType.POSITION_LIMIT

    def test_adding_to_existing_position_bypasses_count_limit(
        self, manager: RiskManager, instrument: InstrumentSpec
    ) -> None:
        positions = tuple(
            [
                Position(
                    symbol="BTCUSDT", side=PositionSide.LONG, quantity=0.1,
                    entry_price=100.0, mark_price=100.0,
                )
            ]
            + [
                Position(
                    symbol=f"SYM{i}USDT", side=PositionSide.LONG, quantity=0.1,
                    entry_price=100.0, mark_price=100.0,
                )
                for i in range(4)
            ]
        )
        crowded = PortfolioView(
            equity=100_000.0, available_margin=90_000.0,
            positions=positions, total_exposure=50.0,
        )
        assessment = manager.evaluate(
            proposal(instrument), crowded, snapshot_with(), now=NOW
        )
        assert assessment.blocked_by is not RiskEventType.POSITION_LIMIT

    def test_portfolio_exposure_limit_blocks(
        self, limits: RiskLimits, state: RiskState, instrument: InstrumentSpec
    ) -> None:
        tight = limits.model_copy(update={"max_portfolio_exposure": 0.5})
        manager = RiskManager(tight, state)
        loaded = PortfolioView(
            equity=100_000.0, available_margin=40_000.0, total_exposure=60_000.0,
        )
        assessment = manager.evaluate(
            proposal(instrument), loaded, snapshot_with(), now=NOW
        )
        assert assessment.blocked_by is RiskEventType.EXPOSURE_LIMIT

    def test_asset_exposure_limit_blocks(
        self, limits: RiskLimits, state: RiskState, instrument: InstrumentSpec
    ) -> None:
        tight = limits.model_copy(update={"max_asset_exposure": 0.05})
        manager = RiskManager(tight, state)
        loaded = PortfolioView(
            equity=100_000.0,
            available_margin=90_000.0,
            positions=(
                Position(
                    symbol="BTCUSDT", side=PositionSide.LONG, quantity=100.0,
                    entry_price=100.0, mark_price=100.0,
                ),
            ),
            total_exposure=10_000.0,
        )
        assessment = manager.evaluate(
            proposal(instrument), loaded, snapshot_with(), now=NOW
        )
        assert assessment.blocked_by is RiskEventType.EXPOSURE_LIMIT

    def test_stale_data_blocks_and_trips_switch(
        self, manager: RiskManager, portfolio: PortfolioView, instrument: InstrumentSpec
    ) -> None:
        stale = snapshot_with(age_seconds=8_000)
        assessment = manager.evaluate(proposal(instrument), portfolio, stale, now=NOW)
        assert assessment.blocked_by is RiskEventType.STALE_DATA
        assert manager.kill_switch.reason is KillSwitchReason.STALE_MARKET_DATA

    def test_wide_spread_blocks(
        self, manager: RiskManager, portfolio: PortfolioView, instrument: InstrumentSpec
    ) -> None:
        wide = snapshot_with(spread_bps=80.0)
        assessment = manager.evaluate(proposal(instrument), portfolio, wide, now=NOW)
        assert assessment.blocked_by in {RiskEventType.SPREAD, RiskEventType.LIMIT_BREACH}

    def test_thin_book_blocks(
        self, limits: RiskLimits, state: RiskState, instrument: InstrumentSpec
    ) -> None:
        strict = limits.model_copy(update={"min_liquidity_multiple": 1000.0})
        manager = RiskManager(strict, state)
        portfolio = PortfolioView(equity=100_000.0, available_margin=100_000.0)
        assessment = manager.evaluate(
            proposal(instrument), portfolio, snapshot_with(depth=10.0), now=NOW
        )
        assert not assessment.approved

    def test_poor_reward_risk_blocks(
        self, limits: RiskLimits, state: RiskState, instrument: InstrumentSpec
    ) -> None:
        strict = limits.model_copy(update={"min_reward_risk": 3.0})
        manager = RiskManager(strict, state)
        portfolio = PortfolioView(equity=100_000.0, available_margin=100_000.0)
        assessment = manager.evaluate(
            proposal(instrument, entry=100.0, stop=98.0, target=101.0),
            portfolio, snapshot_with(), now=NOW,
        )
        assert not assessment.approved
        assert "Reward/risk" in assessment.reason

    def test_missing_target_is_a_warning_not_a_block(
        self, limits: RiskLimits, state: RiskState, instrument: InstrumentSpec
    ) -> None:
        strict = limits.model_copy(update={"min_reward_risk": 2.0})
        manager = RiskManager(strict, state)
        portfolio = PortfolioView(equity=100_000.0, available_margin=100_000.0)
        assessment = manager.evaluate(
            proposal(instrument, target=None), portfolio, snapshot_with(), now=NOW
        )
        assert assessment.approved
        assert any("take-profit" in w for w in assessment.warnings)

    def test_rejected_assessment_raises_when_acted_on(
        self, manager: RiskManager, portfolio: PortfolioView, instrument: InstrumentSpec
    ) -> None:
        manager.emergency_stop("operator")
        assessment = manager.evaluate(
            proposal(instrument), portfolio, snapshot_with(), now=NOW
        )
        with pytest.raises(RiskViolationError):
            assessment.require_approval()

    def test_platform_ceiling_is_applied_at_construction(
        self, state: RiskState, instrument: InstrumentSpec
    ) -> None:
        greedy = RiskLimits(
            risk_per_trade=0.04, max_concurrent_positions=2,
            max_daily_loss=0.1, max_weekly_loss=0.2, max_drawdown=0.3,
        )
        ceiling = RiskLimits(risk_per_trade=0.002)
        manager = RiskManager(greedy, state, platform_ceiling=ceiling)
        assert manager.limits.risk_per_trade == pytest.approx(0.002)

    def test_events_are_recorded_for_rejections(
        self, manager: RiskManager, portfolio: PortfolioView, instrument: InstrumentSpec
    ) -> None:
        manager.emergency_stop("operator")
        manager.evaluate(proposal(instrument), portfolio, snapshot_with(), now=NOW)
        events = manager.drain_events()
        assert len(events) == 1
        assert events[0].event_type is RiskEventType.KILL_SWITCH
        assert manager.drain_events() == []

    def test_api_failures_trip_the_switch(self, manager: RiskManager) -> None:
        for _ in range(5):
            manager.on_api_failure("connection reset")
        assert manager.kill_switch.reason is KillSwitchReason.API_FAILURES

    def test_api_success_resets_the_counter(self, manager: RiskManager) -> None:
        for _ in range(4):
            manager.on_api_failure("timeout")
        manager.on_api_success()
        manager.on_api_failure("timeout")
        assert not manager.kill_switch.is_active

    def test_clock_drift_trips_the_switch(self, manager: RiskManager) -> None:
        manager.on_clock_drift(0.5)
        assert not manager.kill_switch.is_active
        manager.on_clock_drift(30.0)
        assert manager.kill_switch.reason is KillSwitchReason.CLOCK_DRIFT

    def test_reconciliation_failure_trips_the_switch(self, manager: RiskManager) -> None:
        manager.on_reconciliation_failure("local shows 1.5 BTC, exchange shows 0.5 BTC")
        assert manager.kill_switch.reason is KillSwitchReason.EXCHANGE_DESYNC

    def test_describe_reports_headroom(self, manager: RiskManager) -> None:
        manager.state.mark_equity(100_000.0, now=NOW)
        described = manager.describe()
        assert described["headroom"]["trades_remaining_today"] == 30
        assert "limits" in described and "kill_switch" in described


# =========================================================================== #
# Cross-cutting invariants
# =========================================================================== #
@pytest.mark.parametrize("risk_fraction", [0.001, 0.005, 0.01, 0.02])
@pytest.mark.parametrize("stop_distance", [0.5, 2.0, 10.0])
def test_no_configuration_exceeds_the_risk_budget(
    instrument: InstrumentSpec, risk_fraction: float, stop_distance: float
) -> None:
    """Exhaustive property: approved size never risks more than configured."""
    limits = RiskLimits(
        risk_per_trade=risk_fraction,
        max_concurrent_positions=3,
        max_daily_loss=0.02,
        max_weekly_loss=0.06,
        max_drawdown=0.10,
        max_position_fraction=1.0,
        max_portfolio_exposure=5.0,
        cooldown_seconds=0,
        min_reward_risk=0.0,
    )
    manager = RiskManager(limits, RiskState(starting_equity=100_000.0))
    portfolio = PortfolioView(equity=100_000.0, available_margin=100_000.0)
    assessment = manager.evaluate(
        proposal(instrument, entry=100.0, stop=100.0 - stop_distance, target=None),
        portfolio,
        snapshot_with(),
        now=NOW,
    )
    if assessment.approved:
        assert assessment.sizing is not None
        loss = estimate_worst_case_loss(
            assessment.quantity, 100.0, 100.0 - stop_distance, instrument,
            slippage_bps=assessment.sizing.assumed_slippage_bps,
        )
        assert loss <= portfolio.equity * risk_fraction * 1.05


def test_manager_never_mutates_the_portfolio_view(
    manager: RiskManager, instrument: InstrumentSpec
) -> None:
    portfolio = PortfolioView(
        equity=100_000.0,
        available_margin=100_000.0,
        positions=(
            Position(
                symbol="ETHUSDT", side=PositionSide.LONG, quantity=1.0,
                entry_price=100.0, mark_price=100.0,
            ),
        ),
        total_exposure=100.0,
    )
    before = (portfolio.equity, portfolio.position_count, portfolio.total_exposure)
    manager.evaluate(proposal(instrument), portfolio, snapshot_with(), now=NOW)
    assert (portfolio.equity, portfolio.position_count, portfolio.total_exposure) == before
