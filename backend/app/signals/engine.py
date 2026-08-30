"""Signal engine.

The single place where a strategy's opinion becomes a decision. It is the only component
permitted to combine strategy output with regime, news, liquidity, data quality and portfolio
state, and it produces exactly one :class:`Signal` per symbol per bar.

Order of evaluation — each stage can only ever *reduce* what is permitted:

1. **Data quality.** Bad or stale data ⇒ ``NO_TRADE``. Nothing downstream runs.
2. **Regime.** Detected once per bar and handed to the strategy, which applies its own mandate.
3. **Strategy.** Produces an opinion, or declines.
4. **Exit handling.** ``CLOSE`` signals pass through with minimal filtering — reducing exposure
   is always allowed, and blocking an exit because the spread widened would be backwards.
5. **News.** Adjusts confidence and can veto, never originates. A signal that exists only
   because of a headline is not a signal.
6. **Liquidity and cost.** A trade whose expected execution cost swamps its edge is dropped.
7. **Portfolio.** Duplicate and conflicting positions are filtered here, before the risk
   manager, so that risk sees only coherent proposals.

Sizing is *not* here. The engine says what to do; the risk manager says how much.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.core.clock import utcnow
from app.core.domain import Position
from app.core.enums import MarketRegime, PositionSide, SignalAction
from app.core.logging import get_logger
from app.core.numeric import clamp
from app.market_data.models import MarketSnapshot
from app.market_data.validation import MarketDataValidator
from app.news.scoring import NewsAssessment
from app.regimes.detector import MarketRegimeDetector, RegimeDetection
from app.strategies.base import Strategy, StrategyContext, StrategyResult

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Signal:
    """The pipeline's decision for one symbol at one instant."""

    action: SignalAction
    symbol: str
    timestamp: datetime
    strategy_name: str
    confidence: float = 0.0
    entry: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    regime: MarketRegime = MarketRegime.UNKNOWN
    regime_confidence: float = 0.0
    news_score: float = 0.0
    reason: str = ""
    blocked_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.action in {SignalAction.BUY, SignalAction.SELL, SignalAction.CLOSE}

    @property
    def is_entry(self) -> bool:
        return self.action.is_entry

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "strategy": self.strategy_name,
            "confidence": round(self.confidence, 4),
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "regime": self.regime.value,
            "regime_confidence": round(self.regime_confidence, 4),
            "news_score": round(self.news_score, 4),
            "reason": self.reason,
            "blocked_by": self.blocked_by,
        }


@dataclass(slots=True)
class SignalEngineConfig:
    """Tuning for the engine's own filters (distinct from strategy or risk parameters)."""

    #: Minimum confidence after all adjustments for an entry to survive.
    min_confidence: float = 0.35
    #: How strongly news moves confidence. 0 disables news influence entirely.
    news_weight: float = 0.25
    #: News impact at or above this, opposing the signal, vetoes the entry.
    news_veto_threshold: float = 0.6
    #: Maximum acceptable spread for an entry, in basis points.
    max_spread_bps: float = 25.0
    #: Bars of history required before the engine will evaluate anything.
    min_history: int = 100
    #: Allow a signal that would reverse an existing position (close then open opposite).
    allow_reversals: bool = False
    #: Re-detect the regime every bar. Disabling caches it, for expensive backtests.
    detect_regime: bool = True


class SignalEngine:
    """Combines every input into a single decision per symbol."""

    def __init__(
        self,
        strategy: Strategy,
        *,
        config: SignalEngineConfig | None = None,
        regime_detector: MarketRegimeDetector | None = None,
        validator: MarketDataValidator | None = None,
    ) -> None:
        self.strategy = strategy
        self.config = config or SignalEngineConfig()
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self.validator = validator or MarketDataValidator(
            min_required_bars=self.config.min_history
        )
        self._last_regime: dict[str, RegimeDetection] = {}

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        snapshot: MarketSnapshot,
        *,
        position: Position | None = None,
        news: NewsAssessment | None = None,
        bars_since_last_trade: int | None = None,
        now: datetime | None = None,
    ) -> Signal:
        """Produce the decision for one symbol."""
        moment = now or snapshot.timestamp or utcnow()
        symbol = snapshot.symbol

        # --- 1. data quality ------------------------------------------------
        validation = self.validator.validate_snapshot(
            snapshot,
            now=moment,
            required_bars=max(self.config.min_history, self.strategy.required_history),
        )
        if not validation.is_valid:
            return self._no_trade(
                symbol, moment,
                reason=f"market data unusable: {validation.reason()}",
                blocked_by="data_quality",
            )

        # --- 2. regime -------------------------------------------------------
        detection = self._detect_regime(snapshot)

        # --- 3. strategy -----------------------------------------------------
        context = StrategyContext(
            snapshot=snapshot,
            regime=detection.regime,
            regime_confidence=detection.confidence,
            position=position,
            news_score=news.directional_score if news else 0.0,
            bars_since_last_trade=bars_since_last_trade,
            now=moment,
        )
        result = self.strategy.generate_signal(context)

        # --- 4. exits pass through -------------------------------------------
        if result.action is SignalAction.CLOSE:
            return self._from_strategy(
                result, detection, news, moment,
                reason=f"exit: {result.reason}",
            )
        if result.action in {SignalAction.HOLD, SignalAction.NO_TRADE}:
            return self._from_strategy(result, detection, news, moment)

        # --- 5. news ----------------------------------------------------------
        adjusted_confidence, news_note, vetoed = self._apply_news(result, news)
        if vetoed:
            return self._no_trade(
                symbol, moment,
                reason=news_note,
                blocked_by="news_conflict",
                regime=detection,
                news_score=news.directional_score if news else 0.0,
                strategy_name=self.strategy.name,
            )

        # --- 6. execution cost -------------------------------------------------
        cost_block = self._check_execution_cost(snapshot)
        if cost_block is not None:
            return self._no_trade(
                symbol, moment, reason=cost_block, blocked_by="execution_cost",
                regime=detection, strategy_name=self.strategy.name,
            )

        # --- 7. portfolio coherence ---------------------------------------------
        portfolio_block = self._check_portfolio(result, position)
        if portfolio_block is not None:
            return self._no_trade(
                symbol, moment, reason=portfolio_block, blocked_by="portfolio",
                regime=detection, strategy_name=self.strategy.name,
            )

        # --- final confidence gate ----------------------------------------------
        if adjusted_confidence < self.config.min_confidence:
            return self._no_trade(
                symbol, moment,
                reason=(
                    f"confidence {adjusted_confidence:.2f} below the engine threshold "
                    f"{self.config.min_confidence:.2f}{news_note}"
                ),
                blocked_by="low_confidence",
                regime=detection,
                strategy_name=self.strategy.name,
            )

        signal = Signal(
            action=result.action,
            symbol=symbol,
            timestamp=moment,
            strategy_name=result.strategy_name,
            confidence=adjusted_confidence,
            entry=result.entry,
            stop_loss=result.stop_loss,
            take_profit=result.take_profit,
            regime=detection.regime,
            regime_confidence=detection.confidence,
            news_score=news.directional_score if news else 0.0,
            reason=f"{result.reason}{news_note}",
            metadata={
                **result.metadata,
                "regime_reason": detection.reason,
                "strategy_confidence": round(result.confidence, 4),
            },
        )
        logger.info(
            "signal.generated",
            symbol=symbol,
            action=signal.action.value,
            confidence=round(signal.confidence, 3),
            regime=signal.regime.value,
            strategy=signal.strategy_name,
        )
        return signal

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #
    def _detect_regime(self, snapshot: MarketSnapshot) -> RegimeDetection:
        if not self.config.detect_regime:
            cached = self._last_regime.get(snapshot.symbol)
            if cached is not None:
                return cached
        detection = self.regime_detector.detect(snapshot.candles, now=snapshot.timestamp)
        self._last_regime[snapshot.symbol] = detection
        return detection

    def _apply_news(
        self, result: StrategyResult, news: NewsAssessment | None
    ) -> tuple[float, str, bool]:
        """Adjust confidence for news, and decide whether news vetoes the trade.

        News can shift confidence and can block. It can never create a signal, because this
        function is only ever reached when the strategy has already produced one.
        """
        confidence = result.confidence
        if news is None or self.config.news_weight <= 0 or not news.has_coverage:
            return confidence, "", False

        direction = 1.0 if result.action is SignalAction.BUY else -1.0
        alignment = news.directional_score * direction  # +1 fully supportive, -1 opposed

        if (
            alignment < 0
            and news.max_impact >= self.config.news_veto_threshold
            and news.confidence >= 0.5
        ):
            return (
                confidence,
                (
                    f"high-impact news contradicts the {result.action.value} signal "
                    f"(news score {news.directional_score:+.2f}, impact "
                    f"{news.max_impact:.2f}): {news.headline_summary}"
                ),
                True,
            )

        adjustment = alignment * self.config.news_weight * news.confidence
        adjusted = clamp(confidence + adjustment, 0.0, 1.0)
        note = (
            f"; news {news.directional_score:+.2f} "
            f"({'supports' if alignment >= 0 else 'opposes'}, "
            f"confidence {confidence:.2f} -> {adjusted:.2f})"
        )
        return adjusted, note, False

    def _check_execution_cost(self, snapshot: MarketSnapshot) -> str | None:
        ticker = snapshot.ticker
        if ticker is None:
            return None
        spread = ticker.spread_bps
        if spread is not None and spread > self.config.max_spread_bps:
            return (
                f"spread {spread:.1f} bps exceeds the engine limit of "
                f"{self.config.max_spread_bps:.1f} bps"
            )
        return None

    def _check_portfolio(
        self, result: StrategyResult, position: Position | None
    ) -> str | None:
        if position is None or not position.is_open:
            return None
        going_long = result.action is SignalAction.BUY
        aligned = (position.side is PositionSide.LONG) == going_long
        if aligned:
            return (
                f"already holding a {position.side.value} position in this symbol; "
                "the strategy does not pyramid"
            )
        if not self.config.allow_reversals:
            return (
                f"signal would reverse an open {position.side.value} position; "
                "reversals are disabled, close the position first"
            )
        return None

    # ------------------------------------------------------------------ #
    # Builders
    # ------------------------------------------------------------------ #
    def _from_strategy(
        self,
        result: StrategyResult,
        detection: RegimeDetection,
        news: NewsAssessment | None,
        moment: datetime,
        *,
        reason: str | None = None,
    ) -> Signal:
        return Signal(
            action=result.action,
            symbol=result.symbol,
            timestamp=moment,
            strategy_name=result.strategy_name,
            confidence=result.confidence,
            entry=result.entry,
            stop_loss=result.stop_loss,
            take_profit=result.take_profit,
            regime=detection.regime,
            regime_confidence=detection.confidence,
            news_score=news.directional_score if news else 0.0,
            reason=reason or result.reason,
            metadata={**result.metadata, "regime_reason": detection.reason},
        )

    def _no_trade(
        self,
        symbol: str,
        moment: datetime,
        *,
        reason: str,
        blocked_by: str,
        regime: RegimeDetection | None = None,
        news_score: float = 0.0,
        strategy_name: str | None = None,
    ) -> Signal:
        return Signal(
            action=SignalAction.NO_TRADE,
            symbol=symbol,
            timestamp=moment,
            strategy_name=strategy_name or self.strategy.name,
            regime=regime.regime if regime else MarketRegime.UNKNOWN,
            regime_confidence=regime.confidence if regime else 0.0,
            news_score=news_score,
            reason=reason,
            blocked_by=blocked_by,
        )
