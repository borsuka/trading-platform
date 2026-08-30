"""Strategy framework.

A strategy is a **pure opinion function**. Given a market snapshot it returns a
:class:`StrategyResult` describing what it would like to do and why. It has no reference to an
exchange, a portfolio or an order manager, and there is no code path by which it can place a
trade. That is enforced structurally: nothing in this module imports the execution layer.

The pipeline is::

    Strategy -> SignalEngine -> RiskManager -> OrderManager -> Exchange

Every strategy declares:

* a typed, validated parameter model,
* how much history it needs before it can speak (``required_history``),
* which market regimes it is permitted to trade in.

A strategy that is asked for a signal with insufficient history returns ``NO_TRADE`` with a
reason, rather than computing an indicator on a short window and pretending.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.clock import utcnow
from app.core.domain import Position
from app.core.enums import MarketRegime, PositionSide, SignalAction
from app.core.exceptions import StrategyConfigurationError
from app.core.numeric import safe_divide
from app.market_data.models import Candle, MarketSnapshot


class StrategyParameters(BaseModel):
    """Base class for strategy parameter models.

    ``extra="forbid"`` is deliberate: a typo in a parameter name must be an error, not a
    silently ignored key that leaves the strategy running on defaults.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=False)

    risk_reward_ratio: float = Field(
        default=2.0, gt=0.1, le=20.0, description="Take-profit distance / stop distance"
    )
    atr_period: int = Field(default=14, ge=2, le=200)
    atr_stop_multiplier: float = Field(
        default=2.0, gt=0.1, le=20.0, description="Stop distance in ATRs"
    )
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    cooldown_bars: int = Field(default=0, ge=0, le=1000)
    allow_long: bool = True
    allow_short: bool = True

    @model_validator(mode="after")
    def _at_least_one_direction(self) -> StrategyParameters:
        if not self.allow_long and not self.allow_short:
            raise ValueError("At least one of allow_long / allow_short must be enabled")
        return self


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Everything a strategy is allowed to see.

    Deliberately narrow. A strategy cannot reach the portfolio, the balance or the order book
    depth beyond what the snapshot carries — those belong to risk and execution.
    """

    snapshot: MarketSnapshot
    regime: MarketRegime = MarketRegime.UNKNOWN
    regime_confidence: float = 0.0
    position: Position | None = None
    news_score: float = 0.0
    bars_since_last_trade: int | None = None
    now: datetime = field(default_factory=utcnow)

    @property
    def symbol(self) -> str:
        return self.snapshot.symbol

    @property
    def candles(self) -> tuple[Candle, ...]:
        return self.snapshot.candles

    @property
    def price(self) -> float | None:
        return self.snapshot.price

    @property
    def has_position(self) -> bool:
        return self.position is not None and self.position.is_open


@dataclass(frozen=True, slots=True)
class StrategyResult:
    """A strategy's opinion.

    ``entry``, ``stop_loss`` and ``take_profit`` are *proposals*. Position sizing, exposure and
    every limit check happen later; a strategy never decides how much to trade.
    """

    action: SignalAction
    symbol: str
    strategy_name: str
    timestamp: datetime
    confidence: float = 0.0
    entry: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.action.is_entry:
            if self.entry is None:
                raise ValueError(f"{self.action} requires an entry price")
            if self.stop_loss is None:
                raise ValueError(
                    f"{self.action} requires a stop loss: an entry without a defined "
                    "invalidation level cannot be risk-sized"
                )
            self._validate_level_ordering()

    def _validate_level_ordering(self) -> None:
        assert self.entry is not None and self.stop_loss is not None
        if self.action is SignalAction.BUY:
            if self.stop_loss >= self.entry:
                raise ValueError(
                    f"Long stop {self.stop_loss:g} must be below entry {self.entry:g}"
                )
            if self.take_profit is not None and self.take_profit <= self.entry:
                raise ValueError(
                    f"Long target {self.take_profit:g} must be above entry {self.entry:g}"
                )
        else:
            if self.stop_loss <= self.entry:
                raise ValueError(
                    f"Short stop {self.stop_loss:g} must be above entry {self.entry:g}"
                )
            if self.take_profit is not None and self.take_profit >= self.entry:
                raise ValueError(
                    f"Short target {self.take_profit:g} must be below entry {self.entry:g}"
                )

    @property
    def risk_per_unit(self) -> float | None:
        if self.entry is None or self.stop_loss is None:
            return None
        return abs(self.entry - self.stop_loss)

    @property
    def reward_risk_ratio(self) -> float | None:
        risk = self.risk_per_unit
        if risk is None or self.take_profit is None or self.entry is None or risk == 0:
            return None
        return abs(self.take_profit - self.entry) / risk

    @classmethod
    def no_trade(
        cls,
        symbol: str,
        strategy_name: str,
        reason: str,
        *,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StrategyResult:
        return cls(
            action=SignalAction.NO_TRADE,
            symbol=symbol,
            strategy_name=strategy_name,
            timestamp=timestamp or utcnow(),
            reason=reason,
            metadata=metadata or {},
        )

    @classmethod
    def hold(
        cls,
        symbol: str,
        strategy_name: str,
        reason: str = "conditions unchanged",
        *,
        timestamp: datetime | None = None,
    ) -> StrategyResult:
        return cls(
            action=SignalAction.HOLD,
            symbol=symbol,
            strategy_name=strategy_name,
            timestamp=timestamp or utcnow(),
            reason=reason,
        )


class Strategy(ABC):
    """Base class for all strategies."""

    #: Stable identifier used in the registry, the database and the API.
    name: ClassVar[str] = "abstract"
    #: Semantic version. Bump on any behaviour change; running bots pin a version.
    version: ClassVar[str] = "1.0.0"
    #: One-line description shown in the UI.
    description: ClassVar[str] = ""
    #: Parameter model class.
    parameters_model: ClassVar[type[StrategyParameters]] = StrategyParameters
    #: Regimes in which this strategy may produce an entry. Empty means "any known regime".
    allowed_regimes: ClassVar[frozenset[MarketRegime]] = frozenset()

    def __init__(self, parameters: StrategyParameters | dict[str, Any] | None = None) -> None:
        self.params = self._coerce_parameters(parameters)

    @classmethod
    def _coerce_parameters(
        cls, parameters: StrategyParameters | dict[str, Any] | None
    ) -> Any:
        if parameters is None:
            return cls.parameters_model()
        if isinstance(parameters, cls.parameters_model):
            return parameters
        if isinstance(parameters, StrategyParameters):
            parameters = parameters.model_dump()
        try:
            return cls.parameters_model(**parameters)
        except Exception as exc:
            raise StrategyConfigurationError(
                f"Invalid parameters for strategy {cls.name}: {exc}",
                context={"strategy": cls.name},
            ) from exc

    # ------------------------------------------------------------------ #
    # Contract
    # ------------------------------------------------------------------ #
    @property
    @abstractmethod
    def required_history(self) -> int:
        """Minimum number of closed candles before this strategy can produce a signal."""

    @abstractmethod
    def _evaluate(self, context: StrategyContext) -> StrategyResult:
        """Strategy-specific logic. Called only after preconditions are satisfied."""

    def generate_signal(self, context: StrategyContext) -> StrategyResult:
        """Produce a signal for the current bar.

        Applies the universal preconditions — history, regime, cooldown, direction — before
        delegating to :meth:`_evaluate`, so no individual strategy can forget them.
        """
        precondition = self._check_preconditions(context)
        if precondition is not None:
            return precondition

        result = self._evaluate(context)
        return self._apply_direction_filter(result, context)

    def _check_preconditions(self, context: StrategyContext) -> StrategyResult | None:
        if not context.snapshot.has_history(self.required_history):
            return StrategyResult.no_trade(
                context.symbol,
                self.name,
                f"insufficient history: {len(context.candles)}/{self.required_history} bars",
                timestamp=context.now,
            )
        if not context.regime.allows_trading:
            return StrategyResult.no_trade(
                context.symbol,
                self.name,
                "market regime is UNKNOWN; refusing to trade blind",
                timestamp=context.now,
            )
        if self.allowed_regimes and context.regime not in self.allowed_regimes:
            return StrategyResult.no_trade(
                context.symbol,
                self.name,
                f"regime {context.regime.value} is outside this strategy's mandate "
                f"({', '.join(sorted(r.value for r in self.allowed_regimes))})",
                timestamp=context.now,
                metadata={"regime": context.regime.value},
            )
        if (
            self.params.cooldown_bars > 0
            and context.bars_since_last_trade is not None
            and context.bars_since_last_trade < self.params.cooldown_bars
        ):
            return StrategyResult.no_trade(
                context.symbol,
                self.name,
                f"cooldown: {context.bars_since_last_trade}/"
                f"{self.params.cooldown_bars} bars since last trade",
                timestamp=context.now,
            )
        return None

    def _apply_direction_filter(
        self, result: StrategyResult, context: StrategyContext
    ) -> StrategyResult:
        if result.action is SignalAction.BUY and not self.params.allow_long:
            return StrategyResult.no_trade(
                context.symbol, self.name, "long entries disabled", timestamp=context.now
            )
        if result.action is SignalAction.SELL and not self.params.allow_short:
            return StrategyResult.no_trade(
                context.symbol, self.name, "short entries disabled", timestamp=context.now
            )
        if result.action.is_entry and result.confidence < self.params.min_confidence:
            return StrategyResult.no_trade(
                context.symbol,
                self.name,
                f"confidence {result.confidence:.2f} below threshold "
                f"{self.params.min_confidence:.2f}",
                timestamp=context.now,
                metadata=dict(result.metadata),
            )
        return result

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    def compute_levels(
        self,
        *,
        action: SignalAction,
        entry: float,
        atr_value: float,
        stop_multiplier: float | None = None,
        reward_ratio: float | None = None,
    ) -> tuple[float, float]:
        """ATR-based stop and target.

        Volatility-scaled stops are what make a fixed fractional risk model coherent across
        symbols and regimes: risking 0.5% of equity means the same thing on a quiet bar and a
        violent one only if the stop distance moves with volatility.
        """
        if atr_value <= 0 or not np.isfinite(atr_value):
            raise ValueError(f"ATR must be positive and finite, got {atr_value}")
        multiplier = (
            stop_multiplier if stop_multiplier is not None else self.params.atr_stop_multiplier
        )
        ratio = reward_ratio if reward_ratio is not None else self.params.risk_reward_ratio
        distance = atr_value * multiplier
        if action is SignalAction.BUY:
            return entry - distance, entry + distance * ratio
        return entry + distance, entry - distance * ratio

    @staticmethod
    def exit_signal_for(position: Position | None, action: SignalAction) -> bool:
        """True when ``action`` is opposite to the open position and should close it."""
        if position is None or not position.is_open:
            return False
        if position.side is PositionSide.LONG and action is SignalAction.SELL:
            return True
        return position.side is PositionSide.SHORT and action is SignalAction.BUY

    @staticmethod
    def trend_structure_score(candles: tuple[Candle, ...], lookback: int = 20) -> float:
        """Price-structure score in ``[-1, 1]``.

        Positive when the series is making higher highs and higher lows, negative for the
        mirror image. Complements indicator-based trend measures, which can stay positive
        through a structural break.
        """
        if len(candles) < lookback * 2:
            return 0.0
        recent = candles[-lookback:]
        prior = candles[-lookback * 2 : -lookback]
        recent_high, recent_low = max(c.high for c in recent), min(c.low for c in recent)
        prior_high, prior_low = max(c.high for c in prior), min(c.low for c in prior)

        score = 0.0
        score += 0.5 if recent_high > prior_high else -0.5
        score += 0.5 if recent_low > prior_low else -0.5
        return float(np.clip(score, -1.0, 1.0))

    @staticmethod
    def volume_ratio(candles: tuple[Candle, ...], period: int = 20) -> float:
        """Latest volume as a multiple of its recent average. 1.0 means average."""
        if len(candles) < period + 1:
            return 1.0
        window = [c.volume for c in candles[-period - 1 : -1]]
        average = float(np.mean(window)) if window else 0.0
        return safe_divide(candles[-1].volume, average, default=1.0)

    def describe(self) -> dict[str, Any]:
        """Serialisable description for the API and audit log."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "required_history": self.required_history,
            "allowed_regimes": sorted(r.value for r in self.allowed_regimes),
            "parameters": self.params.model_dump(),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(version={self.version})"
