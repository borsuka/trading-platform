"""Strategy framework and shipped strategies."""

from app.strategies.base import (
    Strategy,
    StrategyContext,
    StrategyParameters,
    StrategyResult,
)
from app.strategies.mean_reversion import MeanReversionParameters, MeanReversionStrategy
from app.strategies.momentum_breakout import (
    MomentumBreakoutParameters,
    MomentumBreakoutStrategy,
)
from app.strategies.multi_factor import MultiFactorParameters, MultiFactorStrategy
from app.strategies.registry import (
    available_strategies,
    create_strategy,
    describe_all,
    get_strategy_class,
    register,
)
from app.strategies.trend_following import (
    TrendFollowingParameters,
    TrendFollowingStrategy,
)

__all__ = [
    "MeanReversionParameters",
    "MeanReversionStrategy",
    "MomentumBreakoutParameters",
    "MomentumBreakoutStrategy",
    "MultiFactorParameters",
    "MultiFactorStrategy",
    "Strategy",
    "StrategyContext",
    "StrategyParameters",
    "StrategyResult",
    "TrendFollowingParameters",
    "TrendFollowingStrategy",
    "available_strategies",
    "create_strategy",
    "describe_all",
    "get_strategy_class",
    "register",
]
