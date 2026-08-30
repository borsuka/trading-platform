"""Strategy registry.

Strategies are looked up by name from the database, the API and the backtester. The registry is
the single place that maps a name to a class, so an unknown strategy type fails loudly at
construction rather than producing a bot that silently never trades.
"""

from __future__ import annotations

from typing import Any

from app.core.exceptions import StrategyConfigurationError
from app.strategies.base import Strategy, StrategyParameters

_REGISTRY: dict[str, type[Strategy]] = {}


def register(strategy_cls: type[Strategy]) -> type[Strategy]:
    """Register a strategy class. Usable as a decorator."""
    name = strategy_cls.name
    if name in _REGISTRY and _REGISTRY[name] is not strategy_cls:
        raise StrategyConfigurationError(
            f"Strategy name {name!r} is already registered to "
            f"{_REGISTRY[name].__name__}"
        )
    _REGISTRY[name] = strategy_cls
    return strategy_cls


def get_strategy_class(name: str) -> type[Strategy]:
    """Look up a registered strategy class by name."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise StrategyConfigurationError(
            f"Unknown strategy {name!r}. Available: {', '.join(sorted(_REGISTRY))}",
            context={"requested": name, "available": sorted(_REGISTRY)},
        ) from exc


def create_strategy(
    name: str, parameters: dict[str, Any] | StrategyParameters | None = None
) -> Strategy:
    """Instantiate a registered strategy with validated parameters."""
    return get_strategy_class(name)(parameters)


def available_strategies() -> list[str]:
    return sorted(_REGISTRY)


def describe_all() -> list[dict[str, Any]]:
    """Metadata for every registered strategy, for the API's strategy catalogue."""
    return [
        {
            "name": cls.name,
            "version": cls.version,
            "description": cls.description,
            "allowed_regimes": sorted(r.value for r in cls.allowed_regimes),
            "parameters_schema": cls.parameters_model.model_json_schema(),
            "default_parameters": cls.parameters_model().model_dump(),
        }
        for cls in sorted(_REGISTRY.values(), key=lambda c: c.name)
    ]


def _register_builtins() -> None:
    """Import and register the shipped strategies."""
    from app.strategies.mean_reversion import MeanReversionStrategy
    from app.strategies.momentum_breakout import MomentumBreakoutStrategy
    from app.strategies.multi_factor import MultiFactorStrategy
    from app.strategies.trend_following import TrendFollowingStrategy

    for cls in (
        TrendFollowingStrategy,
        MomentumBreakoutStrategy,
        MeanReversionStrategy,
        MultiFactorStrategy,
    ):
        register(cls)


_register_builtins()
