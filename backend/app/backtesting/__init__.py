"""Event-driven backtesting, metrics and robustness analysis."""

from app.backtesting.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    date_range_slice,
    split_candles,
)
from app.backtesting.metrics import (
    DrawdownInfo,
    PerformanceMetrics,
    calmar_ratio,
    compound_annual_growth_rate,
    compute_drawdown,
    compute_metrics,
    monthly_returns,
    periodic_returns,
    periods_per_year_for,
    sharpe_ratio,
    sortino_ratio,
)
from app.backtesting.validation import (
    MonteCarloReport,
    RobustnessReport,
    SensitivityReport,
    WalkForwardReport,
    monte_carlo,
    parameter_sensitivity,
    validate_strategy,
    walk_forward,
)

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "DrawdownInfo",
    "MonteCarloReport",
    "PerformanceMetrics",
    "RobustnessReport",
    "SensitivityReport",
    "WalkForwardReport",
    "calmar_ratio",
    "compound_annual_growth_rate",
    "compute_drawdown",
    "compute_metrics",
    "date_range_slice",
    "monte_carlo",
    "monthly_returns",
    "parameter_sensitivity",
    "periodic_returns",
    "periods_per_year_for",
    "sharpe_ratio",
    "sortino_ratio",
    "split_candles",
    "validate_strategy",
    "walk_forward",
]
