"""Risk limit configuration.

A bot may configure its own limits, but :meth:`RiskLimits.clamped_to` guarantees a per-bot
configuration can only ever be *tighter* than the platform ceiling. There is no code path by
which a user-supplied value loosens a platform limit — that is the difference between a risk
system and a suggestion.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config.settings import RiskSettings


class RiskLimits(BaseModel):
    """Complete set of risk constraints applied to every order.

    Fractions are of *equity* unless stated otherwise.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- per-trade ---------------------------------------------------------
    risk_per_trade: float = Field(
        default=0.005, gt=0.0, le=0.05,
        description="Fraction of equity lost if the stop is hit",
    )
    max_position_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    max_leverage: float = Field(default=3.0, ge=1.0, le=20.0)
    min_reward_risk: float = Field(
        default=1.0, ge=0.0, le=20.0,
        description="Minimum take-profit/stop ratio; 0 disables the check",
    )

    # --- portfolio ---------------------------------------------------------
    max_portfolio_exposure: float = Field(default=1.0, gt=0.0, le=10.0)
    max_asset_exposure: float = Field(default=0.35, gt=0.0, le=1.0)
    max_concurrent_positions: int = Field(default=5, ge=1, le=100)
    max_correlated_positions: int = Field(
        default=3, ge=1, le=100,
        description="Max simultaneous positions sharing a quote/base asset group",
    )

    # --- loss controls -----------------------------------------------------
    max_daily_loss: float = Field(default=0.02, gt=0.0, le=0.5)
    max_weekly_loss: float = Field(default=0.06, gt=0.0, le=0.8)
    max_drawdown: float = Field(default=0.10, gt=0.0, le=0.9)
    max_loss_streak: int = Field(default=4, ge=1, le=50)
    max_daily_trades: int = Field(default=30, ge=1, le=10_000)
    cooldown_seconds: int = Field(default=300, ge=0, le=86_400)
    cooldown_after_loss_only: bool = True

    # --- execution quality --------------------------------------------------
    max_spread_bps: float = Field(default=15.0, gt=0.0, le=1000.0)
    max_slippage_bps: float = Field(default=25.0, gt=0.0, le=1000.0)
    min_liquidity_multiple: float = Field(default=10.0, ge=1.0, le=1000.0)
    slippage_safety_factor: float = Field(
        default=2.0, ge=1.0, le=10.0,
        description=(
            "Multiplier applied to the estimated slippage when sizing. Sizing on a "
            "point estimate means roughly half of all fills breach the risk budget."
        ),
    )
    min_slippage_bps: float = Field(
        default=2.0, ge=0.0, le=100.0,
        description="Slippage floor, so a momentarily perfect book cannot imply zero cost",
    )
    max_cost_to_risk_ratio: float = Field(
        default=0.35, gt=0.0, le=1.0,
        description=(
            "Reject trades where fees plus slippage exceed this fraction of the stop "
            "distance. Such trades need an implausible hit rate just to break even."
        ),
    )

    # --- data quality -------------------------------------------------------
    max_data_staleness_seconds: float = Field(default=120.0, gt=0.0)
    max_clock_drift_seconds: float = Field(default=2.0, gt=0.0)
    max_consecutive_api_failures: int = Field(default=5, ge=1, le=100)

    # --- behaviour ----------------------------------------------------------
    scale_size_by_confidence: bool = Field(
        default=True,
        description="Reduce size on low-confidence signals. Never increases it.",
    )
    allow_size_reduction: bool = Field(
        default=True,
        description="When a limit binds, submit a smaller order instead of rejecting",
    )

    @model_validator(mode="after")
    def _validate_ordering(self) -> Self:
        if self.max_daily_loss > self.max_weekly_loss:
            raise ValueError(
                f"max_daily_loss ({self.max_daily_loss}) cannot exceed max_weekly_loss "
                f"({self.max_weekly_loss})"
            )
        if self.max_weekly_loss > self.max_drawdown:
            raise ValueError(
                f"max_weekly_loss ({self.max_weekly_loss}) cannot exceed max_drawdown "
                f"({self.max_drawdown}): the weekly limit would never bind"
            )
        if self.risk_per_trade * self.max_concurrent_positions > self.max_drawdown * 2:
            raise ValueError(
                f"risk_per_trade ({self.risk_per_trade:.3f}) x max_concurrent_positions "
                f"({self.max_concurrent_positions}) risks "
                f"{self.risk_per_trade * self.max_concurrent_positions:.1%} of equity at "
                f"once, more than twice the {self.max_drawdown:.1%} drawdown limit. "
                "Reduce the per-trade risk or the position count."
            )
        return self

    @property
    def max_simultaneous_risk(self) -> float:
        """Total equity at risk if every allowed position stops out together."""
        return self.risk_per_trade * self.max_concurrent_positions

    def clamped_to(self, ceiling: RiskLimits) -> RiskLimits:
        """Return these limits, tightened wherever they are looser than ``ceiling``.

        This is the mechanism that stops a per-bot configuration from widening a platform
        limit. It is applied unconditionally when a bot's limits are loaded.
        """
        tighter_is_smaller = (
            "risk_per_trade",
            "max_position_fraction",
            "max_leverage",
            "max_portfolio_exposure",
            "max_asset_exposure",
            "max_concurrent_positions",
            "max_correlated_positions",
            "max_daily_loss",
            "max_weekly_loss",
            "max_drawdown",
            "max_loss_streak",
            "max_daily_trades",
            "max_spread_bps",
            "max_slippage_bps",
            "max_cost_to_risk_ratio",
            "max_data_staleness_seconds",
            "max_clock_drift_seconds",
            "max_consecutive_api_failures",
        )
        tighter_is_larger = (
            "min_liquidity_multiple",
            "min_reward_risk",
            "cooldown_seconds",
            "slippage_safety_factor",
            "min_slippage_bps",
        )

        values: dict[str, Any] = self.model_dump()
        for field in tighter_is_smaller:
            values[field] = min(getattr(self, field), getattr(ceiling, field))
        for field in tighter_is_larger:
            values[field] = max(getattr(self, field), getattr(ceiling, field))
        return RiskLimits(**values)

    @classmethod
    def from_settings(cls, settings: RiskSettings) -> RiskLimits:
        """Build platform-level limits from environment configuration."""
        return cls(
            risk_per_trade=settings.per_trade,
            max_daily_loss=settings.max_daily_loss,
            max_weekly_loss=settings.max_weekly_loss,
            max_drawdown=settings.max_drawdown,
            max_position_fraction=settings.max_position_fraction,
            max_portfolio_exposure=settings.max_portfolio_exposure,
            max_asset_exposure=settings.max_asset_exposure,
            max_concurrent_positions=settings.max_concurrent_positions,
            max_leverage=settings.max_leverage,
            max_spread_bps=settings.max_spread_bps,
            max_slippage_bps=settings.max_slippage_bps,
            min_liquidity_multiple=settings.min_liquidity_multiple,
            cooldown_seconds=settings.cooldown_seconds,
            max_loss_streak=settings.max_loss_streak,
            max_daily_trades=settings.max_daily_trades,
        )

    @classmethod
    def conservative(cls) -> RiskLimits:
        """A deliberately cautious preset, used as the default for new bots."""
        return cls(
            risk_per_trade=0.0025,
            max_position_fraction=0.15,
            max_leverage=1.0,
            max_portfolio_exposure=0.5,
            max_asset_exposure=0.20,
            max_concurrent_positions=3,
            max_daily_loss=0.01,
            max_weekly_loss=0.03,
            max_drawdown=0.06,
            max_loss_streak=3,
            max_daily_trades=10,
            cooldown_seconds=900,
        )
