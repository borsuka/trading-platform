"""Bot runtime. Drives both paper and live trading through one code path."""

from app.paper_trading.factory import (
    build_exchange_adapter,
    build_live_bot,
    build_paper_bot,
    build_paper_exchange,
)
from app.paper_trading.replay import (
    ReplayClock,
    ReplayMarketDataProvider,
    build_replay_bot,
)
from app.paper_trading.runtime import (
    BotConfig,
    BotEvent,
    BotRegistry,
    BotSnapshot,
    TradingBot,
    bot_registry,
    build_symbols,
)

__all__ = [
    "BotConfig",
    "BotEvent",
    "BotRegistry",
    "BotSnapshot",
    "ReplayClock",
    "ReplayMarketDataProvider",
    "TradingBot",
    "bot_registry",
    "build_exchange_adapter",
    "build_live_bot",
    "build_paper_bot",
    "build_paper_exchange",
    "build_replay_bot",
    "build_symbols",
]
