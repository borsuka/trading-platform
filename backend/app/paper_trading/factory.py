"""Bot assembly.

One place that wires a strategy, an exchange, a portfolio and a risk manager into a running
:class:`~app.paper_trading.runtime.TradingBot`. Centralising construction is what makes the
"live is paper with a different adapter" claim true in practice rather than in principle: mode
selection happens here, once, and nothing downstream branches on it.

Live construction additionally refuses to proceed unless every safety precondition is met — see
:func:`build_live_bot` and :mod:`app.exchanges.live_gate`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.config import ExchangeName, Settings, TradingMode, get_settings
from app.core.domain import InstrumentSpec
from app.core.enums import InstrumentType
from app.core.exceptions import ConfigurationError, LiveTradingDisabledError
from app.core.logging import get_logger
from app.exchanges.base import ExchangeAdapter, ExchangeCredentials
from app.exchanges.paper import PaperExchange, PaperExchangeConfig
from app.market_data.providers import (
    ExchangeLiveProvider,
    LiveMarketDataProvider,
    generate_synthetic_candles,
)
from app.news.providers import NewsProvider, build_news_provider
from app.paper_trading.runtime import BotConfig, TradingBot
from app.portfolio.manager import PortfolioManager
from app.risk.limits import RiskLimits
from app.signals.engine import SignalEngineConfig
from app.strategies.registry import create_strategy

logger = get_logger(__name__)

#: Instrument metadata used when a venue does not supply it (paper mode with no seed data).
DEFAULT_SPECS: dict[str, InstrumentSpec] = {
    "BTCUSDT": InstrumentSpec(
        symbol="BTCUSDT", base_asset="BTC", quote_asset="USDT",
        instrument_type=InstrumentType.SPOT,
        tick_size=0.1, lot_size=0.00001, min_quantity=0.00001, min_notional=5.0,
        maker_fee=0.0002, taker_fee=0.00055,
    ),
    "ETHUSDT": InstrumentSpec(
        symbol="ETHUSDT", base_asset="ETH", quote_asset="USDT",
        instrument_type=InstrumentType.SPOT,
        tick_size=0.01, lot_size=0.0001, min_quantity=0.0001, min_notional=5.0,
        maker_fee=0.0002, taker_fee=0.00055,
    ),
    "SOLUSDT": InstrumentSpec(
        symbol="SOLUSDT", base_asset="SOL", quote_asset="USDT",
        instrument_type=InstrumentType.SPOT,
        tick_size=0.001, lot_size=0.01, min_quantity=0.01, min_notional=5.0,
        maker_fee=0.0002, taker_fee=0.00055,
    ),
}


def default_instruments(symbols: Sequence[str]) -> dict[str, InstrumentSpec]:
    """Instrument specs for the requested symbols, inferring unknown ones conservatively."""
    specs: dict[str, InstrumentSpec] = {}
    for symbol in symbols:
        upper = symbol.upper()
        if upper in DEFAULT_SPECS:
            specs[upper] = DEFAULT_SPECS[upper]
            continue
        for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
            if upper.endswith(quote) and len(upper) > len(quote):
                specs[upper] = InstrumentSpec(
                    symbol=upper,
                    base_asset=upper[: -len(quote)],
                    quote_asset=quote,
                    instrument_type=InstrumentType.SPOT,
                    tick_size=0.01,
                    lot_size=0.0001,
                    min_quantity=0.0001,
                    min_notional=5.0,
                )
                break
        else:
            raise ConfigurationError(
                f"Cannot infer instrument metadata for {symbol}; register it explicitly."
            )
    return specs


def build_paper_exchange(
    symbols: Sequence[str],
    *,
    starting_balance: float,
    quote_asset: str = "USDT",
    config: PaperExchangeConfig | None = None,
) -> PaperExchange:
    """Construct a paper exchange preloaded with instrument metadata."""
    return PaperExchange(
        config
        or PaperExchangeConfig(
            starting_balance=starting_balance, quote_asset=quote_asset
        ),
        instruments=default_instruments(symbols),
    )


def default_paper_market_data(
    exchange: PaperExchange, symbols: Sequence[str], interval: str
) -> LiveMarketDataProvider:
    """Where a paper bot gets its prices when the caller does not say.

    Wrapping the paper exchange in an ``ExchangeLiveProvider`` - the obvious-looking choice -
    produces a bot that cannot work: the simulator stores no history, so every cycle fails
    with "no market data has been fed". It is a matching engine, not a price source, and it
    needs feeding from outside.

    The default is therefore a live read of a venue's public endpoints. No credentials are
    involved and the bot still executes against the simulator; only the prices are real. That
    is the combination the live-trading checklist asks you to rehearse with.
    """
    from app.config import get_settings

    settings = get_settings()
    if settings.paper_market_data == "synthetic":
        from app.paper_trading.replay import ReplayMarketDataProvider

        candles = [
            candle
            for symbol in symbols
            for candle in generate_synthetic_candles(symbol, interval=interval, count=1500)
        ]
        return ReplayMarketDataProvider(candles, exchange=exchange)

    from app.paper_trading.live_feed import PublicMarketFeed

    return PublicMarketFeed(exchange, source=settings.paper_market_data)


def build_paper_bot(
    *,
    bot_id: str,
    user_id: str,
    name: str,
    strategy_name: str,
    symbols: Sequence[str],
    interval: str = "15m",
    starting_balance: float = 10_000.0,
    strategy_parameters: dict[str, Any] | None = None,
    risk_limits: RiskLimits | None = None,
    signal_config: SignalEngineConfig | None = None,
    market_data: LiveMarketDataProvider | None = None,
    news_provider: NewsProvider | None = None,
    exchange_config: PaperExchangeConfig | None = None,
    **bot_options: Any,
) -> TradingBot:
    """Assemble a fully wired paper-trading bot.

    ``market_data`` may be supplied to replay a fixed history; otherwise the bot reads from
    the paper exchange itself, which must then be fed with :meth:`PaperExchange.process_candle`.
    """
    normalized = tuple(s.upper() for s in symbols)
    if not normalized:
        raise ConfigurationError("A bot needs at least one symbol")

    exchange = build_paper_exchange(
        normalized, starting_balance=starting_balance, config=exchange_config
    )
    provider = market_data or default_paper_market_data(exchange, normalized, interval)
    portfolio = PortfolioManager(starting_balance=starting_balance)

    return TradingBot(
        BotConfig(
            bot_id=bot_id,
            user_id=user_id,
            name=name,
            symbols=normalized,
            interval=interval,
            **bot_options,
        ),
        strategy=create_strategy(strategy_name, strategy_parameters),
        exchange=exchange,
        market_data=provider,
        portfolio=portfolio,
        risk_limits=risk_limits or RiskLimits.conservative(),
        signal_config=signal_config,
        news_provider=news_provider or build_news_provider(),
    )


async def build_live_bot(
    *,
    bot_id: str,
    user_id: str,
    name: str,
    strategy_name: str,
    symbols: Sequence[str],
    credentials: ExchangeCredentials,
    exchange_name: ExchangeName,
    interval: str = "15m",
    strategy_parameters: dict[str, Any] | None = None,
    risk_limits: RiskLimits | None = None,
    settings: Settings | None = None,
    **bot_options: Any,
) -> TradingBot:
    """Assemble a **live** bot.

    Refuses unless the platform is explicitly configured for live trading. The full preflight
    (credential permissions, balances, connectivity, clock) runs in
    :func:`app.exchanges.live_gate.run_preflight`, which the API calls before this and the bot
    calls again on startup. Two checks, because the cost of getting this wrong is real money.
    """
    resolved = settings or get_settings()
    if not resolved.live_trading_enabled:
        raise LiveTradingDisabledError(
            "Live trading is disabled. Set LIVE_TRADING_ENABLED=true and complete the "
            "activation checklist in docs/live-trading.md before attempting this."
        )
    if resolved.trading_mode is not TradingMode.LIVE:
        raise LiveTradingDisabledError(
            f"TRADING_MODE is {resolved.trading_mode.value}, not live."
        )
    if exchange_name is ExchangeName.PAPER:
        raise ConfigurationError("Live trading requires a real exchange, not 'paper'.")

    exchange = build_exchange_adapter(exchange_name, credentials)

    permissions = await exchange.validate_credentials()
    problem = permissions.rejection_reason()
    if problem is not None:
        await exchange.close()
        raise LiveTradingDisabledError(problem)

    balance = await exchange.get_balance()
    starting = balance.total(resolved.paper_quote_currency)
    if starting <= 0:
        await exchange.close()
        raise LiveTradingDisabledError(
            f"The exchange account holds no {resolved.paper_quote_currency}."
        )

    portfolio = PortfolioManager(
        starting_balance=starting, quote_asset=resolved.paper_quote_currency
    )
    logger.warning(
        "bot.live_constructed",
        bot_id=bot_id,
        exchange=exchange_name.value,
        testnet=credentials.testnet,
        starting_balance=round(starting, 2),
    )
    return TradingBot(
        BotConfig(
            bot_id=bot_id,
            user_id=user_id,
            name=name,
            symbols=tuple(s.upper() for s in symbols),
            interval=interval,
            reconcile_on_start=True,
            **bot_options,
        ),
        strategy=create_strategy(strategy_name, strategy_parameters),
        exchange=exchange,
        market_data=ExchangeLiveProvider(exchange),
        portfolio=portfolio,
        risk_limits=risk_limits or RiskLimits.conservative(),
        news_provider=build_news_provider(),
    )


def build_exchange_adapter(
    name: ExchangeName, credentials: ExchangeCredentials | None = None
) -> ExchangeAdapter:
    """Construct the adapter for a venue.

    The single place where a venue name becomes an adapter instance.
    """
    if name is ExchangeName.PAPER:
        return PaperExchange()
    if credentials is None:
        raise ConfigurationError(f"{name.value} requires API credentials")

    if name is ExchangeName.BYBIT:
        from app.exchanges.bybit import BybitAdapter

        return BybitAdapter(credentials)
    if name is ExchangeName.BINANCE:
        from app.exchanges.binance import BinanceAdapter

        return BinanceAdapter(credentials)
    if name is ExchangeName.COINBASE:
        from app.exchanges.coinbase import CoinbaseAdapter

        return CoinbaseAdapter(credentials)
    if name is ExchangeName.CRYPTOCOM:
        from app.exchanges.cryptocom import CryptoComAdapter

        return CryptoComAdapter(credentials)
    raise ConfigurationError(f"Unsupported exchange: {name}")
