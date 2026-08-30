"""Live adapter and preflight tests.

Run against ``httpx.MockTransport`` reproducing each venue's documented envelopes and error
codes. They verify the properties that make live trading safe:

* a key with withdrawal permission is **rejected**, on both venues;
* a timeout maps to :class:`ExchangeTimeoutError` and never to a silent retry;
* the ``client_order_id`` round-trips, so a timed-out order can be found again;
* the preflight blocks on every individual failure.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.config import AppEnv, Settings, TradingMode
from app.core.domain import OrderRequest
from app.core.enums import OrderSide, OrderStatus, OrderType
from app.core.exceptions import (
    ExchangeAuthError,
    ExchangeConnectionError,
    ExchangeRateLimitError,
    ExchangeTimeoutError,
    InsufficientBalanceError,
    OrderRejectedError,
    PreflightFailedError,
    UnsafeCredentialsError,
)
from app.exchanges.base import ExchangeCredentials
from app.exchanges.binance import BinanceAdapter
from app.exchanges.bybit import BybitAdapter
from app.exchanges.live_gate import CONFIRMATION_PHRASE, run_preflight
from app.exchanges.paper import PaperExchange
from app.market_data.models import Candle
from app.risk.limits import RiskLimits

CREDS = ExchangeCredentials(
    api_key="test-key-1234", api_secret="test-secret-abcd", testnet=True
)


def make_client(routes: dict[str, Any]) -> httpx.AsyncClient:
    """Mock transport dispatching on path, with per-route callables or payloads."""

    def handler(request: httpx.Request) -> httpx.Response:
        route = routes.get(request.url.path)
        if route is None:
            return httpx.Response(404, json={"msg": f"no route for {request.url.path}"})
        if callable(route):
            return route(request)
        return httpx.Response(200, json=route)

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://mock.invalid"
    )


# =========================================================================== #
# Bybit
# =========================================================================== #
BYBIT_SAFE_KEY = {
    "retCode": 0,
    "retMsg": "OK",
    "result": {
        "permissions": {
            "Spot": ["SpotTrade"],
            "Wallet": ["AccountTransfer"],
            "Derivatives": [],
            "Withdraw": [],
        },
        "ips": ["203.0.113.10"],
    },
}

BYBIT_WITHDRAW_KEY = {
    "retCode": 0,
    "retMsg": "OK",
    "result": {
        "permissions": {
            "Spot": ["SpotTrade"],
            "Wallet": ["AccountTransfer", "Withdraw"],
            "Withdraw": ["Withdraw"],
        },
        "ips": ["*"],
    },
}

BYBIT_INSTRUMENTS = {
    "retCode": 0,
    "result": {
        "list": [
            {
                "symbol": "BTCUSDT",
                "baseCoin": "BTC",
                "quoteCoin": "USDT",
                "status": "Trading",
                "lotSizeFilter": {
                    "basePrecision": "0.000001",
                    "minOrderQty": "0.000048",
                    "maxOrderQty": "71.73956243",
                    "minOrderAmt": "1",
                },
                "priceFilter": {"tickSize": "0.01"},
                "leverageFilter": {"maxLeverage": "1"},
            }
        ]
    },
}


class TestBybitAdapter:
    async def test_rejects_a_key_that_can_withdraw(self) -> None:
        """The single most important adapter test."""
        client = make_client({"/v5/user/query-api": BYBIT_WITHDRAW_KEY})
        adapter = BybitAdapter(CREDS, client=client)
        with pytest.raises(UnsafeCredentialsError, match="WITHDRAWAL"):
            await adapter.validate_credentials()
        await client.aclose()

    async def test_accepts_a_trade_only_key(self) -> None:
        client = make_client({"/v5/user/query-api": BYBIT_SAFE_KEY})
        adapter = BybitAdapter(CREDS, client=client)
        permissions = await adapter.validate_credentials()
        assert permissions.is_safe_for_trading
        assert permissions.can_trade and not permissions.can_withdraw
        assert permissions.ip_restricted
        await client.aclose()

    async def test_signs_requests_without_leaking_the_secret(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(request.headers)
            captured["url"] = str(request.url)
            return httpx.Response(200, json=BYBIT_SAFE_KEY)

        client = make_client({"/v5/user/query-api": handler})
        adapter = BybitAdapter(CREDS, client=client)
        await adapter.validate_credentials()

        assert captured["headers"]["x-bapi-api-key"] == "test-key-1234"
        assert len(captured["headers"]["x-bapi-sign"]) == 64  # hex sha256
        assert "test-secret-abcd" not in json.dumps(captured)
        await client.aclose()

    async def test_parses_instruments(self) -> None:
        client = make_client({"/v5/market/instruments-info": BYBIT_INSTRUMENTS})
        adapter = BybitAdapter(CREDS, client=client)
        spec = await adapter.get_instrument("BTCUSDT")
        assert spec.base_asset == "BTC"
        assert spec.tick_size == pytest.approx(0.01)
        assert spec.lot_size == pytest.approx(0.000001)
        assert spec.min_notional == pytest.approx(1.0)
        await client.aclose()

    async def test_parses_candles_oldest_first(self) -> None:
        payload = {
            "retCode": 0,
            "result": {
                "list": [
                    ["1700000060000", "101", "102", "100", "101.5", "10", "1015"],
                    ["1700000000000", "100", "101", "99", "100.5", "12", "1206"],
                ]
            },
        }
        client = make_client({"/v5/market/kline": payload})
        adapter = BybitAdapter(CREDS, client=client)
        candles = await adapter.get_candles("BTCUSDT", "1m")
        assert len(candles) == 2
        assert candles[0].open_time < candles[1].open_time
        assert candles[0].close == pytest.approx(100.5)
        await client.aclose()

    async def test_timeout_is_not_swallowed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        client = make_client({"/v5/order/create": handler})
        adapter = BybitAdapter(CREDS, client=client)
        adapter._instruments["BTCUSDT"] = (
            await BybitAdapter(CREDS, client=make_client(
                {"/v5/market/instruments-info": BYBIT_INSTRUMENTS}
            )).get_instrument("BTCUSDT")
        )
        with pytest.raises(ExchangeTimeoutError, match="outcome is unknown"):
            await adapter.create_order(
                OrderRequest(
                    symbol="BTCUSDT", side=OrderSide.BUY,
                    order_type=OrderType.MARKET, quantity=0.01,
                )
            )
        await client.aclose()

    async def test_client_order_id_round_trips(self) -> None:
        captured: dict[str, Any] = {}
        order_payload = {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "orderId": "ex-1",
                        "orderLinkId": "my-id-42",
                        "symbol": "BTCUSDT",
                        "side": "Buy",
                        "orderType": "Market",
                        "qty": "0.01",
                        "cumExecQty": "0.01",
                        "avgPrice": "50000",
                        "cumExecFee": "0.275",
                        "orderStatus": "Filled",
                        "createdTime": "1700000000000",
                        "updatedTime": "1700000001000",
                    }
                ]
            },
        }

        def create(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"retCode": 0, "result": {"orderId": "ex-1"}})

        client = make_client(
            {
                "/v5/market/instruments-info": BYBIT_INSTRUMENTS,
                "/v5/order/create": create,
                "/v5/order/realtime": order_payload,
            }
        )
        adapter = BybitAdapter(CREDS, client=client)
        order = await adapter.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.MARKET,
                quantity=0.01, client_order_id="my-id-42",
            )
        )
        assert captured["body"]["orderLinkId"] == "my-id-42"
        assert order.client_order_id == "my-id-42"
        assert order.status is OrderStatus.FILLED
        assert order.filled_quantity == pytest.approx(0.01)
        await client.aclose()

    async def test_get_order_returns_none_when_venue_has_no_record(self) -> None:
        """Only a definitive negative permits a resubmission."""
        empty = {"retCode": 0, "result": {"list": []}}
        client = make_client(
            {"/v5/order/realtime": empty, "/v5/order/history": empty}
        )
        adapter = BybitAdapter(CREDS, client=client)
        assert await adapter.get_order("never-sent", symbol="BTCUSDT") is None
        await client.aclose()

    async def test_insufficient_balance_is_typed(self) -> None:
        payload = {"retCode": 110007, "retMsg": "ab not enough for new order"}
        client = make_client({"/v5/order/create": lambda r: httpx.Response(200, json=payload)})
        adapter = BybitAdapter(CREDS, client=client)
        adapter._instruments["BTCUSDT"] = (
            await BybitAdapter(CREDS, client=make_client(
                {"/v5/market/instruments-info": BYBIT_INSTRUMENTS}
            )).get_instrument("BTCUSDT")
        )
        with pytest.raises(InsufficientBalanceError):
            await adapter.create_order(
                OrderRequest(
                    symbol="BTCUSDT", side=OrderSide.BUY,
                    order_type=OrderType.MARKET, quantity=0.01,
                )
            )
        await client.aclose()

    async def test_rate_limit_is_typed(self) -> None:
        client = make_client(
            {"/v5/market/time": lambda r: httpx.Response(
                429, json={"retCode": 10006}, headers={"Retry-After": "5"}
            )}
        )
        adapter = BybitAdapter(CREDS, client=client)
        with pytest.raises(ExchangeRateLimitError) as info:
            await adapter.get_info()
        assert info.value.retry_after_seconds == pytest.approx(5.0)
        await client.aclose()

    async def test_auth_failure_is_typed(self) -> None:
        client = make_client(
            {"/v5/user/query-api": lambda r: httpx.Response(401, json={"retCode": 10003})}
        )
        adapter = BybitAdapter(CREDS, client=client)
        with pytest.raises(ExchangeAuthError):
            await adapter.validate_credentials()
        await client.aclose()

    async def test_server_error_is_safe_to_retry(self) -> None:
        client = make_client(
            {"/v5/market/time": lambda r: httpx.Response(503, json={})}
        )
        adapter = BybitAdapter(CREDS, client=client)
        with pytest.raises(ExchangeConnectionError, match="not processed"):
            await adapter.get_info()
        await client.aclose()

    async def test_reports_itself_as_live_even_on_testnet(self) -> None:
        adapter = BybitAdapter(CREDS, client=make_client({}))
        assert adapter.is_live is True

    async def test_repr_hides_credentials(self) -> None:
        adapter = BybitAdapter(CREDS, client=make_client({}))
        assert "test-secret-abcd" not in repr(adapter)
        assert "test-secret-abcd" not in repr(CREDS)


# =========================================================================== #
# Binance
# =========================================================================== #
BINANCE_SAFE_KEY = {
    "ipRestrict": True,
    "enableReading": True,
    "enableSpotAndMarginTrading": True,
    "enableWithdrawals": False,
    "permitsUniversalTransfer": False,
}

BINANCE_WITHDRAW_KEY = {**BINANCE_SAFE_KEY, "enableWithdrawals": True}
BINANCE_TRANSFER_KEY = {**BINANCE_SAFE_KEY, "permitsUniversalTransfer": True}

BINANCE_EXCHANGE_INFO = {
    "symbols": [
        {
            "symbol": "BTCUSDT",
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "status": "TRADING",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
                {
                    "filterType": "LOT_SIZE",
                    "stepSize": "0.00001000",
                    "minQty": "0.00001000",
                    "maxQty": "9000.00000000",
                },
                {"filterType": "NOTIONAL", "minNotional": "5.00000000"},
            ],
        }
    ]
}


class TestBinanceAdapter:
    async def test_rejects_a_key_that_can_withdraw(self) -> None:
        client = make_client({"/sapi/v1/account/apiRestrictions": BINANCE_WITHDRAW_KEY})
        adapter = BinanceAdapter(CREDS, client=client)
        with pytest.raises(UnsafeCredentialsError, match="withdraw"):
            await adapter.validate_credentials()
        await client.aclose()

    async def test_rejects_a_key_that_can_transfer(self) -> None:
        client = make_client({"/sapi/v1/account/apiRestrictions": BINANCE_TRANSFER_KEY})
        adapter = BinanceAdapter(CREDS, client=client)
        with pytest.raises(UnsafeCredentialsError):
            await adapter.validate_credentials()
        await client.aclose()

    async def test_accepts_a_trade_only_key(self) -> None:
        client = make_client({"/sapi/v1/account/apiRestrictions": BINANCE_SAFE_KEY})
        adapter = BinanceAdapter(CREDS, client=client)
        permissions = await adapter.validate_credentials()
        assert permissions.is_safe_for_trading
        await client.aclose()

    async def test_signature_is_in_the_query_string(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = dict(request.headers)
            return httpx.Response(200, json=BINANCE_SAFE_KEY)

        client = make_client({"/sapi/v1/account/apiRestrictions": handler})
        adapter = BinanceAdapter(CREDS, client=client)
        await adapter.validate_credentials()
        assert "signature=" in captured["url"]
        assert "timestamp=" in captured["url"]
        assert captured["headers"]["x-mbx-apikey"] == "test-key-1234"
        assert "test-secret-abcd" not in captured["url"]
        await client.aclose()

    async def test_parses_filters(self) -> None:
        client = make_client({"/api/v3/exchangeInfo": BINANCE_EXCHANGE_INFO})
        adapter = BinanceAdapter(CREDS, client=client)
        spec = await adapter.get_instrument("BTCUSDT")
        assert spec.tick_size == pytest.approx(0.01)
        assert spec.lot_size == pytest.approx(0.00001)
        assert spec.min_notional == pytest.approx(5.0)
        await client.aclose()

    async def test_order_with_fills_is_parsed(self) -> None:
        payload = {
            "symbol": "BTCUSDT",
            "orderId": 28,
            "clientOrderId": "my-id-7",
            "transactTime": 1700000000000,
            "price": "0.00000000",
            "origQty": "0.01000000",
            "executedQty": "0.01000000",
            "cummulativeQuoteQty": "500.00000000",
            "status": "FILLED",
            "type": "MARKET",
            "side": "BUY",
            "fills": [
                {"price": "50000.00", "qty": "0.01000000",
                 "commission": "0.00001000", "commissionAsset": "BTC", "tradeId": 56},
            ],
        }
        client = make_client(
            {"/api/v3/exchangeInfo": BINANCE_EXCHANGE_INFO, "/api/v3/order": payload}
        )
        adapter = BinanceAdapter(CREDS, client=client)
        order = await adapter.create_order(
            OrderRequest(
                symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.MARKET,
                quantity=0.01, client_order_id="my-id-7",
            )
        )
        assert order.status is OrderStatus.FILLED
        assert order.client_order_id == "my-id-7"
        assert order.average_fill_price == pytest.approx(50_000.0)
        assert len(order.fills) == 1
        await client.aclose()

    async def test_unknown_order_returns_none(self) -> None:
        client = make_client(
            {"/api/v3/order": lambda r: httpx.Response(
                400, json={"code": -2011, "msg": "Unknown order sent."}
            )}
        )
        adapter = BinanceAdapter(CREDS, client=client)
        assert await adapter.get_order("never-sent", symbol="BTCUSDT") is None
        await client.aclose()

    async def test_filter_failure_is_typed(self) -> None:
        client = make_client(
            {
                "/api/v3/exchangeInfo": BINANCE_EXCHANGE_INFO,
                "/api/v3/order": lambda r: httpx.Response(
                    400, json={"code": -1013, "msg": "Filter failure: MIN_NOTIONAL"}
                ),
            }
        )
        adapter = BinanceAdapter(CREDS, client=client)
        with pytest.raises(OrderRejectedError, match="MIN_NOTIONAL"):
            await adapter.create_order(
                OrderRequest(
                    symbol="BTCUSDT", side=OrderSide.BUY,
                    order_type=OrderType.MARKET, quantity=0.01,
                )
            )
        await client.aclose()

    async def test_spot_reports_no_positions(self) -> None:
        adapter = BinanceAdapter(CREDS, client=make_client({}))
        assert await adapter.get_positions() == []

    async def test_balance_skips_empty_assets(self) -> None:
        payload = {
            "balances": [
                {"asset": "BTC", "free": "0.5", "locked": "0.0"},
                {"asset": "DOGE", "free": "0.0", "locked": "0.0"},
            ]
        }
        client = make_client({"/api/v3/account": payload})
        adapter = BinanceAdapter(CREDS, client=client)
        balance = await adapter.get_balance()
        assert "BTC" in balance.balances
        assert "DOGE" not in balance.balances
        await client.aclose()


# =========================================================================== #
# Preflight
# =========================================================================== #
def live_settings(**overrides: Any) -> Settings:
    base = {
        "app_env": AppEnv.DEVELOPMENT,
        "trading_mode": TradingMode.LIVE,
        "live_trading_enabled": True,
        "exchange": "bybit",
        "max_clock_drift_seconds": 5.0,
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def stocked_paper_exchange() -> PaperExchange:
    from datetime import UTC, datetime, timedelta

    from app.paper_trading.factory import build_paper_exchange

    exchange = build_paper_exchange(["BTCUSDT"], starting_balance=10_000.0)
    now = datetime.now(UTC)
    for i in range(120):
        exchange.process_candle(
            Candle(
                symbol="BTCUSDT", interval="15m",
                open_time=now - timedelta(minutes=15 * (120 - i)),
                open=100.0, high=101.0, low=99.0, close=100.0, volume=1_000.0,
            )
        )
    return exchange


class TestPreflight:
    async def test_blocks_when_platform_not_configured(
        self, stocked_paper_exchange: PaperExchange
    ) -> None:
        settings = Settings(trading_mode=TradingMode.PAPER, live_trading_enabled=False)
        report = await run_preflight(
            stocked_paper_exchange, symbols=["BTCUSDT"],
            risk_limits=RiskLimits.conservative(),
            confirmation=CONFIRMATION_PHRASE, settings=settings,
        )
        assert not report.passed
        assert any(c.name == "platform_configuration" for c in report.failures)

    async def test_blocks_without_confirmation(
        self, stocked_paper_exchange: PaperExchange
    ) -> None:
        report = await run_preflight(
            stocked_paper_exchange, symbols=["BTCUSDT"],
            risk_limits=RiskLimits.conservative(),
            confirmation=None, settings=live_settings(),
        )
        assert not report.passed
        assert any(c.name == "user_confirmation" for c in report.failures)

    async def test_blocks_on_wrong_confirmation_phrase(
        self, stocked_paper_exchange: PaperExchange
    ) -> None:
        report = await run_preflight(
            stocked_paper_exchange, symbols=["BTCUSDT"],
            risk_limits=RiskLimits.conservative(),
            confirmation="yes go ahead", settings=live_settings(),
        )
        assert not report.passed

    async def test_blocks_without_risk_config(
        self, stocked_paper_exchange: PaperExchange
    ) -> None:
        report = await run_preflight(
            stocked_paper_exchange, symbols=["BTCUSDT"], risk_limits=None,
            confirmation=CONFIRMATION_PHRASE, settings=live_settings(),
        )
        assert not report.passed
        assert any(c.name == "risk_configuration" for c in report.failures)

    async def test_blocks_on_zero_balance(self) -> None:
        from app.paper_trading.factory import build_paper_exchange

        exchange = build_paper_exchange(["BTCUSDT"], starting_balance=100.0)
        await exchange.withdraw(100.0)
        report = await run_preflight(
            exchange, symbols=["BTCUSDT"], risk_limits=RiskLimits.conservative(),
            confirmation=CONFIRMATION_PHRASE, settings=live_settings(),
        )
        assert not report.passed
        assert any(c.name == "account_balance" for c in report.failures)

    async def test_blocks_on_missing_market_data(self) -> None:
        from app.paper_trading.factory import build_paper_exchange

        exchange = build_paper_exchange(["BTCUSDT"], starting_balance=10_000.0)
        report = await run_preflight(
            exchange, symbols=["BTCUSDT"], risk_limits=RiskLimits.conservative(),
            confirmation=CONFIRMATION_PHRASE, settings=live_settings(),
        )
        assert not report.passed
        assert any(c.name == "market_data" for c in report.failures)

    async def test_report_lists_every_check(
        self, stocked_paper_exchange: PaperExchange
    ) -> None:
        report = await run_preflight(
            stocked_paper_exchange, symbols=["BTCUSDT"],
            risk_limits=RiskLimits.conservative(),
            confirmation=CONFIRMATION_PHRASE, settings=live_settings(),
        )
        names = {c.name for c in report.checks}
        assert names == {
            "platform_configuration",
            "user_confirmation",
            "risk_configuration",
            "api_credentials",
            "api_permissions",
            "account_balance",
            "exchange_connectivity",
            "clock_synchronisation",
            "market_data",
        }
        assert "LIVE TRADING PREFLIGHT" in report.summary()

    async def test_require_pass_raises_on_failure(
        self, stocked_paper_exchange: PaperExchange
    ) -> None:
        report = await run_preflight(
            stocked_paper_exchange, symbols=["BTCUSDT"],
            risk_limits=RiskLimits.conservative(),
            confirmation=None, settings=live_settings(),
        )
        with pytest.raises(PreflightFailedError):
            report.require_pass()

    async def test_serialises(self, stocked_paper_exchange: PaperExchange) -> None:
        report = await run_preflight(
            stocked_paper_exchange, symbols=["BTCUSDT"],
            risk_limits=RiskLimits.conservative(),
            confirmation=CONFIRMATION_PHRASE, settings=live_settings(),
        )
        payload = report.to_dict()
        assert isinstance(payload["passed"], bool)
        assert len(payload["checks"]) == 9


# =========================================================================== #
# Factory safety
# =========================================================================== #
class TestFactorySafety:
    async def test_live_bot_refused_when_disabled(self) -> None:
        from app.config import ExchangeName
        from app.core.exceptions import LiveTradingDisabledError
        from app.paper_trading.factory import build_live_bot

        with pytest.raises(LiveTradingDisabledError, match="Live trading is disabled"):
            await build_live_bot(
                bot_id="b1", user_id="u1", name="test",
                strategy_name="trend_following", symbols=["BTCUSDT"],
                credentials=CREDS, exchange_name=ExchangeName.BYBIT,
                settings=Settings(trading_mode=TradingMode.PAPER),
            )

    async def test_live_bot_refuses_paper_exchange(self) -> None:
        from app.config import ExchangeName
        from app.core.exceptions import ConfigurationError
        from app.paper_trading.factory import build_live_bot

        with pytest.raises(ConfigurationError, match="requires a real exchange"):
            await build_live_bot(
                bot_id="b1", user_id="u1", name="test",
                strategy_name="trend_following", symbols=["BTCUSDT"],
                credentials=CREDS, exchange_name=ExchangeName.PAPER,
                settings=live_settings(),
            )

    def test_paper_bot_is_never_live(self) -> None:
        from app.paper_trading.factory import build_paper_bot

        bot = build_paper_bot(
            bot_id="b1", user_id="u1", name="paper",
            strategy_name="trend_following", symbols=["BTCUSDT"],
        )
        assert bot.mode == "PAPER"
        assert bot.exchange.is_live is False
