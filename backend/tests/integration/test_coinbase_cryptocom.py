"""Coinbase and Crypto.com adapter tests.

Run against ``httpx.MockTransport`` reproducing each venue's documented envelopes and error
bodies. The properties under test are the ones that make a live adapter safe to hand real
money to:

* a key that can move funds off the venue is **refused**, before any order can be placed;
* a timeout maps to :class:`ExchangeTimeoutError` and never to a silent retry;
* ``client_order_id`` round-trips, so a timed-out submission can be resolved by asking;
* an unverifiable permission state is a refusal, not an assumption.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.core.domain import OrderRequest
from app.core.enums import OrderSide, OrderStatus, OrderType
from app.core.exceptions import (
    ConfigurationError,
    ExchangeAuthError,
    ExchangeTimeoutError,
    InstrumentNotSupportedError,
    UnsafeCredentialsError,
)
from app.exchanges.base import ExchangeCredentials
from app.exchanges.coinbase import CoinbaseAdapter
from app.exchanges.cryptocom import CryptoComAdapter
from app.exchanges.symbols import split_symbol

LIVE_CREDS = ExchangeCredentials(
    api_key="test-key-1234", api_secret="test-secret-abcd", testnet=False
)
TESTNET_CREDS = ExchangeCredentials(
    api_key="test-key-1234", api_secret="test-secret-abcd", testnet=True
)


def make_client(routes: dict[str, Any]) -> httpx.AsyncClient:
    """Mock transport dispatching on path.

    A route may be a callable, a ready-made ``httpx.Response`` (for non-200 statuses), or a
    payload to be returned with a 200.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        route = routes.get(request.url.path)
        if route is None:
            return httpx.Response(404, json={"message": f"no route for {request.url.path}"})
        if isinstance(route, httpx.Response):
            # Rebuilt each time: a Response instance cannot be streamed twice.
            return httpx.Response(
                route.status_code, content=route.content, headers=route.headers
            )
        if callable(route):
            return route(request)
        return httpx.Response(200, json=route)

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://mock.invalid"
    )


# =========================================================================== #
# Symbol splitting
# =========================================================================== #
class TestSymbolSplitting:
    @pytest.mark.parametrize(
        ("symbol", "expected"),
        [
            ("BTCUSDT", ("BTC", "USDT")),
            ("BTCUSD", ("BTC", "USD")),
            ("ETHUSDC", ("ETH", "USDC")),
            ("SOLEUR", ("SOL", "EUR")),
            ("ETHBTC", ("ETH", "BTC")),
        ],
    )
    def test_splits_on_the_longest_quote(
        self, symbol: str, expected: tuple[str, str]
    ) -> None:
        """USDT must win over USD, or BTCUSDT splits into BTCT."""
        assert split_symbol(symbol) == expected

    @pytest.mark.parametrize("symbol", ["BTC", "", "WEIRDPAIR", "USDT"])
    def test_refuses_rather_than_guesses(self, symbol: str) -> None:
        """Routing an order to the wrong instrument is worse than not routing one."""
        assert split_symbol(symbol) == (None, None)

    @pytest.mark.parametrize("symbol", ["BTC-USD", "BTC_USDT", "BTC/USD"])
    def test_separated_symbols_are_left_to_the_caller(self, symbol: str) -> None:
        assert split_symbol(symbol) == (None, None)


# =========================================================================== #
# Coinbase
# =========================================================================== #
COINBASE_SAFE_KEY = {
    "can_view": True,
    "can_trade": True,
    "can_transfer": False,
    "portfolio_uuid": "abc-123",
    "portfolio_type": "DEFAULT",
}

COINBASE_TRANSFER_KEY = {
    "can_view": True,
    "can_trade": True,
    "can_transfer": True,
    "portfolio_uuid": "abc-123",
    "portfolio_type": "DEFAULT",
}

COINBASE_PRODUCT = {
    "product_id": "BTC-USD",
    "base_currency_id": "BTC",
    "quote_currency_id": "USD",
    "quote_increment": "0.01",
    "base_increment": "0.00000001",
    "base_min_size": "0.0001",
    "base_max_size": "1000",
    "quote_min_size": "1",
    "trading_disabled": False,
}


class TestCoinbaseAdapter:
    def test_testnet_is_refused_outright(self) -> None:
        """There is no Coinbase sandbox. Pretending otherwise would route real orders."""
        with pytest.raises(ConfigurationError, match="no testnet"):
            CoinbaseAdapter(TESTNET_CREDS)

    @pytest.mark.asyncio
    async def test_safe_key_is_accepted(self) -> None:
        client = make_client({"/api/v3/brokerage/key_permissions": COINBASE_SAFE_KEY})
        adapter = CoinbaseAdapter(LIVE_CREDS, client=client)
        permissions = await adapter.validate_credentials()
        assert permissions.can_trade is True
        assert permissions.can_withdraw is False
        assert permissions.is_safe_for_trading is True

    @pytest.mark.asyncio
    async def test_transfer_permission_is_rejected(self) -> None:
        """Coinbase's transfer right moves money off the account. Same rule as withdrawal."""
        client = make_client({"/api/v3/brokerage/key_permissions": COINBASE_TRANSFER_KEY})
        adapter = CoinbaseAdapter(LIVE_CREDS, client=client)
        with pytest.raises(UnsafeCredentialsError, match="TRANSFER"):
            await adapter.validate_credentials()

    @pytest.mark.asyncio
    async def test_signature_covers_the_query_string(self) -> None:
        """A signature over the bare path fails authentication on every GET with params."""
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request.headers)
            return httpx.Response(200, json={"accounts": []})

        client = make_client({"/api/v3/brokerage/accounts": handler})
        adapter = CoinbaseAdapter(LIVE_CREDS, client=client)
        await adapter.get_balance()

        assert seen["cb-access-key"] == LIVE_CREDS.api_key
        assert len(seen["cb-access-sign"]) == 64  # hex sha256
        expected = adapter._hmac_sha256(
            LIVE_CREDS.api_secret,
            f"{seen['cb-access-timestamp']}GET/api/v3/brokerage/accounts?limit=250",
        )
        assert seen["cb-access-sign"] == expected

    @pytest.mark.asyncio
    async def test_secret_never_appears_in_headers(self) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request.headers)
            return httpx.Response(200, json={"accounts": []})

        client = make_client({"/api/v3/brokerage/accounts": handler})
        await CoinbaseAdapter(LIVE_CREDS, client=client).get_balance()
        assert LIVE_CREDS.api_secret not in json.dumps(dict(seen))

    @pytest.mark.asyncio
    async def test_platform_style_symbol_is_converted(self) -> None:
        client = make_client({"/api/v3/brokerage/products/BTC-USD": COINBASE_PRODUCT})
        adapter = CoinbaseAdapter(LIVE_CREDS, client=client)
        spec = await adapter.get_instrument("BTCUSD")
        assert spec.symbol == "BTC-USD"
        assert spec.base_asset == "BTC"

    @pytest.mark.asyncio
    async def test_unsplittable_symbol_is_refused(self) -> None:
        adapter = CoinbaseAdapter(LIVE_CREDS, client=make_client({}))
        with pytest.raises(InstrumentNotSupportedError, match="BASE-QUOTE"):
            await adapter.get_instrument("WEIRDPAIR")

    @pytest.mark.asyncio
    async def test_balance_parses_available_and_hold(self) -> None:
        client = make_client(
            {
                "/api/v3/brokerage/accounts": {
                    "accounts": [
                        {
                            "currency": "USD",
                            "available_balance": {"value": "900.5"},
                            "hold": {"value": "99.5"},
                        },
                        {
                            "currency": "ZERO",
                            "available_balance": {"value": "0"},
                            "hold": {"value": "0"},
                        },
                    ]
                }
            }
        )
        balance = await CoinbaseAdapter(LIVE_CREDS, client=client).get_balance()
        assert balance.balances["USD"].free == 900.5
        assert balance.balances["USD"].locked == 99.5
        assert "ZERO" not in balance.balances  # empty accounts are noise

    @pytest.mark.asyncio
    async def test_client_order_id_round_trips(self) -> None:
        """The property that makes a timed-out submission recoverable."""
        client = make_client(
            {
                "/api/v3/brokerage/products/BTC-USD": COINBASE_PRODUCT,
                "/api/v3/brokerage/orders": {
                    "success": True,
                    "success_response": {"order_id": "venue-1"},
                },
                "/api/v3/brokerage/orders/historical/batch": {
                    "orders": [
                        {
                            "order_id": "venue-1",
                            "client_order_id": "platform-42",
                            "product_id": "BTC-USD",
                            "side": "BUY",
                            "status": "OPEN",
                            "filled_size": "0",
                            "order_configuration": {
                                "limit_limit_gtc": {
                                    "base_size": "0.01",
                                    "limit_price": "50000.00",
                                }
                            },
                        }
                    ]
                },
            }
        )
        adapter = CoinbaseAdapter(LIVE_CREDS, client=client)
        order = await adapter.create_order(
            OrderRequest(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.01,
                price=50_000.0,
                client_order_id="platform-42",
            )
        )
        assert order.client_order_id == "platform-42"
        assert order.exchange_order_id == "venue-1"
        assert order.status is OrderStatus.OPEN

    @pytest.mark.asyncio
    async def test_market_order_uses_the_ioc_configuration(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"success": True, "success_response": {}})

        client = make_client(
            {
                "/api/v3/brokerage/products/BTC-USD": COINBASE_PRODUCT,
                "/api/v3/brokerage/orders": handler,
                "/api/v3/brokerage/orders/historical/batch": {"orders": []},
            }
        )
        adapter = CoinbaseAdapter(LIVE_CREDS, client=client)
        await adapter.create_order(
            OrderRequest(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.01,
            )
        )
        assert "market_market_ioc" in captured["order_configuration"]

    @pytest.mark.asyncio
    async def test_stop_direction_depends_on_order_type_not_only_side(self) -> None:
        """A sell stop-loss fires downward; a sell take-profit fires upward.

        Choosing on side alone makes every protective order fire the moment it is placed.
        """
        captured: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={"success": True, "success_response": {}})

        client = make_client(
            {
                "/api/v3/brokerage/products/BTC-USD": COINBASE_PRODUCT,
                "/api/v3/brokerage/orders": handler,
                "/api/v3/brokerage/orders/historical/batch": {"orders": []},
            }
        )
        adapter = CoinbaseAdapter(LIVE_CREDS, client=client)
        for order_type in (OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT):
            await adapter.create_order(
                OrderRequest(
                    symbol="BTC-USD",
                    side=OrderSide.SELL,
                    order_type=order_type,
                    quantity=0.01,
                    price=49_000.0,
                    trigger_price=49_500.0,
                )
            )

        stop_loss = captured[0]["order_configuration"]["stop_limit_stop_limit_gtc"]
        take_profit = captured[1]["order_configuration"]["stop_limit_stop_limit_gtc"]
        assert stop_loss["stop_direction"] == "STOP_DIRECTION_STOP_DOWN"
        assert take_profit["stop_direction"] == "STOP_DIRECTION_STOP_UP"

    @pytest.mark.asyncio
    async def test_timeout_is_never_a_silent_retry(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        client = make_client({"/api/v3/brokerage/accounts": handler})
        adapter = CoinbaseAdapter(LIVE_CREDS, client=client)
        with pytest.raises(ExchangeTimeoutError, match="querying"):
            await adapter.get_balance()

    @pytest.mark.asyncio
    async def test_spot_reports_no_positions(self) -> None:
        adapter = CoinbaseAdapter(LIVE_CREDS, client=make_client({}))
        assert await adapter.get_positions() == []


# =========================================================================== #
# Crypto.com
# =========================================================================== #
def crypto_ok(result: Any) -> dict[str, Any]:
    return {"id": 1, "method": "test", "code": 0, "result": result}


def crypto_error(code: int, message: str = "error") -> dict[str, Any]:
    return {"id": 1, "method": "test", "code": code, "message": message}


CRYPTO_INSTRUMENTS = crypto_ok(
    {
        "data": [
            {
                "symbol": "BTC_USDT",
                "base_ccy": "BTC",
                "quote_ccy": "USDT",
                "price_tick_size": "0.01",
                "qty_tick_size": "0.000001",
                "min_quantity": "0.0001",
                "max_quantity": "100",
                "tradable": True,
            }
        ]
    }
)

CRYPTO_BALANCE = crypto_ok(
    {
        "data": [
            {
                "position_balances": [
                    {"instrument_name": "USDT", "quantity": "1000", "reserved_qty": "100"}
                ]
            }
        ]
    }
)

V1 = "/exchange/v1"


class TestCryptoComSigning:
    def test_params_are_flattened_in_sorted_key_order(self) -> None:
        to_string = CryptoComAdapter._params_to_string
        assert to_string({"b": 2, "a": 1}) == "a1b2"

    def test_booleans_are_lowercase(self) -> None:
        """Python's `True` would produce a different string from the venue's `true`."""
        assert CryptoComAdapter._params_to_string({"flag": True}) == "flagtrue"
        assert CryptoComAdapter._params_to_string({"flag": False}) == "flagfalse"

    def test_none_becomes_null(self) -> None:
        assert CryptoComAdapter._params_to_string({"x": None}) == "xnull"

    def test_nested_structures_are_flattened(self) -> None:
        assert CryptoComAdapter._params_to_string({"a": [1, 2], "b": {"c": 3}}) == "a12bc3"

    def test_signature_matches_the_documented_recipe(self) -> None:
        adapter = CryptoComAdapter(LIVE_CREDS, client=make_client({}))
        envelope = adapter._signed_envelope("private/get-order-detail", {"order_id": "1"})
        expected = adapter._hmac_sha256(
            LIVE_CREDS.api_secret,
            f"private/get-order-detail{envelope['id']}{LIVE_CREDS.api_key}"
            f"order_id1{envelope['nonce']}",
        )
        assert envelope["sig"] == expected

    def test_secret_is_not_in_the_envelope(self) -> None:
        adapter = CryptoComAdapter(LIVE_CREDS, client=make_client({}))
        envelope = adapter._signed_envelope("private/user-balance", {})
        assert LIVE_CREDS.api_secret not in json.dumps(envelope)


class TestCryptoComCredentials:
    @pytest.mark.asyncio
    async def test_trade_only_key_is_accepted(self) -> None:
        """A key refused by the withdrawal endpoint is the shape we want."""
        client = make_client(
            {
                f"{V1}/private/user-balance": CRYPTO_BALANCE,
                f"{V1}/private/get-withdrawal-history": httpx.Response(
                    401, json=crypto_error(40101, "Unauthorized")
                ),
            }
        )
        adapter = CryptoComAdapter(LIVE_CREDS, client=client)
        permissions = await adapter.validate_credentials()
        assert permissions.can_withdraw is False
        assert permissions.is_safe_for_trading is True

    @pytest.mark.asyncio
    async def test_key_that_reaches_the_withdrawal_api_is_rejected(self) -> None:
        """Reaching that endpoint at all proves the key holds withdrawal rights."""
        client = make_client(
            {
                f"{V1}/private/user-balance": CRYPTO_BALANCE,
                f"{V1}/private/get-withdrawal-history": crypto_ok({"withdrawal_list": []}),
            }
        )
        adapter = CryptoComAdapter(LIVE_CREDS, client=client)
        with pytest.raises(UnsafeCredentialsError, match="withdrawal"):
            await adapter.validate_credentials()

    @pytest.mark.asyncio
    async def test_unverifiable_permissions_are_refused_not_assumed(self) -> None:
        """"Unknown" and "safe" must not collapse into the same answer."""
        client = make_client(
            {
                f"{V1}/private/user-balance": CRYPTO_BALANCE,
                f"{V1}/private/get-withdrawal-history": httpx.Response(
                    500, json=crypto_error(99999, "server exploded")
                ),
            }
        )
        adapter = CryptoComAdapter(LIVE_CREDS, client=client)
        with pytest.raises(UnsafeCredentialsError, match=r"[Cc]ould not establish"):
            await adapter.validate_credentials()

    @pytest.mark.asyncio
    async def test_unreadable_account_is_refused(self) -> None:
        client = make_client(
            {
                f"{V1}/private/user-balance": httpx.Response(
                    401, json=crypto_error(40101, "Unauthorized")
                )
            }
        )
        adapter = CryptoComAdapter(LIVE_CREDS, client=client)
        with pytest.raises(UnsafeCredentialsError, match="cannot read"):
            await adapter.validate_credentials()


class TestCryptoComAdapter:
    @pytest.mark.asyncio
    async def test_platform_style_symbol_is_converted(self) -> None:
        client = make_client({f"{V1}/public/get-instruments": CRYPTO_INSTRUMENTS})
        adapter = CryptoComAdapter(LIVE_CREDS, client=client)
        spec = await adapter.get_instrument("BTCUSDT")
        assert spec.symbol == "BTC_USDT"
        assert spec.base_asset == "BTC"

    @pytest.mark.asyncio
    async def test_balance_subtracts_reserved_quantity(self) -> None:
        client = make_client({f"{V1}/private/user-balance": CRYPTO_BALANCE})
        balance = await CryptoComAdapter(LIVE_CREDS, client=client).get_balance()
        assert balance.balances["USDT"].free == 900.0
        assert balance.balances["USDT"].locked == 100.0

    @pytest.mark.asyncio
    async def test_error_code_maps_to_insufficient_balance(self) -> None:
        from app.core.exceptions import InsufficientBalanceError

        client = make_client(
            {
                f"{V1}/private/user-balance": httpx.Response(
                    400, json=crypto_error(30004, "insufficient")
                )
            }
        )
        adapter = CryptoComAdapter(LIVE_CREDS, client=client)
        with pytest.raises(InsufficientBalanceError):
            await adapter.get_balance()

    @pytest.mark.asyncio
    async def test_nonzero_code_on_http_200_is_still_an_error(self) -> None:
        """The venue reports failures inside a 200 envelope as often as not."""
        client = make_client({f"{V1}/private/user-balance": crypto_error(40101, "nope")})
        adapter = CryptoComAdapter(LIVE_CREDS, client=client)
        with pytest.raises(ExchangeAuthError):
            await adapter.get_balance()

    @pytest.mark.asyncio
    async def test_client_order_id_round_trips(self) -> None:
        client = make_client(
            {
                f"{V1}/public/get-instruments": CRYPTO_INSTRUMENTS,
                f"{V1}/private/create-order": crypto_ok({"order_id": "venue-9"}),
                f"{V1}/private/get-order-detail": crypto_ok(
                    {
                        "data": [
                            {
                                "order_id": "venue-9",
                                "client_oid": "platform-77",
                                "instrument_name": "BTC_USDT",
                                "side": "BUY",
                                "type": "LIMIT",
                                "status": "ACTIVE",
                                "quantity": "0.01",
                                "price": "50000",
                                "cumulative_quantity": "0",
                            }
                        ]
                    }
                ),
            }
        )
        adapter = CryptoComAdapter(LIVE_CREDS, client=client)
        order = await adapter.create_order(
            OrderRequest(
                symbol="BTC_USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.01,
                price=50_000.0,
                client_order_id="platform-77",
            )
        )
        assert order.client_order_id == "platform-77"
        assert order.exchange_order_id == "venue-9"
        assert order.status is OrderStatus.OPEN

    @pytest.mark.asyncio
    async def test_average_fill_price_comes_from_value_over_quantity(self) -> None:
        client = make_client(
            {
                f"{V1}/private/get-order-detail": crypto_ok(
                    {
                        "data": [
                            {
                                "order_id": "v1",
                                "client_oid": "c1",
                                "instrument_name": "BTC_USDT",
                                "side": "BUY",
                                "type": "MARKET",
                                "status": "FILLED",
                                "quantity": "2",
                                "cumulative_quantity": "2",
                                "cumulative_value": "100000",
                            }
                        ]
                    }
                )
            }
        )
        adapter = CryptoComAdapter(LIVE_CREDS, client=client)
        order = await adapter.get_order("c1")
        assert order is not None
        assert order.average_fill_price == 50_000.0
        assert len(order.fills) == 1

    @pytest.mark.asyncio
    async def test_timeout_is_never_a_silent_retry(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        client = make_client({f"{V1}/private/create-order": handler})
        adapter = CryptoComAdapter(LIVE_CREDS, client=client)
        with pytest.raises(ExchangeTimeoutError, match="querying"):
            await adapter._private("private/create-order", {})

    @pytest.mark.asyncio
    async def test_uat_url_is_used_for_testnet(self) -> None:
        adapter = CryptoComAdapter(TESTNET_CREDS)
        assert "uat" in adapter.url
        await adapter.close()

    def test_testnet_is_still_treated_as_live(self) -> None:
        """The code path is the production one; only the venue's balances are fake."""
        assert CryptoComAdapter(TESTNET_CREDS).is_live is True


# =========================================================================== #
# Registry
# =========================================================================== #
class TestFactoryWiring:
    def test_both_venues_are_constructible(self) -> None:
        from app.config import ExchangeName
        from app.paper_trading.factory import build_exchange_adapter

        coinbase = build_exchange_adapter(ExchangeName.COINBASE, LIVE_CREDS)
        cryptocom = build_exchange_adapter(ExchangeName.CRYPTOCOM, LIVE_CREDS)
        assert coinbase.name == "coinbase"
        assert cryptocom.name == "cryptocom"

    def test_both_venues_require_credentials(self) -> None:
        from app.config import ExchangeName
        from app.paper_trading.factory import build_exchange_adapter

        for venue in (ExchangeName.COINBASE, ExchangeName.CRYPTOCOM):
            with pytest.raises(ConfigurationError, match="credentials"):
                build_exchange_adapter(venue, None)

    def test_default_exchange_is_still_paper(self) -> None:
        """Adding venues must not change which one is reached by default."""
        from app.config import ExchangeName, Settings

        assert Settings().exchange is ExchangeName.PAPER
