"""Bybit V5 adapter.

Implements the platform's :class:`~app.exchanges.base.ExchangeAdapter` contract against
Bybit's unified V5 REST API.

BLOCKED BY EXTERNAL DEPENDENCY: this adapter is fully implemented and tested against a mock
transport that reproduces Bybit's documented request signing, response envelopes and error
codes. It has **not** been verified against the live venue, which requires API credentials.
Before trading real money with it, follow ``docs/live-trading.md``: run it against Bybit
testnet first, verify fills and balances match, then reduce size and go live.

Safety behaviour specific to this adapter:

* :meth:`BybitAdapter.validate_credentials` reads the key's permission set and **rejects** any
  key with withdrawal rights before a single order can be placed.
* ``orderLinkId`` carries the platform's ``client_order_id``, which is what makes
  :meth:`get_order` able to resolve a timed-out submission.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

from app.core.clock import from_epoch_ms, to_epoch_ms, utcnow
from app.core.domain import (
    AccountBalance,
    Balance,
    Fill,
    InstrumentSpec,
    Order,
    OrderRequest,
    Position,
)
from app.core.enums import (
    InstrumentType,
    LiquidityRole,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    TimeInForce,
)
from app.core.exceptions import (
    ExchangeError,
    InstrumentNotSupportedError,
    UnsafeCredentialsError,
)
from app.core.logging import get_logger
from app.exchanges.base import AccountPermissions, ExchangeInfo
from app.exchanges.rest_base import RestExchangeAdapter
from app.market_data.models import (
    Candle,
    OrderBook,
    OrderBookLevel,
    PublicTrade,
    Ticker,
)

logger = get_logger(__name__)

#: Platform interval -> Bybit interval code.
INTERVAL_MAP: dict[str, str] = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
    "1d": "D", "1w": "W",
}

ORDER_TYPE_MAP: dict[OrderType, str] = {
    OrderType.MARKET: "Market",
    OrderType.LIMIT: "Limit",
    OrderType.STOP_MARKET: "Market",
    OrderType.STOP_LIMIT: "Limit",
    OrderType.TAKE_PROFIT_MARKET: "Market",
    OrderType.TAKE_PROFIT_LIMIT: "Limit",
    OrderType.TRAILING_STOP: "Market",
}

STATUS_MAP: dict[str, OrderStatus] = {
    "New": OrderStatus.OPEN,
    "Created": OrderStatus.SUBMITTED,
    "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
    "Filled": OrderStatus.FILLED,
    "Cancelled": OrderStatus.CANCELLED,
    "PartiallyFilledCanceled": OrderStatus.CANCELLED,
    "Rejected": OrderStatus.REJECTED,
    "Deactivated": OrderStatus.CANCELLED,
    "Triggered": OrderStatus.OPEN,
    "Untriggered": OrderStatus.OPEN,
}

TIF_MAP: dict[TimeInForce, str] = {
    TimeInForce.GTC: "GTC",
    TimeInForce.IOC: "IOC",
    TimeInForce.FOK: "FOK",
    TimeInForce.POST_ONLY: "PostOnly",
}


class BybitAdapter(RestExchangeAdapter):
    """Bybit unified V5 REST adapter."""

    name = "bybit"
    base_url = "https://api.bybit.com"
    testnet_url = "https://api-testnet.bybit.com"

    #: Bybit product category. "spot" or "linear" (USDT perpetuals).
    category: str = "spot"

    def __init__(self, *args: Any, category: str = "spot", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if category not in {"spot", "linear", "inverse"}:
            raise ValueError(f"Unsupported Bybit category: {category}")
        self.category = category
        self._instruments: dict[str, InstrumentSpec] = {}

    # ------------------------------------------------------------------ #
    # Signing
    # ------------------------------------------------------------------ #
    def _sign(
        self, method: str, path: str, params: dict[str, Any], body: str
    ) -> tuple[dict[str, str], dict[str, Any], str]:
        """Bybit V5: sign ``timestamp + api_key + recv_window + (query | body)``."""
        timestamp = str(self._timestamp_ms())
        recv_window = str(self.recv_window_ms)
        payload = urlencode(sorted(params.items())) if method == "GET" else body
        signature = self._hmac_sha256(
            self.credentials.api_secret,
            f"{timestamp}{self.credentials.api_key}{recv_window}{payload}",
        )
        headers = {
            "X-BAPI-API-KEY": self.credentials.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
        }
        return headers, params, body

    def _unwrap(self, payload: Any, path: str) -> Any:
        """Bybit wraps everything in ``{retCode, retMsg, result}``; non-zero is an error."""
        if not isinstance(payload, dict):
            raise ExchangeError(f"bybit returned an unexpected body on {path}")
        code = payload.get("retCode")
        if code not in (0, None):
            raise self._map_error(payload, path, 400)
        return payload.get("result", {})

    def _map_error(self, payload: Any, path: str, status: int) -> ExchangeError:
        if isinstance(payload, dict):
            code = payload.get("retCode")
            message = str(payload.get("retMsg", "")) or f"HTTP {status}"
            # Selected Bybit codes worth mapping precisely.
            if code in (110007, 170131):
                from app.core.exceptions import InsufficientBalanceError

                return InsufficientBalanceError(
                    f"bybit: {message}", context={"endpoint": path, "code": code}
                )
            if code in (110001, 110017):  # order does not exist / already closed
                from app.core.exceptions import OrderRejectedError

                return OrderRejectedError(
                    f"bybit: {message}", context={"endpoint": path, "code": code}
                )
        return super()._map_error(payload, path, status)

    # ------------------------------------------------------------------ #
    # Info and credentials
    # ------------------------------------------------------------------ #
    async def get_info(self) -> ExchangeInfo:
        result = await self._request("GET", "/v5/market/time")
        server_ms = int(result.get("timeNano", 0)) // 1_000_000 or int(
            result.get("timeSecond", 0)
        ) * 1000
        return ExchangeInfo(
            name=self.name,
            testnet=self.credentials.testnet,
            server_time=from_epoch_ms(server_ms) if server_ms else utcnow(),
            supports_websocket=True,
            supports_leverage=self.category != "spot",
            rate_limit_per_minute=600,
        )

    async def validate_credentials(self) -> AccountPermissions:
        """Read the key's permissions and refuse anything that can withdraw."""
        result = await self._request(
            "GET", "/v5/user/query-api", signed=True, bucket="private"
        )
        permissions = result.get("permissions", {}) or {}

        wallet = permissions.get("Wallet", []) or []
        withdraw = permissions.get("Withdraw", []) or []
        transfer = permissions.get("Exchange", []) or []
        spot = permissions.get("Spot", []) or []
        derivatives = permissions.get("Derivatives", []) or []
        contract = permissions.get("ContractTrade", []) or []

        can_withdraw = bool(withdraw) or "Withdraw" in wallet
        can_trade = bool(spot or derivatives or contract)
        can_transfer = "SubMemberTransfer" in wallet or bool(transfer)

        account = AccountPermissions(
            can_read=True,
            can_trade=can_trade,
            can_withdraw=can_withdraw,
            can_transfer=can_transfer,
            ip_restricted=bool(result.get("ips")) and result.get("ips") != ["*"],
            raw={"permissions": permissions},
        )
        if account.can_withdraw:
            logger.error("bybit.unsafe_key_rejected", reason="withdrawal permission present")
            raise UnsafeCredentialsError(
                "This Bybit API key has WITHDRAWAL permission. The platform refuses to use "
                "it. Create a new key with Read and Trade permissions only, and consider "
                "adding an IP allow-list."
            )
        return account

    # ------------------------------------------------------------------ #
    # Instruments
    # ------------------------------------------------------------------ #
    async def get_instruments(self) -> dict[str, InstrumentSpec]:
        result = await self._request(
            "GET", "/v5/market/instruments-info", params={"category": self.category}
        )
        specs: dict[str, InstrumentSpec] = {}
        for item in result.get("list", []) or []:
            try:
                spec = self._parse_instrument(item)
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "bybit.instrument_skipped",
                    symbol=item.get("symbol"), error=str(exc),
                )
                continue
            specs[spec.symbol] = spec
        self._instruments = specs
        return specs

    def _parse_instrument(self, item: dict[str, Any]) -> InstrumentSpec:
        lot = item.get("lotSizeFilter", {}) or {}
        price = item.get("priceFilter", {}) or {}
        leverage = item.get("leverageFilter", {}) or {}
        is_spot = self.category == "spot"
        return InstrumentSpec(
            symbol=item["symbol"],
            base_asset=item.get("baseCoin", ""),
            quote_asset=item.get("quoteCoin", "USDT"),
            instrument_type=(
                InstrumentType.SPOT if is_spot else InstrumentType.LINEAR_PERPETUAL
            ),
            tick_size=self._as_float(price.get("tickSize"), 0.01),
            lot_size=self._as_float(
                lot.get("basePrecision") or lot.get("qtyStep"), 0.000001
            ),
            min_quantity=self._as_float(lot.get("minOrderQty"), 0.0),
            max_quantity=self._as_float(lot.get("maxOrderQty"), 1e12),
            min_notional=self._as_float(lot.get("minOrderAmt"), 0.0),
            max_leverage=self._as_float(leverage.get("maxLeverage"), 1.0),
            maker_fee=0.0002,
            taker_fee=0.00055,
            is_active=item.get("status") == "Trading",
        )

    async def get_instrument(self, symbol: str) -> InstrumentSpec:
        if symbol in self._instruments:
            return self._instruments[symbol]
        result = await self._request(
            "GET",
            "/v5/market/instruments-info",
            params={"category": self.category, "symbol": symbol},
        )
        items = result.get("list", []) or []
        if not items:
            raise InstrumentNotSupportedError(
                f"bybit does not list {symbol} in category {self.category}",
                context={"symbol": symbol},
            )
        spec = self._parse_instrument(items[0])
        self._instruments[symbol] = spec
        return spec

    # ------------------------------------------------------------------ #
    # Market data
    # ------------------------------------------------------------------ #
    async def get_ticker(self, symbol: str) -> Ticker:
        result = await self._request(
            "GET",
            "/v5/market/tickers",
            params={"category": self.category, "symbol": symbol},
        )
        items = result.get("list", []) or []
        if not items:
            raise InstrumentNotSupportedError(f"bybit has no ticker for {symbol}")
        item = items[0]
        return Ticker(
            symbol=symbol,
            price=self._as_float(item.get("lastPrice")),
            timestamp=utcnow(),
            bid=self._as_float(item.get("bid1Price")) or None,
            ask=self._as_float(item.get("ask1Price")) or None,
            bid_size=self._as_float(item.get("bid1Size")) or None,
            ask_size=self._as_float(item.get("ask1Size")) or None,
            volume_24h=self._as_float(item.get("volume24h")) or None,
            price_change_24h_pct=self._as_float(item.get("price24hPcnt")) or None,
            funding_rate=self._as_float(item.get("fundingRate")) or None,
            open_interest=self._as_float(item.get("openInterest")) or None,
        )

    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        result = await self._request(
            "GET",
            "/v5/market/orderbook",
            params={"category": self.category, "symbol": symbol, "limit": depth},
        )
        return OrderBook(
            symbol=symbol,
            timestamp=from_epoch_ms(int(result.get("ts", self._timestamp_ms()))),
            bids=tuple(
                OrderBookLevel(price=float(p), quantity=float(q))
                for p, q in result.get("b", []) or []
            ),
            asks=tuple(
                OrderBookLevel(price=float(p), quantity=float(q))
                for p, q in result.get("a", []) or []
            ),
        )

    async def get_candles(
        self,
        symbol: str,
        interval: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[Candle]:
        code = INTERVAL_MAP.get(interval)
        if code is None:
            raise ExchangeError(f"bybit does not support the {interval} interval")
        params: dict[str, Any] = {
            "category": self.category,
            "symbol": symbol,
            "interval": code,
            "limit": min(limit, 1000),
        }
        if start is not None:
            params["start"] = to_epoch_ms(start)
        if end is not None:
            params["end"] = to_epoch_ms(end)

        result = await self._request("GET", "/v5/market/kline", params=params)
        candles: list[Candle] = []
        for row in result.get("list", []) or []:
            try:
                candles.append(
                    Candle(
                        symbol=symbol,
                        interval=interval,
                        open_time=from_epoch_ms(int(row[0])),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                        quote_volume=float(row[6]) if len(row) > 6 else None,
                    )
                )
            except (ValueError, IndexError, TypeError) as exc:
                logger.warning("bybit.candle_skipped", symbol=symbol, error=str(exc))
        # Bybit returns newest first.
        return sorted(candles, key=lambda c: c.open_time)

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[PublicTrade]:
        result = await self._request(
            "GET",
            "/v5/market/recent-trade",
            params={"category": self.category, "symbol": symbol, "limit": limit},
        )
        return [
            PublicTrade(
                symbol=symbol,
                price=float(item["price"]),
                quantity=float(item["size"]),
                side=str(item.get("side", "")).lower(),
                timestamp=from_epoch_ms(int(item["time"])),
                trade_id=item.get("execId"),
            )
            for item in result.get("list", []) or []
        ]

    # ------------------------------------------------------------------ #
    # Account
    # ------------------------------------------------------------------ #
    async def get_balance(self) -> AccountBalance:
        result = await self._request(
            "GET",
            "/v5/account/wallet-balance",
            params={"accountType": "UNIFIED"},
            signed=True,
            bucket="private",
        )
        balances: dict[str, Balance] = {}
        for account in result.get("list", []) or []:
            for coin in account.get("coin", []) or []:
                asset = coin.get("coin")
                if not asset:
                    continue
                total = self._as_float(coin.get("walletBalance"))
                locked = self._as_float(coin.get("locked"))
                balances[asset] = Balance(
                    asset=asset, free=max(0.0, total - locked), locked=locked
                )
        return AccountBalance(balances=balances, timestamp=utcnow())

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        if self.category == "spot":
            return []  # spot has balances, not positions
        params: dict[str, Any] = {"category": self.category, "settleCoin": "USDT"}
        if symbol:
            params["symbol"] = symbol
        result = await self._request(
            "GET", "/v5/position/list", params=params, signed=True, bucket="private"
        )
        positions: list[Position] = []
        for item in result.get("list", []) or []:
            size = self._as_float(item.get("size"))
            if size <= 0:
                continue
            side = (
                PositionSide.LONG
                if str(item.get("side", "")).lower() == "buy"
                else PositionSide.SHORT
            )
            positions.append(
                Position(
                    symbol=item["symbol"],
                    side=side,
                    quantity=size,
                    entry_price=self._as_float(item.get("avgPrice")),
                    mark_price=self._as_float(item.get("markPrice")) or None,
                    leverage=self._as_float(item.get("leverage"), 1.0),
                    stop_loss=self._as_float(item.get("stopLoss")) or None,
                    take_profit=self._as_float(item.get("takeProfit")) or None,
                    realized_pnl=self._as_float(item.get("curRealisedPnl")),
                    opened_at=from_epoch_ms(
                        int(item.get("createdTime", self._timestamp_ms()))
                    ),
                )
            )
        return positions

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        params: dict[str, Any] = {"category": self.category}
        if symbol:
            params["symbol"] = symbol
        elif self.category != "spot":
            params["settleCoin"] = "USDT"
        result = await self._request(
            "GET", "/v5/order/realtime", params=params, signed=True, bucket="private"
        )
        return [self._parse_order(item) for item in result.get("list", []) or []]

    async def get_order(
        self, client_order_id: str, *, symbol: str | None = None
    ) -> Order | None:
        """Look an order up by ``orderLinkId``.

        This is the method that makes timeout recovery safe: it consults open orders and then
        history, and returns ``None`` only when the venue genuinely has no record.
        """
        params: dict[str, Any] = {
            "category": self.category,
            "orderLinkId": client_order_id,
        }
        if symbol:
            params["symbol"] = symbol

        for endpoint in ("/v5/order/realtime", "/v5/order/history"):
            result = await self._request(
                "GET", endpoint, params=params, signed=True, bucket="private"
            )
            items = result.get("list", []) or []
            if items:
                return self._parse_order(items[0])
        return None

    # ------------------------------------------------------------------ #
    # Trading
    # ------------------------------------------------------------------ #
    async def create_order(self, request: OrderRequest) -> Order:
        from app.core.exceptions import OrderRejectedError

        spec = await self.get_instrument(request.symbol)
        quantity = spec.round_quantity(request.quantity)
        if quantity <= 0:
            raise OrderRejectedError(
                f"bybit: quantity rounds to zero at step {spec.lot_size:g}"
            )
        # Notional can only be checked when a price is known. A market order's fill price is
        # not, so the venue applies the minimum-notional filter itself and the platform
        # surfaces that rejection rather than guessing at a price here.
        if request.price is not None:
            problem = spec.validate_order(quantity, request.price)
            if problem is not None:
                raise OrderRejectedError(f"bybit: {problem}")

        qty_decimals = _decimals(spec.lot_size)
        price_decimals = _decimals(spec.tick_size)
        body: dict[str, Any] = {
            "category": self.category,
            "symbol": request.symbol,
            "side": "Buy" if request.side is OrderSide.BUY else "Sell",
            "orderType": ORDER_TYPE_MAP[request.order_type],
            "qty": f"{quantity:.{qty_decimals}f}",
            "orderLinkId": request.client_order_id,
            "timeInForce": TIF_MAP[request.time_in_force],
        }
        if request.price is not None:
            limit_price = spec.round_price(request.price)
            body["price"] = f"{limit_price:.{price_decimals}f}"
        if request.trigger_price is not None:
            trigger = spec.round_price(request.trigger_price)
            body["triggerPrice"] = f"{trigger:.{price_decimals}f}"
            body["triggerDirection"] = 1 if request.side is OrderSide.BUY else 2
        if request.reduce_only:
            body["reduceOnly"] = True

        await self._request(
            "POST", "/v5/order/create", body=body, signed=True, bucket="order", cost=1.0
        )
        # Bybit's create response carries only the ids; read back the full state.
        order = await self.get_order(request.client_order_id, symbol=request.symbol)
        if order is not None:
            return order
        # Acknowledged but not yet queryable: return a local view rather than guessing.
        local = Order.from_request(request)
        local.status = OrderStatus.SUBMITTED
        return local

    async def cancel_order(
        self, client_order_id: str, *, symbol: str | None = None
    ) -> Order:
        existing = await self.get_order(client_order_id, symbol=symbol)
        if existing is not None and existing.status.is_terminal:
            return existing
        body = {
            "category": self.category,
            "symbol": symbol or (existing.symbol if existing else ""),
            "orderLinkId": client_order_id,
        }
        await self._request(
            "POST", "/v5/order/cancel", body=body, signed=True, bucket="order"
        )
        refreshed = await self.get_order(client_order_id, symbol=symbol)
        if refreshed is not None:
            return refreshed
        if existing is not None:
            existing.status = OrderStatus.CANCELLED
            return existing
        raise ExchangeError(f"bybit: cannot resolve order {client_order_id} after cancel")

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #
    def _parse_order(self, item: dict[str, Any]) -> Order:
        filled = self._as_float(item.get("cumExecQty"))
        order = Order(
            client_order_id=item.get("orderLinkId") or item.get("orderId", ""),
            symbol=item.get("symbol", ""),
            side=(
                OrderSide.BUY
                if str(item.get("side", "")).lower() == "buy"
                else OrderSide.SELL
            ),
            order_type=(
                OrderType.LIMIT
                if str(item.get("orderType", "")).lower() == "limit"
                else OrderType.MARKET
            ),
            quantity=self._as_float(item.get("qty")),
            status=STATUS_MAP.get(str(item.get("orderStatus", "")), OrderStatus.UNKNOWN),
            exchange_order_id=item.get("orderId"),
            price=self._as_float(item.get("price")) or None,
            trigger_price=self._as_float(item.get("triggerPrice")) or None,
            filled_quantity=filled,
            average_fill_price=self._as_float(item.get("avgPrice")),
            fees_paid=self._as_float(item.get("cumExecFee")),
            reduce_only=bool(item.get("reduceOnly")),
            reject_reason=item.get("rejectReason") or None,
            created_at=from_epoch_ms(int(item.get("createdTime", self._timestamp_ms()))),
            updated_at=from_epoch_ms(int(item.get("updatedTime", self._timestamp_ms()))),
        )
        if filled > 0:
            order.fills.append(
                Fill(
                    fill_id=item.get("orderId", "") or order.client_order_id,
                    order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=filled,
                    price=order.average_fill_price or order.price or 1.0,
                    fee=order.fees_paid,
                    fee_asset="USDT",
                    role=LiquidityRole.TAKER,
                    timestamp=order.updated_at,
                )
            )
        return order

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    async def subscribe_candles(self, symbol: str, interval: str) -> AsyncIterator[Candle]:
        """Poll for closed candles.

        A REST poll rather than the websocket stream: it is simpler, has no reconnect state
        machine to get wrong, and at bar-close cadence the latency difference does not affect
        a strategy that only acts on closed bars. A websocket implementation belongs here when
        sub-bar reaction time is actually needed.
        """
        import asyncio

        from app.core.clock import interval_to_timedelta

        step = interval_to_timedelta(interval).total_seconds()
        last_open: datetime | None = None
        while True:
            try:
                candles = await self.get_candles(symbol, interval, limit=2)
            except ExchangeError as exc:
                logger.warning("bybit.stream_poll_failed", symbol=symbol, error=str(exc))
                await asyncio.sleep(min(step, 30.0))
                continue
            for candle in candles:
                if last_open is None or candle.open_time > last_open:
                    last_open = candle.open_time
                    yield candle
            await asyncio.sleep(max(1.0, step / 4.0))

    async def subscribe_ticker(self, symbol: str) -> AsyncIterator[Ticker]:
        import asyncio

        while True:
            try:
                yield await self.get_ticker(symbol)
            except ExchangeError as exc:
                logger.warning("bybit.ticker_poll_failed", symbol=symbol, error=str(exc))
            await asyncio.sleep(2.0)


def _decimals(step: float) -> int:
    from app.core.numeric import decimals_for_step

    return decimals_for_step(step)
