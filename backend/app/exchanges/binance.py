"""Binance Spot adapter.

Implements the platform's :class:`~app.exchanges.base.ExchangeAdapter` contract against
Binance's Spot REST API.

BLOCKED BY EXTERNAL DEPENDENCY: implemented and tested against a mock transport reproducing
Binance's documented signing, filters and error codes, but not verified against the live venue,
which requires API credentials. Follow ``docs/live-trading.md`` before using it with real money.

Safety behaviour specific to this adapter:

* :meth:`BinanceAdapter.validate_credentials` reads ``enableWithdrawals`` and
  ``permitsUniversalTransfer`` from the account's API restrictions and **rejects** any key that
  can move funds.
* ``newClientOrderId`` carries the platform's ``client_order_id``, which makes timed-out
  submissions resolvable by query.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

from app.core.clock import from_epoch_ms, interval_to_timedelta, to_epoch_ms, utcnow
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
    TimeInForce,
)
from app.core.exceptions import (
    ExchangeError,
    InstrumentNotSupportedError,
    OrderRejectedError,
    UnsafeCredentialsError,
)
from app.core.logging import get_logger
from app.core.numeric import decimals_for_step
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

INTERVAL_MAP: dict[str, str] = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h",
    "1d": "1d", "1w": "1w",
}

ORDER_TYPE_MAP: dict[OrderType, str] = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.STOP_MARKET: "STOP_LOSS",
    OrderType.STOP_LIMIT: "STOP_LOSS_LIMIT",
    OrderType.TAKE_PROFIT_MARKET: "TAKE_PROFIT",
    OrderType.TAKE_PROFIT_LIMIT: "TAKE_PROFIT_LIMIT",
    OrderType.TRAILING_STOP: "MARKET",
}

STATUS_MAP: dict[str, OrderStatus] = {
    "NEW": OrderStatus.OPEN,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELLED,
    "PENDING_CANCEL": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "EXPIRED": OrderStatus.EXPIRED,
    "EXPIRED_IN_MATCH": OrderStatus.EXPIRED,
}

TIF_MAP: dict[TimeInForce, str] = {
    TimeInForce.GTC: "GTC",
    TimeInForce.IOC: "IOC",
    TimeInForce.FOK: "FOK",
    TimeInForce.POST_ONLY: "GTC",  # Binance expresses post-only as LIMIT_MAKER
}


class BinanceAdapter(RestExchangeAdapter):
    """Binance Spot REST adapter."""

    name = "binance"
    base_url = "https://api.binance.com"
    testnet_url = "https://testnet.binance.vision"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._instruments: dict[str, InstrumentSpec] = {}

    # ------------------------------------------------------------------ #
    # Signing
    # ------------------------------------------------------------------ #
    def _sign(
        self, method: str, path: str, params: dict[str, Any], body: str
    ) -> tuple[dict[str, str], dict[str, Any], str]:
        """Binance signs the full query string, including ``timestamp`` and ``recvWindow``.

        Everything goes in the query string even for POST, which is what the venue expects.
        """
        signed_params = dict(params)
        signed_params["timestamp"] = self._timestamp_ms()
        signed_params["recvWindow"] = self.recv_window_ms
        signature = self._hmac_sha256(
            self.credentials.api_secret, urlencode(signed_params)
        )
        signed_params["signature"] = signature
        headers = {"X-MBX-APIKEY": self.credentials.api_key}
        return headers, signed_params, ""

    def _unwrap(self, payload: Any, path: str) -> Any:
        """Binance returns bare payloads; an error body carries ``code`` and ``msg``."""
        if isinstance(payload, dict) and "code" in payload and "msg" in payload:
            code = payload.get("code")
            if isinstance(code, int) and code < 0:
                raise self._map_error(payload, path, 400)
        return payload

    def _map_error(self, payload: Any, path: str, status: int) -> ExchangeError:
        if isinstance(payload, dict):
            code = payload.get("code")
            message = str(payload.get("msg", "")) or f"HTTP {status}"
            if code == -2010:  # insufficient balance
                from app.core.exceptions import InsufficientBalanceError

                return InsufficientBalanceError(
                    f"binance: {message}", context={"endpoint": path, "code": code}
                )
            if code in (-1013, -1111, -1121):  # filter failure / precision / bad symbol
                return OrderRejectedError(
                    f"binance: {message}", context={"endpoint": path, "code": code}
                )
            if code == -2011:  # unknown order
                return OrderRejectedError(
                    f"binance: {message}", context={"endpoint": path, "code": code}
                )
        return super()._map_error(payload, path, status)

    # ------------------------------------------------------------------ #
    # Info and credentials
    # ------------------------------------------------------------------ #
    async def get_info(self) -> ExchangeInfo:
        payload = await self._request("GET", "/api/v3/time")
        return ExchangeInfo(
            name=self.name,
            testnet=self.credentials.testnet,
            server_time=from_epoch_ms(int(payload.get("serverTime", self._timestamp_ms()))),
            supports_websocket=True,
            supports_leverage=False,
            rate_limit_per_minute=1200,
        )

    async def validate_credentials(self) -> AccountPermissions:
        """Read API restrictions and refuse any key that can move funds off the venue."""
        payload = await self._request(
            "GET",
            "/sapi/v1/account/apiRestrictions",
            signed=True,
            bucket="private",
        )
        can_withdraw = bool(payload.get("enableWithdrawals"))
        can_transfer = bool(
            payload.get("permitsUniversalTransfer")
            or payload.get("enableInternalTransfer")
        )
        account = AccountPermissions(
            can_read=bool(payload.get("enableReading", True)),
            can_trade=bool(payload.get("enableSpotAndMarginTrading")),
            can_withdraw=can_withdraw,
            can_transfer=can_transfer,
            ip_restricted=bool(payload.get("ipRestrict")),
            raw=dict(payload),
        )
        if account.can_withdraw or account.can_transfer:
            logger.error(
                "binance.unsafe_key_rejected",
                withdraw=can_withdraw,
                transfer=can_transfer,
            )
            raise UnsafeCredentialsError(
                "This Binance API key can withdraw or transfer funds. The platform refuses "
                "to use it. Create a key with 'Enable Reading' and 'Enable Spot Trading' "
                "only, with withdrawals and universal transfer disabled."
            )
        return account

    # ------------------------------------------------------------------ #
    # Instruments
    # ------------------------------------------------------------------ #
    async def get_instruments(self) -> dict[str, InstrumentSpec]:
        payload = await self._request("GET", "/api/v3/exchangeInfo")
        specs: dict[str, InstrumentSpec] = {}
        for item in payload.get("symbols", []) or []:
            try:
                spec = self._parse_instrument(item)
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "binance.instrument_skipped",
                    symbol=item.get("symbol"), error=str(exc),
                )
                continue
            specs[spec.symbol] = spec
        self._instruments = specs
        return specs

    def _parse_instrument(self, item: dict[str, Any]) -> InstrumentSpec:
        filters = {f["filterType"]: f for f in item.get("filters", []) or []}
        price_filter = filters.get("PRICE_FILTER", {})
        lot_filter = filters.get("LOT_SIZE", {})
        notional = filters.get("NOTIONAL", filters.get("MIN_NOTIONAL", {}))
        return InstrumentSpec(
            symbol=item["symbol"],
            base_asset=item.get("baseAsset", ""),
            quote_asset=item.get("quoteAsset", "USDT"),
            instrument_type=InstrumentType.SPOT,
            tick_size=self._as_float(price_filter.get("tickSize"), 0.01),
            lot_size=self._as_float(lot_filter.get("stepSize"), 0.000001),
            min_quantity=self._as_float(lot_filter.get("minQty"), 0.0),
            max_quantity=self._as_float(lot_filter.get("maxQty"), 1e12),
            min_notional=self._as_float(
                notional.get("minNotional") or notional.get("notional"), 0.0
            ),
            max_leverage=1.0,
            maker_fee=0.001,
            taker_fee=0.001,
            is_active=item.get("status") == "TRADING",
        )

    async def get_instrument(self, symbol: str) -> InstrumentSpec:
        if symbol in self._instruments:
            return self._instruments[symbol]
        payload = await self._request(
            "GET", "/api/v3/exchangeInfo", params={"symbol": symbol}
        )
        items = payload.get("symbols", []) or []
        if not items:
            raise InstrumentNotSupportedError(
                f"binance does not list {symbol}", context={"symbol": symbol}
            )
        spec = self._parse_instrument(items[0])
        self._instruments[symbol] = spec
        return spec

    # ------------------------------------------------------------------ #
    # Market data
    # ------------------------------------------------------------------ #
    async def get_ticker(self, symbol: str) -> Ticker:
        book = await self._request(
            "GET", "/api/v3/ticker/bookTicker", params={"symbol": symbol}
        )
        stats = await self._request(
            "GET", "/api/v3/ticker/24hr", params={"symbol": symbol}
        )
        bid = self._as_float(book.get("bidPrice")) or None
        ask = self._as_float(book.get("askPrice")) or None
        last = self._as_float(stats.get("lastPrice"))
        return Ticker(
            symbol=symbol,
            price=last or ((bid or 0.0) + (ask or 0.0)) / 2.0,
            timestamp=utcnow(),
            bid=bid,
            ask=ask,
            bid_size=self._as_float(book.get("bidQty")) or None,
            ask_size=self._as_float(book.get("askQty")) or None,
            volume_24h=self._as_float(stats.get("volume")) or None,
            price_change_24h_pct=self._as_float(stats.get("priceChangePercent")) / 100.0
            or None,
        )

    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        payload = await self._request(
            "GET", "/api/v3/depth", params={"symbol": symbol, "limit": depth}
        )
        return OrderBook(
            symbol=symbol,
            timestamp=utcnow(),
            bids=tuple(
                OrderBookLevel(price=float(p), quantity=float(q))
                for p, q in payload.get("bids", []) or []
            ),
            asks=tuple(
                OrderBookLevel(price=float(p), quantity=float(q))
                for p, q in payload.get("asks", []) or []
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
            raise ExchangeError(f"binance does not support the {interval} interval")
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": code,
            "limit": min(limit, 1000),
        }
        if start is not None:
            params["startTime"] = to_epoch_ms(start)
        if end is not None:
            params["endTime"] = to_epoch_ms(end)

        rows = await self._request("GET", "/api/v3/klines", params=params)
        candles: list[Candle] = []
        for row in rows or []:
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
                        quote_volume=float(row[7]) if len(row) > 7 else None,
                        trade_count=int(row[8]) if len(row) > 8 else None,
                    )
                )
            except (ValueError, IndexError, TypeError) as exc:
                logger.warning("binance.candle_skipped", symbol=symbol, error=str(exc))
        return candles

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[PublicTrade]:
        rows = await self._request(
            "GET", "/api/v3/trades", params={"symbol": symbol, "limit": limit}
        )
        return [
            PublicTrade(
                symbol=symbol,
                price=float(item["price"]),
                quantity=float(item["qty"]),
                side="sell" if item.get("isBuyerMaker") else "buy",
                timestamp=from_epoch_ms(int(item["time"])),
                trade_id=str(item.get("id")),
            )
            for item in rows or []
        ]

    # ------------------------------------------------------------------ #
    # Account
    # ------------------------------------------------------------------ #
    async def get_balance(self) -> AccountBalance:
        payload = await self._request(
            "GET", "/api/v3/account", signed=True, bucket="private"
        )
        balances: dict[str, Balance] = {}
        for item in payload.get("balances", []) or []:
            free = self._as_float(item.get("free"))
            locked = self._as_float(item.get("locked"))
            if free <= 0 and locked <= 0:
                continue
            balances[item["asset"]] = Balance(
                asset=item["asset"], free=free, locked=locked
            )
        return AccountBalance(balances=balances, timestamp=utcnow())

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Spot has no positions.

        Returning an empty list rather than synthesising positions from balances is
        deliberate: the platform's reconciliation compares *positions*, and inventing them
        from balances would make every manual deposit look like a phantom trade.
        """
        return []

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        params = {"symbol": symbol} if symbol else {}
        rows = await self._request(
            "GET", "/api/v3/openOrders", params=params, signed=True, bucket="private"
        )
        return [self._parse_order(item) for item in rows or []]

    async def get_order(
        self, client_order_id: str, *, symbol: str | None = None
    ) -> Order | None:
        """Look an order up by ``origClientOrderId``.

        Binance requires a symbol for this endpoint. When the caller does not supply one, the
        open-order list is searched instead — which is enough for the timeout-recovery path,
        since a just-submitted order is either open or immediately terminal.
        """
        if symbol is None:
            for order in await self.get_open_orders():
                if order.client_order_id == client_order_id:
                    return order
            return None
        try:
            payload = await self._request(
                "GET",
                "/api/v3/order",
                params={"symbol": symbol, "origClientOrderId": client_order_id},
                signed=True,
                bucket="private",
            )
        except OrderRejectedError:
            return None  # -2011 "Unknown order sent": definitively not present
        return self._parse_order(payload)

    # ------------------------------------------------------------------ #
    # Trading
    # ------------------------------------------------------------------ #
    async def create_order(self, request: OrderRequest) -> Order:
        spec = await self.get_instrument(request.symbol)
        quantity = spec.round_quantity(request.quantity)
        if quantity <= 0:
            raise OrderRejectedError(
                f"binance: quantity rounds to zero at step {spec.lot_size:g}"
            )

        order_type = ORDER_TYPE_MAP[request.order_type]
        if (
            request.time_in_force is TimeInForce.POST_ONLY
            and request.order_type is OrderType.LIMIT
        ):
            order_type = "LIMIT_MAKER"

        params: dict[str, Any] = {
            "symbol": request.symbol,
            "side": "BUY" if request.side is OrderSide.BUY else "SELL",
            "type": order_type,
            "quantity": f"{quantity:.{decimals_for_step(spec.lot_size)}f}",
            "newClientOrderId": request.client_order_id,
        }
        if request.price is not None:
            limit_price = spec.round_price(request.price)
            params["price"] = f"{limit_price:.{decimals_for_step(spec.tick_size)}f}"
        if order_type in {"LIMIT", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"}:
            params["timeInForce"] = TIF_MAP[request.time_in_force]
        if request.trigger_price is not None:
            price_decimals = decimals_for_step(spec.tick_size)
            trigger = spec.round_price(request.trigger_price)
            params["stopPrice"] = f"{trigger:.{price_decimals}f}"

        payload = await self._request(
            "POST", "/api/v3/order", params=params, signed=True, bucket="order"
        )
        return self._parse_order(payload)

    async def cancel_order(
        self, client_order_id: str, *, symbol: str | None = None
    ) -> Order:
        if symbol is None:
            existing = await self.get_order(client_order_id)
            if existing is None:
                raise ExchangeError(
                    f"binance: cannot cancel {client_order_id} without a symbol"
                )
            symbol = existing.symbol
        try:
            payload = await self._request(
                "DELETE",
                "/api/v3/order",
                params={"symbol": symbol, "origClientOrderId": client_order_id},
                signed=True,
                bucket="order",
            )
        except OrderRejectedError:
            # Already terminal. Report its real state instead of failing the caller.
            existing = await self.get_order(client_order_id, symbol=symbol)
            if existing is not None:
                return existing
            raise
        return self._parse_order(payload)

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #
    def _parse_order(self, item: dict[str, Any]) -> Order:
        filled = self._as_float(item.get("executedQty"))
        quote_filled = self._as_float(item.get("cummulativeQuoteQty"))
        average = quote_filled / filled if filled > 0 else 0.0
        transact_ms = int(
            item.get("transactTime") or item.get("updateTime") or item.get("time")
            or self._timestamp_ms()
        )
        order = Order(
            client_order_id=item.get("clientOrderId") or str(item.get("orderId", "")),
            symbol=item.get("symbol", ""),
            side=(
                OrderSide.BUY
                if str(item.get("side", "")).upper() == "BUY"
                else OrderSide.SELL
            ),
            order_type=(
                OrderType.LIMIT
                if "LIMIT" in str(item.get("type", "")).upper()
                else OrderType.MARKET
            ),
            quantity=self._as_float(item.get("origQty")),
            status=STATUS_MAP.get(str(item.get("status", "")), OrderStatus.UNKNOWN),
            exchange_order_id=str(item.get("orderId")) if item.get("orderId") else None,
            price=self._as_float(item.get("price")) or None,
            trigger_price=self._as_float(item.get("stopPrice")) or None,
            filled_quantity=filled,
            average_fill_price=average,
            created_at=from_epoch_ms(transact_ms),
            updated_at=from_epoch_ms(transact_ms),
        )
        fills = item.get("fills") or []
        for entry in fills:
            order.fills.append(
                Fill(
                    fill_id=str(entry.get("tradeId", "")) or order.client_order_id,
                    order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=self._as_float(entry.get("qty")),
                    price=self._as_float(entry.get("price")),
                    fee=self._as_float(entry.get("commission")),
                    fee_asset=entry.get("commissionAsset", "USDT"),
                    role=LiquidityRole.TAKER,
                    timestamp=order.updated_at,
                )
            )
        if not fills and filled > 0:
            order.fills.append(
                Fill(
                    fill_id=order.client_order_id,
                    order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=filled,
                    price=average or (order.price or 1.0),
                    fee=0.0,
                    fee_asset="USDT",
                    role=LiquidityRole.TAKER,
                    timestamp=order.updated_at,
                )
            )
        order.fees_paid = sum(f.fee for f in order.fills)
        return order

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    async def subscribe_candles(self, symbol: str, interval: str) -> AsyncIterator[Candle]:
        """Poll for closed candles. See the note in the Bybit adapter."""
        step = interval_to_timedelta(interval).total_seconds()
        last_open: datetime | None = None
        while True:
            try:
                candles = await self.get_candles(symbol, interval, limit=2)
            except ExchangeError as exc:
                logger.warning("binance.stream_poll_failed", symbol=symbol, error=str(exc))
                await asyncio.sleep(min(step, 30.0))
                continue
            for candle in candles:
                if last_open is None or candle.open_time > last_open:
                    last_open = candle.open_time
                    yield candle
            await asyncio.sleep(max(1.0, step / 4.0))

    async def subscribe_ticker(self, symbol: str) -> AsyncIterator[Ticker]:
        while True:
            try:
                yield await self.get_ticker(symbol)
            except ExchangeError as exc:
                logger.warning("binance.ticker_poll_failed", symbol=symbol, error=str(exc))
            await asyncio.sleep(2.0)
