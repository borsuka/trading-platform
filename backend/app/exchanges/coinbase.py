"""Coinbase Advanced Trade adapter.

Implements the platform's :class:`~app.exchanges.base.ExchangeAdapter` contract against
Coinbase's Advanced Trade REST API (``/api/v3/brokerage``).

BLOCKED BY EXTERNAL DEPENDENCY: this adapter is fully implemented and tested against a mock
transport reproducing Coinbase's documented signing, envelopes and error bodies. It has **not**
been verified against the live venue, which requires API credentials. Before trading real money
with it, follow ``docs/live-trading.md``.

Three things differ from the other venues and are worth knowing before you use it:

* **There is no testnet.** Coinbase retired the Advanced Trade sandbox, so unlike Bybit and
  Binance there is no way to rehearse against fake balances. Rather than silently sending
  ``testnet=True`` orders to the production venue - which would be the worst possible
  interpretation - the adapter refuses to construct at all. Paper mode is the rehearsal.
* **Symbols are ``BASE-QUOTE``**, e.g. ``BTC-USD``. The platform's ``BTCUSD`` style is accepted
  and converted; anything ambiguous is rejected rather than guessed at.
* **Permissions are explicit.** ``/key_permissions`` reports ``can_trade`` and ``can_transfer``
  directly, and ``can_transfer`` is what moves money off the venue, so it is treated exactly
  like a withdrawal right: refused.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

from app.core.clock import ensure_utc, from_epoch_ms, utcnow
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
    ConfigurationError,
    ExchangeError,
    InstrumentNotSupportedError,
    OrderRejectedError,
    UnsafeCredentialsError,
)
from app.core.logging import get_logger
from app.exchanges.base import AccountPermissions, ExchangeInfo
from app.exchanges.rest_base import RestExchangeAdapter
from app.exchanges.symbols import split_symbol
from app.market_data.models import (
    Candle,
    OrderBook,
    OrderBookLevel,
    PublicTrade,
    Ticker,
)

logger = get_logger(__name__)

API = "/api/v3/brokerage"

#: Platform interval -> Coinbase granularity. Coinbase offers a fixed set; anything outside it
#: is rejected rather than silently served at a different resolution.
GRANULARITY_MAP: dict[str, str] = {
    "1m": "ONE_MINUTE",
    "5m": "FIVE_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "30m": "THIRTY_MINUTE",
    "1h": "ONE_HOUR",
    "2h": "TWO_HOUR",
    "6h": "SIX_HOUR",
    "1d": "ONE_DAY",
}

STATUS_MAP: dict[str, OrderStatus] = {
    "PENDING": OrderStatus.SUBMITTED,
    "OPEN": OrderStatus.OPEN,
    "FILLED": OrderStatus.FILLED,
    "CANCELLED": OrderStatus.CANCELLED,
    "CANCEL_QUEUED": OrderStatus.CANCELLED,
    "EXPIRED": OrderStatus.CANCELLED,
    "FAILED": OrderStatus.REJECTED,
    "UNKNOWN_ORDER_STATUS": OrderStatus.UNKNOWN,
    "QUEUED": OrderStatus.SUBMITTED,
}


class CoinbaseAdapter(RestExchangeAdapter):
    """Coinbase Advanced Trade REST adapter (spot)."""

    name = "coinbase"
    base_url = "https://api.coinbase.com"
    # Deliberately identical: there is no sandbox, and __init__ refuses testnet outright so
    # this is never reached with testnet=True.
    testnet_url = "https://api.coinbase.com"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.credentials.testnet:
            raise ConfigurationError(
                "Coinbase Advanced Trade has no testnet. Connecting with testnet enabled "
                "would send real orders to the production venue while the interface claimed "
                "otherwise. Rehearse in paper mode instead, then connect with testnet "
                "disabled when you are ready to trade real money."
            )
        self._instruments: dict[str, InstrumentSpec] = {}

    # ------------------------------------------------------------------ #
    # Symbols
    # ------------------------------------------------------------------ #
    def _product_id(self, symbol: str) -> str:
        """Convert a platform symbol into a Coinbase product id.

        ``BTC-USD`` passes through. ``BTCUSD`` is split on a known quote asset so that bots
        configured in the platform's usual style keep working. A symbol that cannot be split
        unambiguously raises rather than being guessed at - sending an order for the wrong
        instrument is far worse than refusing to send one.
        """
        if "-" in symbol:
            return symbol.upper()
        base, quote = split_symbol(symbol)
        if base is None or quote is None:
            raise InstrumentNotSupportedError(
                f"Cannot tell where the base ends in {symbol!r}. Coinbase products are "
                "written BASE-QUOTE, for example BTC-USD.",
                context={"symbol": symbol},
            )
        return f"{base}-{quote}"

    # ------------------------------------------------------------------ #
    # Signing
    # ------------------------------------------------------------------ #
    def _sign(
        self, method: str, path: str, params: dict[str, Any], body: str
    ) -> tuple[dict[str, str], dict[str, Any], str]:
        """Coinbase legacy HMAC: sign ``timestamp + method + path + body``.

        The signed path includes the query string, so it has to be rebuilt here exactly as it
        will be sent - a mismatch produces an authentication failure that looks like bad
        credentials.
        """
        timestamp = str(int(time.time()))
        signed_path = path
        if params:
            signed_path = f"{path}?{urlencode(params)}"
        signature = self._hmac_sha256(
            self.credentials.api_secret,
            f"{timestamp}{method.upper()}{signed_path}{body}",
        )
        headers = {
            "CB-ACCESS-KEY": self.credentials.api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
        }
        return headers, params, body

    def _unwrap(self, payload: Any, path: str) -> Any:
        """Coinbase returns the result directly; errors carry an ``error`` field."""
        if isinstance(payload, dict) and payload.get("error"):
            raise self._map_error(payload, path, 400)
        return payload

    def _map_error(self, payload: Any, path: str, status: int) -> ExchangeError:
        if isinstance(payload, dict):
            message = str(
                payload.get("message")
                or payload.get("error_details")
                or payload.get("error")
                or f"HTTP {status}"
            )
            reason = str(payload.get("error") or "")
            if reason in {"INSUFFICIENT_FUND", "INSUFFICIENT_FUNDS"}:
                from app.core.exceptions import InsufficientBalanceError

                return InsufficientBalanceError(
                    f"coinbase: {message}", context={"endpoint": path}
                )
            if reason:
                return OrderRejectedError(
                    f"coinbase: {message}", context={"endpoint": path, "reason": reason}
                )
        return super()._map_error(payload, path, status)

    # ------------------------------------------------------------------ #
    # Info and credentials
    # ------------------------------------------------------------------ #
    async def get_info(self) -> ExchangeInfo:
        result = await self._request("GET", f"{API}/time")
        epoch = self._as_float(result.get("epochSeconds"))
        return ExchangeInfo(
            name=self.name,
            testnet=False,
            server_time=from_epoch_ms(int(epoch * 1000)) if epoch else utcnow(),
            supports_websocket=True,
            supports_leverage=False,
            rate_limit_per_minute=600,
        )

    async def validate_credentials(self) -> AccountPermissions:
        """Read the key's scopes and refuse anything that can move funds.

        Coinbase's ``can_transfer`` covers withdrawals and transfers between portfolios, which
        is exactly the capability this platform must never hold.
        """
        result = await self._request(
            "GET", f"{API}/key_permissions", signed=True, bucket="private"
        )
        can_transfer = bool(result.get("can_transfer"))
        account = AccountPermissions(
            can_read=bool(result.get("can_view", True)),
            can_trade=bool(result.get("can_trade")),
            # Coinbase does not separate the two; transfer rights are withdrawal rights.
            can_withdraw=can_transfer,
            can_transfer=can_transfer,
            raw={
                "portfolio_type": result.get("portfolio_type"),
                "portfolio_uuid": result.get("portfolio_uuid"),
            },
        )
        if account.can_withdraw:
            logger.error(
                "coinbase.unsafe_key_rejected", reason="transfer permission present"
            )
            raise UnsafeCredentialsError(
                "This Coinbase API key has TRANSFER permission, which can move funds off the "
                "account. The platform refuses to use it. Create a new key with View and "
                "Trade permissions only."
            )
        return account

    # ------------------------------------------------------------------ #
    # Instruments
    # ------------------------------------------------------------------ #
    async def get_instruments(self) -> dict[str, InstrumentSpec]:
        result = await self._request(
            "GET", f"{API}/products", params={"product_type": "SPOT"}
        )
        specs: dict[str, InstrumentSpec] = {}
        for item in result.get("products", []) or []:
            try:
                spec = self._parse_instrument(item)
            except (KeyError, ValueError) as exc:
                logger.warning("coinbase.instrument_skipped", error=str(exc))
                continue
            specs[spec.symbol] = spec
        self._instruments.update(specs)
        return specs

    def _parse_instrument(self, item: dict[str, Any]) -> InstrumentSpec:
        product_id = item["product_id"]
        return InstrumentSpec(
            symbol=product_id,
            base_asset=item.get("base_currency_id", "") or product_id.split("-")[0],
            quote_asset=item.get("quote_currency_id", "") or product_id.split("-")[-1],
            instrument_type=InstrumentType.SPOT,
            tick_size=self._as_float(item.get("quote_increment"), 0.01) or 0.01,
            lot_size=self._as_float(item.get("base_increment"), 0.00000001) or 0.00000001,
            min_quantity=self._as_float(item.get("base_min_size")),
            max_quantity=self._as_float(item.get("base_max_size"), 1e12) or 1e12,
            min_notional=self._as_float(item.get("quote_min_size")),
            max_leverage=1.0,
            maker_fee=0.006,
            taker_fee=0.012,
            # `trading_disabled` and friends are the venue telling us not to route here.
            is_active=not (
                item.get("trading_disabled")
                or item.get("is_disabled")
                or item.get("view_only")
            ),
        )

    async def get_instrument(self, symbol: str) -> InstrumentSpec:
        product_id = self._product_id(symbol)
        if product_id in self._instruments:
            return self._instruments[product_id]
        result = await self._request("GET", f"{API}/products/{product_id}")
        if not result or not result.get("product_id"):
            raise InstrumentNotSupportedError(
                f"coinbase does not list {product_id}", context={"symbol": symbol}
            )
        spec = self._parse_instrument(result)
        self._instruments[product_id] = spec
        return spec

    # ------------------------------------------------------------------ #
    # Market data
    # ------------------------------------------------------------------ #
    async def get_ticker(self, symbol: str) -> Ticker:
        product_id = self._product_id(symbol)
        result = await self._request(
            "GET", f"{API}/products/{product_id}/ticker", params={"limit": 1}
        )
        trades = result.get("trades", []) or []
        last = self._as_float(trades[0].get("price")) if trades else 0.0
        bid = self._as_float(result.get("best_bid")) or None
        ask = self._as_float(result.get("best_ask")) or None
        if not last and bid and ask:
            # No print yet on a thin book: the mid is a defensible stand-in, and returning
            # zero would look like a free asset to every downstream calculation.
            last = (bid + ask) / 2.0
        if not last:
            raise InstrumentNotSupportedError(
                f"coinbase has no price for {product_id}", context={"symbol": symbol}
            )
        return Ticker(
            symbol=product_id,
            price=last,
            timestamp=utcnow(),
            bid=bid,
            ask=ask,
            bid_size=self._as_float(result.get("best_bid_quantity")) or None,
            ask_size=self._as_float(result.get("best_ask_quantity")) or None,
        )

    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        product_id = self._product_id(symbol)
        result = await self._request(
            "GET",
            f"{API}/product_book",
            params={"product_id": product_id, "limit": depth},
        )
        book = result.get("pricebook", result) or {}
        return OrderBook(
            symbol=product_id,
            timestamp=_parse_time(book.get("time")) or utcnow(),
            bids=tuple(
                OrderBookLevel(
                    price=self._as_float(level.get("price")),
                    quantity=self._as_float(level.get("size")),
                )
                for level in (book.get("bids", []) or [])
            ),
            asks=tuple(
                OrderBookLevel(
                    price=self._as_float(level.get("price")),
                    quantity=self._as_float(level.get("size")),
                )
                for level in (book.get("asks", []) or [])
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
        product_id = self._product_id(symbol)
        granularity = GRANULARITY_MAP.get(interval)
        if granularity is None:
            raise ExchangeError(
                f"coinbase does not support the {interval} interval; "
                f"available: {', '.join(sorted(GRANULARITY_MAP))}"
            )
        from app.core.clock import interval_to_timedelta

        step = interval_to_timedelta(interval)
        # Coinbase requires an explicit window and caps it at 350 candles per call.
        capped = min(limit, 350)
        finish = ensure_utc(end) if end is not None else utcnow()
        begin = ensure_utc(start) if start is not None else finish - step * capped

        result = await self._request(
            "GET",
            f"{API}/products/{product_id}/candles",
            params={
                "start": str(int(begin.timestamp())),
                "end": str(int(finish.timestamp())),
                "granularity": granularity,
            },
        )
        candles: list[Candle] = []
        for row in result.get("candles", []) or []:
            try:
                candles.append(
                    Candle(
                        symbol=product_id,
                        interval=interval,
                        open_time=from_epoch_ms(int(float(row["start"])) * 1000),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row.get("volume", 0.0)),
                    )
                )
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    "coinbase.candle_skipped", symbol=product_id, error=str(exc)
                )
        candles.sort(key=lambda c: c.open_time)
        return candles[-limit:]

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[PublicTrade]:
        product_id = self._product_id(symbol)
        result = await self._request(
            "GET",
            f"{API}/products/{product_id}/ticker",
            params={"limit": min(limit, 1000)},
        )
        trades: list[PublicTrade] = []
        for item in result.get("trades", []) or []:
            timestamp = _parse_time(item.get("time"))
            if timestamp is None:
                continue
            trades.append(
                PublicTrade(
                    symbol=product_id,
                    price=self._as_float(item.get("price")),
                    quantity=self._as_float(item.get("size")),
                    side=str(item.get("side", "")).lower(),
                    timestamp=timestamp,
                    trade_id=item.get("trade_id"),
                )
            )
        return trades

    # ------------------------------------------------------------------ #
    # Account
    # ------------------------------------------------------------------ #
    async def get_balance(self) -> AccountBalance:
        result = await self._request(
            "GET",
            f"{API}/accounts",
            params={"limit": 250},
            signed=True,
            bucket="private",
        )
        balances: dict[str, Balance] = {}
        for account in result.get("accounts", []) or []:
            asset = account.get("currency")
            if not asset:
                continue
            free = self._as_float((account.get("available_balance") or {}).get("value"))
            locked = self._as_float((account.get("hold") or {}).get("value"))
            if free <= 0 and locked <= 0:
                continue
            balances[asset] = Balance(asset=asset, free=free, locked=locked)
        return AccountBalance(balances=balances, timestamp=utcnow())

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Spot has balances, not positions.

        Returning an empty list is the honest answer, and it is what the portfolio
        reconciliation expects from a spot venue.
        """
        return []

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        params: dict[str, Any] = {"order_status": "OPEN"}
        if symbol:
            params["product_id"] = self._product_id(symbol)
        result = await self._request(
            "GET",
            f"{API}/orders/historical/batch",
            params=params,
            signed=True,
            bucket="private",
        )
        return [self._parse_order(item) for item in result.get("orders", []) or []]

    async def get_order(
        self, client_order_id: str, *, symbol: str | None = None
    ) -> Order | None:
        """Resolve an order by the platform's ``client_order_id``.

        This is what makes a timed-out submission recoverable: Coinbase echoes
        ``client_order_id`` on every order, so the platform can ask what happened instead of
        sending a second order.

        The one constraint worth knowing: Advanced Trade offers no lookup keyed on
        ``client_order_id`` - only on the venue's own order id, which a timed-out request
        never received - so this scans the most recent orders instead. That window is ample
        for its actual purpose, since a submission being resolved was sent seconds ago, but an
        account placing more than ``limit`` orders in that gap could push it out of view. The
        caller treats ``None`` as "no record", so the failure mode is a refusal to assume the
        order exists, not a duplicate.
        """
        params: dict[str, Any] = {"limit": 100}
        if symbol:
            params["product_id"] = self._product_id(symbol)
        result = await self._request(
            "GET",
            f"{API}/orders/historical/batch",
            params=params,
            signed=True,
            bucket="private",
        )
        for item in result.get("orders", []) or []:
            if item.get("client_order_id") == client_order_id:
                return self._parse_order(item)
        return None

    # ------------------------------------------------------------------ #
    # Trading
    # ------------------------------------------------------------------ #
    async def create_order(self, request: OrderRequest) -> Order:
        spec = await self.get_instrument(request.symbol)
        quantity = spec.round_quantity(request.quantity)
        if quantity <= 0:
            raise OrderRejectedError(
                f"coinbase: quantity rounds to zero at step {spec.lot_size:g}"
            )
        if request.price is not None:
            problem = spec.validate_order(quantity, request.price)
            if problem is not None:
                raise OrderRejectedError(f"coinbase: {problem}")

        qty_decimals = _decimals(spec.lot_size)
        price_decimals = _decimals(spec.tick_size)
        configuration = self._order_configuration(
            request, quantity, qty_decimals, price_decimals, spec
        )

        body: dict[str, Any] = {
            "client_order_id": request.client_order_id,
            "product_id": spec.symbol,
            "side": "BUY" if request.side is OrderSide.BUY else "SELL",
            "order_configuration": configuration,
        }
        result = await self._request(
            "POST", f"{API}/orders", body=body, signed=True, bucket="order"
        )
        if not result.get("success", True):
            response = result.get("error_response", {}) or {}
            raise OrderRejectedError(
                f"coinbase: {response.get('message') or response.get('error') or 'rejected'}",
                context={"symbol": spec.symbol},
            )

        order = await self.get_order(request.client_order_id, symbol=spec.symbol)
        if order is not None:
            return order
        # Accepted but not yet queryable. Report the local view rather than inventing a fill.
        local = Order.from_request(request)
        local.status = OrderStatus.SUBMITTED
        local.exchange_order_id = (result.get("success_response") or {}).get("order_id")
        return local

    def _order_configuration(
        self,
        request: OrderRequest,
        quantity: float,
        qty_decimals: int,
        price_decimals: int,
        spec: InstrumentSpec,
    ) -> dict[str, Any]:
        """Build Coinbase's ``order_configuration`` union for this request."""
        size = f"{quantity:.{qty_decimals}f}"
        if request.order_type is OrderType.MARKET:
            return {"market_market_ioc": {"base_size": size}}

        if request.price is None:
            raise OrderRejectedError(
                f"coinbase: {request.order_type.value} requires a limit price"
            )
        price = f"{spec.round_price(request.price):.{price_decimals}f}"

        if request.order_type in {OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT}:
            if request.trigger_price is None:
                raise OrderRejectedError(
                    f"coinbase: {request.order_type.value} requires a trigger price"
                )
            trigger = f"{spec.round_price(request.trigger_price):.{price_decimals}f}"
            # Which way the stop fires depends on the order type, not only the side: a sell
            # stop-loss triggers on the way down, a sell take-profit on the way up.
            takes_profit = request.order_type is OrderType.TAKE_PROFIT_LIMIT
            rises = (
                request.side is OrderSide.BUY
                if not takes_profit
                else request.side is OrderSide.SELL
            )
            return {
                "stop_limit_stop_limit_gtc": {
                    "base_size": size,
                    "limit_price": price,
                    "stop_price": trigger,
                    "stop_direction": (
                        "STOP_DIRECTION_STOP_UP" if rises else "STOP_DIRECTION_STOP_DOWN"
                    ),
                }
            }

        if request.time_in_force is TimeInForce.IOC:
            return {"sor_limit_ioc": {"base_size": size, "limit_price": price}}
        return {
            "limit_limit_gtc": {
                "base_size": size,
                "limit_price": price,
                "post_only": request.time_in_force is TimeInForce.POST_ONLY,
            }
        }

    async def cancel_order(
        self, client_order_id: str, *, symbol: str | None = None
    ) -> Order:
        existing = await self.get_order(client_order_id, symbol=symbol)
        if existing is not None and existing.status.is_terminal:
            return existing
        if existing is None or not existing.exchange_order_id:
            raise ExchangeError(
                f"coinbase: cannot cancel {client_order_id}; the venue has no record of it"
            )
        await self._request(
            "POST",
            f"{API}/orders/batch_cancel",
            body={"order_ids": [existing.exchange_order_id]},
            signed=True,
            bucket="order",
        )
        refreshed = await self.get_order(client_order_id, symbol=symbol)
        if refreshed is not None:
            return refreshed
        existing.status = OrderStatus.CANCELLED
        return existing

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #
    def _parse_order(self, item: dict[str, Any]) -> Order:
        configuration = item.get("order_configuration", {}) or {}
        leg: dict[str, Any] = (
            next(iter(configuration.values()), {}) if configuration else {}
        )
        filled = self._as_float(item.get("filled_size"))
        is_limit = "limit" in "".join(configuration).lower() if configuration else False

        order = Order(
            client_order_id=item.get("client_order_id") or item.get("order_id", ""),
            symbol=item.get("product_id", ""),
            side=(
                OrderSide.BUY
                if str(item.get("side", "")).upper() == "BUY"
                else OrderSide.SELL
            ),
            order_type=OrderType.LIMIT if is_limit else OrderType.MARKET,
            quantity=self._as_float(leg.get("base_size")) or filled,
            status=STATUS_MAP.get(str(item.get("status", "")), OrderStatus.UNKNOWN),
            exchange_order_id=item.get("order_id"),
            price=self._as_float(leg.get("limit_price")) or None,
            trigger_price=self._as_float(leg.get("stop_price")) or None,
            filled_quantity=filled,
            average_fill_price=self._as_float(item.get("average_filled_price")),
            fees_paid=self._as_float(item.get("total_fees")),
            created_at=_parse_time(item.get("created_time")) or utcnow(),
            updated_at=_parse_time(item.get("last_fill_time"))
            or _parse_time(item.get("created_time"))
            or utcnow(),
        )
        if filled > 0:
            order.fills.append(
                Fill(
                    fill_id=item.get("order_id", "") or order.client_order_id,
                    order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=filled,
                    price=order.average_fill_price or order.price or 1.0,
                    fee=order.fees_paid,
                    fee_asset=order.symbol.split("-")[-1] if order.symbol else "USD",
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

        REST polling rather than the websocket feed, for the same reason as the other
        adapters: strategies here act on closed bars, so the added reconnect state machine
        would buy latency nobody uses.
        """
        import asyncio

        from app.core.clock import interval_to_timedelta

        step = interval_to_timedelta(interval).total_seconds()
        last_open: datetime | None = None
        while True:
            try:
                candles = await self.get_candles(symbol, interval, limit=2)
            except ExchangeError as exc:
                logger.warning(
                    "coinbase.stream_poll_failed", symbol=symbol, error=str(exc)
                )
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
                logger.warning(
                    "coinbase.ticker_poll_failed", symbol=symbol, error=str(exc)
                )
            await asyncio.sleep(2.0)


def _parse_time(value: Any) -> datetime | None:
    """Parse Coinbase's RFC3339 timestamps, tolerating the trailing ``Z``."""
    if not value or not isinstance(value, str):
        return None
    try:
        return ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def _decimals(step: float) -> int:
    from app.core.numeric import decimals_for_step

    return decimals_for_step(step)
