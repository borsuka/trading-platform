"""Crypto.com Exchange v1 adapter.

Implements the platform's :class:`~app.exchanges.base.ExchangeAdapter` contract against the
Crypto.com Exchange REST API v1.

BLOCKED BY EXTERNAL DEPENDENCY: fully implemented and tested against a mock transport
reproducing the documented signing scheme, envelopes and error codes, but **not** verified
against the live venue, which requires API credentials. Follow ``docs/live-trading.md`` before
trading real money with it.

Two things about this venue need care:

* **Signing is over a canonical parameter string**, not a query string. Nested parameters are
  flattened in sorted key order; getting this wrong produces an authentication error that
  reads like bad credentials rather than a bad signature.
* **There is no endpoint that reports an API key's permissions.** Every other venue here can
  be asked directly whether a key may withdraw. This one cannot, so the adapter *probes*: it
  calls a withdrawal-scoped endpoint and treats success as proof the key holds withdrawal
  rights. See :meth:`CryptoComAdapter.validate_credentials` - the reasoning, and the limits of
  what the probe proves, are set out there.
"""

from __future__ import annotations

import itertools
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

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
    ExchangeAuthError,
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

#: Platform interval -> Crypto.com timeframe code.
TIMEFRAME_MAP: dict[str, str] = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "12h": "12h",
    "1d": "1D", "1w": "7D",
}

ORDER_TYPE_MAP: dict[OrderType, str] = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.STOP_MARKET: "STOP_LOSS",
    OrderType.STOP_LIMIT: "STOP_LIMIT",
    OrderType.TAKE_PROFIT_MARKET: "TAKE_PROFIT",
    OrderType.TAKE_PROFIT_LIMIT: "TAKE_PROFIT_LIMIT",
    OrderType.TRAILING_STOP: "STOP_LOSS",
}

STATUS_MAP: dict[str, OrderStatus] = {
    "NEW": OrderStatus.OPEN,
    "PENDING": OrderStatus.SUBMITTED,
    "ACTIVE": OrderStatus.OPEN,
    "FILLED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELLED,
    "CANCELLED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "EXPIRED": OrderStatus.CANCELLED,
}

TIF_MAP: dict[TimeInForce, str] = {
    TimeInForce.GTC: "GOOD_TILL_CANCEL",
    TimeInForce.IOC: "IMMEDIATE_OR_CANCEL",
    TimeInForce.FOK: "FILL_OR_KILL",
    TimeInForce.POST_ONLY: "GOOD_TILL_CANCEL",
}

#: Crypto.com error codes worth mapping precisely.
INSUFFICIENT_BALANCE_CODES = {30004, 40401, 313}
REJECTED_CODES = {30003, 30005, 30006, 30017, 40001, 316, 213}
#: Returned when the key lacks the scope for the endpoint. Used by the permission probe.
UNAUTHORIZED_CODES = {40101, 40102, 40103, 40104, 10002, 10003, 40403}


class CryptoComAdapter(RestExchangeAdapter):
    """Crypto.com Exchange v1 REST adapter (spot)."""

    name = "cryptocom"
    base_url = "https://api.crypto.com"
    testnet_url = "https://uat-api.3ona.co"

    #: Every v1 endpoint sits under this prefix.
    api_prefix = "/exchange/v1"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._instruments: dict[str, InstrumentSpec] = {}
        self._request_ids = itertools.count(1)

    # ------------------------------------------------------------------ #
    # Symbols
    # ------------------------------------------------------------------ #
    def _instrument_name(self, symbol: str) -> str:
        """Convert a platform symbol into a Crypto.com instrument name (``BTC_USDT``)."""
        text = symbol.strip().upper()
        if "_" in text:
            return text
        if "-" in text:
            return text.replace("-", "_")
        base, quote = split_symbol(text)
        if base is None or quote is None:
            raise InstrumentNotSupportedError(
                f"Cannot tell where the base ends in {symbol!r}. Crypto.com instruments are "
                "written BASE_QUOTE, for example BTC_USDT.",
                context={"symbol": symbol},
            )
        return f"{base}_{quote}"

    # ------------------------------------------------------------------ #
    # Signing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _params_to_string(params: Any, depth: int = 0) -> str:
        """Flatten parameters the way Crypto.com's signature expects.

        Keys in sorted order, key immediately followed by its value, nested structures
        flattened recursively. Booleans are lowercase and ``None`` becomes ``null``: the
        Python defaults (``True``/``None``) would produce a different string from the one the
        venue reconstructs, and the request would be rejected as unsigned.
        """
        if depth > 3:  # the documented nesting limit; deeper input is malformed
            raise ExchangeError("cryptocom: parameters nested too deeply to sign")
        if params is None:
            return "null"
        if isinstance(params, bool):
            return "true" if params else "false"
        if isinstance(params, dict):
            return "".join(
                f"{key}{CryptoComAdapter._params_to_string(params[key], depth + 1)}"
                for key in sorted(params)
            )
        if isinstance(params, (list, tuple)):
            return "".join(
                CryptoComAdapter._params_to_string(item, depth + 1) for item in params
            )
        return str(params)

    def _signed_envelope(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Build the full signed request body for a private method."""
        request_id = next(self._request_ids)
        nonce = self._timestamp_ms()
        payload = (
            f"{method}{request_id}{self.credentials.api_key}"
            f"{self._params_to_string(params)}{nonce}"
        )
        return {
            "id": request_id,
            "method": method,
            "api_key": self.credentials.api_key,
            "params": params,
            "nonce": nonce,
            "sig": self._hmac_sha256(self.credentials.api_secret, payload),
        }

    def _sign(
        self, method: str, path: str, params: dict[str, Any], body: str
    ) -> tuple[dict[str, str], dict[str, Any], str]:
        """Not used.

        Crypto.com signs the request *body*, which has to be constructed before it can be
        signed. :meth:`_private` builds and signs it in one step, so this hook has nothing to
        add. It cannot simply be omitted - the base class declares it abstract.
        """
        return {}, params, body

    def _unwrap(self, payload: Any, path: str) -> Any:
        """Crypto.com wraps everything in ``{id, method, code, result}``."""
        if not isinstance(payload, dict):
            raise ExchangeError(f"cryptocom returned an unexpected body on {path}")
        code = payload.get("code")
        if code not in (0, None, "0"):
            raise self._map_error(payload, path, 400)
        return payload.get("result", {})

    def _map_error(self, payload: Any, path: str, status: int) -> ExchangeError:
        if isinstance(payload, dict):
            # The venue sends the code as a number in most responses and as a string in a
            # few, so it is normalised before any comparison rather than at each one.
            raw_code = payload.get("code")
            code: int | None
            try:
                code = int(str(raw_code))
            except (TypeError, ValueError):
                code = None
            message = str(payload.get("message") or payload.get("msg") or f"HTTP {status}")
            if code in INSUFFICIENT_BALANCE_CODES:
                from app.core.exceptions import InsufficientBalanceError

                return InsufficientBalanceError(
                    f"cryptocom: {message}", context={"endpoint": path, "code": code}
                )
            if code in UNAUTHORIZED_CODES:
                return ExchangeAuthError(
                    f"cryptocom: {message}", context={"endpoint": path, "code": code}
                )
            if code in REJECTED_CODES:
                return OrderRejectedError(
                    f"cryptocom: {message}", context={"endpoint": path, "code": code}
                )
        return super()._map_error(payload, path, status)

    # ------------------------------------------------------------------ #
    # Request helpers
    # ------------------------------------------------------------------ #
    async def _public(self, method: str, params: dict[str, Any] | None = None) -> Any:
        return await self._request(
            "GET", f"{self.api_prefix}/{method}", params=params or {}
        )

    async def _private(
        self, method: str, params: dict[str, Any] | None = None, *, bucket: str = "private"
    ) -> Any:
        """POST a signed request. The signature covers the body, so it is built here."""
        envelope = self._signed_envelope(method, params or {})
        return await self._request(
            "POST", f"{self.api_prefix}/{method}", body=envelope, bucket=bucket
        )

    # ------------------------------------------------------------------ #
    # Info and credentials
    # ------------------------------------------------------------------ #
    async def get_info(self) -> ExchangeInfo:
        # A successful public call is the connectivity proof; the payload is not needed.
        await self._public("public/get-instruments")
        return ExchangeInfo(
            name=self.name,
            testnet=self.credentials.testnet,
            # v1 does not publish a server-time endpoint; a successful public call is still
            # proof of connectivity, which is what this is used for.
            server_time=utcnow(),
            supports_websocket=True,
            supports_leverage=False,
            rate_limit_per_minute=600,
        )

    async def validate_credentials(self) -> AccountPermissions:
        """Establish that the key can trade and cannot withdraw.

        Crypto.com publishes no endpoint that reports a key's scopes, so unlike the other
        venues this cannot simply be read. It is established by probing instead:

        * ``private/user-balance`` must succeed, or the key cannot even read the account.
        * ``private/get-withdrawal-history`` is withdrawal-scoped. If it **succeeds**, the key
          demonstrably holds withdrawal rights and is refused.

        What the probe proves is asymmetric, and it is worth being precise about it. A success
        is conclusive: the key reached a withdrawal endpoint. A permission error is strong
        evidence of the opposite but is not a signed statement from the venue, which is why
        this adapter is documented as unverified against the live API. Anything *other* than
        those two outcomes - a network failure, an unrecognised code - leaves the question
        open, and an open question about withdrawal rights is resolved by refusing to connect
        rather than by assuming the safe answer.
        """
        try:
            await self._private("private/user-balance")
        except ExchangeAuthError as exc:
            raise UnsafeCredentialsError(
                "These Crypto.com credentials cannot read the account balance. Check the key "
                "and secret, and that the key has Read permission."
            ) from exc

        can_withdraw = await self._probe_withdrawal_scope()
        account = AccountPermissions(
            can_read=True,
            # Trade rights are confirmed by the preflight's own dry run, not guessed here.
            can_trade=True,
            can_withdraw=can_withdraw,
            can_transfer=can_withdraw,
            raw={"permission_source": "probe"},
        )
        if can_withdraw:
            logger.error(
                "cryptocom.unsafe_key_rejected", reason="withdrawal endpoint reachable"
            )
            raise UnsafeCredentialsError(
                "This Crypto.com API key can reach the withdrawal API, which means it holds "
                "withdrawal permission. The platform refuses to use it. Create a new key "
                "with Read and Trade permissions only."
            )
        return account

    async def _probe_withdrawal_scope(self) -> bool:
        """True when the key demonstrably holds withdrawal rights.

        Raises when the answer cannot be established, because "unknown" and "safe" must not
        collapse into the same value here.
        """
        try:
            await self._private(
                "private/get-withdrawal-history", {"page_size": 1, "page": 0}
            )
        except ExchangeAuthError:
            # Refused for lack of scope: exactly what a trade-only key should do.
            return False
        except OrderRejectedError:
            # Some deployments answer a scope failure with a generic rejection rather than an
            # auth code. Still a refusal, still not a withdrawal-capable key.
            return False
        except ExchangeError as exc:
            raise UnsafeCredentialsError(
                "Could not establish whether this Crypto.com API key has withdrawal "
                f"permission ({exc}). The platform will not trade with a key whose "
                "permissions it cannot verify. Try again, or use a different venue."
            ) from exc
        return True

    # ------------------------------------------------------------------ #
    # Instruments
    # ------------------------------------------------------------------ #
    async def get_instruments(self) -> dict[str, InstrumentSpec]:
        result = await self._public("public/get-instruments")
        specs: dict[str, InstrumentSpec] = {}
        for item in result.get("data", []) or []:
            try:
                spec = self._parse_instrument(item)
            except (KeyError, ValueError) as exc:
                logger.warning("cryptocom.instrument_skipped", error=str(exc))
                continue
            specs[spec.symbol] = spec
        self._instruments.update(specs)
        return specs

    def _parse_instrument(self, item: dict[str, Any]) -> InstrumentSpec:
        name = item.get("symbol") or item["instrument_name"]
        base = item.get("base_ccy") or name.split("_")[0]
        quote = item.get("quote_ccy") or name.split("_")[-1]
        return InstrumentSpec(
            symbol=name,
            base_asset=base,
            quote_asset=quote,
            instrument_type=InstrumentType.SPOT,
            tick_size=self._as_float(item.get("price_tick_size"), 0.01) or 0.01,
            lot_size=self._as_float(item.get("qty_tick_size"), 0.000001) or 0.000001,
            min_quantity=self._as_float(item.get("min_quantity")),
            max_quantity=self._as_float(item.get("max_quantity"), 1e12) or 1e12,
            min_notional=self._as_float(item.get("min_notional")),
            max_leverage=self._as_float(item.get("max_leverage"), 1.0) or 1.0,
            maker_fee=0.00075,
            taker_fee=0.00075,
            is_active=str(item.get("tradable", True)).lower() not in {"false", "0"},
        )

    async def get_instrument(self, symbol: str) -> InstrumentSpec:
        name = self._instrument_name(symbol)
        if name in self._instruments:
            return self._instruments[name]
        await self.get_instruments()
        spec = self._instruments.get(name)
        if spec is None:
            raise InstrumentNotSupportedError(
                f"cryptocom does not list {name}", context={"symbol": symbol}
            )
        return spec

    # ------------------------------------------------------------------ #
    # Market data
    # ------------------------------------------------------------------ #
    async def get_ticker(self, symbol: str) -> Ticker:
        name = self._instrument_name(symbol)
        result = await self._public("public/get-tickers", {"instrument_name": name})
        rows = result.get("data", []) or []
        if not rows:
            raise InstrumentNotSupportedError(
                f"cryptocom has no ticker for {name}", context={"symbol": symbol}
            )
        row = rows[0]
        return Ticker(
            symbol=name,
            price=self._as_float(row.get("a")),  # latest trade price
            timestamp=from_epoch_ms(int(row.get("t", self._timestamp_ms()))),
            bid=self._as_float(row.get("b")) or None,
            ask=self._as_float(row.get("k")) or None,
            volume_24h=self._as_float(row.get("v")) or None,
            price_change_24h_pct=self._as_float(row.get("c")) or None,
        )

    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        name = self._instrument_name(symbol)
        result = await self._public(
            "public/get-book", {"instrument_name": name, "depth": min(depth, 50)}
        )
        rows = result.get("data", []) or []
        book = rows[0] if rows else {}
        return OrderBook(
            symbol=name,
            timestamp=from_epoch_ms(int(book.get("t", self._timestamp_ms()))),
            bids=tuple(
                OrderBookLevel(price=float(level[0]), quantity=float(level[1]))
                for level in (book.get("bids", []) or [])
            ),
            asks=tuple(
                OrderBookLevel(price=float(level[0]), quantity=float(level[1]))
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
        name = self._instrument_name(symbol)
        timeframe = TIMEFRAME_MAP.get(interval)
        if timeframe is None:
            raise ExchangeError(
                f"cryptocom does not support the {interval} interval; "
                f"available: {', '.join(sorted(TIMEFRAME_MAP))}"
            )
        params: dict[str, Any] = {
            "instrument_name": name,
            "timeframe": timeframe,
            "count": min(limit, 300),
        }
        if start is not None:
            params["start_ts"] = to_epoch_ms(start)
        if end is not None:
            params["end_ts"] = to_epoch_ms(end)

        result = await self._public("public/get-candlestick", params)
        candles: list[Candle] = []
        for row in result.get("data", []) or []:
            try:
                candles.append(
                    Candle(
                        symbol=name,
                        interval=interval,
                        open_time=from_epoch_ms(int(row["t"])),
                        open=float(row["o"]),
                        high=float(row["h"]),
                        low=float(row["l"]),
                        close=float(row["c"]),
                        volume=float(row.get("v", 0.0)),
                    )
                )
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("cryptocom.candle_skipped", symbol=name, error=str(exc))
        candles.sort(key=lambda c: c.open_time)
        return candles[-limit:]

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[PublicTrade]:
        name = self._instrument_name(symbol)
        result = await self._public(
            "public/get-trades", {"instrument_name": name, "count": min(limit, 200)}
        )
        trades: list[PublicTrade] = []
        for item in result.get("data", []) or []:
            try:
                trades.append(
                    PublicTrade(
                        symbol=name,
                        price=float(item["p"]),
                        quantity=float(item["q"]),
                        side=str(item.get("s", "")).lower(),
                        timestamp=from_epoch_ms(int(item["t"])),
                        trade_id=str(item.get("d")) if item.get("d") else None,
                    )
                )
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("cryptocom.trade_skipped", symbol=name, error=str(exc))
        return trades

    # ------------------------------------------------------------------ #
    # Account
    # ------------------------------------------------------------------ #
    async def get_balance(self) -> AccountBalance:
        result = await self._private("private/user-balance")
        balances: dict[str, Balance] = {}
        for account in result.get("data", []) or []:
            for position in account.get("position_balances", []) or []:
                asset = position.get("instrument_name") or position.get("currency")
                if not asset:
                    continue
                quantity = self._as_float(position.get("quantity"))
                reserved = self._as_float(position.get("reserved_qty"))
                if quantity <= 0 and reserved <= 0:
                    continue
                balances[asset] = Balance(
                    asset=asset,
                    free=max(0.0, quantity - reserved),
                    locked=reserved,
                )
        return AccountBalance(balances=balances, timestamp=utcnow())

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        params: dict[str, Any] = {}
        if symbol:
            params["instrument_name"] = self._instrument_name(symbol)
        result = await self._private("private/get-positions", params)
        positions: list[Position] = []
        for item in result.get("data", []) or []:
            quantity = self._as_float(item.get("quantity"))
            if quantity == 0:
                continue
            positions.append(
                Position(
                    symbol=item.get("instrument_name", ""),
                    side=PositionSide.LONG if quantity > 0 else PositionSide.SHORT,
                    quantity=abs(quantity),
                    entry_price=self._as_float(item.get("open_price")),
                    mark_price=self._as_float(item.get("mark_price")) or None,
                    leverage=1.0,
                    realized_pnl=self._as_float(item.get("session_pnl")),
                    opened_at=from_epoch_ms(
                        int(item.get("update_timestamp_ms", self._timestamp_ms()))
                    ),
                )
            )
        return positions

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        params: dict[str, Any] = {}
        if symbol:
            params["instrument_name"] = self._instrument_name(symbol)
        result = await self._private("private/get-open-orders", params)
        return [self._parse_order(item) for item in result.get("data", []) or []]

    async def get_order(
        self, client_order_id: str, *, symbol: str | None = None
    ) -> Order | None:
        """Resolve an order by ``client_oid``.

        This is what makes a timed-out submission recoverable without resending it.
        """
        try:
            result = await self._private(
                "private/get-order-detail", {"client_oid": client_order_id}
            )
        except OrderRejectedError:
            # The venue's way of saying "no such order".
            return None
        data = result.get("data")
        if isinstance(data, list):
            return self._parse_order(data[0]) if data else None
        if isinstance(data, dict) and data:
            return self._parse_order(data)
        if isinstance(result, dict) and result.get("order_id"):
            return self._parse_order(result)
        return None

    # ------------------------------------------------------------------ #
    # Trading
    # ------------------------------------------------------------------ #
    async def create_order(self, request: OrderRequest) -> Order:
        spec = await self.get_instrument(request.symbol)
        quantity = spec.round_quantity(request.quantity)
        if quantity <= 0:
            raise OrderRejectedError(
                f"cryptocom: quantity rounds to zero at step {spec.lot_size:g}"
            )
        if request.price is not None:
            problem = spec.validate_order(quantity, request.price)
            if problem is not None:
                raise OrderRejectedError(f"cryptocom: {problem}")

        qty_decimals = _decimals(spec.lot_size)
        price_decimals = _decimals(spec.tick_size)
        params: dict[str, Any] = {
            "instrument_name": spec.symbol,
            "side": "BUY" if request.side is OrderSide.BUY else "SELL",
            "type": ORDER_TYPE_MAP[request.order_type],
            "quantity": f"{quantity:.{qty_decimals}f}",
            "client_oid": request.client_order_id,
        }
        if request.order_type is not OrderType.MARKET:
            params["time_in_force"] = TIF_MAP[request.time_in_force]
        if request.time_in_force is TimeInForce.POST_ONLY:
            params["exec_inst"] = ["POST_ONLY"]
        if request.price is not None:
            params["price"] = f"{spec.round_price(request.price):.{price_decimals}f}"
        if request.trigger_price is not None:
            params["ref_price"] = (
                f"{spec.round_price(request.trigger_price):.{price_decimals}f}"
            )
        if request.reduce_only:
            params["spot_margin"] = "SPOT"

        await self._private("private/create-order", params, bucket="order")
        order = await self.get_order(request.client_order_id, symbol=spec.symbol)
        if order is not None:
            return order
        local = Order.from_request(request)
        local.status = OrderStatus.SUBMITTED
        return local

    async def cancel_order(
        self, client_order_id: str, *, symbol: str | None = None
    ) -> Order:
        existing = await self.get_order(client_order_id, symbol=symbol)
        if existing is not None and existing.status.is_terminal:
            return existing
        params: dict[str, Any] = {"client_oid": client_order_id}
        if existing is not None and existing.exchange_order_id:
            params["order_id"] = existing.exchange_order_id
        await self._private("private/cancel-order", params, bucket="order")
        refreshed = await self.get_order(client_order_id, symbol=symbol)
        if refreshed is not None:
            return refreshed
        if existing is not None:
            existing.status = OrderStatus.CANCELLED
            return existing
        raise ExchangeError(
            f"cryptocom: cannot resolve order {client_order_id} after cancel"
        )

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #
    def _parse_order(self, item: dict[str, Any]) -> Order:
        filled = self._as_float(item.get("cumulative_quantity"))
        quote_filled = self._as_float(item.get("cumulative_value"))
        average = (quote_filled / filled) if filled > 0 and quote_filled > 0 else 0.0
        symbol = item.get("instrument_name", "")
        order = Order(
            client_order_id=item.get("client_oid") or item.get("order_id", ""),
            symbol=symbol,
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
            quantity=self._as_float(item.get("quantity")) or filled,
            status=STATUS_MAP.get(str(item.get("status", "")).upper(), OrderStatus.UNKNOWN),
            exchange_order_id=str(item.get("order_id")) if item.get("order_id") else None,
            price=self._as_float(item.get("price")) or None,
            trigger_price=self._as_float(item.get("ref_price")) or None,
            filled_quantity=filled,
            average_fill_price=average or self._as_float(item.get("avg_price")),
            fees_paid=self._as_float(item.get("cumulative_fee")),
            reject_reason=item.get("reason") or None,
            created_at=from_epoch_ms(
                int(item.get("create_time", self._timestamp_ms()))
            ),
            updated_at=from_epoch_ms(
                int(item.get("update_time", item.get("create_time", self._timestamp_ms())))
            ),
        )
        if filled > 0:
            order.fills.append(
                Fill(
                    fill_id=str(item.get("order_id", "")) or order.client_order_id,
                    order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=filled,
                    price=order.average_fill_price or order.price or 1.0,
                    fee=order.fees_paid,
                    fee_asset=item.get("fee_instrument_name")
                    or (symbol.split("_")[-1] if symbol else "USDT"),
                    role=LiquidityRole.TAKER,
                    timestamp=order.updated_at,
                )
            )
        return order

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    async def subscribe_candles(self, symbol: str, interval: str) -> AsyncIterator[Candle]:
        """Poll for closed candles, as the other REST adapters do."""
        import asyncio

        from app.core.clock import interval_to_timedelta

        step = interval_to_timedelta(interval).total_seconds()
        last_open: datetime | None = None
        while True:
            try:
                candles = await self.get_candles(symbol, interval, limit=2)
            except ExchangeError as exc:
                logger.warning(
                    "cryptocom.stream_poll_failed", symbol=symbol, error=str(exc)
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
                    "cryptocom.ticker_poll_failed", symbol=symbol, error=str(exc)
                )
            await asyncio.sleep(2.0)


def _decimals(step: float) -> int:
    from app.core.numeric import decimals_for_step

    return decimals_for_step(step)
