"""Exchange adapter interface.

Everything above this boundary is venue-neutral. That is what makes paper and live trading share
a single code path: mode selection happens exactly once, when the adapter is constructed, and
never again.

Two rules every implementation must honour:

1. :meth:`ExchangeAdapter.validate_credentials` must **reject** any API key that carries
   withdrawal permission. The platform never needs it, and a key that has it turns a bug into a
   theft.
2. :meth:`ExchangeAdapter.create_order` must be idempotent on ``client_order_id``. A timed-out
   submission is resolved by *querying*, never by resending.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from types import TracebackType
from typing import Any, Self

from app.core.domain import (
    AccountBalance,
    InstrumentSpec,
    Order,
    OrderRequest,
    Position,
)
from app.core.enums import OrderStatus
from app.market_data.models import Candle, OrderBook, PublicTrade, Ticker


@dataclass(frozen=True, slots=True)
class ExchangeCredentials:
    """API credentials.

    ``__repr__`` is overridden so a credentials object caught in a traceback or a debug log
    cannot leak the secret.
    """

    api_key: str
    api_secret: str
    passphrase: str | None = None
    testnet: bool = True

    def __repr__(self) -> str:
        suffix = self.api_key[-4:] if len(self.api_key) >= 4 else "?"
        return f"ExchangeCredentials(api_key=...{suffix}, secret=***, testnet={self.testnet})"


@dataclass(frozen=True, slots=True)
class AccountPermissions:
    """What an API key is allowed to do, as reported by the venue."""

    can_read: bool = False
    can_trade: bool = False
    can_withdraw: bool = False
    can_transfer: bool = False
    ip_restricted: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_safe_for_trading(self) -> bool:
        """A key is safe only when it can trade and cannot move funds off the venue."""
        return self.can_read and self.can_trade and not self.can_withdraw

    def rejection_reason(self) -> str | None:
        if self.can_withdraw:
            return (
                "This API key has WITHDRAWAL permission. The platform refuses to use it. "
                "Create a new key with read and trade permissions only."
            )
        if self.can_transfer:
            return (
                "This API key has internal TRANSFER permission, which can move funds between "
                "accounts. Create a key with read and trade permissions only."
            )
        if not self.can_trade:
            return "This API key cannot place orders. Enable trade permission."
        if not self.can_read:
            return "This API key cannot read account state. Enable read permission."
        return None


@dataclass(frozen=True, slots=True)
class ExchangeInfo:
    """Venue identity and connectivity state."""

    name: str
    testnet: bool
    server_time: datetime
    supports_websocket: bool = True
    supports_leverage: bool = False
    rate_limit_per_minute: int = 600


class ExchangeAdapter(ABC):
    """Abstract exchange.

    Implementations must be safe to use from a single asyncio event loop. They are not required
    to be thread-safe.
    """

    #: Venue identifier, e.g. ``"bybit"``.
    name: str = "abstract"
    #: True when orders placed through this adapter move real money.
    is_live: bool = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def connect(self) -> None:
        """Establish transport and load instrument metadata."""

    @abstractmethod
    async def close(self) -> None:
        """Release sockets and background tasks. Must be safe to call twice."""

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    @abstractmethod
    async def get_info(self) -> ExchangeInfo:
        """Venue identity plus server time, used for clock-drift checks."""

    @abstractmethod
    async def validate_credentials(self) -> AccountPermissions:
        """Verify credentials and report permissions.

        Implementations MUST raise
        :class:`~app.core.exceptions.UnsafeCredentialsError` when the key can withdraw.
        """

    # ------------------------------------------------------------------ #
    # Instruments
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def get_instruments(self) -> dict[str, InstrumentSpec]:
        """All tradable instruments, keyed by symbol."""

    @abstractmethod
    async def get_instrument(self, symbol: str) -> InstrumentSpec:
        """One instrument's trading rules.

        Raises :class:`~app.core.exceptions.InstrumentNotSupportedError` if unknown.
        """

    # ------------------------------------------------------------------ #
    # Market data
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker: ...

    @abstractmethod
    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook: ...

    @abstractmethod
    async def get_candles(
        self,
        symbol: str,
        interval: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[Candle]:
        """Closed candles, ascending by open time."""

    @abstractmethod
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[PublicTrade]: ...

    # ------------------------------------------------------------------ #
    # Account
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def get_balance(self) -> AccountBalance: ...

    @abstractmethod
    async def get_positions(self, symbol: str | None = None) -> list[Position]: ...

    @abstractmethod
    async def get_open_orders(self, symbol: str | None = None) -> list[Order]: ...

    @abstractmethod
    async def get_order(
        self, client_order_id: str, *, symbol: str | None = None
    ) -> Order | None:
        """Look an order up by its idempotency key.

        Returning ``None`` means *the venue has no record of it*, which is the only safe basis
        for deciding that a timed-out submission never landed.
        """

    # ------------------------------------------------------------------ #
    # Trading
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def create_order(self, request: OrderRequest) -> Order:
        """Place an order.

        MUST be idempotent on ``request.client_order_id``: submitting the same id twice returns
        the existing order rather than creating a second one.
        """

    @abstractmethod
    async def cancel_order(
        self, client_order_id: str, *, symbol: str | None = None
    ) -> Order:
        """Cancel an order. Cancelling an already-terminal order is not an error."""

    async def cancel_all_orders(self, symbol: str | None = None) -> list[Order]:
        """Cancel every open order. Default implementation walks the open order list."""
        cancelled: list[Order] = []
        for order in await self.get_open_orders(symbol):
            cancelled.append(await self.cancel_order(order.client_order_id, symbol=order.symbol))
        return cancelled

    async def close_position(self, symbol: str) -> Order | None:
        """Flatten a position with a reduce-only market order."""
        from app.core.domain import OrderRequest as _Request
        from app.core.enums import OrderType

        positions = await self.get_positions(symbol)
        open_positions = [p for p in positions if p.is_open]
        if not open_positions:
            return None
        position = open_positions[0]
        return await self.create_order(
            _Request(
                symbol=symbol,
                side=position.side.closing_side,
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                reduce_only=True,
            )
        )

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    @abstractmethod
    def subscribe_candles(self, symbol: str, interval: str) -> AsyncIterator[Candle]:
        """Yield candles as they close."""

    @abstractmethod
    def subscribe_ticker(self, symbol: str) -> AsyncIterator[Ticker]:
        """Yield ticker updates."""

    async def subscribe_market_data(
        self, symbols: list[str], interval: str
    ) -> AsyncIterator[Candle]:
        """Convenience multiplexer over :meth:`subscribe_candles`.

        The default implementation supports a single symbol; adapters with true multiplexed
        sockets should override it.
        """
        if len(symbols) != 1:
            raise NotImplementedError(
                f"{self.name} adapter does not multiplex; subscribe per symbol"
            )
        async for candle in self.subscribe_candles(symbols[0], interval):
            yield candle

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    async def health_check(self) -> bool:
        """Cheap liveness probe."""
        try:
            await self.get_info()
        except Exception:  # probe must never propagate
            return False
        return True

    @staticmethod
    def _resolve_status(filled: float, total: float, cancelled: bool) -> OrderStatus:
        """Map fill progress to a status. Shared so adapters agree on semantics."""
        from app.core.numeric import EPSILON

        if cancelled:
            return OrderStatus.CANCELLED if filled <= EPSILON else OrderStatus.PARTIALLY_FILLED
        if filled <= EPSILON:
            return OrderStatus.OPEN
        if filled >= total - EPSILON:
            return OrderStatus.FILLED
        return OrderStatus.PARTIALLY_FILLED
