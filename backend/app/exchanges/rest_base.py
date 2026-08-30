"""Shared HTTP machinery for live exchange adapters.

Handles the concerns every venue shares: signing, rate limiting, timeout classification and
error mapping. The part that matters most is **timeout classification**.

An exchange request can fail three ways, and treating them alike is how bots double their
positions:

* **Refused before reaching the matching engine** (rate limit, bad signature, malformed
  request). No order exists. Safe to retry.
* **Rejected by the matching engine** (insufficient balance, bad price). No order exists, and
  retrying unchanged will fail identically.
* **Unknown** (timeout, connection reset mid-flight). An order may or may not exist. This maps
  to :class:`~app.core.exceptions.ExchangeTimeoutError`, and the order manager resolves it by
  *querying*, never by resending.

Secrets never appear in logs or exceptions: request signing happens inside
:meth:`RestExchangeAdapter._request` and only the endpoint path is ever logged.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from abc import abstractmethod
from typing import Any

import httpx

from app.core.exceptions import (
    ExchangeAuthError,
    ExchangeConnectionError,
    ExchangeError,
    ExchangeRateLimitError,
    ExchangeTimeoutError,
    InsufficientBalanceError,
    OrderRejectedError,
)
from app.core.logging import get_logger
from app.exchanges.base import ExchangeAdapter, ExchangeCredentials
from app.exchanges.rate_limit import RateLimiter, default_limiter

logger = get_logger(__name__)

#: HTTP statuses that mean "the venue never processed this". Safe to retry.
SAFE_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class RestExchangeAdapter(ExchangeAdapter):
    """Base for REST-based venue adapters."""

    #: Production REST base URL.
    base_url: str = ""
    #: Testnet REST base URL.
    testnet_url: str = ""
    #: Receive window sent with signed requests, in milliseconds.
    recv_window_ms: int = 5_000

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        client: httpx.AsyncClient | None = None,
        limiter: RateLimiter | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.credentials = credentials
        self.timeout_seconds = timeout_seconds
        self.limiter = limiter or default_limiter()
        self._client = client
        self._owns_client = client is None
        self._connected = False

    @property
    def is_live(self) -> bool:  # type: ignore[override]
        """Any REST adapter routes real orders, testnet included.

        Testnet is still "live" for the purposes of the platform's safety gates: the code path
        is the production one, and only the venue's own balances are fake.
        """
        return True

    @property
    def url(self) -> str:
        return self.testnet_url if self.credentials.testnet else self.base_url

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.url,
                timeout=httpx.Timeout(self.timeout_seconds),
                headers={"User-Agent": "trading-platform/1.0"},
            )
        return self._client

    async def connect(self) -> None:
        await self._get_client()
        self._connected = True

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
        self._connected = False

    # ------------------------------------------------------------------ #
    # Signing
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _sign(
        self, method: str, path: str, params: dict[str, Any], body: str
    ) -> tuple[dict[str, str], dict[str, Any], str]:
        """Return ``(headers, params, body)`` with venue-specific authentication applied."""

    @staticmethod
    def _hmac_sha256(secret: str, payload: str) -> str:
        return hmac.new(
            secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()

    @staticmethod
    def _timestamp_ms() -> int:
        return int(time.time() * 1000)

    # ------------------------------------------------------------------ #
    # Request
    # ------------------------------------------------------------------ #
    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        signed: bool = False,
        bucket: str = "public",
        cost: float = 1.0,
    ) -> Any:
        """Perform one request, mapping every failure onto a typed exception."""
        await self.limiter.acquire(bucket, cost)

        request_params = {k: v for k, v in (params or {}).items() if v is not None}
        request_body = json.dumps(body, separators=(",", ":")) if body else ""
        headers: dict[str, str] = {}

        if signed:
            headers, request_params, request_body = self._sign(
                method, path, request_params, request_body
            )
        if request_body:
            headers.setdefault("Content-Type", "application/json")

        client = await self._get_client()
        try:
            response = await client.request(
                method,
                path,
                params=request_params or None,
                content=request_body or None,
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            # The dangerous case: the request may or may not have been applied.
            raise ExchangeTimeoutError(
                f"{self.name} {method} {path} timed out after {self.timeout_seconds}s; "
                "the outcome is unknown and must be resolved by querying, not retrying",
                context={"endpoint": path, "method": method},
            ) from exc
        except (httpx.ConnectError, httpx.ReadError, httpx.WriteError) as exc:
            raise ExchangeConnectionError(
                f"{self.name} connection failed on {method} {path}: {type(exc).__name__}",
                context={"endpoint": path},
            ) from exc
        except httpx.HTTPError as exc:
            raise ExchangeError(
                f"{self.name} request failed: {type(exc).__name__}",
                context={"endpoint": path},
            ) from exc

        return self._handle_response(response, path)

    def _handle_response(self, response: httpx.Response, path: str) -> Any:
        status = response.status_code
        if status == 429:
            retry_after = response.headers.get("Retry-After")
            raise ExchangeRateLimitError(
                f"{self.name} rate limit hit on {path}",
                retry_after_seconds=float(retry_after) if retry_after else None,
                context={"endpoint": path},
            )
        if status in {401, 403}:
            raise ExchangeAuthError(
                f"{self.name} rejected the credentials (HTTP {status})",
                context={"endpoint": path, "status": status},
            )
        if status in SAFE_RETRY_STATUSES:
            raise ExchangeConnectionError(
                f"{self.name} returned HTTP {status} on {path}; the request was not processed",
                context={"endpoint": path, "status": status},
            )

        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ExchangeError(
                f"{self.name} returned a non-JSON body (HTTP {status}) on {path}",
                context={"endpoint": path, "status": status},
            ) from exc

        if status >= 400:
            raise self._map_error(payload, path, status)
        return self._unwrap(payload, path)

    @abstractmethod
    def _unwrap(self, payload: Any, path: str) -> Any:
        """Extract the result from a venue-specific success envelope."""

    def _map_error(self, payload: Any, path: str, status: int) -> ExchangeError:
        """Map a venue error body onto a typed exception.

        The default reads common fields; adapters override for venue-specific codes.
        """
        message = ""
        if isinstance(payload, dict):
            for key in ("msg", "message", "retMsg", "error", "reason"):
                value = payload.get(key)
                if value:
                    message = str(value)
                    break
        message = message or f"HTTP {status}"
        lowered = message.lower()
        if "insufficient" in lowered or "balance" in lowered:
            return InsufficientBalanceError(
                f"{self.name}: {message}", context={"endpoint": path}
            )
        if any(
            token in lowered
            for token in ("reject", "invalid", "too small", "min", "notional", "precision")
        ):
            return OrderRejectedError(
                f"{self.name}: {message}", context={"endpoint": path}
            )
        return ExchangeError(f"{self.name}: {message}", context={"endpoint": path})

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        """Venue numeric fields arrive as strings, empty strings and nulls interchangeably."""
        if value is None or value == "":
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def __repr__(self) -> str:  # pragma: no cover - never leak credentials
        return (
            f"{type(self).__name__}(testnet={self.credentials.testnet}, "
            f"url={self.url!r})"
        )
