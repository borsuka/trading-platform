"""Market data providers.

Two abstractions, deliberately separate because their failure modes differ:

* :class:`HistoricalDataProvider` — bounded, replayable, used by the backtester. Failures are
  "data is missing", handled by aborting the run.
* :class:`LiveMarketDataProvider` — unbounded, stateful, used by the bot runtime. Failures are
  "the feed went quiet", handled by suppressing trading until it recovers.

Concrete implementations here are venue-independent: :class:`CsvHistoricalProvider` for local
files, :class:`InMemoryHistoricalProvider` for tests, and :class:`ExchangeHistoricalProvider`
which delegates to any :class:`~app.exchanges.base.ExchangeAdapter`.
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.clock import ensure_utc, from_epoch_ms, interval_to_timedelta, utcnow
from app.core.exceptions import InsufficientDataError, MarketDataError
from app.core.logging import get_logger
from app.market_data.models import Candle, MarketSnapshot, OrderBook, Ticker
from app.market_data.normalization import CandleNormalizer

if TYPE_CHECKING:
    from app.exchanges.base import ExchangeAdapter

logger = get_logger(__name__)


class HistoricalDataProvider(ABC):
    """Source of closed historical bars."""

    @abstractmethod
    async def get_candles(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        """Return closed candles with ``start <= open_time < end``, ascending."""

    @abstractmethod
    async def available_symbols(self) -> list[str]:
        """Symbols this provider can serve."""

    async def get_recent_candles(
        self, symbol: str, interval: str, count: int, *, end: datetime | None = None
    ) -> list[Candle]:
        """Return the ``count`` most recent bars ending at ``end``."""
        finish = end or utcnow()
        # Over-fetch to absorb gaps, then trim.
        span = interval_to_timedelta(interval) * max(count * 2, count + 50)
        candles = await self.get_candles(symbol, interval, finish - span, finish)
        return candles[-count:]


class LiveMarketDataProvider(ABC):
    """Source of live prices, books and streaming updates."""

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker: ...

    @abstractmethod
    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook: ...

    @abstractmethod
    async def get_snapshot(
        self, symbol: str, interval: str, lookback: int = 200
    ) -> MarketSnapshot:
        """Assemble everything the decision pipeline needs for one symbol."""

    @abstractmethod
    def stream_candles(self, symbol: str, interval: str) -> AsyncIterator[Candle]:
        """Yield candles as they close."""


# --------------------------------------------------------------------------- #
# In-memory
# --------------------------------------------------------------------------- #
class InMemoryHistoricalProvider(HistoricalDataProvider):
    """Serves pre-loaded candles. The backbone of deterministic tests and backtests."""

    def __init__(self, candles: Iterable[Candle] | None = None) -> None:
        self._data: dict[tuple[str, str], list[Candle]] = {}
        if candles:
            self.load(candles)

    def load(self, candles: Iterable[Candle]) -> None:
        """Add candles, normalising each (symbol, interval) series."""
        buckets: dict[tuple[str, str], list[Candle]] = {}
        for candle in candles:
            buckets.setdefault((candle.symbol, candle.interval), []).append(candle)
        normalizer = CandleNormalizer()
        for key, series in buckets.items():
            existing = self._data.get(key, [])
            cleaned, _ = normalizer.normalize([*existing, *series], interval=key[1])
            self._data[key] = cleaned

    async def get_candles(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> list[Candle]:
        series = self._data.get((symbol, interval))
        if series is None:
            raise InsufficientDataError(
                f"No historical data loaded for {symbol} {interval}",
                context={"symbol": symbol, "interval": interval},
            )
        lower = ensure_utc(start)
        upper = ensure_utc(end)
        return [c for c in series if lower <= c.open_time < upper]

    async def available_symbols(self) -> list[str]:
        return sorted({symbol for symbol, _ in self._data})

    def span(self, symbol: str, interval: str) -> tuple[datetime, datetime] | None:
        series = self._data.get((symbol, interval))
        if not series:
            return None
        return series[0].open_time, series[-1].open_time


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
class CsvHistoricalProvider(HistoricalDataProvider):
    """Reads candles from CSV files laid out as ``{root}/{symbol}_{interval}.csv``.

    Expected header: ``open_time,open,high,low,close,volume`` where ``open_time`` is either an
    ISO-8601 timestamp or a millisecond epoch. Files are cached after first read.
    """

    REQUIRED_COLUMNS = ("open_time", "open", "high", "low", "close", "volume")

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._cache: dict[tuple[str, str], list[Candle]] = {}

    def _path(self, symbol: str, interval: str) -> Path:
        return self.root / f"{symbol}_{interval}.csv"

    def _read(self, symbol: str, interval: str) -> list[Candle]:
        key = (symbol, interval)
        if key in self._cache:
            return self._cache[key]
        path = self._path(symbol, interval)
        if not path.exists():
            raise InsufficientDataError(
                f"No CSV data file for {symbol} {interval}",
                context={"path": str(path)},
            )
        candles: list[Candle] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = [c for c in self.REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise MarketDataError(
                    f"{path.name} is missing required columns: {', '.join(missing)}"
                )
            for line_number, row in enumerate(reader, start=2):
                try:
                    candles.append(self._row_to_candle(row, symbol, interval))
                except (ValueError, KeyError, TypeError) as exc:
                    raise MarketDataError(
                        f"{path.name}:{line_number} is malformed: {exc}",
                        context={"path": str(path), "line": line_number},
                    ) from exc
        cleaned, report = CandleNormalizer().normalize(candles, interval=interval)
        if not report.is_clean:
            logger.info("market_data.csv_loaded", file=path.name, summary=report.summary())
        self._cache[key] = cleaned
        return cleaned

    @staticmethod
    def _row_to_candle(row: dict[str, str], symbol: str, interval: str) -> Candle:
        raw_time = row["open_time"].strip()
        if raw_time.isdigit():
            open_time = from_epoch_ms(int(raw_time))
        else:
            open_time = ensure_utc(datetime.fromisoformat(raw_time.replace("Z", "+00:00")))
        return Candle(
            symbol=symbol,
            interval=interval,
            open_time=open_time,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            quote_volume=float(row["quote_volume"]) if row.get("quote_volume") else None,
        )

    async def get_candles(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> list[Candle]:
        series = self._read(symbol, interval)
        lower, upper = ensure_utc(start), ensure_utc(end)
        return [c for c in series if lower <= c.open_time < upper]

    async def available_symbols(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted({path.stem.rsplit("_", 1)[0] for path in self.root.glob("*.csv")})

    @staticmethod
    def write(path: Path, candles: Sequence[Candle]) -> None:
        """Persist candles in the format this provider reads."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CsvHistoricalProvider.REQUIRED_COLUMNS)
            for candle in candles:
                writer.writerow(
                    [
                        candle.open_time.isoformat(),
                        candle.open,
                        candle.high,
                        candle.low,
                        candle.close,
                        candle.volume,
                    ]
                )


# --------------------------------------------------------------------------- #
# Exchange-backed
# --------------------------------------------------------------------------- #
class ExchangeHistoricalProvider(HistoricalDataProvider):
    """Fetches history from any exchange adapter, paginating as needed."""

    def __init__(self, adapter: ExchangeAdapter, *, page_size: int = 1000) -> None:
        self.adapter = adapter
        self.page_size = page_size

    async def get_candles(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> list[Candle]:
        step = interval_to_timedelta(interval)
        cursor = ensure_utc(start)
        finish = ensure_utc(end)
        collected: list[Candle] = []
        guard = 0
        max_pages = 10_000

        while cursor < finish and guard < max_pages:
            guard += 1
            page = await self.adapter.get_candles(
                symbol, interval, start=cursor, end=finish, limit=self.page_size
            )
            if not page:
                break
            collected.extend(page)
            advanced = page[-1].open_time + step
            if advanced <= cursor:  # provider is not advancing; stop rather than spin
                break
            cursor = advanced
            if len(page) < self.page_size:
                break

        if guard >= max_pages:
            logger.warning(
                "market_data.pagination_limit_reached", symbol=symbol, interval=interval
            )
        cleaned, _ = CandleNormalizer().normalize(collected, interval=interval)
        return [c for c in cleaned if ensure_utc(start) <= c.open_time < finish]

    async def available_symbols(self) -> list[str]:
        specs = await self.adapter.get_instruments()
        return sorted(specs)


class ExchangeLiveProvider(LiveMarketDataProvider):
    """Live data backed by an exchange adapter.

    Holds a small rolling candle cache per symbol so that each decision cycle costs one
    incremental request rather than a full history refetch.
    """

    def __init__(self, adapter: ExchangeAdapter, *, cache_size: int = 500) -> None:
        self.adapter = adapter
        self.cache_size = cache_size
        self._candles: dict[tuple[str, str], list[Candle]] = {}

    async def get_ticker(self, symbol: str) -> Ticker:
        return await self.adapter.get_ticker(symbol)

    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        return await self.adapter.get_order_book(symbol, depth=depth)

    async def get_snapshot(
        self, symbol: str, interval: str, lookback: int = 200
    ) -> MarketSnapshot:
        candles = await self._refresh_candles(symbol, interval, lookback)
        ticker: Ticker | None = None
        book: OrderBook | None = None
        try:
            ticker = await self.adapter.get_ticker(symbol)
        except MarketDataError as exc:
            logger.warning("market_data.ticker_unavailable", symbol=symbol, error=str(exc))
        try:
            book = await self.adapter.get_order_book(symbol, depth=20)
        except MarketDataError as exc:
            logger.warning("market_data.book_unavailable", symbol=symbol, error=str(exc))

        return MarketSnapshot(
            symbol=symbol,
            timestamp=utcnow(),
            candles=tuple(candles),
            ticker=ticker,
            order_book=book,
            funding_rate=ticker.funding_rate if ticker else None,
            open_interest=ticker.open_interest if ticker else None,
        )

    async def _refresh_candles(
        self, symbol: str, interval: str, lookback: int
    ) -> list[Candle]:
        key = (symbol, interval)
        cached = self._candles.get(key, [])
        step = interval_to_timedelta(interval)
        if cached:
            since = cached[-1].open_time
            fetched = await self.adapter.get_candles(
                symbol, interval, start=since, end=utcnow() + step, limit=lookback
            )
            merged: dict[datetime, Candle] = {c.open_time: c for c in cached}
            merged.update({c.open_time: c for c in fetched})
            series = [merged[k] for k in sorted(merged)]
        else:
            series = await self.adapter.get_candles(
                symbol, interval, start=utcnow() - step * (lookback + 5), end=utcnow() + step,
                limit=lookback,
            )
        cleaned, _ = CandleNormalizer(drop_unclosed=True).normalize(series, interval=interval)
        trimmed = cleaned[-self.cache_size :]
        self._candles[key] = trimmed
        return trimmed[-lookback:]

    async def stream_candles(self, symbol: str, interval: str) -> AsyncIterator[Candle]:
        async for candle in self.adapter.subscribe_candles(symbol, interval):
            yield candle


# --------------------------------------------------------------------------- #
# Synthetic data for demos and property tests
# --------------------------------------------------------------------------- #
def generate_synthetic_candles(
    symbol: str,
    interval: str,
    count: int,
    *,
    start: datetime | None = None,
    start_price: float = 100.0,
    drift: float = 0.0,
    volatility: float = 0.01,
    seed: int = 42,
    regime_shifts: bool = True,
    gap_volatility: float = 0.0,
) -> list[Candle]:
    """Generate a deterministic synthetic OHLCV series.

    This is **not** a market model and must never be presented as a performance result. It
    exists so the platform can be demonstrated and property-tested end to end without needing
    exchange credentials or network access.

    With ``regime_shifts`` the series alternates trending and ranging segments, which exercises
    the regime detector and the regime-gated strategies.

    ``gap_volatility`` opens each bar away from the previous close. It defaults to zero, giving
    a continuous series where ``open == previous close``; set it above zero to model the
    overnight-style gaps that make the difference between a same-bar fill and a next-bar fill
    observable.
    """
    import numpy as np

    if count < 1:
        raise ValueError("count must be >= 1")
    rng = np.random.default_rng(seed)
    step = interval_to_timedelta(interval)
    begin = ensure_utc(start) if start else utcnow() - step * count

    drifts = np.full(count, drift, dtype=float)
    vols = np.full(count, volatility, dtype=float)
    if regime_shifts:
        segment = max(20, count // 6)
        for index, offset in enumerate(range(0, count, segment)):
            end = min(offset + segment, count)
            phase = index % 3
            if phase == 0:      # trending up
                drifts[offset:end] = abs(drift) + volatility * 0.35
                vols[offset:end] = volatility
            elif phase == 1:    # ranging, quiet
                drifts[offset:end] = 0.0
                vols[offset:end] = volatility * 0.55
            else:               # trending down, volatile
                drifts[offset:end] = -(abs(drift) + volatility * 0.30)
                vols[offset:end] = volatility * 1.6

    candles: list[Candle] = []
    price = start_price
    for index in range(count):
        shock = float(rng.normal(drifts[index], vols[index]))
        gap = float(rng.normal(0.0, gap_volatility)) if gap_volatility > 0 else 0.0
        open_price = max(0.01, price * (1.0 + gap))
        close_price = max(0.01, open_price * (1.0 + shock))
        wick = abs(float(rng.normal(0.0, vols[index] * 0.6))) * open_price
        high = max(open_price, close_price) + wick
        low = max(0.005, min(open_price, close_price) - wick)
        # Volume expands with the size of the move relative to the regime's own volatility.
        # Real markets show this coupling, and without it a breakout strategy's volume filter
        # can never fire, which would make the synthetic series useless for exercising it.
        move_z = abs(shock) / max(vols[index], 1e-9)
        volume = float(abs(rng.normal(1_000.0, 200.0)) * (0.6 + 0.7 * move_z))
        candles.append(
            Candle(
                symbol=symbol,
                interval=interval,
                open_time=begin + step * index,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=volume,
                quote_volume=volume * close_price,
                trade_count=int(volume // 10) + 1,
                is_closed=True,
            )
        )
        price = close_price
    return candles


def build_snapshot(
    candles: Sequence[Candle],
    *,
    spread_bps: float = 2.0,
    book_depth_multiple: float = 50.0,
    now: datetime | None = None,
) -> MarketSnapshot:
    """Build a snapshot (with a synthetic top-of-book) from a candle series.

    Used by the backtester, which has candles but no book, so that liquidity and spread checks
    exercise the same code path they will in live trading.
    """
    from app.market_data.models import OrderBookLevel

    if not candles:
        raise InsufficientDataError("Cannot build a snapshot from an empty candle series")
    last = candles[-1]
    price = last.close
    reference_time = now or last.close_time
    half_spread = price * (spread_bps / 2.0) / 10_000.0
    bid, ask = price - half_spread, price + half_spread
    unit = max(last.volume, 1.0) * book_depth_multiple / 20.0

    bids = tuple(
        OrderBookLevel(price=bid - half_spread * 2 * i, quantity=unit * (1.0 + i * 0.1))
        for i in range(10)
    )
    asks = tuple(
        OrderBookLevel(price=ask + half_spread * 2 * i, quantity=unit * (1.0 + i * 0.1))
        for i in range(10)
    )
    return MarketSnapshot(
        symbol=last.symbol,
        timestamp=reference_time,
        candles=tuple(candles),
        ticker=Ticker(
            symbol=last.symbol,
            price=price,
            timestamp=reference_time,
            bid=bid,
            ask=ask,
            bid_size=unit,
            ask_size=unit,
            volume_24h=sum(c.volume for c in candles[-96:]),
        ),
        order_book=OrderBook(
            symbol=last.symbol, timestamp=reference_time, bids=bids, asks=asks
        ),
    )


def resample_candles(candles: Sequence[Candle], target_interval: str) -> list[Candle]:
    """Aggregate candles into a longer interval.

    Only whole multiples are supported; partial trailing buckets are dropped so that the last
    bar is never a half-formed aggregate.
    """
    if not candles:
        return []
    source_step = interval_to_timedelta(candles[0].interval)
    target_step = interval_to_timedelta(target_interval)
    if target_step <= source_step:
        raise ValueError(
            f"Target interval {target_interval} must be longer than source "
            f"{candles[0].interval}"
        )
    ratio = target_step / source_step
    if ratio != int(ratio):
        raise ValueError(
            f"{target_interval} is not a whole multiple of {candles[0].interval}"
        )
    size = int(ratio)

    buckets: dict[datetime, list[Candle]] = {}
    origin = candles[0].open_time
    for candle in candles:
        offset = int((candle.open_time - origin) / source_step)
        bucket_start = origin + source_step * (offset - offset % size)
        buckets.setdefault(bucket_start, []).append(candle)

    aggregated: list[Candle] = []
    for bucket_start in sorted(buckets):
        group = buckets[bucket_start]
        if len(group) < size:  # incomplete bucket
            continue
        aggregated.append(
            Candle(
                symbol=group[0].symbol,
                interval=target_interval,
                open_time=bucket_start,
                open=group[0].open,
                high=max(c.high for c in group),
                low=min(c.low for c in group),
                close=group[-1].close,
                volume=sum(c.volume for c in group),
                quote_volume=sum(c.quote_volume or 0.0 for c in group) or None,
                trade_count=sum(c.trade_count or 0 for c in group) or None,
                is_closed=True,
            )
        )
    return aggregated
