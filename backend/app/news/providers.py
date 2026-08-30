"""News providers.

The provider interface is deliberately minimal so that a commercial news API, a local file, or
nothing at all are interchangeable. The shipped default is :class:`NullNewsProvider`, because
the platform must run correctly with no news credentials configured — news is an enhancement to
the decision, never a prerequisite for it.

BLOCKED BY EXTERNAL DEPENDENCY: :class:`HttpNewsProvider` implements a generic JSON contract
and is fully tested against a mock transport, but it cannot be verified against a specific
commercial vendor's live API without credentials for that vendor. The field mapping is
configurable so adapting it is configuration, not code.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from app.core.clock import ensure_utc, from_epoch_ms, utcnow
from app.core.exceptions import ConfigurationError, TradingPlatformError
from app.core.logging import get_logger
from app.news.models import NewsArticle

logger = get_logger(__name__)


class NewsProviderError(TradingPlatformError):
    error_code = "news_provider_error"
    status_code = 502
    safe_to_expose = True


class NewsProvider(ABC):
    """Source of news articles."""

    name: str = "abstract"

    @abstractmethod
    async def fetch(
        self,
        *,
        assets: Sequence[str] | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[NewsArticle]:
        """Return articles, newest first."""

    async def close(self) -> None:  # noqa: B027 - optional hook, not every provider
        """Release any transport resources. Safe to call more than once."""

    async def health_check(self) -> bool:
        try:
            await self.fetch(limit=1)
        except Exception:  # probe must never propagate
            return False
        return True


class NullNewsProvider(NewsProvider):
    """Returns nothing. The default when no news source is configured.

    Not a placeholder for missing work: running without news is a supported, tested
    configuration, and every strategy is required to behave correctly under it.
    """

    name = "null"

    async def fetch(
        self,
        *,
        assets: Sequence[str] | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[NewsArticle]:
        return []


class InMemoryNewsProvider(NewsProvider):
    """Serves a preloaded list. Used by tests and by news-aware backtests."""

    name = "memory"

    def __init__(self, articles: Sequence[NewsArticle] | None = None) -> None:
        self._articles: list[NewsArticle] = list(articles or [])

    def add(self, *articles: NewsArticle) -> None:
        self._articles.extend(articles)

    def clear(self) -> None:
        self._articles.clear()

    async def fetch(
        self,
        *,
        assets: Sequence[str] | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[NewsArticle]:
        results = self._articles
        if since is not None:
            cutoff = ensure_utc(since)
            results = [a for a in results if a.published_at >= cutoff]
        if assets:
            wanted = {a.upper() for a in assets}
            results = [
                a for a in results
                if not a.assets or wanted & set(a.assets)
            ]
        return sorted(results, key=lambda a: a.published_at, reverse=True)[:limit]

    def window(self, start: datetime, end: datetime) -> list[NewsArticle]:
        """Articles published in ``[start, end)``. Used to replay news in a backtest."""
        lower, upper = ensure_utc(start), ensure_utc(end)
        return [a for a in self._articles if lower <= a.published_at < upper]


class FileNewsProvider(NewsProvider):
    """Reads articles from a JSON or JSONL file.

    Expected shape per record::

        {"title": "...", "source": "...", "published_at": "2024-01-01T00:00:00Z",
         "description": "...", "url": "...", "assets": ["BTC"]}

    Useful for reproducible news-aware backtests, where the news history has to be fixed.
    """

    name = "file"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._cache: list[NewsArticle] | None = None

    def _load(self) -> list[NewsArticle]:
        if self._cache is not None:
            return self._cache
        if not self.path.exists():
            raise NewsProviderError(
                f"News file not found: {self.path}", context={"path": str(self.path)}
            )
        raw = self.path.read_text(encoding="utf-8").strip()
        if not raw:
            self._cache = []
            return self._cache
        try:
            records = (
                json.loads(raw)
                if raw.lstrip().startswith("[")
                else [json.loads(line) for line in raw.splitlines() if line.strip()]
            )
        except json.JSONDecodeError as exc:
            raise NewsProviderError(
                f"{self.path.name} is not valid JSON: {exc}",
                context={"path": str(self.path)},
            ) from exc

        articles: list[NewsArticle] = []
        for index, record in enumerate(records):
            try:
                articles.append(_article_from_mapping(record, DEFAULT_FIELD_MAP))
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    "news.file_record_skipped",
                    path=str(self.path), index=index, error=str(exc),
                )
        self._cache = articles
        return articles

    async def fetch(
        self,
        *,
        assets: Sequence[str] | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[NewsArticle]:
        provider = InMemoryNewsProvider(self._load())
        return await provider.fetch(assets=assets, since=since, limit=limit)


@dataclass(slots=True)
class FieldMap:
    """Maps a vendor's JSON field names onto :class:`NewsArticle`."""

    title: str = "title"
    description: str = "description"
    source: str = "source"
    published_at: str = "published_at"
    url: str = "url"
    assets: str = "assets"
    external_id: str = "id"
    results_key: str | None = "results"


DEFAULT_FIELD_MAP = FieldMap()


class HttpNewsProvider(NewsProvider):
    """Generic JSON news API client.

    The vendor's response shape is described by a :class:`FieldMap` rather than hard-coded, so
    pointing this at a different provider is a configuration change.
    """

    name = "http"

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        field_map: FieldMap | None = None,
        auth_header: str = "Authorization",
        auth_scheme: str = "Bearer",
        timeout_seconds: float = 10.0,
        client: httpx.AsyncClient | None = None,
        extra_params: dict[str, str] | None = None,
    ) -> None:
        if not base_url:
            raise ConfigurationError("HttpNewsProvider requires a base_url")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.field_map = field_map or DEFAULT_FIELD_MAP
        self.auth_header = auth_header
        self.auth_scheme = auth_scheme
        self.timeout_seconds = timeout_seconds
        self.extra_params = dict(extra_params or {})
        self._client = client
        self._owns_client = client is None

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            value = (
                f"{self.auth_scheme} {self.api_key}" if self.auth_scheme else self.api_key
            )
            headers[self.auth_header] = value
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_seconds)
        return self._client

    async def fetch(
        self,
        *,
        assets: Sequence[str] | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[NewsArticle]:
        params: dict[str, str] = {"limit": str(limit), **self.extra_params}
        if assets:
            params["assets"] = ",".join(a.upper() for a in assets)
        if since is not None:
            params["since"] = ensure_utc(since).isoformat()

        client = await self._get_client()
        try:
            response = await client.get(
                self.base_url, params=params, headers=self._headers()
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            # Never echo the response body: some vendors reflect the API key in errors.
            raise NewsProviderError(
                f"News API returned HTTP {exc.response.status_code}",
                context={"status_code": exc.response.status_code},
            ) from exc
        except httpx.HTTPError as exc:
            raise NewsProviderError(
                f"News API request failed: {type(exc).__name__}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise NewsProviderError("News API returned a malformed JSON body") from exc

        return self._parse(payload)

    def _parse(self, payload: Any) -> list[NewsArticle]:
        records = payload
        key = self.field_map.results_key
        if isinstance(payload, dict):
            if key and key in payload:
                records = payload[key]
            else:
                records = next(
                    (v for v in payload.values() if isinstance(v, list)), []
                )
        if not isinstance(records, list):
            raise NewsProviderError("News API response did not contain a list of articles")

        articles: list[NewsArticle] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            try:
                articles.append(_article_from_mapping(record, self.field_map))
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("news.record_skipped", index=index, error=str(exc))
        return articles

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


def _article_from_mapping(record: dict[str, Any], field_map: FieldMap) -> NewsArticle:
    """Build a :class:`NewsArticle` from a vendor record."""
    title = record.get(field_map.title)
    if not title:
        raise ValueError("record has no title")

    published_raw = record.get(field_map.published_at)
    published = _parse_timestamp(published_raw)

    source_raw = record.get(field_map.source, "unknown")
    if isinstance(source_raw, dict):
        source_raw = source_raw.get("name") or source_raw.get("title") or "unknown"

    assets_raw = record.get(field_map.assets) or []
    if isinstance(assets_raw, str):
        assets = tuple(a.strip().upper() for a in assets_raw.split(",") if a.strip())
    else:
        assets = tuple(
            str(a.get("code") if isinstance(a, dict) else a).upper()
            for a in assets_raw
        )

    return NewsArticle(
        title=str(title),
        description=(
            str(record[field_map.description])
            if record.get(field_map.description)
            else None
        ),
        source=str(source_raw),
        published_at=published,
        url=str(record[field_map.url]) if record.get(field_map.url) else None,
        assets=assets,
        external_id=(
            str(record[field_map.external_id])
            if record.get(field_map.external_id)
            else None
        ),
        raw_metadata={},
    )


def _parse_timestamp(value: Any) -> datetime:
    if value is None:
        return utcnow()
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, (int, float)):
        # Heuristic: values above ~year 2001 in seconds are milliseconds.
        return from_epoch_ms(value if value > 1e11 else value * 1000)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"unparseable timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        from datetime import UTC

        parsed = parsed.replace(tzinfo=UTC)
    return ensure_utc(parsed)


def build_news_provider() -> NewsProvider:
    """Construct the configured provider.

    Falls back to :class:`NullNewsProvider` — with a warning — when news is requested but not
    configured, rather than failing startup. A missing news feed must never stop a bot from
    managing its open positions.
    """
    from app.config import AppEnv, get_settings

    settings = get_settings()
    if not settings.news_enabled:
        return NullNewsProvider()

    if settings.app_env is AppEnv.TEST:
        # Automated tests must not reach the public internet. A test that depends on what
        # CoinDesk published this morning is not a test, and the network calls would make the
        # suite slow and intermittently red for reasons that have nothing to do with the code.
        # Tests that need articles construct a provider explicitly with their own fixtures.
        return NullNewsProvider()

    if settings.news_provider == "rss":
        from app.news.rss import RssNewsProvider

        return RssNewsProvider(cache_seconds=settings.news_cache_seconds)
    if settings.news_provider == "http":
        if not settings.news_api_url:
            logger.warning("news.http_provider_unconfigured", fallback="null")
            return NullNewsProvider()
        return HttpNewsProvider(
            settings.news_api_url,
            api_key=(
                settings.news_api_key.get_secret_value()
                if settings.news_api_key
                else None
            ),
        )
    if settings.news_provider == "file":
        path = settings.data_dir / "news.json"
        if not path.exists():
            logger.warning("news.file_missing", path=str(path), fallback="null")
            return NullNewsProvider()
        return FileNewsProvider(path)
    return NullNewsProvider()


def synthetic_news(
    asset: str,
    *,
    count: int = 5,
    start: datetime | None = None,
    positive: bool = True,
) -> list[NewsArticle]:
    """Deterministic sample articles for demos and tests.

    Clearly synthetic. Never presented as real market data.
    """
    begin = start or utcnow()
    positive_titles = [
        f"{asset} surges as spot ETF sees record inflows",
        f"Major bank announces {asset} custody partnership",
        f"Regulator approves {asset} product after lengthy review",
        f"{asset} network upgrade goes live, adoption jumps",
        f"Analysts turn bullish on {asset} after strong quarter",
    ]
    negative_titles = [
        f"{asset} plunges after exchange halts withdrawals",
        f"Regulator announces crackdown affecting {asset} trading",
        f"Major {asset} bridge exploited, funds stolen",
        f"{asset} sell-off deepens amid hawkish rate outlook",
        f"Lawsuit filed against leading {asset} issuer",
    ]
    titles = positive_titles if positive else negative_titles
    return [
        NewsArticle(
            title=titles[index % len(titles)],
            description="Synthetic article generated for demonstration purposes only.",
            source="reuters" if index % 2 == 0 else "coindesk",
            published_at=begin - timedelta(minutes=index * 20),
            assets=(asset.upper(),),
            url=f"https://example.invalid/synthetic/{index}",
        )
        for index in range(count)
    ]
