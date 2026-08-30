"""Live news from public RSS feeds.

This is the provider that makes news work without an account anywhere. Crypto outlets publish
open RSS feeds that update as they publish, so a running bot sees a story within a poll of it
going out - no API key, no vendor contract, no free-tier quota to exhaust.

What it does *not* do is change what news is allowed to influence. Articles from here go
through the same classifier and the same aggregator as any other source, which means they can
reduce confidence in a signal or veto it, and can never create one. A feed being free does not
make it authoritative, and source reputation is applied to these exactly as it is to a paid
vendor.

Two design points worth stating:

* **Filtering is strict.** When a caller asks for BTC news, only articles that actually mention
  BTC come back. The in-memory provider treats an untagged article as matching everything,
  which is right for a curated test fixture and wrong for a general news feed - it would answer
  "news about BTC" with whatever happened to be on the wire.
* **Results are cached process-wide**, because the API builds a provider per request. Without
  it, opening the news page would hit every publisher several times over.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from xml.etree.ElementTree import Element, ParseError

import httpx

# `defusedxml`, not the standard parser: these documents come from the public internet,
# and a hostile or merely broken one can carry entity-expansion payloads that a plain
# ElementTree will happily inflate until the process runs out of memory.
from defusedxml.ElementTree import fromstring as parse_xml

from app.core.clock import ensure_utc, utcnow
from app.core.logging import get_logger
from app.news.assets import detect_assets, mentions_asset
from app.news.models import NewsArticle
from app.news.providers import NewsProvider, NewsProviderError

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Feed:
    """One publisher's RSS or Atom feed.

    ``source`` is the name the classifier's reputation table is keyed on, so it is set here
    rather than read from the feed's own title, which changes without warning.
    """

    source: str
    url: str


#: Public crypto news feeds. All are open RSS and require no credentials.
DEFAULT_FEEDS: tuple[Feed, ...] = (
    Feed("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    Feed("cointelegraph", "https://cointelegraph.com/rss"),
    Feed("decrypt", "https://decrypt.co/feed"),
    Feed("bitcoin magazine", "https://bitcoinmagazine.com/feed"),
    Feed("cryptoslate", "https://cryptoslate.com/feed/"),
    Feed("the block", "https://www.theblock.co/rss.xml"),
)

#: XML namespaces that appear in these feeds.
NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
}


@dataclass
class _CacheEntry:
    fetched_at: float
    articles: list[NewsArticle] = field(default_factory=list)


class RssNewsProvider(NewsProvider):
    """Aggregates public crypto news feeds."""

    name = "rss"

    #: Shared across instances. The API constructs a provider per request and closes it again,
    #: so an instance-level cache would never be reused and every page view would re-poll
    #: every publisher.
    _cache: dict[str, _CacheEntry] = {}

    def __init__(
        self,
        feeds: Sequence[Feed] | None = None,
        *,
        cache_seconds: float = 60.0,
        timeout_seconds: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.feeds = tuple(feeds) if feeds is not None else DEFAULT_FEEDS
        self.cache_seconds = cache_seconds
        self.timeout_seconds = timeout_seconds
        self._client = client
        self._owns_client = client is None

    @classmethod
    def clear_cache(cls) -> None:
        """Drop cached articles. For tests, and for a manual refresh."""
        cls._cache.clear()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds),
                follow_redirects=True,
                headers={"User-Agent": "trading-platform/1.0 (+news reader)"},
            )
        return self._client

    # ------------------------------------------------------------------ #
    # Fetching
    # ------------------------------------------------------------------ #
    async def fetch(
        self,
        *,
        assets: Sequence[str] | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[NewsArticle]:
        articles = await self._collect()

        if since is not None:
            cutoff = ensure_utc(since)
            articles = [a for a in articles if a.published_at >= cutoff]

        if assets:
            wanted = [a.strip().upper() for a in assets if a.strip()]
            articles = [a for a in articles if _is_about(a, wanted)]

        articles.sort(key=lambda a: a.published_at, reverse=True)
        return articles[:limit]

    async def _collect(self) -> list[NewsArticle]:
        """Fetch every feed, using cached copies that are still fresh."""
        now = time.monotonic()
        stale = [
            feed
            for feed in self.feeds
            if (entry := self._cache.get(feed.url)) is None
            or now - entry.fetched_at >= self.cache_seconds
        ]

        failures = 0
        if stale:
            results = await asyncio.gather(
                *(self._fetch_feed(feed) for feed in stale), return_exceptions=True
            )
            for feed, result in zip(stale, results, strict=True):
                if isinstance(result, BaseException):
                    # One publisher being down must not empty the whole feed. Keep whatever
                    # was cached for it and carry on with the others.
                    failures += 1
                    logger.warning(
                        "news.rss_feed_failed",
                        source=feed.source,
                        error=f"{type(result).__name__}: {result}",
                    )
                    continue
                self._cache[feed.url] = _CacheEntry(fetched_at=now, articles=result)

        collected: list[NewsArticle] = []
        for feed in self.feeds:
            entry = self._cache.get(feed.url)
            if entry is not None:
                collected.extend(entry.articles)

        if not collected and failures:
            # Nothing to show *and* something went wrong. "No news happened" and "news is
            # broken" look identical from the caller's side unless they are separated here,
            # and a silently empty news page is how a broken feed goes unnoticed for weeks.
            # A partial result is not an error: one working publisher is still news.
            raise NewsProviderError(
                f"No news could be retrieved; {failures} feed(s) failed. "
                "Check network access."
            )
        return collected

    async def _fetch_feed(self, feed: Feed) -> list[NewsArticle]:
        client = await self._get_client()
        response = await client.get(feed.url)
        response.raise_for_status()
        return parse_feed(response.text, source=feed.source)

    async def health_check(self) -> bool:
        try:
            return bool(await self.fetch(limit=1))
        except Exception:  # a probe must never propagate
            return False

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


def _is_about(article: NewsArticle, wanted: Sequence[str]) -> bool:
    """Whether an article concerns any of the requested assets.

    Checks the tags first, then the text, so an asset this module has no alias for still
    matches on its bare ticker.
    """
    tagged = set(article.assets)
    for asset in wanted:
        if asset in tagged:
            return True
        if not tagged and mentions_asset(article.text, asset):
            return True
    return False


# =========================================================================== #
# Parsing
# =========================================================================== #
def parse_feed(xml_text: str, *, source: str) -> list[NewsArticle]:
    """Parse an RSS 2.0 or Atom document into articles.

    Malformed individual entries are skipped rather than failing the whole feed: publishers
    ship broken dates and empty titles regularly, and losing one item is better than losing
    the publisher.
    """
    try:
        root = parse_xml(xml_text)
    except (ParseError, ValueError) as exc:
        raise NewsProviderError(f"{source}: feed is not valid XML ({exc})") from exc

    entries = root.findall(".//item") or root.findall(".//atom:entry", NAMESPACES)
    articles: list[NewsArticle] = []
    for entry in entries:
        try:
            article = _entry_to_article(entry, source=source)
        except (ValueError, TypeError) as exc:
            logger.warning("news.rss_entry_skipped", source=source, error=str(exc))
            continue
        if article is not None:
            articles.append(article)
    return articles


def _entry_to_article(entry: Element, *, source: str) -> NewsArticle | None:
    title = _text(entry, "title") or _text(entry, "atom:title")
    if not title:
        return None

    description = (
        _text(entry, "description")
        or _text(entry, "atom:summary")
        or _text(entry, "content:encoded")
    )
    published = (
        _parse_date(_text(entry, "pubDate"))
        or _parse_date(_text(entry, "atom:published"))
        or _parse_date(_text(entry, "atom:updated"))
        or _parse_date(_text(entry, "dc:date"))
    )
    if published is None:
        # An article with no date cannot be aged, and the whole news model is built on
        # freshness decay. Dropping it is safer than dating it "now" and treating a
        # week-old story as breaking.
        return None

    url = _text(entry, "link") or _link_href(entry)
    return NewsArticle(
        title=_clean(title),
        description=_clean(description)[:2000] or None if description else None,
        source=source,
        published_at=published,
        url=url,
        assets=detect_assets(f"{title} {description or ''}"),
        external_id=_text(entry, "guid") or _text(entry, "atom:id") or url,
    )


def _text(entry: Element, tag: str) -> str | None:
    node = entry.find(tag, NAMESPACES) if ":" in tag else entry.find(tag)
    if node is None or node.text is None:
        return None
    text = str(node.text).strip()
    return text or None


def _link_href(entry: Element) -> str | None:
    """Atom puts the URL in an attribute rather than the element text."""
    for node in entry.findall("atom:link", NAMESPACES):
        if node.get("rel") in (None, "alternate"):
            href = node.get("href")
            if href:
                return href
    return None


def _clean(text: str | None) -> str:
    """Strip the HTML that publishers put in RSS descriptions."""
    if not text:
        return ""
    import re

    without_tags = re.sub(r"<[^>]+>", " ", text)
    unescaped = (
        without_tags.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
    )
    return re.sub(r"\s+", " ", unescaped).strip()


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)  # RFC 822, what RSS uses
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))  # Atom
        except ValueError:
            return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed = ensure_utc(parsed)
    # A publisher clock running fast would otherwise produce articles from the future, which
    # freshness decay reads as maximally recent forever.
    return min(parsed, utcnow())
