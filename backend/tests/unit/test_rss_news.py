"""Live news feed tests.

Two things are being protected here.

**Asset detection**, because the whole feature rests on it. A feed says "Bitcoin"; the platform
asks about "BTC". If that translation is wrong in either direction the user gets a news page
that is either empty or full of stories about other coins, and a bot's news assessment is built
on the wrong articles.

**Failure behaviour**, because news is an enhancement and must never become a dependency. One
publisher going down, serving malformed XML, or serving a hostile document has to degrade to
"less news", not to a broken page or a stalled bot.
"""

from __future__ import annotations

from datetime import timedelta

import httpx
import pytest

from app.core.clock import utcnow
from app.news.assets import detect_assets, mentions_asset
from app.news.providers import NewsProviderError
from app.news.rss import Feed, RssNewsProvider, parse_feed


def rss(*items: str, ) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<rss version="2.0"><channel><title>Test</title>'
        + "".join(items)
        + "</channel></rss>"
    )


def item(
    title: str,
    *,
    description: str = "",
    published: str = "Wed, 20 Aug 2025 12:00:00 +0000",
    link: str = "https://example.invalid/a",
) -> str:
    return (
        f"<item><title>{title}</title>"
        f"<description>{description}</description>"
        f"<pubDate>{published}</pubDate>"
        f"<link>{link}</link></item>"
    )


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Ethereum upgrade ships</title>
    <summary>The merge of the day.</summary>
    <published>2025-08-20T12:00:00Z</published>
    <link rel="alternate" href="https://example.invalid/eth"/>
    <id>urn:1</id>
  </entry>
</feed>"""


# =========================================================================== #
# Asset detection
# =========================================================================== #
class TestAssetDetection:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Bitcoin hits a new high", "BTC"),
            ("BTC hits a new high", "BTC"),
            ("Ethereum staking grows", "ETH"),
            ("Solana outage resolved", "SOL"),
            ("Ripple settles with the SEC", "XRP"),
        ],
    )
    def test_names_and_tickers_both_resolve(self, text: str, expected: str) -> None:
        assert expected in detect_assets(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Press the button to continue",  # contains TON
            "An arbitrage opportunity appeared",  # contains ARB
            "The operation was a success",  # contains OP
            "A dotted line separates them",  # contains DOT
        ],
    )
    def test_substrings_of_ordinary_words_are_not_assets(self, text: str) -> None:
        """A substring match would tag routine prose with coins it never mentions."""
        assert detect_assets(text) == ()

    def test_bitcoin_cash_is_distinguished_from_bitcoin(self) -> None:
        detected = detect_assets("Bitcoin Cash sees a hashrate spike")
        assert "BCH" in detected

    def test_several_assets_in_one_headline(self) -> None:
        detected = detect_assets("Bitcoin and Ethereum rally as Solana lags")
        assert set(detected) >= {"BTC", "ETH", "SOL"}

    def test_unknown_ticker_still_matches_itself(self) -> None:
        """An asset with no alias entry must not become unsearchable."""
        assert mentions_asset("WIF surges on listing news", "WIF") is True
        assert mentions_asset("Nothing relevant here", "WIF") is False

    def test_matching_is_case_insensitive(self) -> None:
        assert "BTC" in detect_assets("BITCOIN RALLIES")
        assert "BTC" in detect_assets("bitcoin rallies")


# =========================================================================== #
# Parsing
# =========================================================================== #
class TestFeedParsing:
    def test_parses_rss_items(self) -> None:
        articles = parse_feed(
            rss(item("Bitcoin rallies", description="Strong inflows")),
            source="coindesk",
        )
        assert len(articles) == 1
        assert articles[0].title == "Bitcoin rallies"
        assert articles[0].source == "coindesk"
        assert "BTC" in articles[0].assets

    def test_parses_atom_entries(self) -> None:
        articles = parse_feed(ATOM, source="decrypt")
        assert len(articles) == 1
        assert articles[0].url == "https://example.invalid/eth"
        assert "ETH" in articles[0].assets

    def test_html_is_stripped_from_descriptions(self) -> None:
        articles = parse_feed(
            rss(item("Bitcoin news", description="&lt;p&gt;Bold &amp;amp; clear&lt;/p&gt;")),
            source="coindesk",
        )
        assert articles[0].description == "Bold & clear"

    def test_undated_articles_are_dropped(self) -> None:
        """Freshness decay is the whole model; an undated article cannot be aged."""
        articles = parse_feed(
            rss("<item><title>No date here</title></item>"), source="coindesk"
        )
        assert articles == []

    def test_future_dates_are_clamped_to_now(self) -> None:
        """A publisher clock running fast would otherwise stay 'breaking' forever."""
        articles = parse_feed(
            rss(item("Bitcoin news", published="Wed, 20 Aug 2125 12:00:00 +0000")),
            source="coindesk",
        )
        assert articles[0].published_at <= utcnow() + timedelta(seconds=1)

    def test_one_broken_item_does_not_lose_the_feed(self) -> None:
        articles = parse_feed(
            rss(
                "<item><title></title></item>",
                item("Bitcoin rallies"),
            ),
            source="coindesk",
        )
        assert len(articles) == 1

    def test_malformed_xml_raises_a_provider_error(self) -> None:
        with pytest.raises(NewsProviderError, match="not valid XML"):
            parse_feed("<rss><channel>", source="coindesk")

    def test_entity_expansion_is_refused(self) -> None:
        """A billion-laughs payload must not be expanded. These feeds are untrusted input."""
        bomb = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE lolz [<!ENTITY lol "lol">'
            '<!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
            '<!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">]>'
            "<rss><channel><title>&lol3;</title></channel></rss>"
        )
        with pytest.raises(NewsProviderError):
            parse_feed(bomb, source="hostile")


# =========================================================================== #
# The provider
# =========================================================================== #
@pytest.fixture(autouse=True)
def clean_cache():
    RssNewsProvider.clear_cache()
    yield
    RssNewsProvider.clear_cache()


def provider_for(bodies: dict[str, str], **kwargs) -> RssNewsProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        body = bodies.get(str(request.url))
        if body is None:
            return httpx.Response(404, text="not found")
        return httpx.Response(200, text=body, headers={"Content-Type": "application/xml"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    feeds = tuple(Feed(f"src{i}", url) for i, url in enumerate(bodies))
    return RssNewsProvider(feeds, client=client, **kwargs)


class TestRssProvider:
    @pytest.mark.asyncio
    async def test_returns_articles_from_every_feed(self) -> None:
        provider = provider_for(
            {
                "https://a.invalid/feed": rss(item("Bitcoin rallies")),
                "https://b.invalid/feed": rss(item("Ethereum upgrade lands")),
            }
        )
        articles = await provider.fetch()
        assert {a.title for a in articles} == {
            "Bitcoin rallies",
            "Ethereum upgrade lands",
        }

    @pytest.mark.asyncio
    async def test_filters_to_the_requested_asset(self) -> None:
        """The behaviour the user asked for: news for the currency they named."""
        provider = provider_for(
            {
                "https://a.invalid/feed": rss(
                    item("Bitcoin rallies"),
                    item("Solana outage resolved"),
                    item("Ethereum upgrade lands"),
                ),
            }
        )
        articles = await provider.fetch(assets=["BTC"])
        assert [a.title for a in articles] == ["Bitcoin rallies"]

    @pytest.mark.asyncio
    async def test_unrelated_articles_do_not_match_an_asset_query(self) -> None:
        """A general feed must not answer "BTC news" with whatever is on the wire."""
        provider = provider_for(
            {"https://a.invalid/feed": rss(item("Central bank holds rates steady"))}
        )
        assert await provider.fetch(assets=["BTC"]) == []

    @pytest.mark.asyncio
    async def test_newest_first(self) -> None:
        provider = provider_for(
            {
                "https://a.invalid/feed": rss(
                    item("Bitcoin older", published="Wed, 20 Aug 2025 09:00:00 +0000"),
                    item("Bitcoin newer", published="Wed, 20 Aug 2025 12:00:00 +0000"),
                )
            }
        )
        articles = await provider.fetch()
        assert articles[0].title == "Bitcoin newer"

    @pytest.mark.asyncio
    async def test_since_excludes_older_articles(self) -> None:
        provider = provider_for(
            {
                "https://a.invalid/feed": rss(
                    item("Bitcoin older", published="Wed, 20 Aug 2025 09:00:00 +0000"),
                    item("Bitcoin newer", published="Wed, 20 Aug 2025 12:00:00 +0000"),
                )
            }
        )
        from datetime import UTC, datetime

        cutoff = datetime(2025, 8, 20, 10, 0, tzinfo=UTC)
        titles = [a.title for a in await provider.fetch(since=cutoff)]
        assert titles == ["Bitcoin newer"]

    @pytest.mark.asyncio
    async def test_one_dead_publisher_does_not_empty_the_feed(self) -> None:
        """News is an enhancement. One outlet being down must not remove all of it."""
        provider = provider_for(
            {
                "https://a.invalid/feed": rss(item("Bitcoin rallies")),
                "https://dead.invalid/feed": "",  # 404s via the handler
            }
        )
        articles = await provider.fetch()
        assert [a.title for a in articles] == ["Bitcoin rallies"]

    @pytest.mark.asyncio
    async def test_every_feed_failing_is_reported_not_silently_empty(self) -> None:
        """"No news" and "news is broken" must be distinguishable."""
        provider = provider_for({"https://dead.invalid/feed": ""})
        with pytest.raises(NewsProviderError, match="network access"):
            await provider.fetch()

    @pytest.mark.asyncio
    async def test_results_are_cached_between_calls(self) -> None:
        """The API builds a provider per request; without the cache each page view re-polls."""
        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            return httpx.Response(200, text=rss(item("Bitcoin rallies")))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        feeds = (Feed("a", "https://a.invalid/feed"),)

        await RssNewsProvider(feeds, client=client, cache_seconds=300).fetch()
        await RssNewsProvider(feeds, client=client, cache_seconds=300).fetch()
        assert calls["count"] == 1

    @pytest.mark.asyncio
    async def test_cache_expires(self) -> None:
        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            return httpx.Response(200, text=rss(item("Bitcoin rallies")))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        feeds = (Feed("a", "https://a.invalid/feed"),)

        provider = RssNewsProvider(feeds, client=client, cache_seconds=0)
        await provider.fetch()
        await provider.fetch()
        assert calls["count"] == 2

    @pytest.mark.asyncio
    async def test_limit_is_honoured(self) -> None:
        provider = provider_for(
            {"https://a.invalid/feed": rss(*(item(f"Bitcoin story {i}") for i in range(10)))}
        )
        assert len(await provider.fetch(limit=3)) == 3

    @pytest.mark.asyncio
    async def test_health_check_never_raises(self) -> None:
        provider = provider_for({"https://dead.invalid/feed": ""})
        assert await provider.health_check() is False


# =========================================================================== #
# Integration with the assessment pipeline
# =========================================================================== #
class TestNewsPipeline:
    @pytest.mark.asyncio
    async def test_fetched_articles_produce_an_assessment(self) -> None:
        """The tags have to line up with what the aggregator considers relevant."""
        from app.news.scoring import NewsAggregator

        published = (utcnow() - timedelta(minutes=5)).strftime(
            "%a, %d %b %Y %H:%M:%S +0000"
        )
        provider = provider_for(
            {
                "https://a.invalid/feed": rss(
                    item(
                        "Bitcoin surges as spot ETF sees record inflows",
                        description="Institutional demand accelerates.",
                        published=published,
                    )
                )
            }
        )
        articles = await provider.fetch(assets=["BTC"])
        assert articles, "the fixture article should have been tagged BTC"

        assessment = NewsAggregator().assess(articles, "BTC")
        assert assessment.article_count == 1

    @pytest.mark.asyncio
    async def test_news_cannot_manufacture_a_signal(self) -> None:
        """The platform-wide rule, restated against a live-feed article.

        An assessment is an opinion about existing conviction. Nothing on this object is a
        trade instruction, and there is no code path from here to an order.
        """
        from app.news.scoring import NewsAggregator

        published = (utcnow() - timedelta(minutes=5)).strftime(
            "%a, %d %b %Y %H:%M:%S +0000"
        )
        provider = provider_for(
            {
                "https://a.invalid/feed": rss(
                    item("Bitcoin surges on record inflows", published=published)
                )
            }
        )
        assessment = NewsAggregator().assess(await provider.fetch(assets=["BTC"]), "BTC")
        assert not hasattr(assessment, "side")
        assert not hasattr(assessment, "order")


class TestProviderSelection:
    """What ``build_news_provider`` returns, and when."""

    def test_tests_never_get_a_network_provider(self, monkeypatch) -> None:
        """An automated test must not depend on what CoinDesk published this morning."""
        from app.config import AppEnv, Settings
        from app.news.providers import NullNewsProvider, build_news_provider

        settings = Settings(app_env=AppEnv.TEST, news_provider="rss", news_enabled=True)
        monkeypatch.setattr("app.config.get_settings", lambda: settings)
        assert isinstance(build_news_provider(), NullNewsProvider)

    def test_rss_is_the_default_outside_tests(self, monkeypatch) -> None:
        from app.config import AppEnv, Settings
        from app.news.providers import build_news_provider

        settings = Settings(app_env=AppEnv.STAGING)
        monkeypatch.setattr("app.config.get_settings", lambda: settings)
        provider = build_news_provider()
        assert provider.name == "rss"

    def test_news_can_still_be_switched_off_entirely(self, monkeypatch) -> None:
        """Running without news is a supported configuration, not a degraded one."""
        from app.config import AppEnv, Settings
        from app.news.providers import NullNewsProvider, build_news_provider

        settings = Settings(app_env=AppEnv.STAGING, news_enabled=False)
        monkeypatch.setattr("app.config.get_settings", lambda: settings)
        assert isinstance(build_news_provider(), NullNewsProvider)
