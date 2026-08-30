"""News intelligence tests.

The safety-critical properties tested here:

* deduplication actually collapses syndicated copies, so one story cannot masquerade as
  consensus;
* stale news is excluded and older news is decayed;
* hedged, low-reputation and unconfirmed reports carry less weight;
* the news layer produces a *modifier*, never a trade.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from app.core.enums import NewsEventType, NewsImpact, NewsSentiment
from app.news.classification import NewsClassifier, source_reputation
from app.news.models import NewsArticle, content_hash
from app.news.providers import (
    FileNewsProvider,
    InMemoryNewsProvider,
    NewsProviderError,
    NullNewsProvider,
    synthetic_news,
)
from app.news.scoring import (
    NewsAggregator,
    NewsAssessment,
    NewsScoringConfig,
    combine_assessments,
    half_life_decay,
    title_similarity,
)

NOW = datetime(2024, 6, 12, 12, 0, tzinfo=UTC)


def article(
    title: str,
    *,
    source: str = "reuters",
    minutes_ago: float = 5.0,
    description: str | None = None,
    assets: tuple[str, ...] = ("BTC",),
) -> NewsArticle:
    return NewsArticle(
        title=title,
        description=description,
        source=source,
        published_at=NOW - timedelta(minutes=minutes_ago),
        assets=assets,
    )


# =========================================================================== #
# Models
# =========================================================================== #
class TestNewsArticle:
    def test_requires_a_title(self) -> None:
        with pytest.raises(ValueError, match="must have a title"):
            NewsArticle(title="  ", source="reuters", published_at=NOW)

    def test_requires_aware_timestamp(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            NewsArticle(
                title="x", source="reuters", published_at=datetime(2024, 1, 1)
            )

    def test_assets_are_uppercased(self) -> None:
        assert article("x", assets=("btc", "eth")).assets == ("BTC", "ETH")

    def test_content_hash_ignores_punctuation_and_case(self) -> None:
        assert content_hash("Bitcoin ETF Approved!") == content_hash("bitcoin etf approved")

    def test_content_hash_distinguishes_different_stories(self) -> None:
        assert content_hash("Bitcoin ETF approved") != content_hash("Bitcoin ETF rejected")

    def test_staleness(self) -> None:
        old = article("x", minutes_ago=300)
        assert old.is_stale(timedelta(minutes=180), now=NOW)
        assert not old.is_stale(timedelta(hours=6), now=NOW)


# =========================================================================== #
# Classification
# =========================================================================== #
class TestClassification:
    @pytest.fixture
    def classifier(self) -> NewsClassifier:
        return NewsClassifier()

    def test_hack_is_critical_and_negative(self, classifier: NewsClassifier) -> None:
        analysis = classifier.classify(
            article("Major exchange hacked, $200M stolen in exploit"), "BTC"
        )
        assert analysis.event_type is NewsEventType.HACK
        assert analysis.sentiment in {NewsSentiment.NEGATIVE, NewsSentiment.VERY_NEGATIVE}
        assert analysis.impact is NewsImpact.CRITICAL
        assert analysis.score < 0

    def test_etf_approval_is_positive(self, classifier: NewsClassifier) -> None:
        analysis = classifier.classify(
            article("SEC approves spot Bitcoin ETF, record inflows expected"), "BTC"
        )
        assert analysis.event_type in {NewsEventType.ETF, NewsEventType.REGULATION}
        assert analysis.score > 0

    def test_regulatory_crackdown_is_negative(self, classifier: NewsClassifier) -> None:
        analysis = classifier.classify(
            article("Regulator announces crackdown, trading banned in major market"), "BTC"
        )
        assert analysis.event_type is NewsEventType.REGULATION
        assert analysis.score < 0

    def test_neutral_text_scores_near_zero(self, classifier: NewsClassifier) -> None:
        analysis = classifier.classify(
            article("Bitcoin trades sideways as market awaits data"), "BTC"
        )
        assert abs(analysis.score) < 0.3

    def test_negation_flips_sentiment(self, classifier: NewsClassifier) -> None:
        approved = classifier.classify(article("Regulator approves the ETF"), "BTC")
        denied = classifier.classify(
            article("Regulator denies approval for the ETF"), "BTC"
        )
        assert approved.score > denied.score

    def test_hedged_language_reduces_confidence(self, classifier: NewsClassifier) -> None:
        firm = classifier.classify(
            article("Exchange halted withdrawals after confirmed exploit"), "BTC"
        )
        hedged = classifier.classify(
            article("Exchange may have reportedly halted withdrawals after alleged exploit"),
            "BTC",
        )
        assert hedged.confidence < firm.confidence

    def test_low_reputation_source_reduces_confidence(
        self, classifier: NewsClassifier
    ) -> None:
        wire = classifier.classify(
            article("Exchange hacked, funds stolen", source="reuters"), "BTC"
        )
        forum = classifier.classify(
            article("Exchange hacked, funds stolen", source="reddit"), "BTC"
        )
        assert forum.confidence < wire.confidence

    def test_urgency_detected(self, classifier: NewsClassifier) -> None:
        urgent = classifier.classify(
            article("BREAKING: exchange halted trading, official confirmed"), "BTC"
        )
        calm = classifier.classify(article("Exchange discusses future plans"), "BTC")
        assert urgent.urgency > calm.urgency

    def test_word_boundaries_respected(self, classifier: NewsClassifier) -> None:
        """'ban' must not match inside 'urban' or 'bank'."""
        analysis = classifier.classify(
            article("Urban bank explores blockchain settlement"), "BTC"
        )
        assert analysis.event_type is not NewsEventType.REGULATION

    def test_deterministic(self, classifier: NewsClassifier) -> None:
        item = article("SEC approves spot Bitcoin ETF")
        first = classifier.classify(item, "BTC")
        second = classifier.classify(item, "BTC")
        assert first.score == pytest.approx(second.score)
        assert first.event_type is second.event_type

    def test_novelty_scales_the_score(self, classifier: NewsClassifier) -> None:
        item = article("Major exchange hacked, funds stolen")
        fresh = classifier.classify(item, "BTC", novelty=1.0)
        repeat = classifier.classify(item, "BTC", novelty=0.3)
        assert abs(repeat.score) < abs(fresh.score)

    def test_source_reputation_lookup(self) -> None:
        assert source_reputation("Reuters") > source_reputation("Twitter")
        # Substring matching is deliberate so "Reuters Business" resolves to Reuters.
        assert source_reputation("Reuters Business") == pytest.approx(0.95)
        assert source_reputation("Daily Crypto Gazette") == pytest.approx(0.45)


# =========================================================================== #
# Deduplication
# =========================================================================== #
class TestDeduplication:
    @pytest.fixture
    def aggregator(self) -> NewsAggregator:
        return NewsAggregator()

    def test_identical_stories_collapse(self, aggregator: NewsAggregator) -> None:
        items = [
            article("SEC approves spot Bitcoin ETF", source="reuters"),
            article("SEC approves spot Bitcoin ETF", source="coindesk"),
            article("sec approves spot bitcoin etf!", source="cointelegraph"),
        ]
        unique, removed = aggregator.deduplicate(items)
        assert len(unique) == 1
        assert removed == 2

    def test_highest_reputation_copy_survives(self, aggregator: NewsAggregator) -> None:
        items = [
            article("SEC approves spot Bitcoin ETF", source="reddit"),
            article("SEC approves spot Bitcoin ETF", source="reuters"),
        ]
        unique, _ = aggregator.deduplicate(items)
        assert unique[0].source == "reuters"

    def test_near_duplicates_collapse(self, aggregator: NewsAggregator) -> None:
        items = [
            article("SEC approves the spot Bitcoin ETF today", source="reuters"),
            article("SEC approves spot Bitcoin ETF", source="coindesk"),
        ]
        unique, removed = aggregator.deduplicate(items)
        assert len(unique) == 1
        assert removed == 1

    def test_different_stories_are_kept(self, aggregator: NewsAggregator) -> None:
        items = [
            article("SEC approves spot Bitcoin ETF"),
            article("Major exchange halts withdrawals after exploit"),
        ]
        unique, removed = aggregator.deduplicate(items)
        assert len(unique) == 2
        assert removed == 0

    def test_syndication_does_not_inflate_the_score(
        self, aggregator: NewsAggregator
    ) -> None:
        """The property that stops one story looking like ten independent confirmations."""
        single = aggregator.assess(
            [article("SEC approves spot Bitcoin ETF, record inflows")], "BTC", now=NOW
        )
        syndicated = NewsAggregator().assess(
            [
                article("SEC approves spot Bitcoin ETF, record inflows", source=src)
                for src in ("reuters", "coindesk", "cointelegraph", "decrypt", "the block")
            ],
            "BTC",
            now=NOW,
        )
        assert syndicated.article_count == 1
        assert syndicated.duplicates_removed == 4
        assert syndicated.directional_score == pytest.approx(
            single.directional_score, abs=0.05
        )

    def test_title_similarity(self) -> None:
        assert title_similarity("SEC approves the Bitcoin ETF", "SEC approves Bitcoin ETF") > 0.8
        assert title_similarity("Bitcoin surges", "Ethereum crashes") < 0.3
        assert title_similarity("", "anything") == 0.0


# =========================================================================== #
# Scoring and aggregation
# =========================================================================== #
class TestScoring:
    @pytest.fixture
    def aggregator(self) -> NewsAggregator:
        return NewsAggregator()

    def test_no_articles_gives_empty_assessment(self, aggregator: NewsAggregator) -> None:
        assessment = aggregator.assess([], "BTC", now=NOW)
        assert not assessment.has_coverage
        assert assessment.directional_score == 0.0

    def test_stale_articles_excluded(self, aggregator: NewsAggregator) -> None:
        assessment = aggregator.assess(
            [article("Bitcoin surges on ETF approval", minutes_ago=500)], "BTC", now=NOW
        )
        assert not assessment.has_coverage

    def test_future_timestamps_rejected(self, aggregator: NewsAggregator) -> None:
        assessment = aggregator.assess(
            [article("Bitcoin surges", minutes_ago=-120)], "BTC", now=NOW
        )
        assert not assessment.has_coverage

    def test_recent_news_outweighs_older(self, aggregator: NewsAggregator) -> None:
        recent = aggregator.assess(
            [article("Major exchange hacked, funds stolen", minutes_ago=1)], "BTC", now=NOW
        )
        older = NewsAggregator().assess(
            [article("Major exchange hacked, funds stolen", minutes_ago=150)],
            "BTC", now=NOW,
        )
        assert abs(recent.directional_score) > abs(older.directional_score)

    def test_decay_is_exponential(self) -> None:
        assert half_life_decay(60.0, 60.0) == pytest.approx(0.5)
        assert half_life_decay(120.0, 60.0) == pytest.approx(0.25)
        assert half_life_decay(0.0, 60.0) == pytest.approx(1.0)

    def test_positive_and_negative_news_offset(self, aggregator: NewsAggregator) -> None:
        mixed = aggregator.assess(
            [
                article("SEC approves spot Bitcoin ETF with record inflows"),
                article("Major exchange hacked, $200M stolen in exploit"),
            ],
            "BTC",
            now=NOW,
        )
        assert abs(mixed.directional_score) < 0.6

    def test_irrelevant_assets_filtered(self, aggregator: NewsAggregator) -> None:
        assessment = aggregator.assess(
            [article("Solana network halts", assets=("SOL",))], "BTC", now=NOW
        )
        assert not assessment.has_coverage

    def test_repeat_story_has_lower_novelty(self, aggregator: NewsAggregator) -> None:
        item = article("Major exchange hacked, funds stolen")
        first = aggregator.assess([item], "BTC", now=NOW)
        second = aggregator.assess([item], "BTC", now=NOW + timedelta(minutes=1))
        assert abs(second.directional_score) < abs(first.directional_score)

    def test_score_is_bounded(self, aggregator: NewsAggregator) -> None:
        items = [
            article(f"Major exchange hacked, funds stolen in exploit {i}", minutes_ago=i)
            for i in range(30)
        ]
        assessment = aggregator.assess(items, "BTC", now=NOW)
        assert -1.0 <= assessment.directional_score <= 1.0
        assert 0.0 <= assessment.confidence <= 1.0

    def test_low_reputation_sources_excluded(self) -> None:
        aggregator = NewsAggregator(
            config=NewsScoringConfig(min_source_reputation=0.5)
        )
        assessment = aggregator.assess(
            [article("Bitcoin about to surge, insider says", source="telegram")],
            "BTC", now=NOW,
        )
        assert not assessment.has_coverage

    def test_materiality_threshold(self, aggregator: NewsAggregator) -> None:
        big = aggregator.assess(
            [article("BREAKING: major exchange hacked, $500M stolen in confirmed exploit")],
            "BTC", now=NOW,
        )
        small = NewsAggregator().assess(
            [article("Exchange announces minor interface update")], "BTC", now=NOW
        )
        assert big.max_impact > small.max_impact

    def test_assess_many(self, aggregator: NewsAggregator) -> None:
        items = [
            article("Bitcoin ETF approved", assets=("BTC",)),
            article("Ethereum upgrade goes live", assets=("ETH",)),
        ]
        results = aggregator.assess_many(items, ["BTC", "ETH"], now=NOW)
        assert set(results) == {"BTC", "ETH"}
        assert results["BTC"].article_count == 1

    def test_combine_assessments(self, aggregator: NewsAggregator) -> None:
        per_asset = [
            aggregator.assess([article("Bitcoin ETF approved")], "BTC", now=NOW),
            NewsAggregator().assess(
                [article("Ethereum upgrade live", assets=("ETH",))], "ETH", now=NOW
            ),
        ]
        combined = combine_assessments(per_asset, now=NOW)
        assert combined.asset == "MARKET"
        assert combined.article_count == 2

    def test_combine_empty(self) -> None:
        assert combine_assessments([], now=NOW).article_count == 0

    def test_assessment_serialises(self, aggregator: NewsAggregator) -> None:
        assessment = aggregator.assess(
            [article("SEC approves spot Bitcoin ETF")], "BTC", now=NOW
        )
        payload = assessment.to_dict()
        assert payload["asset"] == "BTC"
        assert isinstance(payload["directional_score"], float)


# =========================================================================== #
# Providers
# =========================================================================== #
class TestProviders:
    async def test_null_provider_returns_nothing(self) -> None:
        assert await NullNewsProvider().fetch() == []

    async def test_in_memory_provider_filters(self) -> None:
        provider = InMemoryNewsProvider(
            [
                article("Bitcoin news", assets=("BTC",), minutes_ago=10),
                article("Ethereum news", assets=("ETH",), minutes_ago=5),
            ]
        )
        btc = await provider.fetch(assets=["BTC"])
        assert len(btc) == 1
        assert btc[0].title == "Bitcoin news"

    async def test_in_memory_provider_respects_since(self) -> None:
        provider = InMemoryNewsProvider(
            [article("old", minutes_ago=200), article("new", minutes_ago=1)]
        )
        recent = await provider.fetch(since=NOW - timedelta(minutes=60))
        assert [a.title for a in recent] == ["new"]

    async def test_in_memory_window(self) -> None:
        provider = InMemoryNewsProvider(
            [article("a", minutes_ago=100), article("b", minutes_ago=10)]
        )
        window = provider.window(NOW - timedelta(minutes=60), NOW)
        assert [a.title for a in window] == ["b"]

    async def test_file_provider_reads_json(self, tmp_path) -> None:
        path = tmp_path / "news.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "title": "SEC approves Bitcoin ETF",
                        "source": "reuters",
                        "published_at": "2024-06-12T11:55:00Z",
                        "assets": ["BTC"],
                    }
                ]
            ),
            encoding="utf-8",
        )
        articles = await FileNewsProvider(path).fetch()
        assert len(articles) == 1
        assert articles[0].assets == ("BTC",)

    async def test_file_provider_reads_jsonl(self, tmp_path) -> None:
        path = tmp_path / "news.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "title": f"Story {i}",
                        "source": "coindesk",
                        "published_at": "2024-06-12T11:00:00Z",
                    }
                )
                for i in range(3)
            ),
            encoding="utf-8",
        )
        assert len(await FileNewsProvider(path).fetch()) == 3

    async def test_file_provider_missing_file(self, tmp_path) -> None:
        with pytest.raises(NewsProviderError, match="not found"):
            await FileNewsProvider(tmp_path / "absent.json").fetch()

    async def test_file_provider_malformed(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(NewsProviderError, match="not valid JSON"):
            await FileNewsProvider(path).fetch()

    async def test_file_provider_skips_bad_records(self, tmp_path) -> None:
        path = tmp_path / "mixed.json"
        path.write_text(
            json.dumps(
                [
                    {"title": "good", "source": "reuters",
                     "published_at": "2024-06-12T11:00:00Z"},
                    {"source": "reuters"},  # no title
                ]
            ),
            encoding="utf-8",
        )
        articles = await FileNewsProvider(path).fetch()
        assert len(articles) == 1

    def test_synthetic_news_is_labelled(self) -> None:
        items = synthetic_news("BTC", count=3)
        assert len(items) == 3
        assert all("Synthetic" in (a.description or "") for a in items)


# =========================================================================== #
# HTTP provider against a mock transport
# =========================================================================== #
class TestHttpProvider:
    async def test_parses_a_standard_response(self) -> None:
        import httpx

        from app.news.providers import HttpNewsProvider

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "1",
                            "title": "SEC approves Bitcoin ETF",
                            "description": "Regulator gives approval.",
                            "source": {"name": "Reuters"},
                            "published_at": "2024-06-12T11:55:00Z",
                            "assets": [{"code": "BTC"}],
                            "url": "https://example.invalid/1",
                        }
                    ]
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = HttpNewsProvider("https://api.invalid/news", client=client)
        articles = await provider.fetch(assets=["BTC"])
        assert len(articles) == 1
        assert articles[0].source == "Reuters"
        assert articles[0].assets == ("BTC",)
        await client.aclose()

    async def test_http_error_does_not_leak_body(self) -> None:
        import httpx

        from app.news.providers import HttpNewsProvider

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "invalid api key sk-secret-12345"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = HttpNewsProvider(
            "https://api.invalid/news", api_key="sk-secret-12345", client=client
        )
        with pytest.raises(NewsProviderError) as exc_info:
            await provider.fetch()
        assert "sk-secret" not in str(exc_info.value)
        await client.aclose()

    async def test_malformed_payload_raises(self) -> None:
        import httpx

        from app.news.providers import HttpNewsProvider

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": "shape"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = HttpNewsProvider("https://api.invalid/news", client=client)
        articles = await provider.fetch()
        assert articles == []
        await client.aclose()

    async def test_api_key_is_sent_as_a_header_not_a_query_param(self) -> None:
        """Secrets must never appear in a URL: they end up in proxy and server logs."""
        import httpx

        from app.news.providers import HttpNewsProvider

        captured: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["auth"] = request.headers.get("Authorization", "")
            return httpx.Response(200, json={"results": []})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = HttpNewsProvider(
            "https://api.invalid/news", api_key="sk-secret-12345", client=client
        )
        await provider.fetch()
        assert "sk-secret-12345" not in captured["url"]
        assert captured["auth"] == "Bearer sk-secret-12345"
        await client.aclose()


# =========================================================================== #
# Invariants
# =========================================================================== #
def test_empty_assessment_is_inert() -> None:
    assessment = NewsAssessment.empty("BTC", now=NOW)
    assert not assessment.has_coverage
    assert not assessment.is_material
    assert assessment.directional_score == 0.0
