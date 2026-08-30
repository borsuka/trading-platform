"""News intelligence: providers, classification, deduplication and scoring."""

from app.news.assets import ASSET_ALIASES, detect_assets, mentions_asset
from app.news.classification import ClassifierConfig, NewsClassifier, source_reputation
from app.news.models import NewsAnalysis, NewsArticle, content_hash
from app.news.providers import (
    FileNewsProvider,
    HttpNewsProvider,
    InMemoryNewsProvider,
    NewsProvider,
    NullNewsProvider,
    build_news_provider,
    synthetic_news,
)
from app.news.rss import DEFAULT_FEEDS, Feed, RssNewsProvider, parse_feed
from app.news.scoring import (
    NewsAggregator,
    NewsAssessment,
    NewsScoringConfig,
    ScoredArticle,
    combine_assessments,
    title_similarity,
)

__all__ = [
    "ASSET_ALIASES",
    "DEFAULT_FEEDS",
    "ClassifierConfig",
    "Feed",
    "FileNewsProvider",
    "HttpNewsProvider",
    "InMemoryNewsProvider",
    "NewsAggregator",
    "NewsAnalysis",
    "NewsArticle",
    "NewsAssessment",
    "NewsClassifier",
    "NewsProvider",
    "NewsScoringConfig",
    "NullNewsProvider",
    "RssNewsProvider",
    "ScoredArticle",
    "build_news_provider",
    "combine_assessments",
    "content_hash",
    "detect_assets",
    "mentions_asset",
    "parse_feed",
    "source_reputation",
    "synthetic_news",
    "title_similarity",
]
