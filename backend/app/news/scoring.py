"""News aggregation, deduplication and scoring.

Turns a stream of articles into one :class:`NewsAssessment` per asset — the object the signal
engine consumes.

Three properties matter more than the scoring maths:

**Deduplication.** Crypto news is heavily syndicated: one wire story appears across a dozen
outlets within minutes. Counting each copy as independent evidence manufactures false
consensus, which is the single most dangerous failure mode of a news-aware trading system.
Duplicates are detected by normalised content hash and by title similarity, and only the
highest-reputation copy survives.

**Decay.** A three-hour-old headline has already been priced. Scores decay exponentially with
age, so stale news cannot keep influencing decisions.

**Novelty.** A story repeating what was already reported yesterday carries less information
than a genuinely new one. Repeats are down-weighted rather than dropped, since escalating
coverage of the same event is itself informative.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.core.clock import utcnow
from app.core.enums import NewsEventType, NewsImpact, NewsSentiment
from app.core.logging import get_logger
from app.core.numeric import clamp, safe_divide
from app.news.classification import NewsClassifier, source_reputation
from app.news.models import NewsAnalysis, NewsArticle

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ScoredArticle:
    """An article plus its analysis and time-decayed weight."""

    article: NewsArticle
    analysis: NewsAnalysis
    weight: float

    @property
    def weighted_score(self) -> float:
        return self.analysis.score * self.weight


@dataclass(frozen=True, slots=True)
class NewsAssessment:
    """Aggregated news view for one asset at one instant.

    This is the *only* news object the trading pipeline sees.
    """

    asset: str
    timestamp: datetime
    directional_score: float = 0.0
    confidence: float = 0.0
    max_impact: float = 0.0
    article_count: int = 0
    duplicates_removed: int = 0
    dominant_event: NewsEventType = NewsEventType.OTHER
    dominant_sentiment: NewsSentiment = NewsSentiment.NEUTRAL
    urgency: float = 0.0
    scored: tuple[ScoredArticle, ...] = ()
    headline_summary: str = ""

    @property
    def has_coverage(self) -> bool:
        return self.article_count > 0

    @property
    def is_material(self) -> bool:
        """True when the news is strong enough to justify modifying a decision."""
        return self.max_impact >= 0.5 and self.confidence >= 0.4

    def to_dict(self) -> dict[str, object]:
        return {
            "asset": self.asset,
            "timestamp": self.timestamp.isoformat(),
            "directional_score": round(self.directional_score, 4),
            "confidence": round(self.confidence, 4),
            "max_impact": round(self.max_impact, 4),
            "article_count": self.article_count,
            "duplicates_removed": self.duplicates_removed,
            "dominant_event": self.dominant_event.value,
            "dominant_sentiment": self.dominant_sentiment.value,
            "urgency": round(self.urgency, 4),
            "headline_summary": self.headline_summary,
        }

    @classmethod
    def empty(cls, asset: str, *, now: datetime | None = None) -> NewsAssessment:
        return cls(asset=asset.upper(), timestamp=now or utcnow())


@dataclass(slots=True)
class NewsScoringConfig:
    """Aggregation parameters."""

    #: Half-life of a news item's influence.
    half_life_minutes: float = 60.0
    #: Articles older than this are ignored entirely.
    max_age_minutes: float = 180.0
    #: Title similarity above which two articles are considered the same story.
    similarity_threshold: float = 0.80
    #: Novelty multiplier applied to a repeat of an already-seen story.
    repeat_novelty: float = 0.35
    #: How long a story is remembered for novelty purposes.
    novelty_window_hours: float = 24.0
    #: Ignore sources below this reputation entirely.
    min_source_reputation: float = 0.2
    #: Cap on how many articles contribute to one assessment.
    max_articles: int = 50


class NewsAggregator:
    """Deduplicates, classifies, decays and aggregates news per asset."""

    def __init__(
        self,
        *,
        classifier: NewsClassifier | None = None,
        config: NewsScoringConfig | None = None,
    ) -> None:
        self.classifier = classifier or NewsClassifier()
        self.config = config or NewsScoringConfig()
        #: content hash -> first-seen timestamp, for novelty scoring.
        self._seen: dict[str, datetime] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def assess(
        self,
        articles: Iterable[NewsArticle],
        asset: str,
        *,
        now: datetime | None = None,
    ) -> NewsAssessment:
        """Build the assessment for one asset from a batch of articles."""
        moment = now or utcnow()
        asset = asset.upper()

        relevant = [
            article
            for article in articles
            if self._is_relevant(article, asset, moment)
        ]
        if not relevant:
            return NewsAssessment.empty(asset, now=moment)

        unique, duplicates_removed = self.deduplicate(relevant)
        unique = sorted(unique, key=lambda a: a.published_at, reverse=True)
        unique = unique[: self.config.max_articles]

        scored: list[ScoredArticle] = []
        for article in unique:
            novelty = self._novelty(article, moment)
            analysis = self.classifier.classify(article, asset, novelty=novelty)
            weight = self._decay_weight(article, moment)
            scored.append(ScoredArticle(article=article, analysis=analysis, weight=weight))
            self._remember(article, moment)

        return self._aggregate(asset, scored, moment, duplicates_removed)

    def assess_many(
        self,
        articles: Sequence[NewsArticle],
        assets: Iterable[str],
        *,
        now: datetime | None = None,
    ) -> dict[str, NewsAssessment]:
        """Assess several assets from one article batch."""
        return {
            asset.upper(): self.assess(articles, asset, now=now) for asset in assets
        }

    # ------------------------------------------------------------------ #
    # Filtering and deduplication
    # ------------------------------------------------------------------ #
    def _is_relevant(
        self, article: NewsArticle, asset: str, now: datetime
    ) -> bool:
        if article.assets and not article.mentions(asset):
            return False
        if not article.assets and asset.lower() not in article.text.lower():
            return False
        if article.is_stale(timedelta(minutes=self.config.max_age_minutes), now=now):
            return False
        if article.published_at > now + timedelta(minutes=5):
            # A timestamp in the future means a broken feed, not a scoop.
            logger.warning(
                "news.future_timestamp",
                source=article.source,
                published_at=article.published_at.isoformat(),
            )
            return False
        return source_reputation(article.source) >= self.config.min_source_reputation

    def deduplicate(
        self, articles: Sequence[NewsArticle]
    ) -> tuple[list[NewsArticle], int]:
        """Collapse syndicated copies of the same story.

        Exact matches collapse on normalised content hash; near-matches on title token
        overlap. The surviving copy is the one from the highest-reputation source, since that
        is the version whose claims are most likely to hold up.
        """
        by_hash: dict[str, NewsArticle] = {}
        removed = 0

        for article in articles:
            key = article.content_hash
            existing = by_hash.get(key)
            if existing is None:
                by_hash[key] = article
                continue
            removed += 1
            if source_reputation(article.source) > source_reputation(existing.source):
                by_hash[key] = article

        candidates = list(by_hash.values())
        survivors: list[NewsArticle] = []
        for article in candidates:
            match_index = next(
                (
                    index
                    for index, kept in enumerate(survivors)
                    if title_similarity(article.title, kept.title)
                    >= self.config.similarity_threshold
                ),
                None,
            )
            if match_index is None:
                survivors.append(article)
                continue
            removed += 1
            if source_reputation(article.source) > source_reputation(
                survivors[match_index].source
            ):
                survivors[match_index] = article
        return survivors, removed

    # ------------------------------------------------------------------ #
    # Weighting
    # ------------------------------------------------------------------ #
    def _decay_weight(self, article: NewsArticle, now: datetime) -> float:
        """Exponential decay by age. A story loses half its weight per half-life."""
        age_minutes = max(0.0, article.age(now).total_seconds() / 60.0)
        if self.config.half_life_minutes <= 0:
            return 1.0
        return float(0.5 ** (age_minutes / self.config.half_life_minutes))

    def _novelty(self, article: NewsArticle, now: datetime) -> float:
        first_seen = self._seen.get(article.content_hash)
        if first_seen is None:
            return 1.0
        if now - first_seen > timedelta(hours=self.config.novelty_window_hours):
            return 1.0
        return self.config.repeat_novelty

    def _remember(self, article: NewsArticle, now: datetime) -> None:
        self._seen.setdefault(article.content_hash, now)
        if len(self._seen) > 10_000:
            cutoff = now - timedelta(hours=self.config.novelty_window_hours)
            self._seen = {k: v for k, v in self._seen.items() if v >= cutoff}

    # ------------------------------------------------------------------ #
    # Aggregation
    # ------------------------------------------------------------------ #
    def _aggregate(
        self,
        asset: str,
        scored: list[ScoredArticle],
        now: datetime,
        duplicates_removed: int,
    ) -> NewsAssessment:
        if not scored:
            return NewsAssessment.empty(asset, now=now)

        total_weight = sum(item.weight for item in scored)

        # Two separate roles for the decay weights, and conflating them is a real trap:
        #
        #  * as *relative* weights inside the mean they decide which articles dominate;
        #  * they must ALSO attenuate the result, or a batch containing only stale news
        #    scores exactly as strongly as fresh news - the weights cancel in a ratio.
        #
        # Freshness is taken from the newest article, because one current story is enough to
        # make the picture current.
        freshness = max(item.weight for item in scored)

        weighted_mean = safe_divide(
            sum(item.weighted_score for item in scored), total_weight
        )
        directional = clamp(weighted_mean * freshness, -1.0, 1.0)
        confidence = clamp(
            safe_divide(
                sum(item.analysis.confidence * item.weight for item in scored),
                total_weight,
            )
            * freshness,
            0.0,
            1.0,
        )
        max_impact = max(item.analysis.impact.weight * item.weight for item in scored)
        urgency = max(item.analysis.urgency for item in scored)

        # Dominant event: the one carrying the most weighted impact.
        impact_by_event: dict[NewsEventType, float] = {}
        for item in scored:
            impact_by_event[item.analysis.event_type] = impact_by_event.get(
                item.analysis.event_type, 0.0
            ) + item.analysis.impact.weight * item.weight
        dominant_event = max(impact_by_event, key=lambda k: impact_by_event[k])

        top = max(scored, key=lambda item: abs(item.weighted_score))
        summary = top.article.title[:200]

        assessment = NewsAssessment(
            asset=asset,
            timestamp=now,
            directional_score=directional,
            confidence=confidence,
            max_impact=clamp(max_impact, 0.0, 1.0),
            article_count=len(scored),
            duplicates_removed=duplicates_removed,
            dominant_event=dominant_event,
            dominant_sentiment=_bucket(directional),
            urgency=urgency,
            scored=tuple(scored),
            headline_summary=summary,
        )
        logger.debug(
            "news.assessed",
            asset=asset,
            score=round(directional, 3),
            confidence=round(confidence, 3),
            articles=len(scored),
            duplicates_removed=duplicates_removed,
        )
        return assessment


def _bucket(score: float) -> NewsSentiment:
    if score >= 0.6:
        return NewsSentiment.VERY_POSITIVE
    if score >= 0.15:
        return NewsSentiment.POSITIVE
    if score <= -0.6:
        return NewsSentiment.VERY_NEGATIVE
    if score <= -0.15:
        return NewsSentiment.NEGATIVE
    return NewsSentiment.NEUTRAL


def title_similarity(left: str, right: str) -> float:
    """Jaccard similarity over lowercase word sets, ignoring stop words.

    Chosen over edit distance because syndicated copy typically reorders and trims words
    rather than altering characters.
    """
    stop = {
        "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
        "as", "at", "by", "with", "from", "after", "amid", "over", "its", "it",
    }
    left_tokens = {w for w in _tokens(left) if w not in stop}
    right_tokens = {w for w in _tokens(right) if w not in stop}
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return intersection / union


def _tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


def combine_assessments(
    assessments: Sequence[NewsAssessment], *, now: datetime | None = None
) -> NewsAssessment:
    """Merge per-asset assessments into a market-wide view.

    Used for macro events (rate decisions, regulation) that affect everything at once.
    """
    if not assessments:
        return NewsAssessment.empty("MARKET", now=now)
    total_articles = sum(a.article_count for a in assessments)
    if total_articles == 0:
        return NewsAssessment.empty("MARKET", now=now)
    weights = [max(a.article_count, 1) * max(a.confidence, 0.01) for a in assessments]
    total = sum(weights)
    score = sum(a.directional_score * w for a, w in zip(assessments, weights, strict=True))
    confidence = sum(a.confidence * w for a, w in zip(assessments, weights, strict=True))
    return NewsAssessment(
        asset="MARKET",
        timestamp=now or utcnow(),
        directional_score=clamp(safe_divide(score, total), -1.0, 1.0),
        confidence=clamp(safe_divide(confidence, total), 0.0, 1.0),
        max_impact=max(a.max_impact for a in assessments),
        article_count=total_articles,
        duplicates_removed=sum(a.duplicates_removed for a in assessments),
        dominant_event=max(
            assessments, key=lambda a: a.max_impact
        ).dominant_event,
        urgency=max(a.urgency for a in assessments),
    )


def impact_from_weight(weight: float) -> NewsImpact:
    """Map a numeric weight back to the nearest impact level."""
    for level in (NewsImpact.CRITICAL, NewsImpact.HIGH, NewsImpact.MEDIUM, NewsImpact.LOW):
        if weight >= level.weight - 1e-9:
            return level
    return NewsImpact.NONE


def half_life_decay(age_minutes: float, half_life_minutes: float) -> float:
    """Standalone decay helper, exposed for reporting and tests."""
    if half_life_minutes <= 0:
        return 1.0
    return float(math.pow(0.5, age_minutes / half_life_minutes))


def analysis_summary(analyses: Sequence[NewsAnalysis]) -> str:
    """One-line description of a set of analyses, for logs and the UI."""
    if not analyses:
        return "no news"
    material = [a for a in analyses if a.is_material]
    return (
        f"{len(analyses)} article(s), {len(material)} material; "
        f"top event {max(analyses, key=lambda a: a.impact.weight).event_type.value}"
    )
