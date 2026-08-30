"""News domain types."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.core.clock import ensure_utc, utcnow
from app.core.enums import NewsEventType, NewsImpact, NewsSentiment


def content_hash(title: str, description: str | None = None) -> str:
    """Stable hash of an article's *normalised* text.

    Normalisation strips punctuation, casing and whitespace so that the same story republished
    with a slightly different headline still collides. Syndicated wire copy is the dominant
    source of duplicate "signals" in crypto news feeds, and counting one story five times is
    how a news-aware system convinces itself a move is well supported.
    """
    text = f"{title} {description or ''}".lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class NewsArticle:
    """A single news item as received from a provider."""

    title: str
    source: str
    published_at: datetime
    description: str | None = None
    url: str | None = None
    assets: tuple[str, ...] = ()
    external_id: str | None = None
    raw_metadata: dict[str, object] = field(default_factory=dict)
    ingested_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("News article must have a title")
        object.__setattr__(
            self, "published_at", ensure_utc(self.published_at, field="published_at")
        )
        object.__setattr__(
            self, "ingested_at", ensure_utc(self.ingested_at, field="ingested_at")
        )
        object.__setattr__(self, "assets", tuple(a.upper() for a in self.assets))

    @property
    def content_hash(self) -> str:
        return content_hash(self.title, self.description)

    @property
    def text(self) -> str:
        return f"{self.title}. {self.description or ''}".strip()

    def age(self, now: datetime | None = None) -> timedelta:
        return (now or utcnow()) - self.published_at

    def is_stale(self, max_age: timedelta, *, now: datetime | None = None) -> bool:
        return self.age(now) > max_age

    def mentions(self, asset: str) -> bool:
        return asset.upper() in self.assets


@dataclass(frozen=True, slots=True)
class NewsAnalysis:
    """Classification of one article with respect to one asset."""

    asset: str
    event_type: NewsEventType
    sentiment: NewsSentiment
    impact: NewsImpact
    confidence: float
    novelty: float = 1.0
    urgency: float = 0.0
    rationale: str = ""
    analyzed_at: datetime = field(default_factory=utcnow)
    article_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("confidence", "novelty", "urgency"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        object.__setattr__(self, "asset", self.asset.upper())

    @property
    def score(self) -> float:
        """Directional score in ``[-1, 1]``.

        Sentiment supplies the direction; impact, confidence and novelty scale the magnitude.
        A confident read of a low-impact story still contributes almost nothing, which is the
        intended behaviour — most news is noise.
        """
        return (
            self.sentiment.score
            * self.impact.weight
            * self.confidence
            * self.novelty
        )

    @property
    def is_material(self) -> bool:
        return self.impact.weight >= 0.5 and self.confidence >= 0.5
