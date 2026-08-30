"""News classification.

Classifies an article into an event type, a sentiment and an impact level using an explicit,
auditable keyword-and-rule model.

Why rules rather than a language model
--------------------------------------
This is a trading system, and the classifier's output modifies position decisions. A rule-based
classifier is deterministic, testable, reproducible in a backtest, cheap enough to run on every
article, and — most importantly — *inspectable*: when a bot declines a trade because of a
headline, the user can see exactly which phrase caused it.

The architecture keeps an LLM classifier as a drop-in alternative
(:class:`~app.news.providers.NewsProvider` consumers accept any callable with this signature),
but the shipped default is the rule engine, and the platform never lets news alone open a
position regardless of which classifier is in use.

Calibration honesty: the keyword weights below are reasonable priors, not fitted parameters.
They have not been validated against realised price moves, and the confidence scores they
produce should be read as "how clearly does this text match a known pattern", not "how likely
is this to move the market".
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from app.core.clock import utcnow
from app.core.enums import NewsEventType, NewsImpact, NewsSentiment
from app.core.numeric import clamp
from app.news.models import NewsAnalysis, NewsArticle

# --------------------------------------------------------------------------- #
# Event-type patterns
# --------------------------------------------------------------------------- #
#: ``event type -> (keywords, baseline impact)``. Matching is on word boundaries.
EVENT_PATTERNS: dict[NewsEventType, tuple[tuple[str, ...], NewsImpact]] = {
    NewsEventType.HACK: (
        ("hack", "hacked", "exploit", "exploited", "breach", "stolen", "drained",
         "attack", "vulnerability", "compromised"),
        NewsImpact.CRITICAL,
    ),
    NewsEventType.EXCHANGE_INCIDENT: (
        ("halt", "halted", "suspend", "suspended", "outage", "downtime", "insolvency",
         "withdrawal", "withdrawals", "freeze", "frozen", "delisting", "delist"),
        NewsImpact.HIGH,
    ),
    NewsEventType.REGULATION: (
        ("sec", "regulator", "regulatory", "regulation", "ban", "banned", "crackdown",
         "compliance", "license", "licence", "sanction", "sanctions", "cftc", "mica"),
        NewsImpact.HIGH,
    ),
    NewsEventType.ETF: (
        ("etf", "etfs", "spot etf", "exchange-traded fund", "inflow", "inflows",
         "outflow", "outflows"),
        NewsImpact.HIGH,
    ),
    NewsEventType.INTEREST_RATES: (
        ("fed", "fomc", "interest rate", "rate cut", "rate hike", "basis points",
         "central bank", "monetary policy", "hawkish", "dovish"),
        NewsImpact.HIGH,
    ),
    NewsEventType.INFLATION: (
        ("inflation", "cpi", "ppi", "deflation", "consumer price"),
        NewsImpact.MEDIUM,
    ),
    NewsEventType.MACRO: (
        ("gdp", "recession", "unemployment", "jobs report", "payroll", "economy",
         "economic growth", "stimulus"),
        NewsImpact.MEDIUM,
    ),
    NewsEventType.LEGAL: (
        ("lawsuit", "sued", "court", "judge", "settlement", "indicted", "charges",
         "fraud", "trial", "verdict", "subpoena"),
        NewsImpact.HIGH,
    ),
    NewsEventType.TOKEN_UNLOCK: (
        ("unlock", "unlocks", "vesting", "token release", "cliff", "emission"),
        NewsImpact.MEDIUM,
    ),
    NewsEventType.ACQUISITION: (
        ("acquire", "acquires", "acquisition", "merger", "merge", "buyout", "takeover"),
        NewsImpact.MEDIUM,
    ),
    NewsEventType.PARTNERSHIP: (
        ("partnership", "partners with", "collaboration", "integration", "adopts",
         "integrates", "teams up"),
        NewsImpact.LOW,
    ),
    NewsEventType.EARNINGS: (
        ("earnings", "revenue", "quarterly results", "profit", "loss report", "guidance"),
        NewsImpact.MEDIUM,
    ),
    NewsEventType.LEADERSHIP_CHANGE: (
        ("ceo", "cto", "cfo", "resigns", "resignation", "steps down", "appoints",
         "new chief", "departure"),
        NewsImpact.LOW,
    ),
    NewsEventType.GEOPOLITICAL: (
        ("war", "conflict", "invasion", "sanctions", "election", "geopolitical",
         "tensions", "military"),
        NewsImpact.MEDIUM,
    ),
}

# --------------------------------------------------------------------------- #
# Sentiment lexicon
# --------------------------------------------------------------------------- #
POSITIVE_TERMS: dict[str, float] = {
    "surge": 1.0, "surges": 1.0, "soar": 1.0, "soars": 1.0, "rally": 0.9, "rallies": 0.9,
    "gain": 0.6, "gains": 0.6, "rise": 0.5, "rises": 0.5, "jump": 0.8, "jumps": 0.8,
    "approve": 0.9, "approved": 0.9, "approval": 0.9, "adopt": 0.7, "adoption": 0.7,
    "bullish": 1.0, "record high": 1.0, "all-time high": 1.0, "breakthrough": 0.8,
    "partnership": 0.5, "launch": 0.4, "launches": 0.4, "upgrade": 0.6, "upgraded": 0.6,
    "inflow": 0.7, "inflows": 0.7, "accumulate": 0.6, "buy": 0.4, "surges past": 1.0,
    "beat": 0.6, "beats": 0.6, "expand": 0.4, "expansion": 0.4, "green light": 0.9,
    "positive": 0.5, "optimistic": 0.6, "recovery": 0.6, "rebound": 0.7, "wins": 0.6,
    "dovish": 0.7, "rate cut": 0.7, "stimulus": 0.6, "milestone": 0.5,
}

NEGATIVE_TERMS: dict[str, float] = {
    "crash": 1.0, "crashes": 1.0, "plunge": 1.0, "plunges": 1.0, "plummet": 1.0,
    "tumble": 0.8, "tumbles": 0.8, "fall": 0.5, "falls": 0.5, "drop": 0.6, "drops": 0.6,
    "slump": 0.8, "decline": 0.5, "declines": 0.5, "bearish": 1.0, "sell-off": 0.9,
    "selloff": 0.9, "hack": 1.0, "hacked": 1.0, "exploit": 1.0, "stolen": 1.0,
    "ban": 0.9, "banned": 0.9, "crackdown": 0.9, "lawsuit": 0.7, "sued": 0.7,
    "fraud": 1.0, "investigation": 0.6, "probe": 0.6, "halt": 0.7, "halted": 0.7,
    "suspend": 0.7, "suspended": 0.7, "outage": 0.6, "insolvency": 1.0, "bankrupt": 1.0,
    "bankruptcy": 1.0, "liquidation": 0.8, "liquidations": 0.8, "outflow": 0.7,
    "outflows": 0.7, "reject": 0.8, "rejected": 0.8, "rejection": 0.8, "delay": 0.5,
    "delayed": 0.5, "warning": 0.6, "risk": 0.3, "concern": 0.4, "concerns": 0.4,
    "downgrade": 0.6, "downgraded": 0.6, "miss": 0.5, "misses": 0.5, "loss": 0.5,
    "losses": 0.5, "hawkish": 0.6, "rate hike": 0.6, "recession": 0.8, "collapse": 1.0,
    "scam": 1.0, "rug pull": 1.0, "exit scam": 1.0, "unlock": 0.4, "dump": 0.8,
}

#: Terms that flip the polarity of a nearby sentiment word.
NEGATORS: frozenset[str] = frozenset(
    {"not", "no", "never", "denies", "denied", "dismissed", "rejects", "without",
     "fails to", "unlikely", "avoids", "halts plans"}
)

#: Terms that mark a claim as unconfirmed, reducing confidence.
HEDGES: frozenset[str] = frozenset(
    {"may", "might", "could", "reportedly", "rumor", "rumour", "rumored", "alleged",
     "allegedly", "speculation", "unconfirmed", "sources say", "expected to",
     "plans to", "considering", "proposal", "proposed"}
)

#: Terms indicating the event is happening now rather than being discussed.
URGENCY_TERMS: frozenset[str] = frozenset(
    {"breaking", "just in", "urgent", "immediately", "now", "confirmed", "official",
     "announced", "live"}
)

#: Source reputation priors, in [0, 1]. Unknown sources get NEUTRAL_REPUTATION.
SOURCE_REPUTATION: dict[str, float] = {
    "reuters": 0.95, "bloomberg": 0.95, "associated press": 0.95, "ap": 0.95,
    "financial times": 0.9, "wall street journal": 0.9, "cnbc": 0.8,
    "coindesk": 0.8, "the block": 0.8, "cointelegraph": 0.65, "decrypt": 0.7,
    "bitcoin magazine": 0.6, "cryptoslate": 0.55, "beincrypto": 0.5,
    "twitter": 0.25, "x": 0.25, "reddit": 0.2, "telegram": 0.2, "unknown": 0.4,
}
NEUTRAL_REPUTATION = 0.45


@dataclass(slots=True)
class ClassifierConfig:
    """Thresholds for the rule-based classifier."""

    #: Sentiment magnitude below this is reported as NEUTRAL.
    neutral_band: float = 0.15
    #: Magnitude above this maps to the VERY_ strong sentiment levels.
    strong_threshold: float = 0.6
    #: Confidence multiplier applied when hedging language is present.
    hedge_penalty: float = 0.5
    #: Weight given to source reputation in the final confidence.
    reputation_weight: float = 0.5
    #: Window, in words, over which a negator flips a sentiment term.
    negation_window: int = 3


class NewsClassifier:
    """Rule-based news classifier.

    Deterministic: the same article always produces the same analysis, which is what makes
    news-aware backtests reproducible.
    """

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self.config = config or ClassifierConfig()

    def classify(
        self, article: NewsArticle, asset: str, *, novelty: float = 1.0
    ) -> NewsAnalysis:
        """Classify ``article`` with respect to ``asset``."""
        text = article.text.lower()
        words = re.findall(r"[a-z0-9'-]+", text)

        event_type, event_impact, event_matches = self._classify_event(text)
        raw_sentiment, sentiment_hits = self._score_sentiment(text, words)
        hedged = self._contains_any(text, HEDGES)
        urgency = self._urgency(text)
        reputation = source_reputation(article.source)

        sentiment = self._bucket_sentiment(raw_sentiment)
        impact = self._adjust_impact(event_impact, abs(raw_sentiment), reputation)
        confidence = self._confidence(
            event_matches=event_matches,
            sentiment_hits=sentiment_hits,
            hedged=hedged,
            reputation=reputation,
        )

        rationale_parts = [f"event={event_type.value}"]
        if event_matches:
            rationale_parts.append(f"matched={','.join(sorted(event_matches)[:4])}")
        rationale_parts.append(f"sentiment={raw_sentiment:+.2f}")
        rationale_parts.append(f"source_reputation={reputation:.2f}")
        if hedged:
            rationale_parts.append("hedged language reduces confidence")

        return NewsAnalysis(
            asset=asset,
            event_type=event_type,
            sentiment=sentiment,
            impact=impact,
            confidence=confidence,
            novelty=clamp(novelty, 0.0, 1.0),
            urgency=urgency,
            rationale="; ".join(rationale_parts),
            analyzed_at=utcnow(),
            article_hash=article.content_hash,
        )

    # ------------------------------------------------------------------ #
    # Components
    # ------------------------------------------------------------------ #
    def _classify_event(
        self, text: str
    ) -> tuple[NewsEventType, NewsImpact, set[str]]:
        """Pick the event type with the most keyword hits, breaking ties on impact."""
        best_type = NewsEventType.OTHER
        best_impact = NewsImpact.LOW
        best_matches: set[str] = set()
        best_score = 0

        for event_type, (keywords, impact) in EVENT_PATTERNS.items():
            matches = {kw for kw in keywords if _contains_phrase(text, kw)}
            if not matches:
                continue
            score = len(matches)
            if score > best_score or (
                score == best_score and impact.weight > best_impact.weight
            ):
                best_type, best_impact, best_matches, best_score = (
                    event_type, impact, matches, score,
                )
        if best_score == 0:
            return NewsEventType.OTHER, NewsImpact.LOW, set()
        return best_type, best_impact, best_matches

    def _score_sentiment(
        self, text: str, words: list[str]
    ) -> tuple[float, int]:
        """Sum lexicon weights, applying negation. Returns ``(score, hit_count)``."""
        total = 0.0
        hits = 0

        for phrase, weight in POSITIVE_TERMS.items():
            occurrences = _count_phrase(text, phrase)
            if occurrences:
                hits += occurrences
                sign = -1.0 if self._is_negated(words, phrase) else 1.0
                total += sign * weight * min(occurrences, 2)
        for phrase, weight in NEGATIVE_TERMS.items():
            occurrences = _count_phrase(text, phrase)
            if occurrences:
                hits += occurrences
                sign = 1.0 if self._is_negated(words, phrase) else -1.0
                total += sign * weight * min(occurrences, 2)

        if hits == 0:
            return 0.0, 0
        # Normalise by hit count so a long article is not automatically more extreme.
        return clamp(total / max(hits, 1) / 1.0, -1.0, 1.0), hits

    def _is_negated(self, words: list[str], phrase: str) -> bool:
        """True when a negator appears shortly before the phrase."""
        head = phrase.split()[0]
        window = self.config.negation_window
        for index, word in enumerate(words):
            if word == head:
                preceding = words[max(0, index - window) : index]
                if any(term in NEGATORS for term in preceding):
                    return True
        return False

    def _bucket_sentiment(self, score: float) -> NewsSentiment:
        magnitude = abs(score)
        if magnitude < self.config.neutral_band:
            return NewsSentiment.NEUTRAL
        if score > 0:
            return (
                NewsSentiment.VERY_POSITIVE
                if magnitude >= self.config.strong_threshold
                else NewsSentiment.POSITIVE
            )
        return (
            NewsSentiment.VERY_NEGATIVE
            if magnitude >= self.config.strong_threshold
            else NewsSentiment.NEGATIVE
        )

    @staticmethod
    def _adjust_impact(
        base: NewsImpact, sentiment_magnitude: float, reputation: float
    ) -> NewsImpact:
        """Downgrade impact for weak sentiment or poor sources."""
        weight = base.weight
        if sentiment_magnitude < 0.2:
            weight *= 0.6
        if reputation < 0.4:
            weight *= 0.7
        for level in (
            NewsImpact.CRITICAL, NewsImpact.HIGH, NewsImpact.MEDIUM, NewsImpact.LOW
        ):
            if weight >= level.weight - 1e-9:
                return level
        return NewsImpact.NONE

    def _confidence(
        self,
        *,
        event_matches: set[str],
        sentiment_hits: int,
        hedged: bool,
        reputation: float,
    ) -> float:
        """How clearly the text matches a known pattern. Not a probability of a price move."""
        evidence = clamp(len(event_matches) / 3.0, 0.0, 1.0)
        lexical = clamp(sentiment_hits / 4.0, 0.0, 1.0)
        base = 0.5 * evidence + 0.5 * lexical
        blended = (
            (1.0 - self.config.reputation_weight) * base
            + self.config.reputation_weight * base * reputation
        )
        if hedged:
            blended *= self.config.hedge_penalty
        return clamp(blended, 0.0, 1.0)

    @staticmethod
    def _contains_any(text: str, terms: Iterable[str]) -> bool:
        return any(_contains_phrase(text, term) for term in terms)

    @staticmethod
    def _urgency(text: str) -> float:
        hits = sum(1 for term in URGENCY_TERMS if _contains_phrase(text, term))
        return clamp(hits / 2.0, 0.0, 1.0)


def source_reputation(source: str) -> float:
    """Reputation prior for a source name, matched case-insensitively."""
    normalized = source.strip().lower()
    if normalized in SOURCE_REPUTATION:
        return SOURCE_REPUTATION[normalized]
    for known, score in SOURCE_REPUTATION.items():
        if known in normalized:
            return score
    return NEUTRAL_REPUTATION


def _contains_phrase(text: str, phrase: str) -> bool:
    """Word-boundary containment, so "ban" does not match "urban"."""
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _count_phrase(text: str, phrase: str) -> int:
    return len(re.findall(rf"\b{re.escape(phrase)}\b", text))
