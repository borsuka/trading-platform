"""News intelligence endpoints.

Read-only. There is no endpoint that turns a news assessment into a trade, because news never
originates a signal in this platform — it modifies one that already exists.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from app.api.dependencies import CurrentUser, SettingsDep
from app.api.schemas import NewsAssessmentResponse
from app.core.logging import get_logger
from app.news.providers import NullNewsProvider, build_news_provider
from app.news.scoring import NewsAggregator

logger = get_logger(__name__)
router = APIRouter(prefix="/news", tags=["news"])

_aggregator = NewsAggregator()


@router.get("/status", response_model=dict)
async def news_status(settings: SettingsDep) -> dict[str, Any]:
    """Whether news is configured, and what it does when it is."""
    provider = build_news_provider()
    configured = not isinstance(provider, NullNewsProvider)
    return {
        "enabled": settings.news_enabled,
        "provider": provider.name,
        "configured": configured,
        "message": (
            "News is active and modifies signal confidence."
            if configured
            else "No news provider is configured. Strategies run on price and volume alone, "
            "which is a fully supported configuration."
        ),
        "policy": (
            "News can reduce confidence in a signal or veto it outright. It can never create "
            "a signal on its own: a headline without a technical setup produces no trade."
        ),
    }


@router.get("/assessment/{asset}", response_model=NewsAssessmentResponse)
async def assessment(
    asset: str,
    user: CurrentUser,
    limit: int = Query(default=50, ge=1, le=200),
) -> NewsAssessmentResponse:
    """Current news assessment for one asset."""
    provider = build_news_provider()
    articles = await provider.fetch(assets=[asset.upper()], limit=limit)
    result = _aggregator.assess(articles, asset.upper())
    await provider.close()
    return NewsAssessmentResponse.model_validate(result.to_dict())


@router.get("/articles", response_model=list[dict])
async def articles(
    user: CurrentUser,
    assets: str | None = Query(default=None, description="Comma-separated asset codes"),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict[str, Any]]:
    """Recent articles with their classifications.

    Deduplicated: syndicated copies of one story appear once, with the count of duplicates
    removed, so the feed is not five outlets repeating the same headline.
    """
    provider = build_news_provider()
    asset_list = (
        [a.strip().upper() for a in assets.split(",") if a.strip()] if assets else None
    )
    raw = await provider.fetch(assets=asset_list, limit=limit)
    await provider.close()
    if not raw:
        return []

    unique, removed = _aggregator.deduplicate(raw)
    primary = (asset_list or ["BTC"])[0]
    results: list[dict[str, Any]] = []
    for article in sorted(unique, key=lambda a: a.published_at, reverse=True):
        analysis = _aggregator.classifier.classify(article, primary)
        results.append(
            {
                "title": article.title,
                "description": article.description,
                "source": article.source,
                "url": article.url,
                "published_at": article.published_at.isoformat(),
                "assets": list(article.assets),
                "event_type": analysis.event_type.value,
                "sentiment": analysis.sentiment.value,
                "impact": analysis.impact.value,
                "confidence": round(analysis.confidence, 3),
                "score": round(analysis.score, 4),
                "rationale": analysis.rationale,
            }
        )
    if results:
        results[0]["duplicates_removed_in_batch"] = removed
    return results
