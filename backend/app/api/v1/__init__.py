"""API v1 routers."""

from fastapi import APIRouter

from app.api.v1 import (
    auth,
    backtests,
    bots,
    exchange_accounts,
    licenses,
    news,
    risk,
    strategies,
    system,
    trading,
)

# `system` is mounted at the application root rather than under the API prefix, so that
# /health and /ready keep their conventional paths for orchestrators.
system_router = system.router

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(strategies.router)
api_router.include_router(bots.router)
api_router.include_router(backtests.router)
api_router.include_router(trading.router)
api_router.include_router(risk.router)
api_router.include_router(exchange_accounts.router)
api_router.include_router(licenses.router)
api_router.include_router(news.router)

__all__ = ["api_router", "system_router"]
