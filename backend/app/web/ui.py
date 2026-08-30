"""Serve the statically exported dashboard from the API process.

The desktop application is a single process on a single port: the API and the user interface
share an origin. That removes CORS from the picture entirely and means there is no second
server to start, fail, or leave running.

The exported site is a client-side application, so a request for ``/positions`` must be
answered with ``positions.html`` and the router takes it from there. What this must *not* do is
answer unknown ``/api/...`` paths with HTML - a JSON client receiving a login page instead of a
401 is a genuinely confusing failure, so those keep returning a real 404.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from starlette.responses import JSONResponse, Response
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.logging import get_logger

logger = get_logger(__name__)

# Prefixes that belong to the API and must never be answered with the dashboard shell.
API_PREFIXES = ("/api", "/health", "/live", "/ready", "/metrics", "/info", "/docs", "/redoc")


def find_ui_directory() -> Path | None:
    """Locate the exported dashboard, or ``None`` if it has not been built.

    ``UI_DIR`` wins so a packaged build can put the files anywhere; otherwise this looks where
    ``next build`` leaves them in a source checkout.
    """
    override = os.environ.get("UI_DIR")
    if override:
        candidate = Path(override).expanduser().resolve()
        return candidate if (candidate / "index.html").is_file() else None

    here = Path(__file__).resolve()
    for base in (here.parents[3], here.parents[2]):  # repo root, then backend/
        candidate = base / "frontend" / "out"
        if (candidate / "index.html").is_file():
            return candidate
    return None


class ExportedSiteFiles(StaticFiles):
    """Static files with the lookup rules a Next.js export needs."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        if response.status_code != 404:
            return response

        # `next build` writes /positions as positions.html, and nested routes such as
        # /bots/view as bots/view.html.
        if not path.endswith(".html"):
            html = await super().get_response(f"{path}.html", scope)
            if html.status_code != 404:
                return html
            index = await super().get_response(f"{path}/index.html", scope)
            if index.status_code != 404:
                return index

        return await super().get_response("404.html", scope)


def mount_ui(app: FastAPI) -> bool:
    """Attach the dashboard at the application root.

    Returns whether it was mounted. Must be called *after* every API router is registered:
    routes are matched in registration order, so anything added later would be shadowed by
    this catch-all.
    """
    directory = find_ui_directory()
    if directory is None:
        logger.info(
            "ui.not_bundled",
            message=(
                "No exported dashboard found; the API runs without one. Build it with "
                "`npm run build` in frontend/ using NEXT_OUTPUT=export."
            ),
        )
        return False

    site = ExportedSiteFiles(directory=directory, html=True)
    # Mounted rather than routed through a handler so that content types, conditional GETs and
    # range requests are handled by Starlette rather than reimplemented here.
    app.mount("/", _guarded(site), name="dashboard")
    logger.info("ui.mounted", directory=str(directory))
    return True


def _guarded(site: ExportedSiteFiles) -> ASGIApp:
    """Wrap the static site so API paths fall through as 404 JSON rather than HTML."""

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        if any(path == p or path.startswith(f"{p}/") for p in API_PREFIXES):
            response = JSONResponse(
                status_code=404,
                content={
                    "error": "not_found",
                    "message": f"No such endpoint: {path}",
                    "context": {},
                },
            )
            await response(scope, receive, send)
            return
        await site(scope, receive, send)

    return app
