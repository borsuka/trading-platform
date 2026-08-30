"""Every read-only endpoint must answer without a server error.

This exists because of a real defect: ``/trades/summary`` built its query with ``func.case``
instead of ``case``, so the endpoint raised a 500 on every call. It shipped because no test
ever invoked it, and a ``# type: ignore`` on the line hid it from mypy as well.

The check is deliberately shallow - it asserts only that the handler does not blow up. The
value is in the breadth: it sweeps the whole GET surface from the OpenAPI schema, so a new
endpoint is covered the moment it is registered, without anyone remembering to add a test.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import AppEnv, Settings, TradingMode, get_settings
from app.main import create_app

EMAIL = "surface@example.com"
PASSWORD = "a-sufficiently-long-passphrase"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        app_env=AppEnv.TEST,
        trading_mode=TradingMode.PAPER,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'surface.db'}",
        log_json=True,
        log_level="ERROR",
    )


@pytest.fixture
def client(settings: Settings, monkeypatch) -> Iterator[TestClient]:
    monkeypatch.setattr("app.config.settings.get_settings", lambda: settings)
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_URL", settings.database_url)
    monkeypatch.setenv("APP_ENV", "test")

    import app.database.session as session_module

    session_module._engine = None
    session_module._sessionmaker = None

    with TestClient(create_app(settings)) as test_client:
        yield test_client

    session_module._engine = None
    session_module._sessionmaker = None
    get_settings.cache_clear()


@pytest.fixture
def auth(client: TestClient) -> dict[str, str]:
    client.post(
        "/api/v1/auth/register", json={"email": EMAIL, "password": PASSWORD}
    ).raise_for_status()
    token = client.post(
        "/api/v1/auth/login", json={"email": EMAIL, "password": PASSWORD}
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def parameterless_get_paths(client: TestClient) -> list[str]:
    """Every registered GET path that needs no path parameter to call.

    Read from the OpenAPI schema rather than ``app.routes``: routers are nested rather than
    flattened, so walking ``app.routes`` finds only the handful of top-level doc routes and
    the sweep silently covers nothing.
    """
    schema = client.app.openapi()
    return sorted(
        path
        for path, operations in schema["paths"].items()
        if "get" in operations and "{" not in path  # needs an id we do not have
    )


#: Stand-in values for path parameters. The point is not that these resolve to real
#: records - most will not - but that the handler answers 404 or 422 rather than blowing up.
PARAMETER_VALUES = {
    "asset": "BTC",
    "symbol": "BTCUSDT",
    "token": "not-a-real-token",
}
MISSING_ID = "00000000-0000-4000-8000-000000000000"


def parameterised_get_paths(client: TestClient) -> list[str]:
    """Every GET path that takes parameters, with plausible values substituted in."""
    import re

    schema = client.app.openapi()
    filled = []
    for path, operations in schema["paths"].items():
        if "get" not in operations or "{" not in path:
            continue
        filled.append(
            re.sub(
                r"\{([^}]+)\}",
                lambda m: PARAMETER_VALUES.get(m.group(1), MISSING_ID),
                path,
            )
        )
    return sorted(set(filled))


class TestReadSurface:
    def test_the_sweep_is_not_empty(self, client: TestClient) -> None:
        """A refactor that stops discovering routes must fail loudly, not pass vacuously."""
        assert len(parameterless_get_paths(client)) >= 10

    def test_no_endpoint_returns_a_server_error(
        self, client: TestClient, auth: dict[str, str]
    ) -> None:
        failures = []
        for path in parameterless_get_paths(client):
            response = client.get(path, headers=auth)
            if response.status_code >= 500:
                failures.append(f"{path} -> {response.status_code}: {response.text[:200]}")
        assert not failures, "Endpoints raised a server error:\n" + "\n".join(failures)

    def test_parameterised_endpoints_do_not_return_a_server_error(
        self, client: TestClient, auth: dict[str, str]
    ) -> None:
        """Endpoints taking an id or an asset code, called with a value that resolves to
        nothing.

        This exists because ``/news/assessment/{asset}`` returned 500 on every call: its
        response schema forbade extra fields and omitted one the assessment always carries.
        The parameterless sweep could never have reached it. A 404 here is a pass - the point
        is that the handler answers rather than raising.
        """
        failures = []
        for path in parameterised_get_paths(client):
            response = client.get(path, headers=auth)
            if response.status_code >= 500:
                failures.append(f"{path} -> {response.status_code}: {response.text[:200]}")
        assert not failures, "Endpoints raised a server error:\n" + "\n".join(failures)

    def test_the_parameterised_sweep_is_not_empty(self, client: TestClient) -> None:
        assert len(parameterised_get_paths(client)) >= 5

    def test_authentication_is_actually_required(self, client: TestClient) -> None:
        """A 500 sweep is worthless if the endpoints are open to anyone."""
        protected = client.get("/api/v1/trades/summary")
        assert protected.status_code in (401, 403), protected.text


class TestTradeSummary:
    """The specific endpoint that regressed."""

    def test_summary_of_an_account_with_no_trades(
        self, client: TestClient, auth: dict[str, str]
    ) -> None:
        response = client.get("/api/v1/trades/summary", headers=auth)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["total_trades"] == 0
        assert body["wins"] == 0
        assert body["win_rate"] == 0.0
        assert body["net_pnl"] == 0.0
