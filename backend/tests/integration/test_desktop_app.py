"""The desktop application: dashboard serving, first-run setup, and startup safety.

The property worth protecting here is that folding the dashboard into the API process did not
quietly change how the API behaves. In particular a catch-all mount at ``/`` is an easy way to
start answering unknown API paths with an HTML login page, which turns a clear 404 into a
baffling one for anything speaking JSON.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import desktop
from app.config import AppEnv, Settings, TradingMode, get_settings
from app.main import create_app
from app.web import find_ui_directory

needs_built_ui = pytest.mark.skipif(
    find_ui_directory() is None,
    reason="dashboard not built; run scripts/build-desktop.ps1",
)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        app_env=AppEnv.TEST,
        trading_mode=TradingMode.PAPER,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'desktop.db'}",
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


# =========================================================================== #
# Serving the dashboard
# =========================================================================== #
@needs_built_ui
class TestDashboardServing:
    def test_root_serves_the_dashboard(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    @pytest.mark.parametrize("route", ["/login", "/positions", "/bots", "/bots/view"])
    def test_client_routes_resolve_to_pages(self, client: TestClient, route: str) -> None:
        """Each dashboard route is a real exported file, not a fallback to the index."""
        response = client.get(route)
        assert response.status_code == 200, route
        assert "text/html" in response.headers["content-type"]

    def test_unknown_api_path_is_json_not_html(self, client: TestClient) -> None:
        """The regression this guard exists for: a JSON client must get JSON.

        Without it the catch-all mount answers /api/v1/typo with the dashboard shell and a
        200, and the caller sees a parse error instead of a 404.
        """
        response = client.get("/api/v1/no-such-endpoint")
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["error"] == "not_found"

    def test_api_routes_still_win_over_the_mount(self, client: TestClient) -> None:
        response = client.get("/api/v1/strategies/catalog")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")

    def test_health_endpoints_are_not_shadowed(self, client: TestClient) -> None:
        assert client.get("/live").status_code == 200
        assert client.get("/info").json()["mode"] == "PAPER"

    def test_dashboard_carries_the_mode_and_security_headers(
        self, client: TestClient
    ) -> None:
        """The page itself must be labelled, not only the API calls it makes."""
        response = client.get("/")
        assert response.headers["x-trading-mode"] == "PAPER"
        assert response.headers["x-live-trading"] == "false"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["x-content-type-options"] == "nosniff"


class TestUiDiscovery:
    def test_missing_directory_is_not_an_error(self, monkeypatch, tmp_path) -> None:
        """A backend-only install runs fine; it just has no dashboard."""
        monkeypatch.setenv("UI_DIR", str(tmp_path / "nowhere"))
        assert find_ui_directory() is None

    def test_ui_dir_override_is_honoured(self, monkeypatch, tmp_path) -> None:
        (tmp_path / "index.html").write_text("<html></html>", encoding="utf-8")
        monkeypatch.setenv("UI_DIR", str(tmp_path))
        assert find_ui_directory() == tmp_path.resolve()

    def test_app_starts_without_a_dashboard(
        self, settings: Settings, monkeypatch
    ) -> None:
        monkeypatch.setenv("UI_DIR", "/definitely/not/a/directory")
        app = create_app(settings)
        with TestClient(app) as client:
            assert client.get("/live").status_code == 200
            assert client.get("/api/v1/strategies/catalog").status_code == 200


# =========================================================================== #
# First run
# =========================================================================== #
class TestGeneratedConfiguration:
    @pytest.fixture
    def env_file(self, tmp_path, monkeypatch) -> Path:
        target = tmp_path / ".env"
        monkeypatch.setattr(desktop, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(desktop, "ENV_FILE", target)
        return target

    def test_first_run_writes_configuration(self, env_file: Path) -> None:
        assert desktop.ensure_configuration() is True
        assert env_file.exists()

    def test_second_run_leaves_it_alone(self, env_file: Path) -> None:
        """Regenerating ENCRYPTION_KEY would orphan every stored exchange credential."""
        desktop.ensure_configuration()
        original = env_file.read_text(encoding="utf-8")
        assert desktop.ensure_configuration() is False
        assert env_file.read_text(encoding="utf-8") == original

    def test_generated_configuration_is_paper_and_unique(self, env_file: Path) -> None:
        desktop.ensure_configuration()
        values = dict(
            line.split("=", 1)
            for line in env_file.read_text(encoding="utf-8").splitlines()
            if "=" in line and not line.startswith("#")
        )
        assert values["TRADING_MODE"] == "paper"
        assert values["LIVE_TRADING_ENABLED"] == "false"
        assert values["EXCHANGE"] == "paper"
        # A shipped default key would mean every install shares one, which is the same as
        # having none at all.
        assert len(values["SECRET_KEY"]) >= 40
        assert len(values["ENCRYPTION_KEY"]) >= 40

    def test_generated_configuration_actually_loads(self, env_file: Path) -> None:
        """A file the application cannot parse is worse than no file."""
        desktop.ensure_configuration()
        values = dict(
            line.split("=", 1)
            for line in env_file.read_text(encoding="utf-8").splitlines()
            if "=" in line and not line.startswith("#")
        )
        loaded = Settings(**{k.lower(): v for k, v in values.items()})
        assert loaded.is_live is False
        assert loaded.describe_mode() == "PAPER"
        # A desktop install has no mail server; requiring verification would lock the only
        # operator out of their own machine.
        assert loaded.email_verification_required is False


class TestPortSelection:
    """Port selection is tested against a port this test owns.

    Asserting anything about port 8000 itself would make the result depend on whatever else
    happens to be running on the developer's machine - including a copy of this very
    application.
    """

    @staticmethod
    def _free_port() -> int:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])

    def test_prefers_the_given_port_when_it_is_free(self) -> None:
        port = self._free_port()
        assert desktop.choose_port(port) == port

    def test_falls_back_when_the_preferred_port_is_taken(self) -> None:
        """Something already on the port must not stop the application from starting."""
        import socket

        preferred = self._free_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind(("127.0.0.1", preferred))
            held.listen(1)
            port = desktop.choose_port(preferred)
        assert port != preferred
        assert port > 0

    def test_the_default_is_the_conventional_port(self) -> None:
        assert desktop.PREFERRED_PORT == 8000


class TestWindowFallback:
    """What happens when the native window does not work out.

    The failure that motivated these: the window appeared, WebView2 tore itself down a few
    seconds later, and the launcher treated that as "the user quit" and exited. From the
    outside the application simply vanished. Falling back to the browser is always better than
    disappearing, so every one of these paths has to return False rather than True.
    """

    @staticmethod
    def _fake_webview(monkeypatch, *, fire_shown: bool, blocks_for: float):
        import sys
        import time
        import types

        class Events:
            def __init__(self) -> None:
                self.handlers: list = []

            def __iadd__(self, handler):
                self.handlers.append(handler)
                return self

            def fire(self) -> None:
                for handler in self.handlers:
                    handler()

        class Window:
            def __init__(self) -> None:
                self.events = types.SimpleNamespace(shown=Events())

        window = Window()
        module = types.ModuleType("webview")

        def start() -> None:
            if fire_shown:
                window.events.shown.fire()
            time.sleep(blocks_for)

        module.create_window = lambda *a, **k: window  # type: ignore[attr-defined]
        module.start = start  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "webview", module)
        return module

    def test_missing_toolkit_falls_back(self, monkeypatch) -> None:
        import sys

        # A None entry in sys.modules makes `import webview` raise ImportError.
        monkeypatch.setitem(sys.modules, "webview", None)
        assert desktop.open_window("http://127.0.0.1:9/") is False

    def test_window_that_never_appears_falls_back(self, monkeypatch) -> None:
        self._fake_webview(monkeypatch, fire_shown=False, blocks_for=0.0)
        assert desktop.open_window("http://127.0.0.1:9/") is False

    def test_window_that_closes_instantly_falls_back(self, monkeypatch) -> None:
        """The actual observed failure: shown fires, then it dies unprompted."""
        self._fake_webview(monkeypatch, fire_shown=True, blocks_for=0.0)
        assert desktop.open_window("http://127.0.0.1:9/") is False

    def test_toolkit_exception_falls_back(self, monkeypatch) -> None:
        import sys
        import types

        module = types.ModuleType("webview")
        module.create_window = lambda *a, **k: None  # type: ignore[attr-defined]

        def explode() -> None:
            raise RuntimeError("no display")

        module.start = explode  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "webview", module)
        assert desktop.open_window("http://127.0.0.1:9/") is False

    def test_a_window_the_user_actually_used_is_not_a_failure(self, monkeypatch) -> None:
        """Otherwise closing the app would pop open a browser tab on the way out."""
        monkeypatch.setattr(desktop, "MIN_WINDOW_LIFETIME_SECONDS", 0.0)
        self._fake_webview(monkeypatch, fire_shown=True, blocks_for=0.05)
        assert desktop.open_window("http://127.0.0.1:9/") is True


class TestStaleBotStatus:
    """A bot's record outlives the process that ran it.

    The runtime lives in memory and dies with the process; the row saying RUNNING does not.
    After a restart the dashboard showed a green RUNNING badge next to a bot that would never
    act again - the worst kind of wrong, because it looks like everything is fine.
    """

    @pytest.fixture
    def logged_in(self, client: TestClient) -> dict[str, str]:
        client.post(
            "/api/v1/auth/register",
            json={"email": "stale@example.com", "password": "a-long-enough-passphrase"},
        ).raise_for_status()
        token = client.post(
            "/api/v1/auth/login",
            json={"email": "stale@example.com", "password": "a-long-enough-passphrase"},
        ).json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_a_bot_left_running_is_marked_stopped_on_restart(
        self, settings: Settings, client: TestClient, logged_in: dict[str, str]
    ) -> None:
        created = client.post(
            "/api/v1/bots",
            headers=logged_in,
            json={
                "name": "left-running",
                "strategy_type": "trend_following",
                "symbols": ["BTCUSDT"],
                "interval": "15m",
            },
        )
        assert created.status_code == 201, created.text
        bot_id = created.json()["id"]

        # Put the row into the state a crash or restart leaves behind: recorded as running,
        # with no runtime anywhere.
        import asyncio

        from sqlalchemy import update

        from app.core.enums import BotStatus
        from app.database.models import Bot
        from app.database.session import session_scope
        from app.main import _reconcile_bot_status
        from app.paper_trading.runtime import bot_registry

        async def mark_running() -> None:
            async with session_scope() as session:
                await session.execute(
                    update(Bot).where(Bot.id == bot_id).values(status=BotStatus.RUNNING)
                )

        asyncio.run(mark_running())
        assert not list(bot_registry.all()), "no runtime should exist in a fresh process"

        listed = client.get("/api/v1/bots", headers=logged_in).json()
        assert listed[0]["status"] == "running"  # the lie, before reconciliation

        asyncio.run(_reconcile_bot_status())

        listed = client.get("/api/v1/bots", headers=logged_in).json()
        assert listed[0]["status"] == "stopped"
        # And it says why, rather than leaving the user to guess.
        assert "exited" in (listed[0].get("last_error") or "")

    def test_reconciliation_leaves_stopped_bots_alone(
        self, client: TestClient, logged_in: dict[str, str]
    ) -> None:
        created = client.post(
            "/api/v1/bots",
            headers=logged_in,
            json={
                "name": "never-started",
                "strategy_type": "trend_following",
                "symbols": ["BTCUSDT"],
                "interval": "15m",
            },
        )
        assert created.status_code == 201

        import asyncio

        from app.main import _reconcile_bot_status

        asyncio.run(_reconcile_bot_status())

        listed = client.get("/api/v1/bots", headers=logged_in).json()
        assert listed[0]["status"] == "created"
        assert not listed[0].get("last_error")
