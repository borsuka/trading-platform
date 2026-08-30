"""Desktop application entry point.

Starting the platform should be one action, not a checklist. This module is that action: it
prepares configuration on first run, brings the schema up to date, starts the API with the
dashboard served from the same port, and opens a native window pointed at it.

Three deliberate choices:

* **One process, one port.** The dashboard is a static export served by the API itself, so
  there is no Node process, no second port and no CORS.
* **Loopback only.** The server binds 127.0.0.1. A trading dashboard reachable from the local
  network is not a feature, and a desktop install has no business listening on one.
* **Paper mode, still.** Nothing here changes the trading defaults. A desktop launcher that
  quietly enabled live trading would defeat every other safeguard in this codebase.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"

WINDOW_TITLE = "Trading Platform"
PREFERRED_PORT = 8000
STARTUP_TIMEOUT_SECONDS = 60.0
# A native window that closes sooner than this did not get closed by a person. Treating it as
# a toolkit failure costs a browser tab in the rare case someone really did quit instantly;
# treating it as a deliberate quit costs them an application that vanishes with no explanation.
MIN_WINDOW_LIFETIME_SECONDS = 3.0


# =========================================================================== #
# First run
# =========================================================================== #
def write_initial_env() -> Path:
    """Create ``.env`` with freshly generated keys.

    Generating the keys here rather than asking the user to run a command is the difference
    between an application and a development checkout. The values are written once and never
    regenerated: ENCRYPTION_KEY protects stored exchange credentials, and replacing it would
    make every one of them undecryptable.
    """
    from app.core.crypto import generate_encryption_key

    lines = [
        "# Generated on first run of the desktop application.",
        "# Keep this file private - it contains the keys protecting your data.",
        "",
        "# `staging` rather than `development`: the schema is managed by migrations instead",
        "# of being created implicitly, which is the same path a server install takes. The",
        "# API documentation stays available at /docs.",
        "APP_ENV=staging",
        "",
        "# Paper trading. Switching to live needs BOTH of the next two lines changed, a",
        "# trade-only API key, and the in-app preflight to pass. Read docs/live-trading.md",
        "# before you touch either of them.",
        "TRADING_MODE=paper",
        "LIVE_TRADING_ENABLED=false",
        "EXCHANGE=paper",
        "",
        f"SECRET_KEY={secrets.token_urlsafe(48)}",
        f"ENCRYPTION_KEY={generate_encryption_key()}",
        "",
        "# A desktop install has one operator and no mail server, so requiring email",
        "# verification here would lock that operator out of their own machine.",
        "REQUIRE_EMAIL_VERIFICATION=false",
        "",
        "DATABASE_URL=sqlite+aiosqlite:///./data/trading.db",
        "LOG_JSON=false",
        "LOG_LEVEL=INFO",
        "",
    ]
    ENV_FILE.write_text("\n".join(lines), encoding="utf-8")
    # Best effort. Windows permissions do not map onto this cleanly, and failing to
    # tighten the mode is not a reason to refuse to start.
    with contextlib.suppress(OSError):
        ENV_FILE.chmod(0o600)
    return ENV_FILE


def ensure_configuration() -> bool:
    """Prepare configuration and storage. Returns whether this was a first run."""
    (REPO_ROOT / "data").mkdir(parents=True, exist_ok=True)
    if ENV_FILE.exists():
        return False
    write_initial_env()
    return True


async def _inspect_schema() -> tuple[bool, bool]:
    """Return ``(alembic has stamped this database, application tables exist)``."""
    from sqlalchemy import inspect

    from app.database.session import dispose_engine, get_engine

    engine = get_engine()
    try:
        async with engine.connect() as connection:
            names = await connection.run_sync(lambda sync: inspect(sync).get_table_names())
    finally:
        await dispose_engine()
    return "alembic_version" in names, "users" in names


def run_migrations() -> None:
    """Bring the database schema to head.

    One wrinkle deserves the extra code. Running the API with ``APP_ENV=development`` creates
    the schema directly from the models and never writes an ``alembic_version`` row, so a
    database that is already fully built looks brand new to Alembic - which then tries to
    CREATE TABLE over tables that exist and fails. That database is at head; it simply has no
    record saying so. Stamping is the honest repair, and it is only ever applied when the
    application's own tables are already present.

    Raises on failure. Running against a half-migrated schema is exactly the kind of
    inconsistent state the platform is supposed to refuse to trade from.
    """
    import asyncio

    from alembic import command
    from alembic.config import Config

    stamped, has_tables = asyncio.run(_inspect_schema())

    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))

    if has_tables and not stamped:
        print("  Adopting an existing database created before migrations were in use")
        command.stamp(config, "head")
        return

    command.upgrade(config, "head")


# =========================================================================== #
# Server
# =========================================================================== #
def choose_port(preferred: int = PREFERRED_PORT) -> int:
    """Return a free loopback port, preferring the conventional one.

    The dashboard is served from the same origin as the API, so any port works and there is
    no reason to refuse to start because something else already holds 8000.
    """
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return int(probe.getsockname()[1])
    raise RuntimeError("No free loopback port available")


class BackgroundServer:
    """A uvicorn server on its own thread.

    The window must own the main thread on Windows, so the server cannot have it.
    """

    def __init__(self, port: int) -> None:
        import uvicorn

        self.port = port
        config = uvicorn.Config(
            "app.main:app",
            host="127.0.0.1",
            port=port,
            log_config=None,  # structlog owns logging
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="api", daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self._thread.start()

    def wait_until_ready(self, timeout: float = STARTUP_TIMEOUT_SECONDS) -> None:
        """Block until the server answers, or raise.

        Opening a window against a server that never came up shows the user a blank page and
        no explanation, so the failure is surfaced here instead.
        """
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if not self._thread.is_alive():
                raise RuntimeError("the API stopped while starting up")
            try:
                with urllib.request.urlopen(f"{self.url}/live", timeout=2) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, OSError, ValueError) as exc:
                last_error = exc
            time.sleep(0.25)
        raise TimeoutError(f"the API was not ready within {timeout:.0f}s ({last_error})")

    def stop(self) -> None:
        """Ask the server to shut down and give its lifespan time to run.

        That shutdown stops running bots; killing the process instead would leave their state
        unwritten.
        """
        self._server.should_exit = True
        self._thread.join(timeout=20)


# =========================================================================== #
# Window
# =========================================================================== #
def open_window(url: str) -> bool:
    """Open the dashboard in a native window.

    Returns False when no window toolkit is available or it fails to start, in which case the
    caller should fall back to the browser: the application still works there, and refusing to
    run would be a worse outcome than a less polished one.
    """
    try:
        import webview
    except ImportError:
        return False

    window = webview.create_window(
        WINDOW_TITLE,
        url,
        width=1440,
        height=900,
        min_size=(1024, 700),
    )

    # `webview.start()` returns both when the user closes the window and when the window never
    # managed to appear - and those need opposite responses: shut down in the first case, fall
    # back to the browser in the second. Nothing in the return value distinguishes them, so
    # record whether the window was ever actually shown.
    appeared = threading.Event()
    if window is not None:
        window.events.shown += appeared.set

    started = time.monotonic()
    try:
        webview.start()
    except Exception as exc:  # any toolkit failure means "use the browser"
        print(f"  Could not open a native window ({exc}); using the browser instead.")
        return False

    lifetime = time.monotonic() - started
    if not appeared.is_set():
        print(f"  The native window did not open (after {lifetime:.1f}s).")
        return False
    if lifetime < MIN_WINDOW_LIFETIME_SECONDS:
        print(f"  The native window closed itself after {lifetime:.1f}s.")
        return False
    return True


def open_in_browser(url: str) -> None:
    """Fallback: the default browser, with this console acting as the application."""
    import webbrowser

    webbrowser.open(url)
    print(f"\nThe dashboard is open at {url}")
    print("Keep this window open while you use it. Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


# =========================================================================== #
# Entry point
# =========================================================================== #
def main() -> int:
    os.chdir(REPO_ROOT)
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))

    print(f"{WINDOW_TITLE} - starting")

    if ensure_configuration():
        print(f"  Created {ENV_FILE.name} with newly generated keys (first run)")

    try:
        run_migrations()
        print("  Database schema is up to date")
    except Exception as exc:  # reported to the user, never swallowed
        print(f"\nThe database could not be prepared: {exc}")
        print("Refusing to start: an unknown schema state is not safe to trade from.")
        return 1

    from app.config import get_settings
    from app.web import find_ui_directory

    if find_ui_directory() is None:
        print("\nThe dashboard has not been built yet.")
        print("Build it once with:  powershell -File scripts/build-desktop.ps1")
        return 1

    settings = get_settings()
    server = BackgroundServer(choose_port())
    server.start()
    try:
        server.wait_until_ready()
    except (TimeoutError, RuntimeError) as exc:
        print(f"\nThe application failed to start: {exc}")
        server.stop()
        return 1

    print(f"  Mode: {settings.describe_mode()}")
    print(f"  Listening on {server.url} (this machine only)")
    print()
    print("  This software does not provide investment advice and does not guarantee any")
    print("  return. Past performance and backtest results do not guarantee future")
    print("  performance. You can lose money.")
    print()

    try:
        if not open_window(server.url):
            print("  Falling back to your browser.")
            open_in_browser(server.url)
    finally:
        print("Shutting down...")
        server.stop()
        print("Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
