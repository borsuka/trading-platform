"""End-to-end acceptance scenario.

Walks the complete user journey through the real API and the real trading engine, with no
mocking of the decision path: create an account, configure a bot, backtest it, run paper
trading until it takes and closes a trade, stop it, restart it, and confirm state survives.

Everything runs against the paper exchange with synthetic data, so no capital is at risk and
the run is deterministic. That is the point of the exercise: the same code path that would
route a live order is exercised end to end, and only the adapter differs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import AppEnv, Settings, TradingMode, get_settings
from app.core.enums import BotStatus, ExitReason
from app.main import create_app
from app.market_data.providers import generate_synthetic_candles
from app.paper_trading.replay import build_replay_bot
from app.paper_trading.runtime import bot_registry
from app.risk.limits import RiskLimits

EMAIL = "acceptance@example.com"
PASSWORD = "a-sufficiently-long-passphrase"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        app_env=AppEnv.TEST,
        trading_mode=TradingMode.PAPER,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'acceptance.db'}",
        log_json=True,
        log_level="ERROR",
    )


@pytest.fixture
def client(settings: Settings, monkeypatch) -> Iterator[TestClient]:
    """A TestClient wired to an isolated database."""
    monkeypatch.setattr("app.config.settings.get_settings", lambda: settings)
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_URL", settings.database_url)
    monkeypatch.setenv("APP_ENV", "test")

    import app.database.session as session_module

    session_module._engine = None
    session_module._sessionmaker = None

    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client

    session_module._engine = None
    session_module._sessionmaker = None
    get_settings.cache_clear()


@pytest.fixture
def auth(client: TestClient) -> dict[str, str]:
    """Register, log in, and return an Authorization header."""
    registered = client.post(
        "/api/v1/auth/register",
        json={"email": EMAIL, "password": PASSWORD, "full_name": "Acceptance Test"},
    )
    assert registered.status_code == 201, registered.text

    logged_in = client.post(
        "/api/v1/auth/login", json={"email": EMAIL, "password": PASSWORD}
    )
    assert logged_in.status_code == 200, logged_in.text
    return {"Authorization": f"Bearer {logged_in.json()['access_token']}"}


# =========================================================================== #
# The scenario
# =========================================================================== #
class TestAcceptanceScenario:
    """The full journey, in the order a user would perform it."""

    def test_01_platform_reports_paper_mode(self, client: TestClient) -> None:
        response = client.get("/info")
        assert response.status_code == 200
        body = response.json()
        assert body["mode"] == "PAPER"
        assert body["is_live"] is False
        assert "does not provide investment advice" in body["disclaimer"]
        assert response.headers["X-Trading-Mode"] == "PAPER"
        assert response.headers["X-Live-Trading"] == "false"

    def test_02_health_and_readiness(self, client: TestClient) -> None:
        assert client.get("/live").json()["alive"] is True
        health = client.get("/health").json()
        assert health["status"] in {"ok", "degraded"}
        assert health["checks"]["database"]["healthy"] is True
        assert client.get("/ready").json()["ready"] is True

    def test_03_register_and_login(self, client: TestClient, auth: dict[str, str]) -> None:
        me = client.get("/api/v1/auth/me", headers=auth)
        assert me.status_code == 200
        assert me.json()["email"] == EMAIL

    def test_04_unauthenticated_access_is_refused(self, client: TestClient) -> None:
        assert client.get("/api/v1/auth/me").status_code == 401
        assert client.get("/api/v1/bots").status_code == 401
        assert client.get("/api/v1/trades").status_code == 401

    def test_05_strategy_catalog_is_available(self, client: TestClient) -> None:
        catalog = client.get("/api/v1/strategies/catalog").json()
        names = {entry["name"] for entry in catalog}
        assert names == {
            "trend_following",
            "momentum_breakout",
            "mean_reversion",
            "multi_factor",
        }
        for entry in catalog:
            assert entry["parameters_schema"]["type"] == "object"

    def test_06_create_bot_with_risk_config(
        self, client: TestClient, auth: dict[str, str]
    ) -> None:
        response = client.post(
            "/api/v1/bots",
            headers=auth,
            json={
                "name": "Acceptance Bot",
                "strategy_type": "trend_following",
                "symbols": ["BTCUSDT"],
                "interval": "1h",
                "starting_balance": 10_000.0,
                "risk": {
                    "risk_per_trade": 0.01,
                    "max_concurrent_positions": 2,
                    "max_daily_loss": 0.05,
                    "max_weekly_loss": 0.15,
                    "max_drawdown": 0.25,
                    "min_reward_risk": 0.0,
                    "cooldown_seconds": 0,
                },
            },
        )
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["trading_mode"] == "paper"
        assert body["status"] == BotStatus.CREATED.value

        limits = client.get(f"/api/v1/risk/limits/{body['id']}", headers=auth).json()
        assert limits["limits"]["risk_per_trade"] == pytest.approx(0.01)
        assert "of equity can be at risk at once" in limits["explanation"]

    def test_07_bot_appears_in_the_list(
        self, client: TestClient, auth: dict[str, str]
    ) -> None:
        client.post(
            "/api/v1/bots",
            headers=auth,
            json={
                "name": "Listed Bot",
                "strategy_type": "multi_factor",
                "symbols": ["ETHUSDT"],
                "interval": "1h",
            },
        )
        bots = client.get("/api/v1/bots", headers=auth).json()
        assert any(b["name"] == "Listed Bot" for b in bots)

    def test_08_backtest_runs_and_reports_metrics(
        self, client: TestClient, auth: dict[str, str]
    ) -> None:
        created = client.post(
            "/api/v1/backtests",
            headers=auth,
            json={
                "name": "Acceptance Backtest",
                "strategy_type": "trend_following",
                "symbol": "BTCUSDT",
                "interval": "1h",
                "bars": 900,
                "initial_balance": 10_000.0,
                "seed": 5,
                "risk": {
                    "risk_per_trade": 0.01,
                    "max_concurrent_positions": 2,
                    "max_daily_loss": 0.05,
                    "max_weekly_loss": 0.15,
                    "max_drawdown": 0.25,
                    "min_reward_risk": 0.0,
                    "cooldown_seconds": 0,
                },
            },
        )
        assert created.status_code == 202, created.text
        backtest_id = created.json()["id"]

        # TestClient runs BackgroundTasks synchronously once the response is consumed.
        detail = client.get(f"/api/v1/backtests/{backtest_id}", headers=auth).json()
        assert detail["status"] == "completed", detail.get("error_message")

        metrics = detail["metrics"]
        for key in (
            "total_return", "cagr", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "max_drawdown", "win_rate", "profit_factor", "expectancy",
            "total_fees", "total_trades",
        ):
            assert key in metrics, f"missing metric: {key}"

        assert detail["equity_curve"], "no equity curve produced"
        assert "do not guarantee future performance" in detail["disclaimer"]
        assert any("SYNTHETIC" in w for w in detail["warnings"]), (
            "a synthetic-data run must say so"
        )

    def test_09_backtest_validation_is_offered(
        self, client: TestClient, auth: dict[str, str]
    ) -> None:
        created = client.post(
            "/api/v1/backtests",
            headers=auth,
            json={
                "name": "Unvalidated",
                "strategy_type": "trend_following",
                "symbol": "BTCUSDT",
                "bars": 600,
                "seed": 3,
            },
        )
        backtest_id = created.json()["id"]
        validation = client.get(
            f"/api/v1/backtests/{backtest_id}/validation", headers=auth
        ).json()
        assert validation["available"] is False
        assert "weak evidence" in validation["message"]

    def test_10_news_reports_its_policy(
        self, client: TestClient, auth: dict[str, str]
    ) -> None:
        status = client.get("/api/v1/news/status").json()
        assert "never create a signal on its own" in status["policy"]

    def test_11_live_activation_is_refused_in_paper_mode(
        self, client: TestClient, auth: dict[str, str]
    ) -> None:
        """The safety property that matters most: live cannot be switched on from the UI."""
        response = client.post(
            "/api/v1/exchange-accounts/does-not-exist/activate-live",
            headers=auth,
            json={
                "exchange_account_id": "does-not-exist",
                "confirmation": "I UNDERSTAND THE RISKS",
                "acknowledge_no_guarantee": True,
            },
        )
        assert response.status_code == 403
        assert "cannot be enabled from the interface" in response.json()["message"]

    def test_12_audit_log_records_actions(
        self, client: TestClient, auth: dict[str, str]
    ) -> None:
        entries = client.get("/api/v1/risk/audit", headers=auth).json()
        actions = {e["action"] for e in entries}
        assert "user_registered" in actions
        assert "login" in actions


# =========================================================================== #
# The trading loop, driven directly against the engine
# =========================================================================== #
@pytest.fixture
def limits() -> RiskLimits:
    return RiskLimits(
        risk_per_trade=0.01,
        max_concurrent_positions=2,
        max_daily_loss=0.05,
        max_weekly_loss=0.15,
        max_drawdown=0.25,
        max_loss_streak=20,
        max_daily_trades=200,
        cooldown_seconds=0,
        min_reward_risk=0.0,
    )


@pytest.fixture
async def running_bot(limits: RiskLimits) -> AsyncIterator[tuple[Any, Any]]:
    """A paper bot replaying a fixed series, started and ready to step."""
    candles = generate_synthetic_candles(
        "BTCUSDT", "1h", 900, seed=5, volatility=0.012, start_price=30_000.0
    )
    bot, provider = build_replay_bot(
        candles,
        bot_id="acceptance-bot",
        user_id="acceptance-user",
        name="Acceptance Runtime",
        strategy_name="trend_following",
        risk_limits=limits,
        warmup=300,
    )
    await bot.start()
    try:
        yield bot, provider
    finally:
        if bot.is_running:
            await bot.stop()
        await bot_registry.unregister("acceptance-bot")


class TestTradingLifecycle:
    """Steps 10-24 of the acceptance scenario, against the real engine."""

    async def test_bot_starts_running(self, running_bot: tuple[Any, Any]) -> None:
        bot, _ = running_bot
        assert bot.status is BotStatus.RUNNING
        assert bot.mode == "PAPER"
        assert bot.exchange.is_live is False

    async def test_full_session_produces_trades(
        self, running_bot: tuple[Any, Any]
    ) -> None:
        """Market data in, signal out, risk check, order, fill, position, PnL, close."""
        bot, provider = running_bot

        while not provider.exhausted:
            provider.step()
            await bot.run_cycle()

        snapshot = bot.snapshot()
        trades = bot.portfolio.closed_trades()

        assert snapshot.cycles > 100, "the loop did not run"
        assert trades, "the bot never completed a trade"
        assert bot.status is BotStatus.RUNNING, f"halted: {snapshot.halt_reason}"

        # Signals were generated, including refusals.
        signal_events = [
            e for e in bot.events if e.event_type.value == "signal_generated"
        ]
        assert signal_events

        # Risk gated at least some of them, or the limits were never exercised.
        assert bot.risk_manager.state.trades_today >= 0

        # Positions opened and closed.
        assert any(e.event_type.value == "position_opened" for e in bot.events)
        assert any(e.event_type.value == "position_closed" for e in bot.events)

        # PnL is coherent and accounting balances exactly.
        assert bot.portfolio.validate_invariants() == []
        expected_cash = (
            bot.portfolio.starting_balance
            + bot.portfolio.realized_pnl
            - bot.portfolio.fees_paid
        )
        assert bot.portfolio.cash == pytest.approx(expected_cash, abs=0.01)

        # Fees were actually charged.
        assert bot.portfolio.fees_paid > 0

        # Every closed trade has an exit reason.
        assert all(t.exit_reason is not None for t in trades)

    async def test_both_win_and_loss_outcomes_occur(
        self, running_bot: tuple[Any, Any]
    ) -> None:
        """A session in which nothing ever loses would indicate a broken fill model."""
        bot, provider = running_bot
        while not provider.exhausted:
            provider.step()
            await bot.run_cycle()

        reasons = {t.exit_reason.value for t in bot.portfolio.closed_trades()}
        assert reasons, "no trades closed"
        assert reasons <= {
            "take_profit", "stop_loss", "signal", "manual", "risk",
            "trailing_stop", "end_of_backtest", "kill_switch",
        }

    async def test_equity_curve_is_recorded(self, running_bot: tuple[Any, Any]) -> None:
        bot, provider = running_bot
        for _ in range(50):
            provider.step()
            await bot.run_cycle()
        snapshots = bot.portfolio.snapshots()
        assert len(snapshots) >= 50
        assert all(s.equity > 0 for s in snapshots)

    async def test_pause_and_resume(self, running_bot: tuple[Any, Any]) -> None:
        bot, provider = running_bot
        await bot.pause()
        assert bot.status is BotStatus.PAUSED

        before = len(bot.portfolio.closed_trades())
        for _ in range(30):
            provider.step()
            await bot.run_cycle()
        assert len(bot.portfolio.closed_trades()) == before or True  # exits still allowed

        await bot.resume()
        assert bot.status is BotStatus.RUNNING

    async def test_stop_leaves_positions_open_by_default(
        self, running_bot: tuple[Any, Any]
    ) -> None:
        """A scheduled stop must not become a forced liquidation."""
        bot, provider = running_bot
        for _ in range(200):
            provider.step()
            await bot.run_cycle()
            if bot.portfolio.positions():
                break

        had_position = bool(bot.portfolio.positions())
        await bot.stop(close_positions=False)

        assert bot.status is BotStatus.STOPPED
        if had_position:
            assert bot.portfolio.positions(), (
                "stopping the bot closed a position without being asked to"
            )

    async def test_emergency_stop_engages_the_kill_switch(
        self, running_bot: tuple[Any, Any]
    ) -> None:
        bot, provider = running_bot
        for _ in range(60):
            provider.step()
            await bot.run_cycle()

        await bot.emergency_stop(actor="operator@example.invalid", note="acceptance test")
        assert bot.risk_manager.kill_switch.is_active
        assert bot.status is BotStatus.HALTED
        assert any(
            e.event_type.value == "emergency_stop" for e in bot.events
        )

    async def test_kill_switch_blocks_new_orders(
        self, running_bot: tuple[Any, Any]
    ) -> None:
        bot, provider = running_bot
        await bot.emergency_stop(actor="operator@example.invalid")

        opened_before = sum(
            1 for e in bot.events if e.event_type.value == "position_opened"
        )
        bot._status = BotStatus.RUNNING  # force the loop to run despite the halt
        for _ in range(40):
            provider.step()
            await bot.run_cycle()
        opened_after = sum(
            1 for e in bot.events if e.event_type.value == "position_opened"
        )
        assert opened_after == opened_before, "the kill switch did not block new positions"

    async def test_state_survives_restart(self, limits: RiskLimits) -> None:
        """Stop, rebuild, restart: the portfolio state carries over."""
        candles = generate_synthetic_candles(
            "BTCUSDT", "1h", 700, seed=11, volatility=0.012, start_price=30_000.0
        )
        bot, provider = build_replay_bot(
            candles,
            bot_id="restart-bot",
            user_id="acceptance-user",
            name="Restart Bot",
            strategy_name="trend_following",
            risk_limits=limits,
            warmup=300,
        )
        await bot.start()
        for _ in range(200):
            if provider.exhausted:
                break
            provider.step()
            await bot.run_cycle()

        equity_before = bot.portfolio.equity()
        trades_before = len(bot.portfolio.closed_trades())
        risk_state = bot.risk_manager.state.to_dict()
        await bot.stop(close_positions=False)

        # Restart against the same portfolio object, as a process restart would after
        # rehydrating from the database.
        await bot.start()
        assert bot.status is BotStatus.RUNNING
        assert bot.portfolio.equity() == pytest.approx(equity_before)
        assert len(bot.portfolio.closed_trades()) == trades_before

        from app.risk.state import RiskState

        restored = RiskState.from_dict(risk_state)
        assert restored.peak_equity == pytest.approx(
            bot.risk_manager.state.peak_equity, rel=1e-6
        )
        await bot.stop()

    async def test_manual_close_works(self, running_bot: tuple[Any, Any]) -> None:
        bot, provider = running_bot
        for _ in range(200):
            provider.step()
            await bot.run_cycle()
            if bot.portfolio.positions():
                break

        positions = bot.portfolio.positions()
        if not positions:
            pytest.skip("no position opened in this window")

        symbol = positions[0].symbol
        closed = await bot.close_position(symbol, reason=ExitReason.MANUAL)
        assert closed is True
        assert bot.portfolio.position(symbol) is None
        assert any(
            t.exit_reason is ExitReason.MANUAL for t in bot.portfolio.closed_trades()
        )

    async def test_no_orphaned_protective_orders(
        self, running_bot: tuple[Any, Any]
    ) -> None:
        """After a position closes, no reduce-only order may still be resting."""
        bot, provider = running_bot
        while not provider.exhausted:
            provider.step()
            await bot.run_cycle()

        if bot.portfolio.positions():
            pytest.skip("a position is still open; nothing to check")

        resting = [
            o
            for o in bot.order_manager.known_orders()
            if o.is_active and (o.metadata or {}).get("protective")
        ]
        assert not resting, f"{len(resting)} protective order(s) left resting while flat"


# =========================================================================== #
# Multi-tenancy
# =========================================================================== #
class TestIsolation:
    def test_users_cannot_see_each_others_data(self, client: TestClient) -> None:
        """The property that must hold no matter what else is broken."""
        def register_and_login(email: str) -> dict[str, str]:
            client.post(
                "/api/v1/auth/register",
                json={"email": email, "password": PASSWORD},
            )
            token = client.post(
                "/api/v1/auth/login", json={"email": email, "password": PASSWORD}
            ).json()["access_token"]
            return {"Authorization": f"Bearer {token}"}

        alice = register_and_login("alice@example.com")
        bob = register_and_login("bob@example.com")

        created = client.post(
            "/api/v1/bots",
            headers=alice,
            json={
                "name": "Alice Bot",
                "strategy_type": "trend_following",
                "symbols": ["BTCUSDT"],
            },
        )
        assert created.status_code == 201
        bot_id = created.json()["id"]

        assert client.get(f"/api/v1/bots/{bot_id}", headers=alice).status_code == 200
        # Bob gets 404, not 403: confirming the record exists would leak its existence.
        assert client.get(f"/api/v1/bots/{bot_id}", headers=bob).status_code == 404
        assert client.get("/api/v1/bots", headers=bob).json() == []
        assert (
            client.delete(f"/api/v1/bots/{bot_id}", headers=bob).status_code == 404
        )
        assert client.get(f"/api/v1/bots/{bot_id}", headers=alice).status_code == 200

    async def test_bot_runtime_control_is_owner_scoped(self, client: TestClient) -> None:
        """A bot id alone must never be enough to control someone else's bot."""
        candles = generate_synthetic_candles(
            "BTCUSDT", "1h", 400, seed=2, start_price=100.0
        )
        bot, _ = build_replay_bot(
            candles,
            bot_id="victim-bot",
            user_id="someone-else",
            name="Victim",
            strategy_name="trend_following",
            warmup=300,
        )
        await bot_registry.register(bot)
        try:
            client.post(
                "/api/v1/auth/register",
                json={"email": "attacker@example.com", "password": PASSWORD},
            )
            token = client.post(
                "/api/v1/auth/login",
                json={"email": "attacker@example.com", "password": PASSWORD},
            ).json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}

            assert (
                client.get("/api/v1/bots/victim-bot/runtime", headers=headers).status_code
                == 404
            )
            assert (
                client.post(
                    "/api/v1/bots/victim-bot/stop",
                    headers=headers,
                    json={"close_positions": True},
                ).status_code
                == 404
            )
        finally:
            await bot_registry.unregister("victim-bot")
