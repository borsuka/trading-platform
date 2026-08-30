"""Configuration tests.

These guard the safety defaults and the parsing that broke a container start once already:
a `list[str]` settings field is JSON-decoded by pydantic-settings before any validator runs,
so `CORS_ORIGINS=http://localhost:3000` — the obvious thing to write in a `.env` file — was a
startup crash rather than a working configuration.
"""

from __future__ import annotations

import pytest

from app.config import AppEnv, ExchangeName, Settings, TradingMode
from app.config.settings import RiskSettings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test in this module against an empty environment.

    ``_env_file=None`` stops pydantic-settings reading the ``.env`` file, but it does not stop
    it reading real environment variables - and CI exports ``SECRET_KEY`` for the test job.
    That turned "production refuses the shipped development secret" into a test of whatever
    happened to be exported, and it passed locally while failing in CI.

    The names are derived from the model rather than listed by hand, so a field added later is
    covered without anyone remembering to update this.
    """
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    for name in RiskSettings.model_fields:
        monkeypatch.delenv(f"RISK_{name.upper()}", raising=False)


class TestSafetyDefaults:
    """The defaults that stop an installation trading real money by accident."""

    def test_defaults_to_paper(self) -> None:
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.trading_mode is TradingMode.PAPER
        assert settings.live_trading_enabled is False
        assert settings.exchange is ExchangeName.PAPER
        assert settings.is_live is False

    def test_live_mode_requires_the_master_switch(self) -> None:
        with pytest.raises(ValueError, match="LIVE_TRADING_ENABLED"):
            Settings(
                _env_file=None,  # type: ignore[call-arg]
                trading_mode=TradingMode.LIVE,
                live_trading_enabled=False,
            )

    def test_live_mode_rejects_the_paper_exchange(self) -> None:
        with pytest.raises(ValueError, match="incompatible with EXCHANGE=paper"):
            Settings(
                _env_file=None,  # type: ignore[call-arg]
                trading_mode=TradingMode.LIVE,
                live_trading_enabled=True,
                exchange=ExchangeName.PAPER,
            )

    def test_live_mode_accepted_when_fully_configured(self) -> None:
        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            trading_mode=TradingMode.LIVE,
            live_trading_enabled=True,
            exchange=ExchangeName.BYBIT,
        )
        assert settings.is_live is True
        assert settings.describe_mode() == "LIVE"

    def test_production_rejects_the_dev_secret(self) -> None:
        with pytest.raises(ValueError, match="SECRET_KEY"):
            Settings(
                _env_file=None,  # type: ignore[call-arg]
                app_env=AppEnv.PRODUCTION,
                database_url="postgresql+asyncpg://u:p@host/db",
            )

    def test_production_rejects_sqlite(self) -> None:
        with pytest.raises(ValueError, match="SQLite is not supported in production"):
            Settings(
                _env_file=None,  # type: ignore[call-arg]
                app_env=AppEnv.PRODUCTION,
                secret_key="a-real-production-secret",
                database_url="sqlite+aiosqlite:///./x.db",
            )

    def test_describe_mode_never_claims_live_without_the_switch(self) -> None:
        settings = Settings(_env_file=None, trading_mode=TradingMode.PAPER)  # type: ignore[call-arg]
        assert settings.describe_mode() == "PAPER"


class TestCorsOriginParsing:
    """Regression tests for a startup crash on a perfectly ordinary env var."""

    def test_comma_separated(self, monkeypatch) -> None:
        monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000,https://app.example.com")
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.cors_origins == [
            "http://localhost:3000",
            "https://app.example.com",
        ]

    def test_single_value(self, monkeypatch) -> None:
        monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000")
        assert Settings(_env_file=None).cors_origins == ["http://localhost:3000"]  # type: ignore[call-arg]

    def test_json_array_still_works(self, monkeypatch) -> None:
        """The form pydantic-settings would have accepted by default."""
        monkeypatch.setenv("CORS_ORIGINS", '["http://a.test","http://b.test"]')
        assert Settings(_env_file=None).cors_origins == [  # type: ignore[call-arg]
            "http://a.test",
            "http://b.test",
        ]

    def test_whitespace_is_trimmed(self, monkeypatch) -> None:
        monkeypatch.setenv("CORS_ORIGINS", "  http://a.test ,  http://b.test  ")
        assert Settings(_env_file=None).cors_origins == [  # type: ignore[call-arg]
            "http://a.test",
            "http://b.test",
        ]

    def test_malformed_json_is_reported_clearly(self, monkeypatch) -> None:
        monkeypatch.setenv("CORS_ORIGINS", '["unterminated')
        with pytest.raises(ValueError, match="looks like JSON but is not valid"):
            Settings(_env_file=None)  # type: ignore[call-arg]

    def test_default_when_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
        assert Settings(_env_file=None).cors_origins == ["http://localhost:3000"]  # type: ignore[call-arg]


class TestRiskSettings:
    def test_defaults_are_conservative(self) -> None:
        risk = RiskSettings(_env_file=None)  # type: ignore[call-arg]
        assert risk.per_trade <= 0.01
        assert risk.max_daily_loss <= 0.05
        assert risk.max_drawdown <= 0.20

    def test_per_trade_is_capped(self) -> None:
        with pytest.raises(ValueError):
            RiskSettings(_env_file=None, per_trade=0.5)  # type: ignore[call-arg]

    def test_reads_prefixed_env(self, monkeypatch) -> None:
        monkeypatch.setenv("RISK_PER_TRADE", "0.002")
        assert RiskSettings(_env_file=None).per_trade == pytest.approx(0.002)  # type: ignore[call-arg]


class TestSecretHandling:
    def test_secrets_are_not_in_the_string_form(self) -> None:
        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            secret_key="super-secret-signing-key",
            exchange_api_secret="super-secret-exchange-value",
        )
        rendered = f"{settings!r} {settings.model_dump_json()}"
        assert "super-secret-signing-key" not in rendered
        assert "super-secret-exchange-value" not in rendered

    def test_secret_value_is_still_reachable_deliberately(self) -> None:
        settings = Settings(_env_file=None, secret_key="known-value")  # type: ignore[call-arg]
        assert settings.secret_key.get_secret_value() == "known-value"

    def test_quote_currency_is_normalised(self) -> None:
        settings = Settings(_env_file=None, paper_quote_currency="usdt")  # type: ignore[call-arg]
        assert settings.paper_quote_currency == "USDT"
