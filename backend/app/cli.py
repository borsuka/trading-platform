"""Command-line interface.

The operational entry point for a desktop or VPS install: generate keys, run migrations,
inspect configuration, run a backtest, and drive a paper session from a terminal.

Every command that could touch real money states the mode it is running in before doing
anything, because "which environment am I pointed at" is the question behind most costly
operator mistakes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from app.config import Settings, get_settings


def _banner(settings: Settings) -> str:
    mode = settings.describe_mode()
    marker = "!! LIVE - REAL MONEY !!" if settings.is_live else "paper - no real money"
    return (
        f"{settings.app_name} v{settings.app_version} | "
        f"env={settings.app_env.value} | mode={mode} ({marker})"
    )


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_generate_key(args: argparse.Namespace) -> int:
    """Generate secrets for a new installation."""
    import secrets

    from app.core.crypto import generate_encryption_key

    print("Add these to your .env file. Keep them secret and back them up.\n")
    print(f"SECRET_KEY={secrets.token_urlsafe(48)}")
    print(f"ENCRYPTION_KEY={generate_encryption_key()}")
    print(
        "\nENCRYPTION_KEY encrypts stored exchange credentials. If you lose it, every\n"
        "connected exchange account must be reconnected. If it leaks, rotate it and\n"
        "reconnect every account."
    )
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Print the effective configuration, with secrets redacted."""
    settings = get_settings()
    print(_banner(settings))
    print()
    payload = json.loads(settings.model_dump_json())

    def redact(node: Any, key: str = "") -> Any:
        from app.core.logging import is_sensitive_key

        if isinstance(node, dict):
            return {k: redact(v, k) for k, v in node.items()}
        if is_sensitive_key(key) and node:
            return "***REDACTED***"
        return node

    print(json.dumps(redact(payload), indent=2, default=str))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Verify that the installation is ready to run."""

    async def run() -> int:
        settings = get_settings()
        print(_banner(settings))
        print()

        from app.monitoring.health import build_health_report

        report = await build_health_report(settings)
        for component in report.components:
            mark = "OK  " if component.healthy else ("FAIL" if component.required else "WARN")
            print(f"[{mark}] {component.name}: {component.detail}")

        problems: list[str] = []
        if settings.encryption_key is None:
            problems.append(
                "ENCRYPTION_KEY is not set: exchange credentials cannot be stored. "
                "Run `trading-bot generate-key`."
            )
        if settings.secret_key.get_secret_value().startswith("dev-only"):
            problems.append(
                "SECRET_KEY is the insecure development default. Run `trading-bot "
                "generate-key` before exposing this to a network."
            )
        if settings.is_live:
            problems.append(
                "LIVE TRADING IS ENABLED. Real orders will be placed with real money."
            )

        if problems:
            print("\nAttention:")
            for problem in problems:
                print(f"  - {problem}")

        print(f"\nResult: {report.status.upper()}")
        return 0 if report.ready else 1

    return asyncio.run(run())


def cmd_migrate(args: argparse.Namespace) -> int:
    """Apply database migrations."""
    from alembic import command
    from alembic.config import Config

    settings = get_settings()
    print(_banner(settings))
    print(f"Applying migrations to: {_safe_url(settings.database_url)}\n")

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", settings.database_url)
    if args.action == "up":
        command.upgrade(config, args.revision)
    elif args.action == "down":
        command.downgrade(config, args.revision)
    elif args.action == "current":
        command.current(config, verbose=True)
    elif args.action == "history":
        command.history(config, verbose=True)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the API server."""
    import uvicorn

    settings = get_settings()
    print(_banner(settings))
    if settings.is_live:
        print("\n!! This instance routes REAL orders. Ctrl-C now if that is not intended. !!\n")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_config=None,
        access_log=False,
    )
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run a backtest from the command line."""

    async def run() -> int:
        from app.backtesting.engine import BacktestConfig, BacktestEngine
        from app.market_data.providers import (
            CsvHistoricalProvider,
            generate_synthetic_candles,
        )
        from app.risk.limits import RiskLimits
        from app.strategies.registry import create_strategy

        settings = get_settings()
        print(_banner(settings))

        if args.csv:
            from datetime import UTC, datetime

            provider = CsvHistoricalProvider(args.csv)
            candles = await provider.get_candles(
                args.symbol,
                args.interval,
                datetime(1970, 1, 1, tzinfo=UTC),
                datetime(2100, 1, 1, tzinfo=UTC),
            )
            source = f"CSV: {args.csv}"
        else:
            candles = generate_synthetic_candles(
                args.symbol,
                args.interval,
                args.bars,
                seed=args.seed,
                volatility=0.01,
                start_price=30_000.0 if args.symbol.upper().startswith("BTC") else 100.0,
            )
            source = "SYNTHETIC data (not market data; results are not a performance claim)"

        print(f"Strategy : {args.strategy}")
        print(f"Symbol   : {args.symbol} {args.interval}")
        print(f"Data     : {len(candles)} bars from {source}")
        print()

        parameters = json.loads(args.parameters) if args.parameters else {}
        engine = BacktestEngine(
            create_strategy(args.strategy, parameters),
            config=BacktestConfig(
                initial_balance=args.balance,
                warmup_bars=min(300, max(120, len(candles) // 8)),
            ),
            risk_limits=RiskLimits(
                risk_per_trade=args.risk,
                max_concurrent_positions=args.max_positions,
                max_daily_loss=0.05,
                max_weekly_loss=0.15,
                max_drawdown=0.25,
                cooldown_seconds=0,
                min_reward_risk=0.0,
            ),
        )
        result = await engine.run(candles)
        print(result.summary())

        if args.validate:
            from app.backtesting.validation import monte_carlo, walk_forward

            print("\nRunning robustness analysis...\n")
            report = await walk_forward(
                args.strategy,
                parameters,
                candles,
                train_bars=max(500, len(candles) // 4),
                test_bars=max(200, len(candles) // 12),
                config=engine.config,
                risk_limits=engine.risk_limits,
            )
            print(f"Walk-forward: {report.verdict()}")
            print(f"Monte Carlo : {monte_carlo(result, simulations=500).verdict()}")

        if args.json:
            print("\n" + json.dumps(result.to_dict(), indent=2, default=str))
        return 0

    return asyncio.run(run())


def cmd_paper(args: argparse.Namespace) -> int:
    """Run a paper-trading session over replayed data."""

    async def run() -> int:
        from app.market_data.providers import generate_synthetic_candles
        from app.paper_trading.replay import build_replay_bot
        from app.risk.limits import RiskLimits

        settings = get_settings()
        print(_banner(settings))

        candles = generate_synthetic_candles(
            args.symbol,
            args.interval,
            args.bars,
            seed=args.seed,
            volatility=0.012,
            start_price=30_000.0 if args.symbol.upper().startswith("BTC") else 100.0,
        )
        bot, provider = build_replay_bot(
            candles,
            bot_id="cli-paper",
            user_id="cli",
            name="CLI Paper Session",
            strategy_name=args.strategy,
            interval=args.interval,
            starting_balance=args.balance,
            warmup=min(300, len(candles) // 3),
            risk_limits=RiskLimits(
                risk_per_trade=args.risk,
                max_concurrent_positions=args.max_positions,
                max_daily_loss=0.05,
                max_weekly_loss=0.15,
                max_drawdown=0.25,
                max_loss_streak=20,
                cooldown_seconds=0,
                min_reward_risk=0.0,
            ),
        )
        print(f"Replaying {len(candles)} bars of SYNTHETIC {args.symbol} data\n")

        await bot.start()
        while not provider.exhausted:
            provider.step()
            await bot.run_cycle()

        snapshot = bot.snapshot()
        stats = bot.portfolio.statistics()
        print(f"Cycles          : {snapshot.cycles}")
        print(f"Status          : {snapshot.status.value}")
        print(f"Equity          : {snapshot.equity:,.2f} (started {args.balance:,.2f})")
        print(f"Realised PnL    : {stats['realized_pnl']:+,.2f}")
        print(f"Fees paid       : {stats['fees_paid']:,.2f}")
        print(f"Closed trades   : {stats['closed_trades']}")
        print(f"Win rate        : {stats['win_rate']:.1%}")
        print(f"Profit factor   : {stats['profit_factor']:.2f}")
        print(f"Max drawdown    : {stats['drawdown']:.2%}")
        print(f"Open positions  : {snapshot.open_positions}")

        problems = bot.portfolio.validate_invariants()
        print(f"Invariants      : {'OK' if not problems else '; '.join(problems)}")
        print(
            "\nThis was a simulation on synthetic data. It is not a performance claim, and "
            "past performance does not guarantee future performance."
        )
        await bot.stop()
        return 0

    return asyncio.run(run())


def cmd_strategies(args: argparse.Namespace) -> int:
    """List the shipped strategies."""
    from app.strategies.registry import describe_all

    for entry in describe_all():
        print(f"\n{entry['name']} v{entry['version']}")
        print(f"  {entry['description']}")
        regimes = ", ".join(entry["allowed_regimes"]) or "any known regime"
        print(f"  Regimes: {regimes}")
        if args.verbose:
            print("  Parameters:")
            for key, value in sorted(entry["default_parameters"].items()):
                print(f"    {key} = {value}")
    return 0


def _safe_url(url: str) -> str:
    """Strip credentials from a database URL before printing it."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}"


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trading-bot",
        description=(
            "Algorithmic trading platform. Defaults to paper mode; live trading requires "
            "explicit configuration plus a passing preflight."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate-key", help="generate SECRET_KEY and ENCRYPTION_KEY").set_defaults(
        func=cmd_generate_key
    )
    sub.add_parser("config", help="print the effective configuration").set_defaults(
        func=cmd_config
    )
    sub.add_parser("check", help="verify the installation is ready").set_defaults(
        func=cmd_check
    )

    migrate = sub.add_parser("migrate", help="apply database migrations")
    migrate.add_argument(
        "action", choices=["up", "down", "current", "history"], default="up", nargs="?"
    )
    migrate.add_argument("--revision", default="head")
    migrate.set_defaults(func=cmd_migrate)

    serve = sub.add_parser("serve", help="run the API server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    backtest = sub.add_parser("backtest", help="run a backtest")
    backtest.add_argument("strategy")
    backtest.add_argument("--symbol", default="BTCUSDT")
    backtest.add_argument("--interval", default="1h")
    backtest.add_argument("--bars", type=int, default=2000)
    backtest.add_argument("--balance", type=float, default=10_000.0)
    backtest.add_argument("--risk", type=float, default=0.01)
    backtest.add_argument("--max-positions", type=int, default=3, dest="max_positions")
    backtest.add_argument("--seed", type=int, default=42)
    backtest.add_argument("--csv", help="directory of {SYMBOL}_{INTERVAL}.csv files")
    backtest.add_argument("--parameters", help="strategy parameters as JSON")
    backtest.add_argument("--validate", action="store_true", help="run robustness analysis")
    backtest.add_argument("--json", action="store_true", help="print the full result as JSON")
    backtest.set_defaults(func=cmd_backtest)

    paper = sub.add_parser("paper", help="run a paper-trading session over replayed data")
    paper.add_argument("strategy")
    paper.add_argument("--symbol", default="BTCUSDT")
    paper.add_argument("--interval", default="1h")
    paper.add_argument("--bars", type=int, default=1500)
    paper.add_argument("--balance", type=float, default=10_000.0)
    paper.add_argument("--risk", type=float, default=0.01)
    paper.add_argument("--max-positions", type=int, default=2, dest="max_positions")
    paper.add_argument("--seed", type=int, default=42)
    paper.set_defaults(func=cmd_paper)

    strategies = sub.add_parser("strategies", help="list the shipped strategies")
    strategies.add_argument("-v", "--verbose", action="store_true")
    strategies.set_defaults(func=cmd_strategies)

    return parser


def main(argv: list[str] | None = None) -> int:
    from app.core.logging import configure_logging

    configure_logging(json_output=False)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        print(f"\nError: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
