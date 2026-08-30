# Architecture

> **Risk notice.** This is trading software. It defaults to **paper** mode and must never be
> pointed at real capital without the explicit, documented activation flow in
> [`docs/live-trading.md`](live-trading.md). Past performance and backtest results do not
> guarantee future performance.

## 1. Purpose

A modular, production-grade algorithmic trading platform. The trading engine is designed to run
on the customer's own machine (desktop/VPS) against the customer's own exchange account. A
separate license server handles activation, device binding and subscription state — it never
receives exchange credentials.

## 2. Deployment topology

```
  Customer machine / VPS                         Vendor infrastructure
 ┌───────────────────────────────┐              ┌──────────────────────┐
 │  TradingBot (backend + worker)│──license────▶│  License server      │
 │  ├── FastAPI control plane    │   validate   │  (activation, device │
 │  ├── Bot runtime (asyncio)    │◀─────────────│   binding, plans)    │
 │  ├── PostgreSQL               │              └──────────────────────┘
 │  └── Redis                    │
 └──────────────┬────────────────┘
                │ API key: trade + read only, NEVER withdraw
                ▼
        Customer exchange account
```

The customer holds their own funds. The platform is software; it is not a broker, not a fund and
not an investment adviser. See [`docs/legal/`](legal/).

## 3. Layering

Dependencies point downward only. Nothing in a lower layer imports from a higher one.

| Layer | Packages | Responsibility |
|---|---|---|
| 6 — Interface | `api`, `frontend`, `desktop` | HTTP/REST, WebSocket, dashboard |
| 5 — Orchestration | `paper_trading` (bot runtime), `backtesting`, `worker` | Drives the loop |
| 4 — Decision | `signals`, `risk`, `portfolio`, `execution` | Signal → risk → order |
| 3 — Analysis | `strategies`, `regimes`, `news`, `indicators` | Turn data into opinions |
| 2 — Data | `market_data`, `exchanges` | Fetch, normalise, validate |
| 1 — Platform | `core`, `config`, `database`, `monitoring` | Config, logging, errors, DB |

`core.domain` holds pure dataclasses/enums with **no** SQLAlchemy or FastAPI imports, so the
analysis and decision layers are unit-testable without a database.

## 4. The trading pipeline

```
MARKET DATA ─▶ VALIDATION ─▶ INDICATORS ─▶ REGIME
                                            │
                    NEWS INTELLIGENCE ──────┤
                                            ▼
                                       STRATEGIES
                                            ▼
                                     SIGNAL ENGINE
                                            ▼
                                    PORTFOLIO CHECK
                                            ▼
                                     RISK MANAGER  ◀── kill switch
                                            ▼
                                     ORDER MANAGER  ── idempotency keys
                                            ▼
                                        EXCHANGE
                                            ▼
                                    FILL ─▶ POSITION ─▶ PORTFOLIO
                                            ▼
                                DATABASE ─▶ MONITORING ─▶ UI / NOTIFICATIONS
```

Hard rules encoded in the code:

* A `Strategy` returns a `StrategyResult`. It has no reference to an exchange, a portfolio or an
  order manager. It cannot place a trade.
* Every order passes through `RiskManager.evaluate()`. There is no second code path.
* `SignalEngine` is the only component allowed to combine strategy output with regime, news,
  liquidity and portfolio state.
* Paper and live share **one** implementation of everything above the `ExchangeAdapter`
  interface. Mode selection happens exactly once, when the adapter is constructed.

## 5. Exchange abstraction

`exchanges.base.ExchangeAdapter` is an ABC. Implementations:

* `PaperExchange` — a real stateful simulator: order book aware fills, partial fills, maker/taker
  fees, slippage, margin, liquidation-free spot + linear perp accounting, stop/TP triggering.
* `BybitAdapter`, `BinanceAdapter`, `CoinbaseAdapter`, `CryptoComAdapter` — REST adapters.
  All refuse to initialise if the
  API key reports withdrawal permission.

The adapter interface is deliberately narrow and exchange-neutral; symbol metadata
(`InstrumentSpec`: tick size, lot size, min notional) is normalised at the adapter boundary so
position sizing never has to special-case a venue.

## 6. Safety architecture

| Control | Where | Behaviour |
|---|---|---|
| Default mode | `config.Settings.trading_mode` | `paper`; live requires an explicit flag |
| Withdrawal permission | adapter `validate_credentials()` | hard reject |
| Kill switch | `risk.kill_switch` | manual + automatic; blocks new orders, never auto-liquidates unless configured |
| Reconciliation | `portfolio.reconciliation` | local vs exchange on startup and on schedule; mismatch ⇒ halt new orders |
| Stale data | `market_data.validation` | candle age > threshold ⇒ `NO_TRADE` |
| Clock drift | `monitoring.clock` | drift > threshold ⇒ halt live trading |
| Idempotency | `execution.order_manager` | client order IDs; timeouts trigger a *state query*, never a blind resend |
| Rate limits | `exchanges.rate_limit` | token bucket per venue per endpoint class |

The system's failure bias is always **NO NEW ORDERS**.

## 7. Persistence

PostgreSQL via SQLAlchemy 2 (async) + Alembic. Schema is never auto-created at startup in
non-development environments; migrations are explicit. Redis is used for caching, rate-limit
buckets, pub/sub of bot events and the worker queue.

At live startup the database is *not* treated as the source of truth:
`LOCAL STATE → EXCHANGE STATE → RECONCILE → VALIDATE → ENABLE TRADING`.

## 8. Multi-tenancy

Every user-owned table carries `user_id`. Repositories take an explicit owner scope and the API
dependency layer injects the authenticated principal; there is no query path that omits it.

## 9. Testing

* Unit: indicators, sizing, risk rules, PnL maths, strategy signals — no I/O.
* Integration: paper exchange order lifecycle, bot runtime, repositories against SQLite/Postgres.
* E2E: the full acceptance scenario in `tests/e2e/` — backtest → paper session → trade → recovery.
* Failure: stale data, desync, repeated order failures, kill-switch trips.
