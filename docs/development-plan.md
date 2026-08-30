# Development plan and status

Status: `[x]` done and tested · `[~]` partial · `[B]` blocked by external dependency

## Phase 1 — Foundation `[x]`
- [x] Repository layout, `pyproject.toml`, pinned dependency set
- [x] `config.Settings` with secret handling and cross-field validation
- [x] Structured logging with **mandatory** secret redaction
- [x] Exception hierarchy with safe public messages
- [x] Async SQLAlchemy engine and session; Redis optional and degradable
- [x] FastAPI app factory, `/health` `/ready` `/live`
- [x] pytest, ruff and mypy configured and green

## Phase 2 — Domain `[x]`
- [x] Pure domain types (`core/domain`, `core/enums`) with no framework imports
- [x] 28 ORM tables
- [x] Alembic environment and initial migration; upgrade/downgrade round-trip verified
- [x] Repositories that **cannot** build an unscoped query

## Phase 3 — Market data `[x]`
- [x] `Candle` / `Ticker` / `OrderBook` value objects, UTC enforced at construction
- [x] Historical and live provider abstractions
- [x] Normaliser: dedupe, ordering, grid alignment, gap handling
- [x] Validator: staleness, continuity, sanity, cross-source divergence
- [x] 11 indicators, all pure and lookahead-free

## Phase 4 — Strategies `[x]`
- [x] `Strategy` ABC with typed parameters and framework-level preconditions
- [x] Trend following
- [x] Momentum breakout
- [x] Mean reversion (four brakes against trading a trend)
- [x] Multi-factor scoring
- [x] Market regime detector, with `UNKNOWN` as a real outcome

## Phase 5 — Risk `[x]`
- [x] `RiskLimits`, clamped so per-bot config can only tighten
- [x] Position sizing including fees, slippage, rounding, margin
- [x] Exposure, drawdown, loss-streak, cooldown, trade-count limits
- [x] Kill switch: manual + seven automatic triggers, human reset required
- [x] Independent post-sizing verification of the risk budget

## Phase 6 — Paper execution `[x]`
- [x] Stateful `PaperExchange`: fills, partials, fees, slippage, margin, rejections
- [x] Order manager with idempotency and query-based timeout recovery
- [x] Portfolio manager, PnL, invariants, reconciliation

## Phase 7 — Backtesting `[x]`
- [x] Event-driven engine reusing the live decision path
- [x] Full metric suite with built-in reliability warnings
- [x] Equity, drawdown, monthly and distribution outputs
- [x] Walk-forward, parameter sensitivity, Monte Carlo (permutation + bootstrap)
- [x] **Bias detection**: a coin-flip strategy must lose money

## Phase 8 — News `[x]`
- [x] Provider abstraction; null, in-memory, file and HTTP implementations
- [x] Deduplication by content hash and title similarity
- [x] Rule-based event classification and sentiment
- [x] Decay, novelty and source reputation
- [x] Wired as a signal modifier that can never originate a trade
- [B] A specific commercial news vendor — needs credentials

## Phase 9 — API `[x]`
- [x] Auth: register, login, refresh with reuse detection, logout, verify, reset
- [x] Bots, strategies, backtests, orders, positions, trades, signals, portfolio
- [x] Risk, exchange accounts, licences, news, health
- [x] Audit logging on every security-relevant action
- [~] 2FA — storage and provisioning scaffolded; login verification not wired (see below)

## Phase 10 — Frontend `[x]`
- [x] Next.js 15 dashboard, 15 pages, builds clean
- [x] PAPER/LIVE banner driven by response headers
- [x] Typed confirmation on every irreversible action
- [x] Light and dark, accessible P&L colours with explicit signs

## Phase 11 — Licensing `[x]`
- [x] Separate licence server with activation, device binding, revocation
- [x] HMAC-signed validation responses
- [x] Client-side licence endpoints and self-hosted trial
- [x] Grace period so a licence outage cannot halt trading

## Phase 12 — Live exchange `[~]`
- [x] Shared REST base with timeout classification
- [x] Bybit V5 adapter
- [x] Binance Spot adapter
- [x] Coinbase Advanced Trade adapter
- [x] Crypto.com Exchange v1 adapter
- [x] Withdrawal-permission rejection on both
- [x] Nine-point preflight
- [B] **Verification against the real venues — requires API credentials**

## Phase 13 — Deployment `[x]`
- [x] Multi-stage Dockerfiles (backend and frontend), non-root
- [x] docker-compose with Postgres, Redis, worker, optional dashboard
- [x] Entrypoint with database wait and migrations
- [x] CI with a blocking safety gate
- [x] Backup script, runbook, security and deployment docs

---

## Remaining limitations

These are stated in the README too. They are not hidden.

| Item | Status | What it needs |
|---|---|---|
| Live venue verification | `[B]` | Bybit/Binance/Coinbase/Crypto.com API credentials; use testnet first where offered (Coinbase has none) |
| Email, Telegram, Discord | `[B]` | SMTP credentials, bot token, webhook URL |
| Web push | `[B]` | VAPID keys and a subscription store |
| Commercial news feed | `[B]` | Vendor API key |
| 2FA login verification | `[~]` | Deliberately not half-enabled — a skippable 2FA prompt advertises protection that is not there |
| News classifier calibration | `[~]` | Keyword weights are reasonable priors, not fitted to realised price moves |
| Multi-node bot scheduling | `[~]` | Correct as-is for the desktop/VPS target; a service deployment needs a distributed scheduler |

## Verification status

Everything below was executed, not assumed:

* Test suite: **624 passing**, including the full end-to-end acceptance scenario
* `ruff check` clean, `mypy app` clean across 101 source files
* Docker image builds; container serves the API and reports healthy
* Migrations upgrade and downgrade cleanly
* Frontend type-checks and builds (15 routes), both as a server bundle and as the
  static export the desktop application serves
* Licence server tests pass (20)
* CLI verified: `check`, `strategies`, `backtest`, `paper`, `migrate`, `generate-key`
* Backup script produces a restorable archive
* Desktop application launches, serves the dashboard on one port, and shuts down cleanly
* News feeds fetched live: 131 articles across 6 publishers, filtered per asset
