# Trading Platform

A modular algorithmic trading platform: strategy research, backtesting with robustness
analysis, paper trading, and — behind an explicit, multi-step gate — live execution against
your own exchange account.

> **Read this first.**
>
> This is software that places orders. It is **not** investment advice, it is **not** a managed
> fund, and it **does not guarantee any return**. You can lose money, including all of it.
> Past performance and backtest results do not guarantee future performance.
>
> The platform defaults to **paper mode** and cannot be switched to live from the user
> interface. See [Live trading](#live-trading).

---

## What it does

```
MARKET DATA ─▶ VALIDATION ─▶ INDICATORS ─▶ REGIME DETECTION
                                              │
                        NEWS INTELLIGENCE ────┤
                                              ▼
                                         STRATEGIES
                                              ▼
                                       SIGNAL ENGINE
                                              ▼
                                     PORTFOLIO CHECK
                                              ▼
                                       RISK MANAGER ◀── kill switch
                                              ▼
                                       ORDER MANAGER ─── idempotency keys
                                              ▼
                                          EXCHANGE
                                              ▼
                                 FILL ─▶ POSITION ─▶ PORTFOLIO
                                              ▼
                                DATABASE ─▶ MONITORING ─▶ UI / NOTIFICATIONS
```

Four load-bearing design decisions:

1. **A strategy cannot place a trade.** It returns an opinion. The signal engine, the risk
   manager and the order manager decide what — if anything — happens next. There is no bypass
   and no "trusted strategy" path.
2. **Paper and live share one code path.** The only difference is which exchange adapter was
   constructed. Duplicating the loop for live mode would guarantee the two drift apart, and the
   whole value of paper trading rests on them being identical.
3. **A timed-out order is never retried.** It is resolved by *querying* the venue. Blind retry
   is how bots end up with double the intended position, and it happens exactly when conditions
   are worst.
4. **The failure bias is always NO NEW ORDERS.** Stale data, clock drift, a reconciliation
   mismatch, repeated API failures — all of them stop new orders rather than guessing.

---

## Quick start

### Desktop application (Windows)

The whole platform as one program: its own window, its own database, nothing to configure.

```powershell
powershell -ExecutionPolicy Bypass -File scriptsuild-desktop.ps1
```

Then double-click **`TradingPlatform.bat`**, or pin it to the taskbar.

The first launch generates `.env` with fresh keys, applies the migrations, and opens the
dashboard. There is no second server: the interface is a static build served by the API on the
same port, bound to `127.0.0.1` and reachable only from that machine. It starts in **paper
mode**, and nothing in the launcher can change that.

Rebuild with the same script after changing the dashboard source. If no window toolkit is
available the launcher opens your default browser instead — the application works either way.

### Docker (recommended for a server)

```bash
cp .env.example .env
docker compose run --rm backend python -m app.cli generate-key   # paste output into .env
docker compose up -d
curl http://localhost:8000/health
```

Open <http://localhost:8000/docs> for the interactive API.

### Local development

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
cd backend && pip install -e ".[dev]"

cp ../.env.example ../.env
python -m app.cli generate-key       # paste output into .env

python -m app.cli migrate up
python -m app.cli serve --reload
```

If `frontend/out` exists (see the desktop build above), the same server also serves the
dashboard at `/`. Otherwise it runs as a bare API, which is what the frontend's own
`npm run dev` expects.

### Try it without any setup

```bash
cd backend

python -m app.cli strategies                             # what ships
python -m app.cli backtest trend_following --bars 2000   # run a backtest
python -m app.cli paper trend_following --bars 1500      # run a paper session
python -m app.cli check                                  # is this install ready?
```

Both commands use **synthetic** data — a deterministic random walk, not market data. Their
results demonstrate that the engine works; they are not a performance claim.

---

## Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.12 | 3.12+ |
| Database | SQLite (single user) | PostgreSQL 16 |
| Cache | none | Redis 7 (needed for multi-worker) |
| RAM | 512 MB | 2 GB |
| Node (frontend) | 20 | 22 |

---

## Configuration

Everything is environment variables; see [`.env.example`](.env.example) for the annotated set.
The ones that matter most:

| Variable | Default | Notes |
|---|---|---|
| `TRADING_MODE` | `paper` | `backtest` / `paper` / `live` |
| `LIVE_TRADING_ENABLED` | `false` | Live needs **both** this and `TRADING_MODE=live` |
| `SECRET_KEY` | insecure dev value | Signs access tokens. Generate before exposing to a network. |
| `ENCRYPTION_KEY` | unset | Encrypts stored exchange credentials. **Without it, exchange accounts cannot be stored at all** — the platform refuses to fall back to plaintext. |
| `DATABASE_URL` | SQLite file | PostgreSQL required when `APP_ENV=production` |
| `RISK_PER_TRADE` | `0.005` | Platform ceiling. Per-bot limits can only be tighter. |
| `RISK_MAX_DRAWDOWN` | `0.10` | Kill switch trips here |

Inspect the effective configuration, with secrets redacted:

```bash
python -m app.cli config
```

---

## Strategies

Four ship in the box. Each documents its hypothesis, entry and exit rules, and — importantly —
its failure modes. See [`docs/strategies.md`](docs/strategies.md).

| Strategy | Trades | Regimes | Primary failure mode |
|---|---|---|---|
| `trend_following` | EMA + ADX + structure | trending, low-vol | whipsaw in a range |
| `momentum_breakout` | Donchian + volume expansion | trending, high-vol | false breakouts |
| `mean_reversion` | Bollinger + RSI + z-score | **ranging only** | trading into a trend |
| `multi_factor` | six weighted factors | any known regime | factor correlation, overfitting |

`mean_reversion` is gated hardest, with four independent brakes against trading a trend. Fading
a trend is the classic way this style blows up.

---

## Backtesting

```bash
python -m app.cli backtest trend_following --bars 3000 --validate
```

The engine reuses the live decision path — the same signal engine, risk manager, order manager
and paper exchange. Lookahead is prevented structurally, not by convention:

* bars are replayed one at a time, sliced so a strategy cannot reach a later bar;
* a decision made on the close of bar *N* executes at the **open of bar N+1**;
* the invariant is re-checked every bar and raises `LookaheadError` if violated;
* news is replayed by publication timestamp.

The backtester is itself tested for bias: a **coin-flip strategy must lose money** at the rate
fees imply. If it did not, the fill model would be flattering and every result meaningless.

### Robustness analysis

A single backtest number is close to worthless. `--validate` adds four independent checks:

| Check | Question |
|---|---|
| Train/test split | Does it work on data not used to choose it? |
| Walk-forward | Does it keep working as the market changes? |
| Parameter sensitivity | Is it a plateau, or a spike that vanishes if a knob moves? |
| Monte Carlo | How much of the equity curve was luck in trade ordering? |

Metrics carry their own honesty warnings — small sample size, short period, fee dominance, an
implausibly high win rate — printed with the results rather than left for the reader to notice.

---

## Paper trading

Paper trading runs the **real engine** against a stateful simulator that models fills, partial
fills, maker/taker fees, slippage, margin, venue tick/lot constraints, and order rejection.
Ambiguity inside a bar resolves pessimistically: if both the stop and the target are inside one
bar's range, the **stop** fills.

```bash
python -m app.cli paper multi_factor --bars 2000 --risk 0.01
```

Through the API:

```bash
POST /api/v1/bots            # create
POST /api/v1/bots/{id}/start # start
POST /api/v1/bots/{id}/cycle # step one decision cycle
GET  /api/v1/portfolio       # watch
```

---

## Live trading

**Live trading is disabled by default and cannot be enabled from the user interface.**

Enabling it requires all of:

1. `LIVE_TRADING_ENABLED=true` **and** `TRADING_MODE=live` in the host environment.
2. An exchange API key with **trade** permission and **no withdrawal permission**. A key that
   can withdraw is rejected at connection time, before it is stored.
3. A passing nine-point preflight: credentials, permissions, balance, risk configuration,
   market data, clock synchronisation, connectivity, platform configuration, and a typed
   confirmation phrase.
4. Explicit acknowledgement that no return is guaranteed.

Read [`docs/live-trading.md`](docs/live-trading.md) before going anywhere near this. It
describes the full checklist, what each check protects against, and the order to do things in.

Four venues are supported: **Bybit V5**, **Binance Spot**, **Coinbase Advanced Trade** and
**Crypto.com Exchange v1**. Each adapter is implemented and tested against a mock transport
reproducing that venue's documented signing, envelopes and error codes. None has been verified
against the live venue, which requires credentials — see
[Known limitations](#known-limitations).

| Venue | Symbols | Testnet | Withdrawal check |
|---|---|---|---|
| Bybit | `BTCUSDT` | yes | permission set read from the venue |
| Binance | `BTCUSDT` | yes | permission set read from the venue |
| Coinbase | `BTC-USD` | **no** | `can_transfer` read from the venue |
| Crypto.com | `BTC_USDT` | yes (UAT) | probed; unverifiable means refused |

Coinbase has no sandbox, so an account there cannot be created with testnet enabled — the
platform refuses rather than routing real orders while the interface says otherwise.

---

## Testing

```bash
cd backend
pytest                      # everything
pytest tests/unit           # fast
pytest tests/e2e            # full acceptance scenario
ruff check . && mypy app    # lint and types
```

The suite covers indicators (including no-lookahead properties), the paper exchange's
accounting invariants, every risk limit, order-manager timeout recovery, news deduplication,
backtester bias detection, multi-tenant isolation, and an end-to-end scenario that creates an
account, backtests, paper-trades to a closed trade, stops, restarts and verifies state.

---

## Security

* Argon2id password hashing; opaque refresh tokens stored only as hashes.
* Refresh-token rotation with reuse detection — replaying a revoked token revokes every session.
* Exchange secrets encrypted with Fernet before they touch the database; never returned by any
  endpoint, only a mask and a fingerprint.
* Mandatory secret redaction in logs, applied as a structlog processor that cannot be
  configured away.
* Every user-owned query is owner-scoped by the repository layer, which **cannot** build an
  unscoped query.
* Append-only audit log for logins, licence changes, credential connections, risk changes,
  bot start/stop, live activation and emergency stops.
* Security headers, CORS allow-list, per-IP auth rate limiting and progressive account lockout.

See [`docs/security.md`](docs/security.md).

---

## Deployment

```bash
docker compose up -d                    # backend + postgres + redis + worker
docker compose --profile full up -d     # ...plus the dashboard
```

Migrations run from the entrypoint on start, once per deploy rather than once per replica. The
application never auto-creates schema outside development.

See [`docs/deployment.md`](docs/deployment.md) for VPS setup, TLS, backups and the operational
runbook.

---

## Project layout

```
backend/app/
  core/          domain types, errors, logging, time, numerics  (no framework imports)
  config/        settings
  database/      ORM models, repositories, session
  market_data/   candles, normalisation, validation, providers
  indicators/    pure, lookahead-free indicator functions
  regimes/       market regime detection
  strategies/    framework + four shipped strategies
  news/          providers, classification, deduplication, scoring
  signals/       the signal engine
  risk/          limits, sizing, state, kill switch, manager
  portfolio/     positions, PnL, reconciliation
  execution/     order manager (idempotency, timeout recovery)
  exchanges/     adapter interface, paper simulator, Bybit, Binance,
                 Coinbase, Crypto.com, live gate
  backtesting/   event-driven engine, metrics, robustness analysis
  paper_trading/ bot runtime, factory, deterministic replay
  api/           HTTP layer
frontend/        Next.js dashboard
license-server/  activation and device binding
docs/            architecture, strategies, live trading, security, deployment, legal
```

---

## Known limitations

Stated plainly rather than buried:

* **Live venue verification is blocked by external dependency.** The Bybit, Binance, Coinbase
  and Crypto.com adapters are complete and tested against mock transports, but no test has run
  against the real exchanges, which requires API credentials. Use testnet first where the venue
  offers one; Coinbase does not.
* **Crypto.com's key permissions are probed, not read.** That venue publishes no endpoint
  reporting a key's scopes. A key that can reach the withdrawal API is refused, and a probe
  that cannot be resolved either way is also refused — but this is inference from an endpoint's
  behaviour rather than a statement from the venue, and it is the one withdrawal check here
  that is not a direct read.
* **Email, Telegram, Discord and web push are unconfigured by default.** Each is implemented;
  each requires credentials. Unconfigured channels report themselves as such rather than
  silently dropping notifications. Web push additionally needs VAPID keys and a subscription
  store.
* **The bundled news classifier is rule-based**, with hand-set keyword weights that have not
  been calibrated against realised price moves. Its confidence score means "how clearly does
  this text match a known pattern", not "how likely is this to move the market".
* **Two-factor authentication is scaffolded, not enabled.** The secret storage and provisioning
  URI exist; the login-time verification loop does not. It is deliberately not half-enabled — a
  2FA prompt that can be skipped is worse than none.
* **Synthetic market data is not market data.** Every result produced from it says so.
* **The bot registry is single-process.** Correct for the desktop/VPS deployment this targets;
  a multi-node deployment needs a distributed scheduler.

---

## Licence and legal

Proprietary. See [`docs/legal/`](docs/legal/) for the Terms, Privacy Policy and Risk Disclosure
**placeholders**. They are drafting aids, not legal documents, and must be reviewed by a
qualified lawyer for your jurisdiction before any commercial launch.

The architecture is designed so that the customer holds their own funds in their own exchange
account and grants trade-only API access. The platform never takes custody.
