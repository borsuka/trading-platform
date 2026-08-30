# Deployment and operations

## Topology

```
  Customer machine / VPS                          Vendor infrastructure
 ┌────────────────────────────────┐              ┌──────────────────────┐
 │  TradingBot                    │──license────▶│  License server      │
 │  ├── FastAPI (control plane)   │   validate   │  (activation, device │
 │  ├── Bot runtime (asyncio)     │◀─────────────│   binding, plans)    │
 │  ├── Worker (jobs, retries)    │              └──────────────────────┘
 │  ├── PostgreSQL                │
 │  └── Redis (optional)          │
 └───────────────┬────────────────┘
                 │ API key: trade + read only, NEVER withdraw
                 ▼
         Customer exchange account
```

The trading engine runs on the customer's machine. The vendor's licence server never receives
exchange credentials, positions or balances.

## Docker Compose

```bash
cp .env.example .env
docker compose run --rm backend python -m app.cli generate-key   # paste into .env
docker compose up -d

docker compose ps
curl http://localhost:8000/health
```

With the dashboard:

```bash
docker compose --profile full up -d
```

Migrations run from the entrypoint on start — once per deploy, not once per replica. The
application never auto-creates schema outside development.

## VPS setup

```bash
# 1. Non-root user
sudo adduser --disabled-password --gecos "" trading
sudo usermod -aG docker trading

# 2. Firewall: nothing but SSH and HTTPS
sudo ufw allow OpenSSH
sudo ufw allow 443/tcp
sudo ufw enable

# 3. Clock. Not optional: signed orders are rejected when timestamps drift.
sudo timedatectl set-ntp true
timedatectl status

# 4. Deploy
sudo -u trading -i
git clone <repo> trading-platform && cd trading-platform
cp .env.example .env && $EDITOR .env
docker compose up -d
```

### TLS

Terminate TLS at a reverse proxy; do not expose the application port directly.

```nginx
server {
    listen 443 ssl http2;
    server_name trading.example.com;

    ssl_certificate     /etc/letsencrypt/live/trading.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/trading.example.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;   # HSTS depends on this
    }
}

server {
    listen 80;
    server_name trading.example.com;
    return 301 https://$host$request_uri;
}
```

`X-Forwarded-Proto` matters: the application only sends HSTS when it believes the request
arrived over HTTPS, and it reads the client IP from `X-Forwarded-For` for rate limiting and the
audit log.

## Health probes

| Endpoint | Checks | Use for |
|---|---|---|
| `/live` | nothing external | Liveness. **Never** point this at the database — a database blip would restart every container at once. |
| `/ready` | database, and Redis when required | Readiness / load-balancer membership |
| `/health` | everything, with detail | Dashboards and humans |

Kubernetes:

```yaml
livenessProbe:
  httpGet: { path: /live, port: 8000 }
  initialDelaySeconds: 20
  periodSeconds: 30
readinessProbe:
  httpGet: { path: /ready, port: 8000 }
  initialDelaySeconds: 10
  periodSeconds: 10
```

## Backups

Two things must be backed up, **separately**:

1. **The database** — trading history, configuration, encrypted credentials.
2. **`ENCRYPTION_KEY`** — without it the encrypted credentials are unrecoverable.

Storing them together defeats the point of encrypting at rest.

```bash
# Database
docker compose exec -T postgres pg_dump -U trading trading \
  | gzip > "backup-$(date +%F).sql.gz"

# Restore
gunzip -c backup-2024-01-15.sql.gz \
  | docker compose exec -T postgres psql -U trading trading
```

```cron
0 3 * * * cd /home/trading/trading-platform && ./scripts/backup.sh >> /var/log/trading-backup.log 2>&1
```

**Test the restore.** An untested backup is a hypothesis.

## Monitoring

Watch these, in priority order:

| Signal | Where | Why it matters |
|---|---|---|
| Kill switch engaged | `/api/v1/risk/status/{bot}` | Trading has stopped |
| Bot status `halted` / `error` | `/health` → `bots` | Something needs a human |
| Reconciliation mismatch | Bot events | Local and venue state disagree |
| Clock drift | Bot events | Orders will be rejected |
| Order manager halted | Bot runtime | An order's state is unknown |
| Database unreachable | `/ready` | Nothing can be recorded |
| Drawdown approaching limit | `/api/v1/risk/status/{bot}` | Advance warning before the switch trips |

Logs are structured JSON on stdout with mandatory secret redaction, so any collector works:

```bash
docker compose logs -f backend | jq 'select(.level == "error")'
```

## Upgrades

```bash
docker compose exec backend python -c "
import asyncio
from app.paper_trading.runtime import bot_registry
print([b.config.name for b in bot_registry.all() if b.is_running])"

git pull
docker compose build
docker compose up -d          # migrations run from the entrypoint
curl -fsS localhost:8000/health
```

Stopping a container stops its bots. **It does not close their positions** — a deploy must not
become a liquidation. Decide about open positions deliberately before a long maintenance window.

## Runbook

### Bot halted

1. `GET /api/v1/risk/events` and the bot's event feed — find the first cause, not the cascade.
2. Fix the underlying condition.
3. `POST /api/v1/risk/kill-switch/{bot}/reset` (requires your identity; audited).
4. Start the bot separately. Clearing the switch does not resume trading.

### Reconciliation mismatch

1. **Do not restart.** The halt is correct; restarting would trade on a wrong position size.
2. Compare the platform's positions against the venue's own interface.
3. Work out where they diverged — a manual trade, a missed fill, a liquidation.
4. Only then adopt the venue state (`adopted_by` is required and recorded), and resume.

### Order manager halted

An order's state could not be determined. There may or may not be an order on the venue.

1. Look up the `client_order_id` in the venue's own interface.
2. Cancel it there if it exists and is unwanted.
3. `clear_halt(cleared_by=...)` once you know what actually happened.

### Database unreachable

Readiness fails; the container stays up by design. Restore connectivity — bots resume on the
next cycle. Positions and protective orders resting at the exchange are unaffected.

## Scaling

The single-process bot registry is correct for one customer's installation. For a multi-tenant
service:

* run the API stateless behind a load balancer with `WEB_CONCURRENCY > 1`;
* move rate limiting to Redis (`REDIS_URL`, `REDIS_REQUIRED=true`);
* replace the in-process registry with a distributed scheduler — the `TradingBot` interface
  does not change;
* run one bot process per customer for isolation, so one customer's failure cannot affect
  another's positions.
