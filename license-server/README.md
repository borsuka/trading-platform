# License server

Issues, validates and revokes licences for the trading platform. Runs on the **vendor's**
infrastructure, separate from the trading engine, which runs on the customer's machine.

## What it never receives

Exchange credentials. Positions. Balances. Trades. There is no field for any of them, and no
reason for one. Compromising this service must not expose anyone's exchange account.

## Running it

```bash
export LICENSE_SIGNING_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export LICENSE_ADMIN_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export LICENSE_DB_PATH=./license.db

pip install -e .
license-server
```

Administrative endpoints are **disabled** when `LICENSE_ADMIN_TOKEN` is unset, rather than left
open.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | — | Liveness |
| POST | `/licenses` | admin | Issue a licence |
| GET | `/licenses/{key}` | admin | Inspect a licence and its devices |
| POST | `/licenses/{key}/revoke` | admin | Revoke |
| POST | `/activate` | — | Bind a licence to a device |
| POST | `/validate` | — | Check a licence; refresh last-seen |
| POST | `/deactivate` | — | Release a device slot |

## Response signing

Validation responses are HMAC-SHA256 signed with `LICENSE_SIGNING_SECRET`, so a client can
verify a reply came from this server and was not forged in transit.

## Grace period

A client that cannot reach this server keeps working for `LICENSE_GRACE_PERIOD_HOURS`
(72 by default). Turning a vendor-side outage into a customer-side trading halt — potentially
while they hold open positions — would be a worse failure than a few days of unlicensed use.

## Deployment notes

* Put it behind TLS. Activation requests carry licence keys.
* Back up `license.db`; it is the record of what every customer is entitled to.
* SQLite is adequate for tens of thousands of licences. Beyond that, move to PostgreSQL — the
  schema is portable.
