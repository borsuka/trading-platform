# Security model

## Threat model

What this platform is actually protecting, ranked by how bad the loss would be:

| Asset | Worst case | Primary control |
|---|---|---|
| Exchange API secrets | Attacker trades or drains the account | Trade-only keys enforced; Fernet encryption at rest; never returned by any endpoint |
| User sessions | Attacker controls someone's bots | Short-lived access tokens; hashed refresh tokens; rotation with reuse detection |
| Trading data | Competitive loss, privacy breach | Owner-scoped queries enforced in the data layer |
| The bot's ability to trade | Unwanted orders | Risk manager, kill switch, no manual-order endpoint |

## Exchange credentials

The most dangerous data here, handled accordingly.

**Withdrawal permission is rejected.** Both adapters read the key's permission set and refuse
any key that can withdraw or transfer — *before* it is stored. This is not a warning the user
can dismiss. A key that can move funds turns any bug, anywhere in the stack, into a theft.

**Encrypted at rest.** Fernet (AES-128-CBC + HMAC-SHA256) with `ENCRYPTION_KEY`, which lives in
the environment, not the database. Without it configured, exchange accounts **cannot be stored
at all** — the platform refuses to fall back to plaintext rather than degrading quietly.

**Never returned.** No API response contains a key or secret. Responses carry a mask
(`abcd******wxyz`) and a fingerprint (a truncated SHA-256) so a user can recognise a key
without anyone being able to use it.

**Re-validated on demand.** A user can widen a key's permissions on the venue after connecting
it. `POST /api/v1/exchange-accounts/{id}/validate` re-reads them and disables the account if it
has become dangerous.

## Authentication

**Argon2id** for passwords — memory-hard, so a leaked hash database cannot be attacked with GPU
parallelism the way bcrypt or PBKDF2 can. Parameters: 3 iterations, 64 MiB, 4 lanes. Hashes are
transparently upgraded on login when the cost parameters are raised.

**Split token model.** Access tokens are short-lived JWTs so most requests need no database
round-trip. Refresh tokens are opaque random strings whose **hash** is stored, so a database
leak does not hand an attacker live sessions. Revocation works on refresh tokens, which is
where it matters.

**Rotation with reuse detection.** Each refresh mints a new token and revokes the old one.
Presenting an already-revoked token means either a client bug or a replayed stolen token — the
service cannot tell which, so it revokes **every** session for that user. A legitimate user
signs in again; an attacker loses the credential.

**Uniform failures.** Login never reveals whether an email exists, and unknown accounts still
run a dummy Argon2 verification so response timing does not leak it either.

**Rate limiting and lockout.** Per-IP sliding window on authentication, plus progressive
per-account lockout that doubles with each failure past the threshold. The two are
complementary: the limiter throttles one source, the lockout protects one account against a
distributed attempt.

**2FA is scaffolded, not enabled.** Secret storage and provisioning URIs exist; the login-time
verification loop does not. It is deliberately not half-enabled — a 2FA prompt that can be
skipped advertises protection that is not there.

## Multi-tenancy

Enforced in the data layer, not by convention. `OwnedRepository` **cannot** build a query
without an owner scope: every read and write goes through a method that unconditionally adds
`WHERE user_id = :owner`, and no method returns rows across users.

The consequence is that "user A sees user B's trades" cannot be caused by a forgotten filter at
a call site — the filter is not the caller's responsibility. Accessing someone else's record
returns **404, not 403**: a 403 would confirm the record exists.

Bot runtime control re-checks ownership against the runtime object, not just the database, so a
bot id alone is never enough to stop someone else's bot.

## Secret redaction

A structlog processor runs on **every** log event before rendering. It is not optional and not
configurable away.

* Field names matching `password`, `secret`, `token`, `api_key` (and variants) are replaced.
* Free-text patterns are scrubbed: `key=value` forms, `Bearer ...`, JWT-shaped strings.
* Nested dictionaries and lists are walked to a bounded depth.

Trading logs routinely carry request payloads and exchange responses; one unredacted field
would be enough to put an API secret in a log file that a customer later emails to support.

The audit log gets the same treatment on its `detail` column, because it is read by support
staff and exported for compliance.

## No manual-order endpoint

There is deliberately no API for placing an arbitrary order. Every order originates from a
strategy signal that has passed the risk manager. A manual-order endpoint would be a second
execution path that skips sizing, exposure checks and the kill switch — exactly the bypass the
architecture exists to prevent.

Closing a position **is** exposed, and is not gated on risk approval: reducing exposure is never
the risky direction.

## Transport and headers

Set on every response:

* `X-Content-Type-Options: nosniff`
* `X-Frame-Options: DENY`
* `Referrer-Policy: strict-origin-when-cross-origin`
* `Permissions-Policy: geolocation=(), microphone=(), camera=()`
* `Strict-Transport-Security` — **only** over HTTPS, since sending it over plain HTTP is
  meaningless and would break local development.

CORS is an explicit allow-list (`CORS_ORIGINS`), never a wildcard with credentials.

## Error handling

No bare `except: pass` anywhere in the codebase. Every exception is either handled with context
or propagated.

API errors expose only what is safe: `TradingPlatformError.public_message()` returns the real
message for client errors (a user needs to know their stop was below the minimum) and a generic
string for server errors (which can contain connection strings, paths or query fragments). The
full detail always goes to the log.

## Audit trail

Append-only. No update or delete method exists on the repository. Recorded: logins and failures,
logout, registration, password changes, licence activation and deactivation, exchange account
connection and removal, strategy changes, risk configuration changes, bot start/stop, live
trading activation, emergency stops, and kill-switch resets.

Actions that could hide a problem require an identified actor and refuse to proceed without one:

* `kill_switch.reset(reset_by=...)`
* `portfolio.resume(resumed_by=...)`
* `portfolio.adopt_exchange_state(adopted_by=...)`
* `order_manager.clear_halt(cleared_by=...)`

## Dependencies

Kept deliberately small. Every dependency is code that runs with access to exchange
credentials, so "popular" is not a sufficient reason to add one.

```bash
pip install pip-audit && pip-audit
```

## Reporting a vulnerability

[Security contact address and disclosure policy — operator must complete before launch.]
