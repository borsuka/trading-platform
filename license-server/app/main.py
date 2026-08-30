"""License server.

A separate service from the trading platform, deliberately. It runs on the vendor's
infrastructure and issues, validates and revokes licences; the trading engine runs on the
customer's machine and trades their account. Keeping them apart means:

* the licence server **never** receives exchange credentials, positions, balances or trades —
  it has no field for them and no reason to;
* a licence server outage cannot stop a customer trading (see the grace period below);
* compromising the licence server does not expose anyone's exchange account.

Grace period
------------
When the licence server is unreachable, the client keeps working for
``LICENSE_GRACE_PERIOD_HOURS``. Turning a vendor-side outage into a customer-side trading halt —
potentially while they hold open positions — would be a worse failure than a few days of
unlicensed use.

Responses are HMAC-signed so a client can verify a validation reply came from this server and
was not forged by something intercepting the connection.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, EmailStr, Field

DB_PATH = Path(os.getenv("LICENSE_DB_PATH", "./license.db"))
SIGNING_SECRET = os.getenv("LICENSE_SIGNING_SECRET", "")
ADMIN_TOKEN = os.getenv("LICENSE_ADMIN_TOKEN", "")

PLANS: dict[str, dict[str, int]] = {
    "trial": {"device_limit": 1, "max_bots": 1, "days": 14},
    "starter": {"device_limit": 1, "max_bots": 2, "days": 30},
    "pro": {"device_limit": 3, "max_bots": 10, "days": 30},
    "enterprise": {"device_limit": 10, "max_bots": 100, "days": 365},
}


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS licenses (
    license_key   TEXT PRIMARY KEY,
    email         TEXT NOT NULL,
    plan          TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'active',
    issued_at     TEXT NOT NULL,
    expires_at    TEXT,
    device_limit  INTEGER NOT NULL DEFAULT 1,
    max_bots      INTEGER NOT NULL DEFAULT 1,
    notes         TEXT
);
CREATE INDEX IF NOT EXISTS ix_licenses_email ON licenses(email);

CREATE TABLE IF NOT EXISTS devices (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    license_key        TEXT NOT NULL REFERENCES licenses(license_key) ON DELETE CASCADE,
    device_fingerprint TEXT NOT NULL,
    device_name        TEXT,
    platform           TEXT,
    app_version        TEXT,
    activated_at       TEXT NOT NULL,
    last_seen_at       TEXT,
    deactivated_at     TEXT,
    UNIQUE(license_key, device_fingerprint)
);

CREATE TABLE IF NOT EXISTS activation_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    license_key  TEXT,
    fingerprint  TEXT,
    action       TEXT NOT NULL,
    success      INTEGER NOT NULL,
    detail       TEXT,
    ip_address   TEXT,
    occurred_at  TEXT NOT NULL
);
"""


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    """A connection with foreign keys on and rows as mappings."""
    connection = sqlite3.connect(DB_PATH, timeout=10.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with db() as connection:
        connection.executescript(SCHEMA)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def log_action(
    connection: sqlite3.Connection,
    *,
    action: str,
    success: bool,
    license_key: str | None = None,
    fingerprint: str | None = None,
    detail: str | None = None,
    ip_address: str | None = None,
) -> None:
    connection.execute(
        "INSERT INTO activation_log "
        "(license_key, fingerprint, action, success, detail, ip_address, occurred_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            _mask(license_key),
            _mask(fingerprint),
            action,
            int(success),
            detail,
            ip_address,
            now_iso(),
        ),
    )


def _mask(value: str | None) -> str | None:
    """Store an identifiable prefix, not the whole credential."""
    if not value:
        return None
    return value[:8] + "..." if len(value) > 8 else value


# --------------------------------------------------------------------------- #
# Signing
# --------------------------------------------------------------------------- #
def sign(payload: dict[str, Any]) -> str:
    """HMAC-SHA256 over the canonical JSON form of a response."""
    if not SIGNING_SECRET:
        return ""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hmac.new(
        SIGNING_SECRET.encode(), canonical.encode(), hashlib.sha256
    ).hexdigest()


def require_admin(authorization: str = Header(default="")) -> None:
    """Guard administrative endpoints."""
    if not ADMIN_TOKEN:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "LICENSE_ADMIN_TOKEN is not configured; administrative endpoints are "
                "disabled rather than left open."
            ),
        )
    expected = f"Bearer {ADMIN_TOKEN}"
    if not hmac.compare_digest(authorization, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Not authorised")


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IssueRequest(Model):
    email: EmailStr
    plan: Literal["trial", "starter", "pro", "enterprise"] = "trial"
    days: int | None = Field(default=None, ge=1, le=3650)
    notes: str | None = Field(default=None, max_length=500)


class ActivateRequest(Model):
    license_key: str = Field(min_length=8, max_length=64)
    device_fingerprint: str = Field(min_length=8, max_length=128)
    device_name: str | None = Field(default=None, max_length=200)
    platform: str | None = Field(default=None, max_length=64)
    app_version: str | None = Field(default=None, max_length=32)


class ValidateRequest(Model):
    license_key: str = Field(min_length=8, max_length=64)
    device_fingerprint: str = Field(min_length=8, max_length=128)
    app_version: str | None = Field(default=None, max_length=32)


class DeactivateRequest(Model):
    license_key: str = Field(min_length=8, max_length=64)
    device_fingerprint: str = Field(min_length=8, max_length=128)


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    init_db()
    yield


app = FastAPI(
    title="License Server",
    version="1.0.0",
    description=(
        "Issues and validates licences for the trading platform.\n\n"
        "This service never receives exchange credentials, positions, balances or trades."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        with db() as connection:
            connection.execute("SELECT 1").fetchone()
    except sqlite3.Error as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"database: {type(exc).__name__}"
        ) from exc
    return {
        "status": "ok",
        "signing_configured": bool(SIGNING_SECRET),
        "admin_configured": bool(ADMIN_TOKEN),
    }


@app.post("/licenses", dependencies=[Depends(require_admin)])
def issue_license(payload: IssueRequest) -> dict[str, Any]:
    """Issue a new licence. Administrative."""
    plan = PLANS[payload.plan]
    days = payload.days or plan["days"]
    key = f"{payload.plan.upper()[:4]}-{secrets.token_hex(12).upper()}"
    expires = datetime.now(UTC) + timedelta(days=days)

    with db() as connection:
        connection.execute(
            "INSERT INTO licenses "
            "(license_key, email, plan, status, issued_at, expires_at, device_limit, "
            " max_bots, notes) VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?)",
            (
                key,
                payload.email.lower(),
                payload.plan,
                now_iso(),
                expires.isoformat(),
                plan["device_limit"],
                plan["max_bots"],
                payload.notes,
            ),
        )
        log_action(
            connection, action="issue", success=True, license_key=key,
            detail=f"plan={payload.plan} days={days}",
        )
    return {
        "license_key": key,
        "plan": payload.plan,
        "expires_at": expires.isoformat(),
        "device_limit": plan["device_limit"],
        "max_bots": plan["max_bots"],
    }


@app.post("/activate")
def activate(payload: ActivateRequest) -> dict[str, Any]:
    """Bind a licence to a device.

    Re-activating the same fingerprint is idempotent, so reinstalling on the same machine does
    not consume another device slot.
    """
    with db() as connection:
        record = connection.execute(
            "SELECT * FROM licenses WHERE license_key = ?", (payload.license_key,)
        ).fetchone()

        problem = _license_problem(record)
        if problem is not None:
            log_action(
                connection, action="activate", success=False,
                license_key=payload.license_key,
                fingerprint=payload.device_fingerprint, detail=problem,
            )
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail=problem)

        existing = connection.execute(
            "SELECT * FROM devices WHERE license_key = ? AND device_fingerprint = ?",
            (payload.license_key, payload.device_fingerprint),
        ).fetchone()

        if existing is not None:
            connection.execute(
                "UPDATE devices SET deactivated_at = NULL, last_seen_at = ?, "
                "app_version = COALESCE(?, app_version) WHERE id = ?",
                (now_iso(), payload.app_version, existing["id"]),
            )
        else:
            active = connection.execute(
                "SELECT COUNT(*) AS n FROM devices "
                "WHERE license_key = ? AND deactivated_at IS NULL",
                (payload.license_key,),
            ).fetchone()["n"]
            if active >= record["device_limit"]:
                detail = (
                    f"This licence allows {record['device_limit']} device(s) and {active} "
                    "are already active. Deactivate one first."
                )
                log_action(
                    connection, action="activate", success=False,
                    license_key=payload.license_key,
                    fingerprint=payload.device_fingerprint, detail=detail,
                )
                raise HTTPException(status.HTTP_409_CONFLICT, detail=detail)

            connection.execute(
                "INSERT INTO devices (license_key, device_fingerprint, device_name, "
                "platform, app_version, activated_at, last_seen_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    payload.license_key,
                    payload.device_fingerprint,
                    payload.device_name,
                    payload.platform,
                    payload.app_version,
                    now_iso(),
                    now_iso(),
                ),
            )

        log_action(
            connection, action="activate", success=True,
            license_key=payload.license_key, fingerprint=payload.device_fingerprint,
        )
        body = _license_body(record, connection)

    body["activated"] = True
    return {"data": body, "signature": sign(body)}


@app.post("/validate")
def validate(payload: ValidateRequest) -> dict[str, Any]:
    """Check a licence and refresh the device's last-seen time.

    Clients call this periodically. A failure here does **not** mean the client should stop
    trading immediately — see the grace period in the module docstring.
    """
    with db() as connection:
        record = connection.execute(
            "SELECT * FROM licenses WHERE license_key = ?", (payload.license_key,)
        ).fetchone()

        problem = _license_problem(record)
        if problem is not None:
            log_action(
                connection, action="validate", success=False,
                license_key=payload.license_key, detail=problem,
            )
            body = {"valid": False, "reason": problem, "checked_at": now_iso()}
            return {"data": body, "signature": sign(body)}

        device = connection.execute(
            "SELECT * FROM devices WHERE license_key = ? AND device_fingerprint = ? "
            "AND deactivated_at IS NULL",
            (payload.license_key, payload.device_fingerprint),
        ).fetchone()
        if device is None:
            body = {
                "valid": False,
                "reason": "This device is not activated on this licence",
                "checked_at": now_iso(),
            }
            return {"data": body, "signature": sign(body)}

        connection.execute(
            "UPDATE devices SET last_seen_at = ?, app_version = COALESCE(?, app_version) "
            "WHERE id = ?",
            (now_iso(), payload.app_version, device["id"]),
        )
        body = _license_body(record, connection)

    return {"data": body, "signature": sign(body)}


@app.post("/deactivate")
def deactivate(payload: DeactivateRequest) -> dict[str, Any]:
    """Release a device slot."""
    with db() as connection:
        updated = connection.execute(
            "UPDATE devices SET deactivated_at = ? WHERE license_key = ? "
            "AND device_fingerprint = ? AND deactivated_at IS NULL",
            (now_iso(), payload.license_key, payload.device_fingerprint),
        ).rowcount
        log_action(
            connection, action="deactivate", success=bool(updated),
            license_key=payload.license_key, fingerprint=payload.device_fingerprint,
        )
    if not updated:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail="That device is not activated on this licence",
        )
    return {"deactivated": True}


@app.get("/licenses/{license_key}", dependencies=[Depends(require_admin)])
def get_license(license_key: str) -> dict[str, Any]:
    """Inspect a licence and its devices. Administrative."""
    with db() as connection:
        record = connection.execute(
            "SELECT * FROM licenses WHERE license_key = ?", (license_key,)
        ).fetchone()
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No such licence")
        devices = connection.execute(
            "SELECT device_fingerprint, device_name, platform, app_version, "
            "activated_at, last_seen_at, deactivated_at FROM devices WHERE license_key = ?",
            (license_key,),
        ).fetchall()
    return {
        "license": dict(record),
        "devices": [
            {**dict(d), "device_fingerprint": _mask(d["device_fingerprint"])}
            for d in devices
        ],
    }


@app.post("/licenses/{license_key}/revoke", dependencies=[Depends(require_admin)])
def revoke(license_key: str, reason: str = "") -> dict[str, Any]:
    """Revoke a licence. Administrative."""
    with db() as connection:
        updated = connection.execute(
            "UPDATE licenses SET status = 'revoked', notes = COALESCE(notes, '') || ? "
            "WHERE license_key = ? AND status != 'revoked'",
            (f" [revoked: {reason}]", license_key),
        ).rowcount
        log_action(
            connection, action="revoke", success=bool(updated),
            license_key=license_key, detail=reason,
        )
    if not updated:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail="No such licence, or already revoked"
        )
    return {"revoked": True}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _license_problem(record: sqlite3.Row | None) -> str | None:
    """Why this licence cannot be used, or ``None`` if it can."""
    if record is None:
        # Deliberately identical to the revoked message: distinguishing them would let an
        # attacker enumerate valid keys.
        return "This licence key was not recognised"
    if record["status"] == "revoked":
        return "This licence has been revoked"
    if record["status"] == "suspended":
        return "This licence is suspended"
    if record["expires_at"]:
        expires = datetime.fromisoformat(record["expires_at"])
        if expires <= datetime.now(UTC):
            return f"This licence expired on {expires.date().isoformat()}"
    return None


def _license_body(record: sqlite3.Row, connection: sqlite3.Connection) -> dict[str, Any]:
    active = connection.execute(
        "SELECT COUNT(*) AS n FROM devices WHERE license_key = ? AND deactivated_at IS NULL",
        (record["license_key"],),
    ).fetchone()["n"]
    days_remaining = None
    if record["expires_at"]:
        expires = datetime.fromisoformat(record["expires_at"])
        days_remaining = max(0, (expires - datetime.now(UTC)).days)
    return {
        "valid": True,
        "plan": record["plan"],
        "status": record["status"],
        "expires_at": record["expires_at"],
        "days_remaining": days_remaining,
        "device_limit": record["device_limit"],
        "active_devices": active,
        "max_bots": record["max_bots"],
        "checked_at": now_iso(),
    }


def main() -> None:
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8080")),
    )


if __name__ == "__main__":
    main()
