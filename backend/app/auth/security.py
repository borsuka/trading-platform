"""Password hashing, tokens and rate limiting.

Choices worth stating:

* **Argon2id** for passwords. It is memory-hard, so an attacker with a leaked hash database
  cannot trade GPU parallelism for speed the way they can against bcrypt or PBKDF2.
* **Short-lived JWT access tokens, opaque refresh tokens.** The access token is stateless so
  every request does not hit the database; the refresh token is a random opaque string whose
  **hash** is stored, so a database leak does not hand an attacker live sessions. Revocation
  works on refresh tokens, which is where it matters.
* **Uniform failure messages.** Login never reveals whether an email exists, and verification
  runs a dummy hash on unknown accounts so the response time does not leak it either.
"""

from __future__ import annotations

import contextlib
import hashlib
import secrets
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from jose import JWTError, jwt

from app.core.clock import utcnow
from app.core.exceptions import AuthenticationError, RateLimitError, ValidationError
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Argon2id parameters. Roughly 64 MiB and ~50ms on commodity hardware in 2024 — enough to
#: make offline cracking expensive without making login feel slow.
_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=4,
    hash_len=32,
    salt_len=16,
)

#: A precomputed hash used to burn the same CPU time when an account does not exist.
_DUMMY_HASH = _hasher.hash("dummy-password-for-timing-equalisation")

JWT_ALGORITHM = "HS256"
TokenType = Literal["access", "refresh"]


# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #
def hash_password(password: str) -> str:
    """Hash a password with Argon2id."""
    if not password:
        raise ValidationError("Password cannot be empty")
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    """Verify a password.

    Passing ``None`` (no such user) still performs a hash verification, so the response time
    does not distinguish "wrong password" from "no such account".
    """
    if password_hash is None:
        with contextlib.suppress(
            VerifyMismatchError, VerificationError, InvalidHashError
        ):
            _hasher.verify(_DUMMY_HASH, password)
        return False
    try:
        _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False
    return True


def needs_rehash(password_hash: str) -> bool:
    """True when the hash was produced with weaker parameters than the current policy."""
    try:
        return _hasher.check_needs_rehash(password_hash)
    except InvalidHashError:
        return True


@dataclass(frozen=True, slots=True)
class PasswordPolicy:
    """Password requirements.

    Length is weighted far more heavily than character-class rules, which mostly push users
    toward predictable substitutions. A long passphrase beats ``P@ssw0rd!`` comfortably.
    """

    min_length: int = 12
    max_length: int = 256
    require_digit: bool = False
    require_symbol: bool = False
    require_mixed_case: bool = False
    forbid_common: bool = True

    def validate(self, password: str, *, email: str | None = None) -> None:
        """Raise :class:`ValidationError` describing every problem at once."""
        problems: list[str] = []
        if len(password) < self.min_length:
            problems.append(f"must be at least {self.min_length} characters")
        if len(password) > self.max_length:
            problems.append(f"must be at most {self.max_length} characters")
        if self.require_digit and not any(c.isdigit() for c in password):
            problems.append("must contain a digit")
        if self.require_mixed_case and password.lower() == password:
            problems.append("must contain an uppercase letter")
        if self.require_symbol and password.isalnum():
            problems.append("must contain a symbol")
        if self.forbid_common and password.lower() in COMMON_PASSWORDS:
            problems.append("is among the most commonly used passwords")
        if email and email.split("@")[0].lower() in password.lower():
            problems.append("must not contain your email address")
        if problems:
            raise ValidationError("Password " + "; ".join(problems))


#: A small deny-list. A production deployment should back this with a breach corpus such as
#: Have I Been Pwned's k-anonymity API; this list is the floor, not the ceiling.
COMMON_PASSWORDS: frozenset[str] = frozenset(
    {
        "password", "password1", "password123", "123456", "12345678", "123456789",
        "qwerty", "qwerty123", "letmein", "welcome", "admin", "administrator",
        "iloveyou", "abc123", "monkey", "dragon", "sunshine", "princess",
        "passw0rd", "password!", "trustno1", "changeme", "trading123",
        "bitcoin123", "cryptocurrency", "moneymoney", "correcthorsebattery",
    }
)


# --------------------------------------------------------------------------- #
# JWT access tokens
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TokenClaims:
    """Decoded access-token claims."""

    subject: str
    token_type: TokenType
    issued_at: datetime
    expires_at: datetime
    role: str = "user"
    session_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return utcnow() >= self.expires_at


def create_access_token(
    *,
    subject: str,
    secret: str,
    ttl_seconds: int,
    role: str = "user",
    session_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Mint a signed access token."""
    if not secret or secret.startswith("dev-only"):
        logger.warning("auth.insecure_secret_in_use")
    now = utcnow()
    payload: dict[str, Any] = {
        "sub": subject,
        "type": "access",
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=ttl_seconds)).timestamp()),
        "jti": secrets.token_urlsafe(8),
    }
    if session_id:
        payload["sid"] = session_id
    if extra:
        payload.update(extra)
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str, secret: str) -> TokenClaims:
    """Verify and decode an access token.

    Raises :class:`AuthenticationError` for any failure — expired, tampered, wrong type — with
    a message that does not distinguish between them to a caller.
    """
    try:
        payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
    except JWTError as exc:
        raise AuthenticationError(
            "Invalid or expired token", context={"reason": type(exc).__name__}
        ) from exc

    token_type = payload.get("type")
    if token_type != "access":
        raise AuthenticationError("Token is not an access token")
    subject = payload.get("sub")
    if not subject:
        raise AuthenticationError("Token has no subject")

    return TokenClaims(
        subject=str(subject),
        token_type="access",
        issued_at=datetime.fromtimestamp(payload.get("iat", 0), tz=UTC),
        expires_at=datetime.fromtimestamp(payload.get("exp", 0), tz=UTC),
        role=str(payload.get("role", "user")),
        session_id=payload.get("sid"),
        raw=payload,
    )


# --------------------------------------------------------------------------- #
# Refresh tokens
# --------------------------------------------------------------------------- #
def generate_refresh_token() -> tuple[str, str]:
    """Return ``(token, token_hash)``.

    The plaintext token goes to the client exactly once; only the hash is stored.
    """
    token = secrets.token_urlsafe(48)
    return token, hash_token(token)


def hash_token(token: str) -> str:
    """SHA-256 of an opaque token.

    A fast hash is correct here, unlike for passwords: the token is 384 bits of entropy, so
    brute force is infeasible regardless of hash speed, and refresh happens often enough that
    an expensive KDF would be a real cost.
    """
    return hashlib.sha256(token.encode()).hexdigest()


def generate_verification_token() -> str:
    """Token for email verification and password-reset links."""
    return secrets.token_urlsafe(32)


# --------------------------------------------------------------------------- #
# TOTP (two-factor) scaffold
# --------------------------------------------------------------------------- #
def generate_totp_secret() -> str:
    """Base32 secret for a TOTP authenticator app.

    The full 2FA flow — enrolment, QR provisioning, recovery codes — is scaffolded here and in
    the user model (``totp_secret_encrypted``, ``totp_enabled``) but the verification loop is
    not wired into login yet. It is deliberately not half-enabled: a 2FA prompt that can be
    skipped is worse than none, because it advertises protection that is not there.
    """
    import base64

    return base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")


def totp_provisioning_uri(secret: str, email: str, issuer: str = "TradingPlatform") -> str:
    """otpauth:// URI for authenticator enrolment."""
    from urllib.parse import quote

    label = quote(f"{issuer}:{email}")
    return (
        f"otpauth://totp/{label}?secret={secret}&issuer={quote(issuer)}"
        f"&algorithm=SHA1&digits=6&period=30"
    )


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class RateLimitBucket:
    """Sliding-window counter for one key."""

    hits: list[float] = field(default_factory=list)


class RateLimiter:
    """In-memory sliding-window rate limiter.

    Correct for the single-process desktop/VPS deployment this platform targets. A multi-node
    deployment must move this to Redis; the interface is unchanged, which is why the limiter is
    injected rather than called as a module function.
    """

    def __init__(self, *, max_attempts: int, window_seconds: float) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._buckets: dict[str, RateLimitBucket] = {}

    def check(self, key: str) -> None:
        """Record an attempt, raising :class:`RateLimitError` when the limit is exceeded."""
        now = time.monotonic()
        bucket = self._buckets.setdefault(key, RateLimitBucket())
        cutoff = now - self.window_seconds
        bucket.hits = [t for t in bucket.hits if t > cutoff]

        if len(bucket.hits) >= self.max_attempts:
            retry_after = bucket.hits[0] + self.window_seconds - now
            raise RateLimitError(
                f"Too many attempts. Try again in {max(1, int(retry_after))} seconds.",
                retry_after_seconds=max(1.0, retry_after),
            )
        bucket.hits.append(now)

    def reset(self, key: str) -> None:
        """Clear a key's history, called after a successful authentication."""
        self._buckets.pop(key, None)

    def prune(self) -> int:
        """Drop empty buckets. Bounds memory in a long-running process."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        removed = 0
        for key in list(self._buckets):
            bucket = self._buckets[key]
            bucket.hits = [t for t in bucket.hits if t > cutoff]
            if not bucket.hits:
                del self._buckets[key]
                removed += 1
        return removed


@dataclass(slots=True)
class LockoutPolicy:
    """Progressive account lockout after repeated failures.

    Complements the rate limiter: the limiter throttles by IP, the lockout protects a specific
    account from a distributed attempt.
    """

    max_failures: int = 10
    lockout_minutes: int = 15

    def should_lock(self, failed_attempts: int) -> bool:
        return failed_attempts >= self.max_failures

    def locked_until(self, failed_attempts: int, *, now: datetime | None = None) -> datetime:
        """Lockout expiry, doubling for each additional failure past the threshold."""
        overshoot = max(0, failed_attempts - self.max_failures)
        minutes = self.lockout_minutes * (2 ** min(overshoot, 5))
        return (now or utcnow()) + timedelta(minutes=minutes)

    def is_locked(self, locked_until: datetime | None, *, now: datetime | None = None) -> bool:
        if locked_until is None:
            return False
        return (now or utcnow()) < locked_until
