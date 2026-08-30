"""Authentication service.

Owns registration, login, refresh, logout, email verification and password reset. Every
security-relevant decision lives here rather than in the routers, so an API change cannot
accidentally skip a check.

Refresh-token rotation
----------------------
Each refresh mints a new token and revokes the old one. If a revoked token is presented again,
that means either a client bug or a stolen token being replayed — the service cannot tell which,
so it takes the safe action and revokes **every** session for that user. A legitimate user
re-authenticates; an attacker loses the stolen credential.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import (
    LockoutPolicy,
    PasswordPolicy,
    RateLimiter,
    create_access_token,
    generate_refresh_token,
    generate_verification_token,
    hash_password,
    hash_token,
    needs_rehash,
    verify_password,
)
from app.config import Settings, get_settings
from app.core.clock import utcnow
from app.core.enums import AuditAction, UserRole, UserStatus
from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    NotFoundError,
    ValidationError,
)
from app.core.logging import get_logger
from app.database.models import RefreshToken, User
from app.database.repositories import (
    AuditLogRepository,
    RefreshTokenRepository,
    UserRepository,
)

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AuthTokens:
    """Token pair returned to a client."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 900

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
        }


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Where a request came from. Recorded in the audit log."""

    ip_address: str | None = None
    user_agent: str | None = None

    @property
    def rate_limit_key(self) -> str:
        return self.ip_address or "unknown"


class AuthService:
    """Registration, login and session management."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        settings: Settings | None = None,
        login_limiter: RateLimiter | None = None,
        password_policy: PasswordPolicy | None = None,
        lockout_policy: LockoutPolicy | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.users = UserRepository(session)
        self.tokens = RefreshTokenRepository(session)
        self.audit = AuditLogRepository(session)
        self.login_limiter = login_limiter or RateLimiter(
            max_attempts=self.settings.auth_rate_limit_per_minute, window_seconds=60.0
        )
        self.password_policy = password_policy or PasswordPolicy(
            min_length=self.settings.password_min_length
        )
        self.lockout_policy = lockout_policy or LockoutPolicy()

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    async def register(
        self,
        *,
        email: str,
        password: str,
        full_name: str | None = None,
        context: RequestContext | None = None,
        auto_verify: bool = False,
    ) -> tuple[User, str]:
        """Create an account. Returns ``(user, verification_token)``.

        ``auto_verify`` exists for development and for the desktop build, where there is no
        mail server and the account holder is the machine's owner.
        """
        normalized = email.strip().lower()
        if not normalized or "@" not in normalized:
            raise ValidationError("A valid email address is required")
        self.password_policy.validate(password, email=normalized)

        if await self.users.email_exists(normalized):
            # The address is already registered. Saying so is a small enumeration leak, and
            # the alternative - silently succeeding - breaks the signup flow for a user who
            # simply forgot they had an account. Rate limiting bounds the leak.
            raise ConflictError("An account with this email address already exists")

        verification_token = generate_verification_token()
        user = User(
            email=normalized,
            password_hash=hash_password(password),
            full_name=full_name,
            role=UserRole.USER,
            status=UserStatus.ACTIVE if auto_verify else UserStatus.PENDING_VERIFICATION,
            email_verified=auto_verify,
            email_verification_token=None if auto_verify else verification_token,
        )
        await self.users.add(user)
        await self.audit.record(
            action=AuditAction.USER_REGISTERED,
            user_id=user.id,
            resource_type="user",
            resource_id=user.id,
            ip_address=context.ip_address if context else None,
            user_agent=context.user_agent if context else None,
        )
        logger.info("auth.registered", user_id=user.id, auto_verified=auto_verify)
        return user, verification_token

    async def verify_email(self, token: str) -> User:
        user = await self.users.get_by_verification_token(token)
        if user is None:
            raise NotFoundError("This verification link is invalid or has already been used")
        user.email_verified = True
        user.email_verification_token = None
        if user.status is UserStatus.PENDING_VERIFICATION:
            user.status = UserStatus.ACTIVE
        await self.session.flush()
        logger.info("auth.email_verified", user_id=user.id)
        return user

    # ------------------------------------------------------------------ #
    # Login
    # ------------------------------------------------------------------ #
    async def login(
        self,
        *,
        email: str,
        password: str,
        context: RequestContext | None = None,
    ) -> tuple[User, AuthTokens]:
        """Authenticate and issue a token pair."""
        ctx = context or RequestContext()
        self.login_limiter.check(ctx.rate_limit_key)

        normalized = email.strip().lower()
        user = await self.users.get_by_email(normalized)

        # Always run verification, even for an unknown account, so timing does not leak.
        password_ok = verify_password(password, user.password_hash if user else None)

        if user is None or not password_ok:
            if user is not None:
                await self._record_failure(user, ctx)
            await self.audit.record(
                action=AuditAction.LOGIN_FAILED,
                user_id=user.id if user else None,
                success=False,
                ip_address=ctx.ip_address,
                user_agent=ctx.user_agent,
                detail={"email": normalized},
            )
            raise AuthenticationError("Incorrect email address or password")

        if self.lockout_policy.is_locked(user.locked_until):
            raise AuthenticationError(
                "This account is temporarily locked after repeated failed sign-ins. "
                "Try again later or reset your password."
            )
        if user.status is UserStatus.SUSPENDED:
            raise AuthorizationError("This account has been suspended")
        if user.status is UserStatus.DELETED:
            raise AuthenticationError("Incorrect email address or password")
        if user.status is UserStatus.PENDING_VERIFICATION:
            raise AuthenticationError(
                "Please verify your email address before signing in"
            )

        # Transparently upgrade the hash if the cost parameters have been raised.
        if needs_rehash(user.password_hash):
            user.password_hash = hash_password(password)

        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login_at = utcnow()

        tokens = await self._issue_tokens(user, ctx)
        self.login_limiter.reset(ctx.rate_limit_key)
        await self.audit.record(
            action=AuditAction.LOGIN,
            user_id=user.id,
            ip_address=ctx.ip_address,
            user_agent=ctx.user_agent,
        )
        logger.info("auth.login", user_id=user.id)
        return user, tokens

    async def _record_failure(self, user: User, context: RequestContext) -> None:
        user.failed_login_attempts += 1
        if self.lockout_policy.should_lock(user.failed_login_attempts):
            user.locked_until = self.lockout_policy.locked_until(
                user.failed_login_attempts
            )
            logger.warning(
                "auth.account_locked",
                user_id=user.id,
                attempts=user.failed_login_attempts,
                until=user.locked_until.isoformat() if user.locked_until else None,
                source_ip=context.ip_address,
            )
        await self.session.flush()

    # ------------------------------------------------------------------ #
    # Tokens
    # ------------------------------------------------------------------ #
    async def _issue_tokens(self, user: User, context: RequestContext) -> AuthTokens:
        raw_token, token_hash = generate_refresh_token()
        record = RefreshToken(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=utcnow()
            + timedelta(seconds=self.settings.refresh_token_ttl_seconds),
            user_agent=context.user_agent,
            ip_address=context.ip_address,
        )
        await self.tokens.add(record)
        access = create_access_token(
            subject=user.id,
            secret=self.settings.secret_key.get_secret_value(),
            ttl_seconds=self.settings.access_token_ttl_seconds,
            role=user.role.value,
            session_id=record.id,
        )
        return AuthTokens(
            access_token=access,
            refresh_token=raw_token,
            expires_in=self.settings.access_token_ttl_seconds,
        )

    async def refresh(
        self, refresh_token: str, *, context: RequestContext | None = None
    ) -> tuple[User, AuthTokens]:
        """Rotate a refresh token.

        Presenting an already-revoked token triggers a full session wipe for that user: it is
        the signature of a replayed credential, and the safe response is to invalidate
        everything rather than guess.
        """
        ctx = context or RequestContext()
        record = await self.tokens.get_by_hash(hash_token(refresh_token))
        if record is None:
            raise AuthenticationError("Invalid refresh token")

        if record.revoked_at is not None:
            revoked = await self.tokens.revoke_all_for_user(record.user_id)
            logger.error(
                "auth.refresh_token_reuse_detected",
                user_id=record.user_id,
                sessions_revoked=revoked,
            )
            raise AuthenticationError(
                "This session has been invalidated. Please sign in again."
            )

        if record.expires_at <= utcnow():
            raise AuthenticationError("Refresh token has expired")

        user = await self.users.get(record.user_id)
        if user is None or user.status is not UserStatus.ACTIVE:
            raise AuthenticationError("Account is not active")

        record.revoked_at = utcnow()
        tokens = await self._issue_tokens(user, ctx)
        await self.session.flush()
        return user, tokens

    async def logout(
        self, refresh_token: str, *, context: RequestContext | None = None
    ) -> None:
        """Revoke one session. ``context`` is recorded in the audit trail."""
        """Revoke one session. Silent when the token is already gone."""
        record = await self.tokens.get_by_hash(hash_token(refresh_token))
        if record is None or record.revoked_at is not None:
            return
        record.revoked_at = utcnow()
        await self.session.flush()
        await self.audit.record(
            action=AuditAction.LOGOUT,
            user_id=record.user_id,
            ip_address=context.ip_address if context else None,
        )

    async def logout_all(self, user_id: str) -> int:
        """Revoke every session for a user."""
        count = await self.tokens.revoke_all_for_user(user_id)
        await self.audit.record(
            action=AuditAction.LOGOUT,
            user_id=user_id,
            detail={"sessions_revoked": count, "scope": "all"},
        )
        return count

    # ------------------------------------------------------------------ #
    # Passwords
    # ------------------------------------------------------------------ #
    async def request_password_reset(self, email: str) -> str | None:
        """Start a reset. Returns the token, or ``None`` when the address is unknown.

        The caller must respond identically either way; the ``None`` is for the mail sender,
        not for the API response.
        """
        user = await self.users.get_by_email(email.strip().lower())
        if user is None:
            logger.info("auth.password_reset_unknown_email")
            return None
        token = generate_verification_token()
        user.password_reset_token = token
        user.password_reset_expires_at = utcnow() + timedelta(hours=1)
        await self.session.flush()
        logger.info("auth.password_reset_requested", user_id=user.id)
        return token

    async def reset_password(self, token: str, new_password: str) -> User:
        user = await self.users.get_by_reset_token(token)
        if user is None:
            raise NotFoundError("This reset link is invalid or has already been used")
        if (
            user.password_reset_expires_at is None
            or user.password_reset_expires_at <= utcnow()
        ):
            raise ValidationError("This reset link has expired. Request a new one.")

        self.password_policy.validate(new_password, email=user.email)
        user.password_hash = hash_password(new_password)
        user.password_reset_token = None
        user.password_reset_expires_at = None
        user.failed_login_attempts = 0
        user.locked_until = None
        await self.session.flush()

        # A password reset must not leave old sessions alive: the point of resetting is often
        # that someone else has access.
        revoked = await self.tokens.revoke_all_for_user(user.id)
        await self.audit.record(
            action=AuditAction.PASSWORD_CHANGED,
            user_id=user.id,
            detail={"via": "reset", "sessions_revoked": revoked},
        )
        logger.info("auth.password_reset", user_id=user.id, sessions_revoked=revoked)
        return user

    async def change_password(
        self, user: User, *, current_password: str, new_password: str
    ) -> None:
        if not verify_password(current_password, user.password_hash):
            raise AuthenticationError("Current password is incorrect")
        if current_password == new_password:
            raise ValidationError("The new password must differ from the current one")
        self.password_policy.validate(new_password, email=user.email)

        user.password_hash = hash_password(new_password)
        await self.session.flush()
        revoked = await self.tokens.revoke_all_for_user(user.id)
        await self.audit.record(
            action=AuditAction.PASSWORD_CHANGED,
            user_id=user.id,
            detail={"via": "change", "sessions_revoked": revoked},
        )

    # ------------------------------------------------------------------ #
    # Maintenance
    # ------------------------------------------------------------------ #
    async def purge_expired_tokens(self, *, now: datetime | None = None) -> int:
        removed = await self.tokens.purge_expired(now=now)
        self.login_limiter.prune()
        return removed
