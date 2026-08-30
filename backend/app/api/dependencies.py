"""FastAPI dependencies.

The authentication dependency is the choke point for multi-tenancy: handlers receive a
:class:`~app.database.models.User`, and every repository call requires that user's id. A
handler physically cannot query another user's data without writing a different user id, which
is a visible, reviewable act rather than an omission.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Header, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import decode_access_token
from app.auth.service import AuthService, RequestContext
from app.config import Settings, get_settings
from app.core.enums import UserRole, UserStatus
from app.core.exceptions import AuthenticationError, AuthorizationError
from app.database.models import User
from app.database.repositories import UserRepository
from app.database.session import session_scope

bearer_scheme = HTTPBearer(auto_error=False, description="Bearer access token")


async def get_session() -> AsyncIterator[AsyncSession]:
    """Request-scoped database session; commits on success, rolls back on error."""
    async with session_scope() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_request_context(
    request: Request,
    user_agent: Annotated[str | None, Header()] = None,
) -> RequestContext:
    """Client address and user agent, for the audit log and rate limiting."""
    forwarded = request.headers.get("x-forwarded-for")
    ip = (
        forwarded.split(",")[0].strip()
        if forwarded
        else (request.client.host if request.client else None)
    )
    return RequestContext(ip_address=ip, user_agent=user_agent)


ContextDep = Annotated[RequestContext, Depends(get_request_context)]


def get_auth_service(session: SessionDep, settings: SettingsDep) -> AuthService:
    return AuthService(session, settings=settings)


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]


async def get_current_user(
    session: SessionDep,
    settings: SettingsDep,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
) -> User:
    """Resolve the authenticated user, or raise 401."""
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Authentication required")

    claims = decode_access_token(
        credentials.credentials, settings.secret_key.get_secret_value()
    )
    user = await UserRepository(session).get(claims.subject)
    if user is None:
        raise AuthenticationError("Account no longer exists")
    if user.status is UserStatus.SUSPENDED:
        raise AuthorizationError("This account has been suspended")
    if user.status is not UserStatus.ACTIVE:
        raise AuthenticationError("Account is not active")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_admin_user(user: CurrentUser) -> User:
    """Require an administrator."""
    if user.role is not UserRole.ADMIN:
        raise AuthorizationError("Administrator access is required")
    return user


AdminUser = Annotated[User, Depends(get_admin_user)]


async def get_optional_user(
    session: SessionDep,
    settings: SettingsDep,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
) -> User | None:
    """Resolve the user when a token is present, without requiring one."""
    if credentials is None or not credentials.credentials:
        return None
    try:
        return await get_current_user(session, settings, credentials)
    except (AuthenticationError, AuthorizationError):
        return None


OptionalUser = Annotated[User | None, Depends(get_optional_user)]
