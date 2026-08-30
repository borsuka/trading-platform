"""Authentication endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.dependencies import AuthServiceDep, ContextDep, CurrentUser
from app.api.schemas import (
    LoginRequest,
    MessageResponse,
    PasswordChangeRequest,
    PasswordResetConfirm,
    PasswordResetRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    payload: RegisterRequest,
    service: AuthServiceDep,
    context: ContextDep,
) -> UserResponse:
    """Create an account.

    Auto-verified unless ``REQUIRE_EMAIL_VERIFICATION`` says otherwise (it defaults to true
    only in production). A desktop or development install has no mail server and the account
    holder is the machine's owner, so requiring verification there would just lock them out.
    """
    settings = get_settings()
    auto_verify = not settings.email_verification_required

    user, verification_token = await service.register(
        email=payload.email,
        password=payload.password,
        full_name=payload.full_name,
        context=context,
        auto_verify=auto_verify,
    )
    if not auto_verify:
        # BLOCKED BY EXTERNAL DEPENDENCY: delivery needs SMTP credentials. The token is
        # generated and stored; app.notifications.email sends it once configured.
        from app.notifications.dispatcher import queue_verification_email

        await queue_verification_email(service.session, user, verification_token)
    return UserResponse.model_validate(user)


@router.post("/login", response_model=TokenResponse)
async def login(
    payload: LoginRequest,
    service: AuthServiceDep,
    context: ContextDep,
) -> TokenResponse:
    """Authenticate and receive an access/refresh token pair."""
    _, tokens = await service.login(
        email=payload.email, password=payload.password, context=context
    )
    return TokenResponse(**tokens.to_dict())


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    payload: RefreshRequest,
    service: AuthServiceDep,
    context: ContextDep,
) -> TokenResponse:
    """Rotate a refresh token.

    Replaying an already-used token invalidates every session for that account.
    """
    _, tokens = await service.refresh(payload.refresh_token, context=context)
    return TokenResponse(**tokens.to_dict())


@router.post("/logout", response_model=MessageResponse)
async def logout(
    payload: RefreshRequest,
    service: AuthServiceDep,
    context: ContextDep,
) -> MessageResponse:
    """Revoke one session."""
    await service.logout(payload.refresh_token, context=context)
    return MessageResponse(message="Signed out")


@router.post("/logout-all", response_model=MessageResponse)
async def logout_all(user: CurrentUser, service: AuthServiceDep) -> MessageResponse:
    """Revoke every session for the current account."""
    count = await service.logout_all(user.id)
    return MessageResponse(message=f"Signed out of {count} session(s)")


@router.get("/me", response_model=UserResponse)
async def current_user(user: CurrentUser) -> UserResponse:
    return UserResponse.model_validate(user)


@router.post("/verify-email/{token}", response_model=MessageResponse)
async def verify_email(token: str, service: AuthServiceDep) -> MessageResponse:
    await service.verify_email(token)
    return MessageResponse(message="Email address verified")


@router.post("/password-reset", response_model=MessageResponse)
async def request_password_reset(
    payload: PasswordResetRequest, service: AuthServiceDep
) -> MessageResponse:
    """Start a password reset.

    Always reports success. Confirming whether an address is registered would turn this
    endpoint into an account-enumeration oracle.
    """
    token = await service.request_password_reset(payload.email)
    if token is not None:
        from app.notifications.dispatcher import queue_password_reset_email

        user = await service.users.get_by_email(payload.email)
        if user is not None:
            await queue_password_reset_email(service.session, user, token)
    return MessageResponse(
        message="If an account exists for that address, a reset link has been sent."
    )


@router.post("/password-reset/confirm", response_model=MessageResponse)
async def confirm_password_reset(
    payload: PasswordResetConfirm, service: AuthServiceDep
) -> MessageResponse:
    await service.reset_password(payload.token, payload.new_password)
    return MessageResponse(
        message="Password updated. All existing sessions have been signed out."
    )


@router.post("/password", response_model=MessageResponse)
async def change_password(
    payload: PasswordChangeRequest, user: CurrentUser, service: AuthServiceDep
) -> MessageResponse:
    await service.change_password(
        user,
        current_password=payload.current_password,
        new_password=payload.new_password,
    )
    return MessageResponse(
        message="Password updated. All existing sessions have been signed out."
    )
