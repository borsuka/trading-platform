"""Exchange account management and the live-trading activation gate.

Credential handling rules enforced here:

* Secrets are encrypted with Fernet before they touch the database.
* No response ever contains a key or secret — only a mask and a fingerprint.
* A key with withdrawal permission is **rejected at connection time**, before it is stored.
* Live activation requires a passing preflight *and* a typed confirmation phrase.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, status

from app.api.dependencies import CurrentUser, SessionDep, SettingsDep
from app.api.schemas import (
    ExchangeAccountCreateRequest,
    ExchangeAccountResponse,
    LiveActivationRequest,
    LivePreflightRequest,
    MessageResponse,
    PreflightResponse,
)
from app.config import ExchangeName
from app.core.clock import utcnow
from app.core.crypto import fingerprint, get_cipher, mask_secret
from app.core.enums import AuditAction
from app.core.exceptions import (
    ConfigurationError,
    ConflictError,
    LiveTradingDisabledError,
    UnsafeCredentialsError,
    ValidationError,
)
from app.core.logging import get_logger
from app.database.models import ExchangeAccount
from app.database.repositories import AuditLogRepository, ExchangeAccountRepository
from app.exchanges.base import ExchangeCredentials
from app.exchanges.live_gate import CONFIRMATION_PHRASE, run_preflight
from app.paper_trading.factory import build_exchange_adapter
from app.risk.limits import RiskLimits

logger = get_logger(__name__)
router = APIRouter(prefix="/exchange-accounts", tags=["exchange-accounts"])


def _credentials_for(account: ExchangeAccount) -> ExchangeCredentials:
    """Decrypt stored credentials for use by an adapter.

    The plaintext exists only inside the adapter instance for the life of the request.
    """
    if not account.api_key_encrypted or not account.api_secret_encrypted:
        raise ConfigurationError(
            f"Exchange account {account.name!r} has no stored credentials"
        )
    cipher = get_cipher()
    return ExchangeCredentials(
        api_key=cipher.decrypt(account.api_key_encrypted),
        api_secret=cipher.decrypt(account.api_secret_encrypted),
        testnet=account.is_testnet,
    )


@router.get("", response_model=list[ExchangeAccountResponse])
async def list_accounts(
    user: CurrentUser, session: SessionDep
) -> list[ExchangeAccountResponse]:
    records = await ExchangeAccountRepository(session).list_for_owner(user.id)
    return [ExchangeAccountResponse.model_validate(r) for r in records]


@router.post(
    "", response_model=ExchangeAccountResponse, status_code=status.HTTP_201_CREATED
)
async def connect_account(
    payload: ExchangeAccountCreateRequest, user: CurrentUser, session: SessionDep
) -> ExchangeAccountResponse:
    """Connect an exchange account.

    The credentials are validated against the venue **before** they are stored, and a key with
    withdrawal permission is refused outright. Storing it first and validating later would
    leave a dangerous key sitting in the database.
    """
    repository = ExchangeAccountRepository(session)
    if await repository.get_by_name(user.id, payload.name) is not None:
        raise ConflictError(f"You already have an exchange account named {payload.name!r}")

    try:
        cipher = get_cipher()
    except ConfigurationError as exc:
        raise ConfigurationError(
            "This deployment cannot store exchange credentials because ENCRYPTION_KEY is "
            "not configured. Generate one with `python -m app.cli generate-key` and set it "
            "before connecting an exchange."
        ) from exc

    credentials = ExchangeCredentials(
        api_key=payload.api_key,
        api_secret=payload.api_secret,
        testnet=payload.testnet,
    )
    adapter = build_exchange_adapter(ExchangeName(payload.exchange), credentials)
    try:
        await adapter.connect()
        permissions = await adapter.validate_credentials()
    except UnsafeCredentialsError:
        await adapter.close()
        # Deliberately not stored. Re-raised as-is: the message tells the user exactly what
        # to change about their key.
        raise
    except Exception as exc:
        await adapter.close()
        raise ValidationError(
            f"Could not validate these credentials with {payload.exchange}: "
            f"{type(exc).__name__}. Check the key, the secret, and whether the key is "
            f"restricted to a different IP address."
        ) from exc
    finally:
        await adapter.close()

    record = ExchangeAccount(
        user_id=user.id,
        name=payload.name,
        exchange=payload.exchange,
        is_testnet=payload.testnet,
        api_key_encrypted=cipher.encrypt(payload.api_key),
        api_secret_encrypted=cipher.encrypt(payload.api_secret),
        api_key_masked=mask_secret(payload.api_key),
        api_key_fingerprint=fingerprint(payload.api_key),
        permissions={
            "can_read": permissions.can_read,
            "can_trade": permissions.can_trade,
            "ip_restricted": permissions.ip_restricted,
        },
        can_withdraw=False,
        is_validated=True,
        validated_at=utcnow(),
    )
    await repository.add(record)
    await AuditLogRepository(session).record(
        action=AuditAction.EXCHANGE_ACCOUNT_CONNECTED,
        user_id=user.id,
        resource_type="exchange_account",
        resource_id=record.id,
        detail={
            "exchange": payload.exchange,
            "testnet": payload.testnet,
            "fingerprint": record.api_key_fingerprint,
        },
    )
    logger.info(
        "api.exchange_account_connected",
        user_id=user.id,
        exchange=payload.exchange,
        testnet=payload.testnet,
    )
    return ExchangeAccountResponse.model_validate(record)


@router.post("/{account_id}/validate", response_model=dict)
async def revalidate(
    account_id: str, user: CurrentUser, session: SessionDep
) -> dict[str, Any]:
    """Re-check a stored key's permissions.

    Worth running periodically: a user can widen a key's permissions on the venue after
    connecting it, and this is how the platform notices.
    """
    repository = ExchangeAccountRepository(session)
    record = await repository.require_for_owner(account_id, user.id)
    adapter = build_exchange_adapter(
        ExchangeName(record.exchange), _credentials_for(record)
    )
    try:
        await adapter.connect()
        permissions = await adapter.validate_credentials()
    except UnsafeCredentialsError as exc:
        record.is_validated = False
        record.is_active = False
        record.can_withdraw = True
        record.last_error = exc.message
        await session.flush()
        logger.error(
            "api.exchange_account_became_unsafe", account_id=account_id, user_id=user.id
        )
        return {
            "valid": False,
            "safe": False,
            "message": exc.message,
            "account_disabled": True,
        }
    except Exception as exc:
        record.is_validated = False
        record.last_error = f"{type(exc).__name__}"
        await session.flush()
        return {"valid": False, "safe": None, "message": f"Validation failed: {exc}"}
    finally:
        await adapter.close()

    record.is_validated = True
    record.validated_at = utcnow()
    record.last_error = None
    record.permissions = {
        "can_read": permissions.can_read,
        "can_trade": permissions.can_trade,
        "ip_restricted": permissions.ip_restricted,
    }
    await session.flush()
    return {
        "valid": True,
        "safe": True,
        "permissions": record.permissions,
        "message": "Credentials are valid and cannot withdraw funds",
    }


@router.post("/{account_id}/preflight", response_model=PreflightResponse)
async def live_preflight(
    account_id: str,
    payload: LivePreflightRequest,
    user: CurrentUser,
    session: SessionDep,
    settings: SettingsDep,
) -> PreflightResponse:
    """Run the live-trading preflight without activating anything.

    Safe to call at any time: it places no orders and changes no state. The frontend uses it to
    render the activation checklist.
    """
    record = await ExchangeAccountRepository(session).require_for_owner(
        account_id, user.id
    )
    adapter = build_exchange_adapter(
        ExchangeName(record.exchange), _credentials_for(record)
    )
    try:
        await adapter.connect()
        report = await run_preflight(
            adapter,
            symbols=payload.symbols,
            interval=payload.interval,
            risk_limits=RiskLimits.conservative(),
            confirmation=CONFIRMATION_PHRASE,  # not the real confirmation; this is a dry run
            settings=settings,
        )
    finally:
        await adapter.close()
    return PreflightResponse.model_validate(report.to_dict())


@router.post("/{account_id}/activate-live", response_model=dict)
async def activate_live(
    account_id: str,
    payload: LiveActivationRequest,
    user: CurrentUser,
    session: SessionDep,
    settings: SettingsDep,
) -> dict[str, Any]:
    """Activate live trading for this account.

    Every one of these must hold:

    1. ``LIVE_TRADING_ENABLED`` is true in the host configuration — which the UI cannot change.
    2. ``TRADING_MODE`` is ``live``.
    3. The user typed the confirmation phrase exactly.
    4. The user acknowledged that no return is guaranteed.
    5. The full preflight passed.

    Activation is recorded in the audit log with the operator's identity.
    """
    if not settings.live_trading_enabled:
        raise LiveTradingDisabledError(
            "Live trading is disabled on this installation. It cannot be enabled from the "
            "interface: set LIVE_TRADING_ENABLED=true and TRADING_MODE=live in the host "
            "environment, then restart. See docs/live-trading.md."
        )
    if payload.confirmation.strip().upper() != CONFIRMATION_PHRASE:
        raise ValidationError(
            f'Confirmation phrase did not match. Type exactly: "{CONFIRMATION_PHRASE}"'
        )
    if not payload.acknowledge_no_guarantee:
        raise ValidationError(
            "You must acknowledge that this software does not guarantee any return and that "
            "you can lose money."
        )

    record = await ExchangeAccountRepository(session).require_for_owner(
        account_id, user.id
    )
    adapter = build_exchange_adapter(
        ExchangeName(record.exchange), _credentials_for(record)
    )
    try:
        await adapter.connect()
        report = await run_preflight(
            adapter,
            symbols=[],  # symbols are checked per bot at start time
            risk_limits=RiskLimits.conservative(),
            confirmation=payload.confirmation,
            settings=settings,
        )
    finally:
        await adapter.close()

    blocking = [c for c in report.failures if c.name != "market_data"]
    if blocking:
        await AuditLogRepository(session).record(
            action=AuditAction.LIVE_TRADING_ACTIVATED,
            user_id=user.id,
            resource_type="exchange_account",
            resource_id=account_id,
            success=False,
            detail={"failed_checks": [c.name for c in blocking]},
        )
        return {
            "activated": False,
            "preflight": report.to_dict(),
            "message": "Live trading was not activated: " + "; ".join(
                c.detail for c in blocking
            ),
        }

    record.permissions = {**(record.permissions or {}), "live_activated": True}
    await session.flush()
    await AuditLogRepository(session).record(
        action=AuditAction.LIVE_TRADING_ACTIVATED,
        user_id=user.id,
        resource_type="exchange_account",
        resource_id=account_id,
        success=True,
        detail={"exchange": record.exchange, "testnet": record.is_testnet},
    )
    logger.warning(
        "api.live_trading_activated",
        user_id=user.id,
        account_id=account_id,
        exchange=record.exchange,
        testnet=record.is_testnet,
    )
    return {
        "activated": True,
        "preflight": report.to_dict(),
        "message": (
            "Live trading activated for this account. Start with the smallest size your "
            "venue allows and verify fills before increasing it."
        ),
    }


@router.delete("/{account_id}", response_model=MessageResponse)
async def remove_account(
    account_id: str, user: CurrentUser, session: SessionDep
) -> MessageResponse:
    """Remove an exchange account and its stored credentials."""
    repository = ExchangeAccountRepository(session)
    record = await repository.require_for_owner(account_id, user.id)

    from sqlalchemy import select

    from app.database.models import Bot

    result = await session.execute(
        select(Bot).where(
            Bot.exchange_account_id == record.id, Bot.user_id == user.id
        )
    )
    bots = list(result.scalars().all())
    if bots:
        raise ConflictError(
            f"This account is used by {len(bots)} bot(s). Delete them first."
        )

    await AuditLogRepository(session).record(
        action=AuditAction.EXCHANGE_ACCOUNT_REMOVED,
        user_id=user.id,
        resource_type="exchange_account",
        resource_id=account_id,
        detail={"exchange": record.exchange},
    )
    await repository.delete(record)
    return MessageResponse(message="Exchange account removed and credentials deleted")
