"""Licensing and subscription endpoints (client side).

The authoritative license record lives on the vendor's license server; this router is the
client's view of it plus local activation state. See ``license-server/`` for the server.

Enforcement policy: when ``LICENSE_ENFORCEMENT`` is false — the default for development and
self-hosted builds — these endpoints report status but never block trading. Turning a licensing
outage into a trading outage would be worse for the customer than for the vendor.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from fastapi import APIRouter, status

from app.api.dependencies import CurrentUser, SessionDep, SettingsDep
from app.api.schemas import (
    LicenseActivateRequest,
    LicenseResponse,
    MessageResponse,
    SubscriptionResponse,
)
from app.core.clock import utcnow
from app.core.crypto import mask_secret
from app.core.enums import AuditAction, LicensePlan, LicenseStatus
from app.core.exceptions import (
    LicenseDeviceLimitError,
    LicenseExpiredError,
    LicenseInvalidError,
    NotFoundError,
)
from app.core.logging import get_logger
from app.database.models import License, LicenseDevice
from app.database.repositories import (
    AuditLogRepository,
    LicenseRepository,
    SubscriptionRepository,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/licenses", tags=["licenses"])


def _to_response(record: License) -> LicenseResponse:
    active_devices = sum(1 for d in record.devices if d.deactivated_at is None)
    days = None
    if record.expires_at is not None:
        days = max(0, (record.expires_at - utcnow()).days)
    return LicenseResponse(
        id=record.id,
        license_key_masked=mask_secret(record.license_key, visible=4),
        plan=record.plan,
        status=record.status,
        expires_at=record.expires_at,
        device_limit=record.device_limit,
        activated_devices=active_devices,
        days_remaining=days,
    )


@router.get("", response_model=list[LicenseResponse])
async def list_licenses(user: CurrentUser, session: SessionDep) -> list[LicenseResponse]:
    records = await LicenseRepository(session).list_for_owner(user.id)
    return [_to_response(r) for r in records]


@router.get("/current", response_model=dict)
async def current_license(
    user: CurrentUser, session: SessionDep, settings: SettingsDep
) -> dict[str, Any]:
    """The active license, and what its absence means on this installation."""
    record = await LicenseRepository(session).active_for_owner(user.id)
    if record is None:
        return {
            "license": None,
            "enforcement": settings.license_enforcement,
            "message": (
                "No active license."
                + (
                    " Trading is blocked until a license is activated."
                    if settings.license_enforcement
                    else " Licence enforcement is disabled on this installation, so "
                    "trading is unaffected."
                )
            ),
        }
    return {
        "license": _to_response(record).model_dump(mode="json"),
        "enforcement": settings.license_enforcement,
        "message": "License active",
    }


@router.post("/activate", response_model=LicenseResponse, status_code=status.HTTP_200_OK)
async def activate(
    payload: LicenseActivateRequest,
    user: CurrentUser,
    session: SessionDep,
) -> LicenseResponse:
    """Activate a license on this device.

    Device binding is by fingerprint. Re-activating the same fingerprint is idempotent, so a
    user reinstalling on the same machine does not burn a device slot.
    """
    repository = LicenseRepository(session)
    record = await repository.get_by_key(payload.license_key)
    if record is None:
        raise LicenseInvalidError("This license key was not recognised")
    if record.user_id != user.id:
        # Same message as "not recognised": confirming that a key exists but belongs to
        # someone else would let an attacker enumerate valid keys.
        raise LicenseInvalidError("This license key was not recognised")
    if record.status in {LicenseStatus.REVOKED, LicenseStatus.SUSPENDED}:
        raise LicenseInvalidError(f"This license is {record.status.value}")
    if record.expires_at is not None and record.expires_at <= utcnow():
        record.status = LicenseStatus.EXPIRED
        await session.flush()
        raise LicenseExpiredError(
            f"This license expired on {record.expires_at.date().isoformat()}"
        )

    existing = next(
        (
            d
            for d in record.devices
            if d.device_fingerprint == payload.device_fingerprint
        ),
        None,
    )
    if existing is not None:
        existing.deactivated_at = None
        existing.last_seen_at = utcnow()
        existing.app_version = payload.app_version or existing.app_version
    else:
        active = sum(1 for d in record.devices if d.deactivated_at is None)
        if active >= record.device_limit:
            raise LicenseDeviceLimitError(
                f"This license allows {record.device_limit} device(s) and {active} are "
                "already active. Deactivate one before adding another."
            )
        session.add(
            LicenseDevice(
                license_id=record.id,
                device_fingerprint=payload.device_fingerprint,
                device_name=payload.device_name,
                platform=payload.platform,
                app_version=payload.app_version,
                activated_at=utcnow(),
                last_seen_at=utcnow(),
            )
        )

    record.status = LicenseStatus.ACTIVE
    if record.issued_at is None:
        record.issued_at = utcnow()
    await session.flush()
    await session.refresh(record)

    await AuditLogRepository(session).record(
        action=AuditAction.LICENSE_ACTIVATED,
        user_id=user.id,
        resource_type="license",
        resource_id=record.id,
        detail={"device": payload.device_name or payload.device_fingerprint[:12]},
    )
    return _to_response(record)


@router.post("/{license_id}/deactivate-device", response_model=MessageResponse)
async def deactivate_device(
    license_id: str,
    device_fingerprint: str,
    user: CurrentUser,
    session: SessionDep,
) -> MessageResponse:
    """Release a device slot."""
    record = await LicenseRepository(session).require_for_owner(license_id, user.id)
    device = next(
        (d for d in record.devices if d.device_fingerprint == device_fingerprint), None
    )
    if device is None or device.deactivated_at is not None:
        raise NotFoundError("That device is not activated on this license")

    device.deactivated_at = utcnow()
    await session.flush()
    await AuditLogRepository(session).record(
        action=AuditAction.LICENSE_DEACTIVATED,
        user_id=user.id,
        resource_type="license",
        resource_id=license_id,
        detail={"device": device_fingerprint[:12]},
    )
    return MessageResponse(message="Device deactivated; the slot is now free")


@router.get("/{license_id}/devices", response_model=list[dict])
async def list_devices(
    license_id: str, user: CurrentUser, session: SessionDep
) -> list[dict[str, Any]]:
    record = await LicenseRepository(session).require_for_owner(license_id, user.id)
    return [
        {
            "device_fingerprint": mask_secret(d.device_fingerprint, visible=6),
            "device_name": d.device_name,
            "platform": d.platform,
            "app_version": d.app_version,
            "activated_at": d.activated_at.isoformat() if d.activated_at else None,
            "last_seen_at": d.last_seen_at.isoformat() if d.last_seen_at else None,
            "active": d.deactivated_at is None,
        }
        for d in record.devices
    ]


@router.get("/subscription/current", response_model=dict)
async def current_subscription(
    user: CurrentUser, session: SessionDep
) -> dict[str, Any]:
    record = await SubscriptionRepository(session).current_for_owner(user.id)
    if record is None:
        return {"subscription": None, "message": "No active subscription"}
    return {
        "subscription": SubscriptionResponse(
            id=record.id,
            plan=record.plan,
            status=record.status.value,
            current_period_end=record.current_period_end,
            cancel_at_period_end=record.cancel_at_period_end,
        ).model_dump(mode="json"),
        "message": "Subscription active",
    }


@router.post("/trial", response_model=LicenseResponse, status_code=status.HTTP_201_CREATED)
async def start_trial(user: CurrentUser, session: SessionDep) -> LicenseResponse:
    """Issue a self-hosted trial license.

    Present so a fresh install has a working licensing path without a license server. A
    commercial deployment issues keys from ``license-server/`` instead and would disable this
    endpoint.
    """
    from app.core.crypto import generate_token

    repository = LicenseRepository(session)
    existing = await repository.list_for_owner(user.id, limit=1)
    if existing:
        raise LicenseInvalidError("This account already has a license")

    record = License(
        license_key=f"TRIAL-{generate_token(12).upper()[:20]}",
        user_id=user.id,
        plan=LicensePlan.TRIAL,
        status=LicenseStatus.ACTIVE,
        issued_at=utcnow(),
        expires_at=utcnow() + timedelta(days=14),
        device_limit=LicensePlan.TRIAL.default_device_limit,
        notes="self-hosted trial",
    )
    await repository.add(record)
    await session.refresh(record)
    return _to_response(record)
