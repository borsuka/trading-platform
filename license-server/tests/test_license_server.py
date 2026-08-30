"""License server tests.

The properties that matter: an unknown key and a revoked key are indistinguishable to a caller
(so keys cannot be enumerated), device limits are enforced, re-activating the same device is
idempotent, and administrative endpoints are closed when no admin token is configured.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

ADMIN_TOKEN = "test-admin-token"
SIGNING_SECRET = "test-signing-secret"


@pytest.fixture
def client(tmp_path, monkeypatch) -> Iterator[TestClient]:
    monkeypatch.setenv("LICENSE_DB_PATH", str(tmp_path / "license.db"))
    monkeypatch.setenv("LICENSE_ADMIN_TOKEN", ADMIN_TOKEN)
    monkeypatch.setenv("LICENSE_SIGNING_SECRET", SIGNING_SECRET)

    import app.main as module

    importlib.reload(module)
    with TestClient(module.app) as test_client:
        yield test_client


@pytest.fixture
def admin() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


def issue(client: TestClient, admin: dict[str, str], plan: str = "pro") -> str:
    response = client.post(
        "/licenses", headers=admin, json={"email": "user@example.com", "plan": plan}
    )
    assert response.status_code == 200, response.text
    return response.json()["license_key"]


class TestHealth:
    def test_reports_configuration(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["signing_configured"] is True
        assert body["admin_configured"] is True


class TestAdminAuth:
    def test_issue_requires_admin(self, client: TestClient) -> None:
        response = client.post(
            "/licenses", json={"email": "user@example.com", "plan": "pro"}
        )
        assert response.status_code == 401

    def test_wrong_token_refused(self, client: TestClient) -> None:
        response = client.post(
            "/licenses",
            headers={"Authorization": "Bearer wrong"},
            json={"email": "user@example.com", "plan": "pro"},
        )
        assert response.status_code == 401

    def test_admin_disabled_when_unconfigured(self, tmp_path, monkeypatch) -> None:
        """No admin token means the endpoints are closed, not open."""
        monkeypatch.setenv("LICENSE_DB_PATH", str(tmp_path / "l.db"))
        monkeypatch.setenv("LICENSE_ADMIN_TOKEN", "")

        import app.main as module

        importlib.reload(module)
        with TestClient(module.app) as test_client:
            response = test_client.post(
                "/licenses", json={"email": "user@example.com", "plan": "pro"}
            )
            assert response.status_code == 503
            assert "disabled" in response.json()["detail"]


class TestIssuance:
    def test_issues_a_key_with_plan_defaults(
        self, client: TestClient, admin: dict[str, str]
    ) -> None:
        body = client.post(
            "/licenses", headers=admin, json={"email": "user@example.com", "plan": "pro"}
        ).json()
        assert body["license_key"].startswith("PRO-")
        assert body["device_limit"] == 3
        assert body["max_bots"] == 10
        assert body["expires_at"]

    def test_custom_duration(self, client: TestClient, admin: dict[str, str]) -> None:
        body = client.post(
            "/licenses",
            headers=admin,
            json={"email": "user@example.com", "plan": "starter", "days": 90},
        ).json()
        assert body["expires_at"]

    def test_invalid_plan_rejected(self, client: TestClient, admin: dict[str, str]) -> None:
        response = client.post(
            "/licenses", headers=admin, json={"email": "u@example.com", "plan": "unlimited"}
        )
        assert response.status_code == 422


class TestActivation:
    def test_activates_a_device(self, client: TestClient, admin: dict[str, str]) -> None:
        key = issue(client, admin)
        response = client.post(
            "/activate",
            json={
                "license_key": key,
                "device_fingerprint": "device-fingerprint-1",
                "device_name": "Trading VPS",
                "platform": "linux",
                "app_version": "1.0.0",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["activated"] is True
        assert body["data"]["active_devices"] == 1
        assert body["signature"]

    def test_reactivation_is_idempotent(
        self, client: TestClient, admin: dict[str, str]
    ) -> None:
        """Reinstalling on the same machine must not consume another slot."""
        key = issue(client, admin, plan="starter")  # device_limit = 1
        payload = {"license_key": key, "device_fingerprint": "same-device"}
        first = client.post("/activate", json=payload)
        second = client.post("/activate", json=payload)
        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json()["data"]["active_devices"] == 1

    def test_device_limit_enforced(
        self, client: TestClient, admin: dict[str, str]
    ) -> None:
        key = issue(client, admin, plan="starter")  # device_limit = 1
        client.post(
            "/activate", json={"license_key": key, "device_fingerprint": "device-one"}
        )
        response = client.post(
            "/activate", json={"license_key": key, "device_fingerprint": "device-two"}
        )
        assert response.status_code == 409
        assert "device(s)" in response.json()["detail"]

    def test_unknown_key_refused(self, client: TestClient) -> None:
        response = client.post(
            "/activate",
            json={"license_key": "NOPE-000000", "device_fingerprint": "device-xxxx"},
        )
        assert response.status_code == 403
        assert "not recognised" in response.json()["detail"]

    def test_revoked_key_is_indistinguishable_from_unknown(
        self, client: TestClient, admin: dict[str, str]
    ) -> None:
        """Different messages here would let an attacker enumerate valid keys."""
        key = issue(client, admin)
        client.post(f"/licenses/{key}/revoke", headers=admin, params={"reason": "test"})

        revoked = client.post(
            "/activate", json={"license_key": key, "device_fingerprint": "device-aaaa"}
        )
        unknown = client.post(
            "/activate",
            json={
                "license_key": "PRO-DOESNOTEXIST",
                "device_fingerprint": "device-aaaa",
            },
        )
        assert revoked.status_code == unknown.status_code == 403


class TestValidation:
    def test_valid_license_and_device(
        self, client: TestClient, admin: dict[str, str]
    ) -> None:
        key = issue(client, admin)
        client.post(
            "/activate", json={"license_key": key, "device_fingerprint": "device-1"}
        )
        body = client.post(
            "/validate",
            json={
                "license_key": key,
                "device_fingerprint": "device-1",
                "app_version": "1.0.1",
            },
        ).json()
        assert body["data"]["valid"] is True
        assert body["data"]["days_remaining"] is not None
        assert body["signature"]

    def test_unactivated_device_is_invalid(
        self, client: TestClient, admin: dict[str, str]
    ) -> None:
        key = issue(client, admin)
        body = client.post(
            "/validate", json={"license_key": key, "device_fingerprint": "never-activated"}
        ).json()
        assert body["data"]["valid"] is False
        assert "not activated" in body["data"]["reason"]

    def test_validation_of_unknown_key_returns_200_not_error(
        self, client: TestClient
    ) -> None:
        """Validation is a status check, not an authorisation.

        A client polls it on a schedule; returning an error status would make routine
        monitoring noisy and encourage clients to ignore it.
        """
        response = client.post(
            "/validate", json={"license_key": "NOPE-000000", "device_fingerprint": "device-aaaa"}
        )
        assert response.status_code == 200
        assert response.json()["data"]["valid"] is False

    def test_signature_is_verifiable(
        self, client: TestClient, admin: dict[str, str]
    ) -> None:
        import hashlib
        import hmac
        import json

        key = issue(client, admin)
        client.post("/activate", json={"license_key": key, "device_fingerprint": "device-0001"})
        body = client.post(
            "/validate", json={"license_key": key, "device_fingerprint": "device-0001"}
        ).json()

        canonical = json.dumps(body["data"], sort_keys=True, separators=(",", ":"))
        expected = hmac.new(
            SIGNING_SECRET.encode(), canonical.encode(), hashlib.sha256
        ).hexdigest()
        assert body["signature"] == expected


class TestDeactivation:
    def test_frees_a_slot(self, client: TestClient, admin: dict[str, str]) -> None:
        key = issue(client, admin, plan="starter")
        client.post("/activate", json={"license_key": key, "device_fingerprint": "device-old-1"})

        blocked = client.post(
            "/activate", json={"license_key": key, "device_fingerprint": "device-new-1"}
        )
        assert blocked.status_code == 409

        client.post(
            "/deactivate", json={"license_key": key, "device_fingerprint": "device-old-1"}
        )
        allowed = client.post(
            "/activate", json={"license_key": key, "device_fingerprint": "device-new-1"}
        )
        assert allowed.status_code == 200

    def test_deactivating_unknown_device(
        self, client: TestClient, admin: dict[str, str]
    ) -> None:
        key = issue(client, admin)
        response = client.post(
            "/deactivate", json={"license_key": key, "device_fingerprint": "device-absent"}
        )
        assert response.status_code == 404


class TestAdminInspection:
    def test_masks_device_fingerprints(
        self, client: TestClient, admin: dict[str, str]
    ) -> None:
        key = issue(client, admin)
        client.post(
            "/activate",
            json={
                "license_key": key,
                "device_fingerprint": "a-very-long-device-fingerprint-value",
            },
        )
        body = client.get(f"/licenses/{key}", headers=admin).json()
        assert body["license"]["license_key"] == key
        assert len(body["devices"]) == 1
        assert body["devices"][0]["device_fingerprint"].endswith("...")

    def test_unknown_license(self, client: TestClient, admin: dict[str, str]) -> None:
        assert client.get("/licenses/NOPE", headers=admin).status_code == 404
