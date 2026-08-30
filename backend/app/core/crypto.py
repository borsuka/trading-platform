"""Encryption at rest for exchange credentials.

Exchange API secrets are the most dangerous data the platform stores. They are encrypted with
Fernet (AES-128-CBC + HMAC-SHA256) before they touch the database, and the key never lives in
the same store as the ciphertext.

A deployment that has not configured ``ENCRYPTION_KEY`` cannot persist credentials at all: the
functions below raise rather than falling back to plaintext.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

from cryptography.fernet import Fernet, InvalidToken

from app.core.exceptions import ConfigurationError, TradingPlatformError


class DecryptionError(TradingPlatformError):
    error_code = "decryption_failed"
    status_code = 500
    default_public_message = "Stored credentials could not be decrypted."


def generate_encryption_key() -> str:
    """Generate a new Fernet key. Used by ``scripts/generate_keys.py``."""
    return Fernet.generate_key().decode()


def _build_fernet(key: str) -> Fernet:
    try:
        return Fernet(key.encode() if isinstance(key, str) else key)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(
            "ENCRYPTION_KEY is not a valid Fernet key. "
            "Generate one with: python -m app.cli generate-key"
        ) from exc


class CredentialCipher:
    """Encrypts and decrypts credential material.

    Instances are cheap; construct one per request or hold one per process, either is fine.
    """

    __slots__ = ("_fernet", "key")

    def __init__(self, key: str) -> None:
        self.key = key
        self._fernet = _build_fernet(key)

    def encrypt(self, plaintext: str) -> str:
        if not plaintext:
            raise ValueError("Refusing to encrypt an empty credential")
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        try:
            return self._fernet.decrypt(ciphertext.encode()).decode()
        except InvalidToken as exc:
            raise DecryptionError(
                "Credential ciphertext is invalid; the encryption key may have been rotated.",
            ) from exc

    def __repr__(self) -> str:  # pragma: no cover - never leak the key
        return "CredentialCipher(key=***REDACTED***)"


def get_cipher() -> CredentialCipher:
    """Build a cipher from settings, failing loudly when unconfigured."""
    from app.config import get_settings

    settings = get_settings()
    if settings.encryption_key is None:
        raise ConfigurationError(
            "ENCRYPTION_KEY is not configured. Exchange credentials cannot be stored "
            "without it; the platform refuses to fall back to plaintext."
        )
    return CredentialCipher(settings.encryption_key.get_secret_value())


def mask_secret(value: str, *, visible: int = 4) -> str:
    """Render a credential for display: ``"abcd...wxyz"``.

    Used by the UI and audit log. Never reveals enough to be useful to an attacker.
    """
    if not value:
        return ""
    if len(value) <= visible * 2:
        return "*" * len(value)
    return f"{value[:visible]}{'*' * 6}{value[-visible:]}"


def fingerprint(value: str) -> str:
    """Stable non-reversible identifier for a credential.

    Lets the platform say "this is the same key you connected before" without storing or
    comparing the secret itself.
    """
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def generate_token(length: int = 32) -> str:
    """Cryptographically secure URL-safe token (verification links, API tokens)."""
    return secrets.token_urlsafe(length)


def constant_time_compare(left: str, right: str) -> bool:
    """Timing-attack resistant string comparison."""
    return hmac.compare_digest(left.encode(), right.encode())


def sign_payload(payload: bytes, secret: str) -> str:
    """HMAC-SHA256 signature, base64url encoded. Used for license responses and webhooks."""
    digest = hmac.new(secret.encode(), payload, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def verify_payload(payload: bytes, secret: str, signature: str) -> bool:
    """Verify a signature produced by :func:`sign_payload`."""
    return constant_time_compare(sign_payload(payload, secret), signature)
