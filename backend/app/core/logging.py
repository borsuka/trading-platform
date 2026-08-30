"""Structured logging with mandatory secret redaction.

The redaction processor is not optional and not configurable away: it runs on every event
before rendering. Trading logs routinely carry request payloads and exchange responses, so a
single unredacted field would be enough to leak an API secret into a log file that customers
routinely email to support.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import structlog
from structlog.typing import EventDict, WrappedLogger

REDACTED = "***REDACTED***"

#: Field names whose values are always replaced.
SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "password",
        "new_password",
        "old_password",
        "password_hash",
        "secret",
        "secret_key",
        "api_secret",
        "apisecret",
        "api_key",
        "apikey",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "authorization",
        "auth",
        "cookie",
        "set-cookie",
        "session",
        "private_key",
        "encryption_key",
        "license_key",
        "signature",
        "sign",
        "webhook_url",
        "bot_token",
        "credentials",
        "mnemonic",
        "seed",
    }
)

#: Patterns for secrets that appear inside free-text messages.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(api[_-]?(?:key|secret)|secret|token|password)\b\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]+=*"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),  # JWT
)

_MAX_DEPTH = 6


def _redact_value(value: Any, depth: int = 0) -> Any:
    if depth >= _MAX_DEPTH:
        return value
    if isinstance(value, Mapping):
        return {k: _redact_pair(str(k), v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)) and not isinstance(value, (str, bytes)):
        rendered = [_redact_value(item, depth + 1) for item in value]
        return type(value)(rendered) if isinstance(value, (list, tuple)) else rendered
    if isinstance(value, str):
        return _scrub_text(value)
    return value


def _redact_pair(key: str, value: Any, depth: int) -> Any:
    if is_sensitive_key(key):
        return REDACTED
    return _redact_value(value, depth)


def is_sensitive_key(key: str) -> bool:
    """True when a field name should never have its value logged."""
    lowered = key.lower().replace("-", "_")
    if lowered in SENSITIVE_KEYS:
        return True
    return any(marker in lowered for marker in ("password", "secret", "api_key", "token"))


def _scrub_text(text: str) -> str:
    if len(text) > 20_000:  # avoid pathological regex cost on huge payloads
        text = text[:20_000] + "...<truncated>"
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: f"{m.group(0).split(':')[0].split('=')[0]}={REDACTED}", text)
    return text


def redact_processor(
    _logger: WrappedLogger, _name: str, event_dict: EventDict
) -> MutableMapping[str, Any]:
    """structlog processor that removes secrets from every event."""
    scrubbed: dict[str, Any] = {}
    for key, value in event_dict.items():
        if key == "event" and isinstance(value, str):
            scrubbed[key] = _scrub_text(value)
        else:
            scrubbed[key] = _redact_pair(str(key), value, 0)
    return scrubbed


def _add_service_context(
    _logger: WrappedLogger, _name: str, event_dict: EventDict
) -> MutableMapping[str, Any]:
    from app.config import get_settings

    settings = get_settings()
    event_dict.setdefault("service", settings.app_name)
    event_dict.setdefault("version", settings.app_version)
    event_dict.setdefault("mode", settings.describe_mode())
    return event_dict


_configured = False


def configure_logging(
    *,
    level: str | None = None,
    json_output: bool | None = None,
    force: bool = False,
) -> None:
    """Configure structlog + stdlib logging. Idempotent unless ``force`` is set."""
    global _configured
    if _configured and not force:
        return

    from app.config import get_settings

    settings = get_settings()
    resolved_level = (level or settings.log_level).upper()
    resolved_json = settings.log_json if json_output is None else json_output

    shared: Sequence[Any] = (
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _add_service_context,
        redact_processor,
    )

    renderer: Any
    if resolved_json:
        renderer = structlog.processors.JSONRenderer()
        formatter_processors = [structlog.processors.format_exc_info, renderer]
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
        formatter_processors = [structlog.dev.set_exc_info, renderer]

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=list(shared),
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *formatter_processors,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(resolved_level)

    # Third-party loggers that are noisy or that echo payloads.
    for noisy in ("uvicorn.access", "sqlalchemy.engine", "httpx", "httpcore", "websockets"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, logging.getLevelName(resolved_level))
                                          if resolved_level == "DEBUG" else logging.WARNING)

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    configure_logging()
    return structlog.stdlib.get_logger(name or "app")
