"""Gateway settings: core knobs from the environment, integrations from the registry file.

Core settings (where Synapse is, where to listen, trust and cache knobs) are env vars so the
gateway runs the same from a shell, a systemd unit or a compose service. Everything beyond
Synapse's own memory and skills is the deployment's choice, declared in the JSON registry
named by ``SYNAPSE_GATEWAY_CONFIG_FILE`` (see :mod:`mcp_gateway.registry`). Without that file
the gateway serves Synapse memory + skills only.

Upstream secrets never touch the Synapse identity path: Synapse only ever sees the CALLER's
own bearer. Nothing here logs a secret or keyed URL: upstreams are displayed as
``scheme://host/…`` and :class:`SecretRedactor` scrubs every configured secret from
messages and tracebacks.
"""

from __future__ import annotations

import logging
import math
import os
import traceback
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from mcp_gateway.registry import (
    TRUST_LEVELS,
    Registry,
    RegistryError,
    load_registry,
    redact_url,
)

ENV_PREFIX = "SYNAPSE_GATEWAY_"

__all__ = ["ConfigError", "GatewaySettings", "load_settings", "redact_url"]


class ConfigError(ValueError):
    """Invalid gateway configuration (raised at startup, never at request time)."""


@dataclass(frozen=True)
class GatewaySettings:
    synapse_url: str = "http://127.0.0.1:8765"
    host: str = "127.0.0.1"
    port: int = 8766
    public_url: str = ""
    #: Minimum Synapse trust a caller needs to see Synapse's skill:// resources. Synapse's
    #: skills provider is not caller-scoped (every active skill, personal ones included),
    #: so the gateway withholds the whole class from restricted devices by default.
    skills_trust: str = "full"
    #: How long a verified caller identity is reused before asking Synapse again.
    auth_cache_ttl: float = 30.0
    #: Per-identity lifetime of the Synapse component listing (tools/resources/templates).
    synapse_cache_ttl: float = 30.0
    synapse_discovery_timeout: float = 10.0
    synapse_call_timeout: float = 120.0
    whoami_timeout: float = 5.0
    #: Deployment-configured upstreams and skill directories (empty: Synapse only).
    registry: Registry = field(default_factory=Registry)

    @property
    def synapse_mcp_url(self) -> str:
        return self.synapse_url.rstrip("/") + "/mcp"

    @property
    def whoami_url(self) -> str:
        return self.synapse_url.rstrip("/") + "/auth/whoami"

    def secrets(self) -> list[str]:
        return self.registry.secrets()


def _get(env: Mapping[str, str], key: str, default: str = "") -> str:
    return (env.get(ENV_PREFIX + key) or default).strip()


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = _get(env, key)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as e:
        raise ConfigError(f"{ENV_PREFIX}{key} must be a number") from e
    if not math.isfinite(value) or value < 0:
        raise ConfigError(f"{ENV_PREFIX}{key} must be finite and >= 0")
    return value


def _trust(env: Mapping[str, str], key: str) -> str:
    value = _get(env, key, "full").lower()
    if value not in TRUST_LEVELS:
        raise ConfigError(f"{ENV_PREFIX}{key} must be one of {TRUST_LEVELS}")
    return value


def load_settings(env: Mapping[str, str] | None = None) -> GatewaySettings:
    env = os.environ if env is None else env
    synapse_url = _get(env, "SYNAPSE_URL", "http://127.0.0.1:8765").rstrip("/")
    if synapse_url.endswith("/mcp"):
        synapse_url = synapse_url[: -len("/mcp")]
    if not synapse_url.startswith(("https://", "http://")):
        raise ConfigError(f"{ENV_PREFIX}SYNAPSE_URL must be an http(s) base URL")
    try:
        port = int(_get(env, "PORT", "8766"))
    except ValueError as e:
        raise ConfigError(f"{ENV_PREFIX}PORT must be an integer") from e
    config_file = _get(env, "CONFIG_FILE")
    try:
        registry = load_registry(config_file, env) if config_file else Registry()
    except RegistryError as e:
        raise ConfigError(str(e)) from None
    return GatewaySettings(
        synapse_url=synapse_url,
        host=_get(env, "HOST", "127.0.0.1"),
        port=port,
        public_url=_get(env, "PUBLIC_URL"),
        skills_trust=_trust(env, "SKILLS_TRUST"),
        auth_cache_ttl=_float(env, "AUTH_CACHE_TTL", 30.0),
        synapse_cache_ttl=_float(env, "SYNAPSE_CACHE_TTL", 30.0),
        synapse_discovery_timeout=_float(env, "SYNAPSE_DISCOVERY_TIMEOUT", 10.0),
        synapse_call_timeout=_float(env, "SYNAPSE_CALL_TIMEOUT", 120.0),
        whoami_timeout=_float(env, "WHOAMI_TIMEOUT", 5.0),
        registry=registry,
    )


class SecretRedactor:
    """Replace configured secrets with ``***`` in arbitrary text."""

    def __init__(self, secrets: Iterable[str]) -> None:
        # Longest first so a secret that contains another is scrubbed whole. The URL-quoted
        # form is included because a keyed URL carries the quoted key, not the raw one.
        forms = {f for s in secrets if s for f in (s, quote(s, safe=""))}
        self._secrets = sorted(forms, key=len, reverse=True)

    def __call__(self, text: str) -> str:
        for s in self._secrets:
            if s in text:
                text = text.replace(s, "***")
        return text


def _scrub(record: logging.LogRecord, redact: SecretRedactor) -> None:
    """Rewrite a record in place iff its message or traceback contains a secret.

    Records without secrets are left untouched (uvicorn's access formatter unpacks
    ``record.args``). A traceback is flattened into the message because handlers such as
    FastMCP's RichHandler render ``exc_info`` themselves, bypassing any Formatter.
    """
    try:
        text = record.getMessage()
    except Exception:  # malformed %-args: leave it to logging's own error path
        return
    if record.exc_info and record.exc_info[1] is not None:
        text += "\n" + "".join(traceback.format_exception(*record.exc_info))
    if record.stack_info:
        text += "\n" + record.stack_info
    clean = redact(text)
    if clean == text:
        return
    record.msg, record.args = clean, None
    record.exc_info = record.exc_text = record.stack_info = None


def install_log_redaction(secrets: Iterable[str]) -> None:
    """Scrub configured secrets from every log record, whichever logger or handler emits it.

    Implemented as a LogRecord factory, so it also covers handlers attached later (uvicorn
    configures its own at startup; FastMCP's do not propagate to root). httpx logs each
    request URL at INFO — a keyed upstream URL is a credential — so it is pinned to WARNING.
    Calling it again replaces the secret set rather than stacking factories.
    """
    global _BASE_FACTORY
    redact = SecretRedactor(secrets)
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if _BASE_FACTORY is None:
        _BASE_FACTORY = logging.getLogRecordFactory()
    base = _BASE_FACTORY

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = base(*args, **kwargs)
        _scrub(record, redact)
        return record

    logging.setLogRecordFactory(factory)


_BASE_FACTORY: Callable[..., logging.LogRecord] | None = None
