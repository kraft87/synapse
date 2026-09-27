"""Gateway configuration from the environment, plus the secret redaction it implies.

Every knob is an env var so the gateway runs the same way from a shell, a systemd unit, or a
compose service. Research credentials come from ``<PREFIX>_API_KEY`` or, preferably,
``<PREFIX>_API_KEY_FILE`` (a docker/systemd secret path) and are never part of the Synapse
identity: the Synapse upstream only ever sees the CALLER's own bearer.

A research upstream is enabled by setting its URL. Hosted research MCP servers differ in
where they want the key, so both shapes are supported and exactly one must be used:

* ``{api_key}`` placeholder in the URL (URL-quoted on substitution), or
* ``<PREFIX>_AUTH_HEADER`` naming the header (``Authorization`` gets ``Bearer <key>``).

A keyed URL is a credential. Nothing here logs one: :func:`redact_url` keeps scheme + host,
and :class:`SecretRedactor` scrubs every configured key out of messages and tracebacks.
"""

from __future__ import annotations

import logging
import os
import traceback
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

ENV_PREFIX = "SYNAPSE_GATEWAY_"

#: Research upstreams the gateway knows how to configure, in listing order. The name is
#: also the tool namespace (``exa_<tool>``), so it is part of the public tool surface.
RESEARCH_UPSTREAMS = ("exa", "firecrawl")

#: Namespace for the Synapse upstream's tools (``synapse_recall``, ...). Resources are NOT
#: namespaced: ``skill://`` URIs stay canonical so stock ``sync_skills`` keeps working.
SYNAPSE_NAMESPACE = "synapse"

TRUST_LEVELS = ("full", "restricted")


class ConfigError(ValueError):
    """Invalid gateway configuration (raised at startup, never at request time)."""


@dataclass(frozen=True)
class ResearchUpstream:
    name: str
    url_template: str
    api_key: str = field(default="", repr=False)
    auth_header: str = ""
    discovery_timeout: float = 5.0
    call_timeout: float = 120.0

    def url(self) -> str:
        """The real endpoint. Contains the key when the provider wants it in the URL."""
        if "{api_key}" in self.url_template:
            return self.url_template.replace("{api_key}", quote(self.api_key, safe=""))
        return self.url_template

    def headers(self) -> dict[str, str]:
        if not (self.auth_header and self.api_key):
            return {}
        if self.auth_header.lower() == "authorization":
            return {"Authorization": f"Bearer {self.api_key}"}
        return {self.auth_header: self.api_key}

    @property
    def display_url(self) -> str:
        return redact_url(self.url_template)


@dataclass(frozen=True)
class GatewaySettings:
    synapse_url: str = "http://127.0.0.1:8765"
    host: str = "127.0.0.1"
    port: int = 8766
    public_url: str = ""
    #: Minimum Synapse trust a caller needs to see research tools and the research skill.
    research_trust: str = "full"
    #: How long a verified caller identity is reused before asking Synapse again.
    auth_cache_ttl: float = 30.0
    #: Per-identity lifetime of the Synapse component listing (tools/resources/templates).
    synapse_cache_ttl: float = 30.0
    #: Shared lifetime of research component listings (gateway-owned credential).
    research_cache_ttl: float = 300.0
    #: After a failed research discovery, skip that upstream for this long.
    failure_backoff: float = 30.0
    synapse_discovery_timeout: float = 10.0
    synapse_call_timeout: float = 120.0
    whoami_timeout: float = 5.0
    local_skills: bool = True
    research: tuple[ResearchUpstream, ...] = ()

    @property
    def synapse_mcp_url(self) -> str:
        return self.synapse_url.rstrip("/") + "/mcp"

    @property
    def whoami_url(self) -> str:
        return self.synapse_url.rstrip("/") + "/auth/whoami"

    def secrets(self) -> list[str]:
        return [u.api_key for u in self.research if u.api_key]


def redact_url(url: str) -> str:
    """``scheme://host[:port]/…`` — enough to identify an upstream, never a keyed path."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<invalid url>"
    if not parts.scheme or not parts.hostname:
        return "<invalid url>"
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{parts.hostname}{port}/…"


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
    if value < 0:
        raise ConfigError(f"{ENV_PREFIX}{key} must be >= 0")
    return value


def _secret(env: Mapping[str, str], key: str) -> str:
    """``<key>`` or the contents of ``<key>_FILE``; setting both is ambiguous and refused."""
    inline = _get(env, key)
    path = _get(env, key + "_FILE")
    if inline and path:
        raise ConfigError(f"set only one of {ENV_PREFIX}{key} and {ENV_PREFIX}{key}_FILE")
    if path:
        try:
            return Path(path).expanduser().read_text(encoding="utf-8").strip()
        except OSError as e:
            raise ConfigError(f"cannot read {ENV_PREFIX}{key}_FILE ({e.strerror})") from None
    return inline


def _research(
    env: Mapping[str, str], name: str, discovery_timeout: float
) -> ResearchUpstream | None:
    key = name.upper()
    url = _get(env, f"{key}_URL")
    if not url:
        return None
    if not url.startswith(("https://", "http://")):
        raise ConfigError(f"{ENV_PREFIX}{key}_URL must be an http(s) URL")
    api_key = _secret(env, f"{key}_API_KEY")
    header = _get(env, f"{key}_AUTH_HEADER")
    templated = "{api_key}" in url
    if templated and header:
        raise ConfigError(
            f"{name}: use the URL {{api_key}} placeholder OR an auth header, not both"
        )
    if (templated or header) and not api_key:
        raise ConfigError(f"{name}: {ENV_PREFIX}{key}_API_KEY(_FILE) is required")
    if api_key and not (templated or header):
        raise ConfigError(
            f"{name}: an API key is set but would never be sent; add {{api_key}} to "
            f"{ENV_PREFIX}{key}_URL or set {ENV_PREFIX}{key}_AUTH_HEADER"
        )
    return ResearchUpstream(
        name=name,
        url_template=url,
        api_key=api_key,
        auth_header=header,
        discovery_timeout=_float(env, f"{key}_DISCOVERY_TIMEOUT", discovery_timeout),
        call_timeout=_float(env, f"{key}_CALL_TIMEOUT", 120.0),
    )


def load_settings(env: Mapping[str, str] | None = None) -> GatewaySettings:
    env = os.environ if env is None else env
    research_trust = _get(env, "RESEARCH_TRUST", "full").lower()
    if research_trust not in TRUST_LEVELS:
        raise ConfigError(f"{ENV_PREFIX}RESEARCH_TRUST must be one of {TRUST_LEVELS}")
    synapse_url = _get(env, "SYNAPSE_URL", "http://127.0.0.1:8765").rstrip("/")
    if synapse_url.endswith("/mcp"):
        synapse_url = synapse_url[: -len("/mcp")]
    if not synapse_url.startswith(("https://", "http://")):
        raise ConfigError(f"{ENV_PREFIX}SYNAPSE_URL must be an http(s) base URL")
    try:
        port = int(_get(env, "PORT", "8766"))
    except ValueError as e:
        raise ConfigError(f"{ENV_PREFIX}PORT must be an integer") from e
    discovery = _float(env, "DISCOVERY_TIMEOUT", 5.0)
    research = tuple(
        u for u in (_research(env, name, discovery) for name in RESEARCH_UPSTREAMS) if u
    )
    return GatewaySettings(
        synapse_url=synapse_url,
        host=_get(env, "HOST", "127.0.0.1"),
        port=port,
        public_url=_get(env, "PUBLIC_URL"),
        research_trust=research_trust,
        auth_cache_ttl=_float(env, "AUTH_CACHE_TTL", 30.0),
        synapse_cache_ttl=_float(env, "SYNAPSE_CACHE_TTL", 30.0),
        research_cache_ttl=_float(env, "RESEARCH_CACHE_TTL", 300.0),
        failure_backoff=_float(env, "FAILURE_BACKOFF", 30.0),
        synapse_discovery_timeout=_float(env, "SYNAPSE_DISCOVERY_TIMEOUT", 10.0),
        synapse_call_timeout=_float(env, "SYNAPSE_CALL_TIMEOUT", 120.0),
        whoami_timeout=_float(env, "WHOAMI_TIMEOUT", 5.0),
        local_skills=_get(env, "LOCAL_SKILLS", "1") not in ("0", "false", "no"),
        research=research,
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
    request URL at INFO — a keyed research URL is a credential — so it is pinned to WARNING.
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
