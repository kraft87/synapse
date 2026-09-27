"""The deployment's own MCP upstream registry: a small, strict JSON file.

Synapse memory and skills are built in. Everything else the gateway fronts is chosen by
the deployment and listed in the file named by ``SYNAPSE_GATEWAY_CONFIG_FILE``::

    {
      "upstreams": [
        {
          "namespace": "tracker",
          "description": "Project tracker: issues and milestones",
          "url": "https://tracker.example.com/mcp",
          "auth": {"type": "bearer", "secret_env": "TRACKER_MCP_TOKEN"},
          "min_trust": "full",
          "discovery_timeout": 5, "call_timeout": 120, "cache_ttl": 300, "failure_backoff": 30
        }
      ],
      "skills_dirs": [{"path": "/srv/team-skills", "min_trust": "full"}]
    }

Only streamable-HTTP MCP servers are supported, authenticated with a static secret sent as
a bearer token, a named header, or a ``{secret}`` placeholder in the URL. OAuth enrollment
and stdio servers are not supported. Secrets are REFERENCES only (``secret_env`` names an
environment variable, ``secret_file`` a file); an inline value is refused, and so is any
key the schema does not know, so a typo can't silently disable a setting and a pasted
credential can't hide in an unknown field. Errors name keys, variables and paths, never
values.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

TRUST_LEVELS = ("full", "restricted")

#: Namespaces that would shadow or be confused with the gateway's own surface: Synapse's
#: memory tools keep their unprefixed names (recall, recall_full_turns, fetch_session, …),
#: the resources bridge adds list_resources/read_resource, and issue_machine_token is a
#: hidden Synapse tool. ``<ns>_…`` must never be mistakable for any of them.
RESERVED_NAMESPACES = frozenset(
    {
        "synapse",
        "gateway",
        "skill",
        "skills",
        "mcp",
        "recall",
        "fetch",
        "remember",
        "list",
        "read",
        "issue",
    }
)
_NAMESPACE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_MAX_NAMESPACE = 32
_MAX_DESCRIPTION = 120
_HEADER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")
_FORBIDDEN_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "content-type",
        "transfer-encoding",
        "connection",
        "upgrade",
        "cookie",
        "mcp-session-id",
        "mcp-protocol-version",
    }
)
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PLACEHOLDER = "{secret}"
AUTH_TYPES = ("none", "bearer", "header", "url")

_TOP_KEYS = {"upstreams", "skills_dirs"}
_UPSTREAM_KEYS = {
    "namespace",
    "url",
    "auth",
    "min_trust",
    "discovery_timeout",
    "call_timeout",
    "cache_ttl",
    "failure_backoff",
    "description",
}
_AUTH_KEYS = {"type", "header", "secret_env", "secret_file"}
_SKILLS_DIR_KEYS = {"path", "min_trust"}


class RegistryError(ValueError):
    """The registry file is unusable; raised at startup, never at request time."""


@dataclass(frozen=True)
class Upstream:
    namespace: str
    url_template: str
    auth_type: str = "none"
    header: str = ""
    secret: str = field(default="", repr=False)
    min_trust: str = "full"
    discovery_timeout: float = 5.0
    call_timeout: float = 120.0
    cache_ttl: float = 300.0
    failure_backoff: float = 30.0
    description: str = ""

    def url(self) -> str:
        """The real endpoint; contains the secret for ``auth.type == "url"``."""
        if self.auth_type == "url":
            return self.url_template.replace(_PLACEHOLDER, quote(self.secret, safe=""))
        return self.url_template

    def headers(self) -> dict[str, str]:
        if self.auth_type == "bearer":
            return {"Authorization": f"Bearer {self.secret}"}
        if self.auth_type == "header":
            return {self.header: self.secret}
        return {}

    @property
    def display_url(self) -> str:
        return redact_url(self.url_template)


@dataclass(frozen=True)
class SkillsDir:
    path: Path
    min_trust: str = "full"


@dataclass(frozen=True)
class Registry:
    upstreams: tuple[Upstream, ...] = ()
    skills_dirs: tuple[SkillsDir, ...] = ()

    def secrets(self) -> list[str]:
        return [u.secret for u in self.upstreams if u.secret]


def redact_url(url: str) -> str:
    """``scheme://host[:port]/…`` — enough to identify an upstream, never a keyed path."""
    try:
        parts = urlsplit(url)
        port = f":{parts.port}" if parts.port else ""
    except ValueError:
        return "<invalid url>"
    if not parts.scheme or not parts.hostname:
        return "<invalid url>"
    return f"{parts.scheme}://{parts.hostname}{port}/…"


def _unknown(where: str, obj: Mapping[str, Any], allowed: set[str]) -> None:
    extra = sorted(set(obj) - allowed)
    if extra:
        # Key names only: an unknown key is exactly where a pasted credential would sit.
        raise RegistryError(f"{where}: unknown key(s) {extra}; allowed: {sorted(allowed)}")


def _number(where: str, obj: Mapping[str, Any], key: str, default: float) -> float:
    value = obj.get(key, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise RegistryError(f"{where}: {key} must be a finite non-negative number")
    return float(value)


def _trust(where: str, obj: Mapping[str, Any]) -> str:
    value = obj.get("min_trust", "full")
    if value not in TRUST_LEVELS:
        raise RegistryError(f"{where}: min_trust must be one of {list(TRUST_LEVELS)}")
    return str(value)


def _secret(where: str, auth: Mapping[str, Any], env: Mapping[str, str], base: Path) -> str:
    env_name, path = auth.get("secret_env"), auth.get("secret_file")
    if bool(env_name) == bool(path):
        raise RegistryError(f"{where}: set exactly one of secret_env or secret_file")
    if env_name:
        if not isinstance(env_name, str) or not _ENV_NAME.match(env_name):
            raise RegistryError(f"{where}: secret_env must be an environment variable name")
        value = env.get(env_name, "").strip()
        if not value:
            raise RegistryError(f"{where}: environment variable {env_name} is unset or empty")
        return value
    if not isinstance(path, str):
        raise RegistryError(f"{where}: secret_file must be a path")
    resolved = (base / os.path.expanduser(path)).resolve()
    try:
        value = resolved.read_text(encoding="utf-8").strip()
    except OSError as e:
        raise RegistryError(f"{where}: cannot read secret_file {resolved} ({e.strerror})") from None
    if not value:
        raise RegistryError(f"{where}: secret_file {resolved} is empty")
    return value


def _url(where: str, raw: Any, auth_type: str) -> str:
    if not isinstance(raw, str):
        raise RegistryError(f"{where}: url must be a string")
    try:
        parts = urlsplit(raw.replace(_PLACEHOLDER, "x"))
        _ = parts.port  # urlsplit validates the port only when this property is read.
    except ValueError:
        raise RegistryError(f"{where}: url is not a valid URL") from None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise RegistryError(f"{where}: url must be an http(s) URL (only HTTP MCP is supported)")
    if parts.username or parts.password:
        raise RegistryError(f"{where}: url must not embed credentials; use auth")
    if (_PLACEHOLDER in raw) != (auth_type == "url"):
        raise RegistryError(
            f"{where}: the url {_PLACEHOLDER} placeholder is required for auth type 'url' "
            "and not allowed otherwise"
        )
    return raw


def _upstream(i: int, raw: Any, env: Mapping[str, str], base: Path) -> Upstream:
    where = f"upstreams[{i}]"
    if not isinstance(raw, dict):
        raise RegistryError(f"{where}: must be an object")
    _unknown(where, raw, _UPSTREAM_KEYS)
    ns = raw.get("namespace")
    if not isinstance(ns, str) or not _NAMESPACE.match(ns) or len(ns) > _MAX_NAMESPACE:
        raise RegistryError(
            f"{where}: namespace must be 1-{_MAX_NAMESPACE} chars of lowercase letters, digits "
            "and single hyphens, starting with a letter (it prefixes tool names as <ns>_<tool>)"
        )
    if ns in RESERVED_NAMESPACES:
        raise RegistryError(f"{where}: namespace {ns!r} is reserved")
    where = f"upstream {ns!r}"
    auth = raw.get("auth", {"type": "none"})
    if not isinstance(auth, dict):
        raise RegistryError(f"{where}: auth must be an object")
    _unknown(f"{where} auth", auth, _AUTH_KEYS)
    auth_type = auth.get("type", "none")
    if auth_type not in AUTH_TYPES:
        raise RegistryError(f"{where}: auth.type must be one of {list(AUTH_TYPES)}")
    url = _url(where, raw.get("url"), auth_type)
    header, secret = "", ""
    if auth_type == "none":
        if set(auth) - {"type"}:
            raise RegistryError(f"{where}: auth type 'none' takes no other keys")
    else:
        secret = _secret(where, auth, env, base)
        if auth_type == "header":
            header = auth.get("header", "")
            if not isinstance(header, str) or not _HEADER.match(header):
                raise RegistryError(f"{where}: auth.header must be a header name")
            if header.lower() in _FORBIDDEN_HEADERS:
                raise RegistryError(f"{where}: auth.header {header!r} is not allowed")
        elif "header" in auth:
            raise RegistryError(f"{where}: auth.header only applies to type 'header'")
    description = raw.get("description", "")
    if not isinstance(description, str) or len(description) > _MAX_DESCRIPTION:
        raise RegistryError(
            f"{where}: description must be a string of at most {_MAX_DESCRIPTION} chars"
        )
    return Upstream(
        namespace=ns,
        description=description.strip(),
        url_template=url,
        auth_type=auth_type,
        header=header,
        secret=secret,
        min_trust=_trust(where, raw),
        discovery_timeout=_number(where, raw, "discovery_timeout", 5.0),
        call_timeout=_number(where, raw, "call_timeout", 120.0),
        cache_ttl=_number(where, raw, "cache_ttl", 300.0),
        failure_backoff=_number(where, raw, "failure_backoff", 30.0),
    )


def _skills_dir(i: int, raw: Any, base: Path) -> SkillsDir:
    where = f"skills_dirs[{i}]"
    if not isinstance(raw, dict):
        raise RegistryError(f"{where}: must be an object")
    _unknown(where, raw, _SKILLS_DIR_KEYS)
    path = raw.get("path")
    if not isinstance(path, str) or not path:
        raise RegistryError(f"{where}: path must be a directory path")
    resolved = (base / os.path.expanduser(path)).resolve()
    if not resolved.is_dir():
        raise RegistryError(f"{where}: {resolved} is not a directory")
    return SkillsDir(path=resolved, min_trust=_trust(where, raw))


def parse_registry(data: Any, env: Mapping[str, str], base: Path) -> Registry:
    if not isinstance(data, dict):
        raise RegistryError("registry must be a JSON object")
    _unknown("registry", data, _TOP_KEYS)
    ups_raw = data.get("upstreams", [])
    dirs_raw = data.get("skills_dirs", [])
    if not isinstance(ups_raw, list) or not isinstance(dirs_raw, list):
        raise RegistryError("registry: upstreams and skills_dirs must be lists")
    upstreams = tuple(_upstream(i, u, env, base) for i, u in enumerate(ups_raw))
    seen: set[str] = set()
    for u in upstreams:
        if u.namespace in seen:
            raise RegistryError(f"duplicate namespace {u.namespace!r}")
        seen.add(u.namespace)
    return Registry(
        upstreams=upstreams,
        skills_dirs=tuple(_skills_dir(i, d, base) for i, d in enumerate(dirs_raw)),
    )


def load_registry(path: str, env: Mapping[str, str]) -> Registry:
    """Parse the registry file; relative ``secret_file``/``path`` resolve against its dir."""
    file = Path(os.path.expanduser(path)).resolve()
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as e:
        raise RegistryError(f"cannot read registry {file} ({e.strerror})") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        # Position only: the decoder's message can quote the offending text.
        raise RegistryError(
            f"registry {file} is not valid JSON (line {e.lineno}, column {e.colno})"
        ) from None
    return parse_registry(data, env, file.parent)
