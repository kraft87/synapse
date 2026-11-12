"""This machine's Synapse settings as one immutable value.

Every setting resolves env var -> ~/.synapse/config.json -> the caller's extra sources
-> default, the order plugin/scripts/config.py uses. config.json is the file `synapse
login` and `synapse config` write (``SYNAPSE_HOME`` moves it); its keys map onto the
env-style names in ``KEYS``. The extra sources are how a harness adapter contributes a
legacy store of its own (the Claude plugin's saved options) without the core knowing
about it.

``Settings.load`` reads the environment and config.json once and resolves the URL, the
token and the enrollment state from that snapshot. Nothing here writes config.json, and
nothing is read at import: a host that serves several sessions can hold several values.
"""

from __future__ import annotations

import json
import os
import re
import types
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import credentials

#: config.json key -> the env-style name the rest of the client resolves by. Mirrors
#: plugin/scripts/synapse_home.py (a test keeps the two equal).
KEYS: Mapping[str, str] = types.MappingProxyType(
    {
        "url": "SYNAPSE_URL",
        "token": "SYNAPSE_INGEST_TOKEN",
        "role": "SYNAPSE_MACHINE_ROLE",
        "skills_sync": "SYNAPSE_SKILLS_SYNC",
        "config_sync": "SYNAPSE_CONFIG_SYNC",
        "config_paths": "SYNAPSE_CONFIG_PATHS",
    }
)
_BY_OPTION: Mapping[str, str] = types.MappingProxyType({opt: key for key, opt in KEYS.items()})
_TOKEN = "SYNAPSE_INGEST_TOKEN"
_ENDPOINTS = ("/ingest", "/recall", "/mcp", "/skills")
# A base URL every shell and harness config format carries without quoting: a scheme, a
# host[:port] and an optional plain path (plugin/scripts/harness.py).
_URL_RE = re.compile(
    r"^https?://(?:[A-Za-z0-9.\-]+|\[[0-9A-Fa-f:]+\])(?::\d{1,5})?(?:/[A-Za-z0-9._~\-/]*)?$"
)


def option(data: Mapping[str, Any], name: str) -> str:
    """``data``'s value for an env-style name (``SYNAPSE_URL``) as the string the env var
    would carry: booleans as "1"/"0", lists comma-joined, "" when unset."""
    key = _BY_OPTION.get(name)
    value = data.get(key) if key else None
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)


def has(data: Mapping[str, Any], name: str) -> bool:
    """Whether ``data`` sets the env-style name at all. An explicitly empty value still
    counts: it must not fall through to a legacy one."""
    key = _BY_OPTION.get(name)
    return key is not None and data.get(key) is not None


def base_url(url: str) -> str:
    """``url`` without a trailing slash or a pasted endpoint suffix (``/ingest``,
    ``/recall``, ``/mcp``, ``/skills``); "" stays "". Lenient, as config.py's
    ``_base_url`` is: an unusual URL is kept, not rejected."""
    out = (url or "").rstrip("/")
    for suffix in _ENDPOINTS:
        # a path segment only: a host that happens to be named "mcp" stays
        if out.endswith(suffix) and "://" in out[: -len(suffix)]:
            out = out[: -len(suffix)]
    return out.rstrip("/")


def normalize_url(url: str) -> str:
    """The base URL to store: no trailing slash, no endpoint suffix (``/install`` too).
    Raises ValueError when it isn't a plain http(s) URL. A port of
    plugin/scripts/harness.py's ``normalize_url``, for installers."""
    out = (url or "").strip().rstrip("/")
    for suffix in (*_ENDPOINTS, "/install"):
        if out.endswith(suffix) and "://" in out[: -len(suffix)]:
            out = out[: -len(suffix)]
    out = out.rstrip("/")
    if not _URL_RE.match(out):
        raise ValueError(f"not a plain http(s) base URL: {url!r}")
    return out


def _expand(text: str, environ: Mapping[str, str]) -> Path:
    """``~`` expanded against ``environ``'s home (HOME, USERPROFILE on Windows), so a
    caller that passes its own environment gets paths from it; else the process's."""
    user_home = environ.get("USERPROFILE" if os.name == "nt" else "HOME")
    if user_home and (text == "~" or text[:2] in ("~/", "~\\")):
        return Path(user_home + text[1:])
    return Path(os.path.expanduser(text))


def _read_config(path: Path) -> dict[str, Any]:
    """config.json parsed, or ``{}`` when it is missing, unreadable or not an object."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _frozen(data: Mapping[str, Any]) -> Mapping[str, Any]:
    return types.MappingProxyType(dict(data))


def _lookup(
    environ: Mapping[str, str],
    config: Mapping[str, Any],
    extras: Iterable[tuple[str, Mapping[str, str]]],
    name: str,
    default: str = "",
) -> str:
    """``Settings.flag`` over explicit sources (see there)."""
    value = environ.get(name)
    if value:
        return value
    if has(config, name):
        return option(config, name) or default
    for _source, values in extras:
        value = values.get(name)
        if value:
            return value
    return default


@dataclass(frozen=True)
class Settings:
    """Resolved settings for one Synapse client. Build it with ``Settings.load``.

    ``base_url`` is "" when no URL is configured anywhere (config.py falls back to
    http://localhost:8765; the core sends nothing instead). ``token`` is the bearer to
    send and ``token_note`` what it overrode (credentials.resolve). ``device`` is the
    enrollment record (config.json's, else a legacy one the caller passed) and
    ``enrolled`` whether it names a surface.
    ``device_origins`` are where the device token may go (empty when unenrolled).
    The token and the raw sources are left out of ``repr``.
    """

    home: Path
    base_url: str
    ingest_url: str
    token: str = field(repr=False)
    token_note: Mapping[str, Any]
    device: Mapping[str, Any]
    enrolled: bool
    private_dir: Path
    device_origins: frozenset[tuple[str, str]]
    environ: Mapping[str, str] = field(repr=False, compare=False)
    config: Mapping[str, Any] = field(repr=False, compare=False)
    extras: tuple[tuple[str, Mapping[str, str]], ...] = field(repr=False, compare=False)

    @property
    def config_file(self) -> Path:
        return self.home / "config.json"

    def flag(self, name: str, default: str = "") -> str:
        """The string config.py's ``_cfg(name, default)`` would return: a non-empty env
        value; else config.json's when it sets the name at all (``default`` when that is
        empty); else the first extra source with a non-empty value; else ``default``.

        Only ``SYNAPSE_*`` names are kept from the environment."""
        return _lookup(self.environ, self.config, self.extras, name, default)

    @classmethod
    def load(
        cls,
        env: Mapping[str, str] | None = None,
        home: str | os.PathLike[str] | None = None,
        extra_candidates: Iterable[tuple[str, Mapping[str, Any]]] = (),
        *,
        legacy_device: Mapping[str, Any] | None = None,
        legacy_token: str = "",
    ) -> Settings:
        """Read ``env`` (default: the process environment) and config.json, and resolve.

        ``home`` overrides the Synapse home (else ``SYNAPSE_HOME``, else ~/.synapse).
        ``extra_candidates`` are ``(source, values)`` pairs ranked below config.json, in
        order: ``values`` maps env-style names to values, and ``source`` is the name the
        token rules know it by (credentials.PLUGIN_SOURCES for the Claude plugin's
        stores). Empty values count as unset.

        ``legacy_device`` and ``legacy_token`` are the Claude plugin's pre-0.20 stores
        (its data dir's device.json and credentials.json), for an adapter that still
        honours them: the record is used when config.json holds none, the token is the
        last fallback. The core reads neither file itself.

        Never raises over a missing or malformed config.json: that reads as nothing
        configured, as it does for the hooks."""
        environ = os.environ if env is None else env
        snapshot = {k: v for k, v in environ.items() if k.startswith("SYNAPSE_")}
        home_path = (
            Path(home)
            if home is not None
            else _expand(snapshot.get("SYNAPSE_HOME") or "~/.synapse", environ)
        )
        config = _read_config(home_path / "config.json")
        extras = tuple(
            (str(source), {str(k): str(v) for k, v in values.items() if v not in (None, "")})
            for source, values in extra_candidates
        )
        url = base_url(
            _lookup(snapshot, config, extras, "SYNAPSE_URL")
            or _lookup(snapshot, config, extras, "SYNAPSE_INGEST_URL")
        )
        ingest = _lookup(snapshot, config, extras, "SYNAPSE_INGEST_URL") or (
            url + "/ingest" if url else ""
        )
        record = config.get("device")
        if not (isinstance(record, dict) and record):
            record = legacy_device
        device: dict[str, Any] = dict(record) if isinstance(record, Mapping) else {}
        candidates = [
            ("env", snapshot.get(_TOKEN, "")),
            ("config_file", option(config, _TOKEN)),
            *((source, values.get(_TOKEN, "")) for source, values in extras),
        ]
        plugin_origins = {
            credentials.origin(values[key])
            for source, values in extras
            if source in credentials.PLUGIN_SOURCES
            for key in ("SYNAPSE_URL", "SYNAPSE_INGEST_URL")
            if values.get(key)
        }
        saved = option(config, "SYNAPSE_URL")
        token, note = credentials.resolve(
            candidates,
            device,
            [url, ingest],
            saved_url=saved,
            plugin_origins=plugin_origins,
            legacy_token=legacy_token,
        )
        allowed = credentials.enrolled_origins(
            device, candidates, saved_url=saved, plugin_origins=plugin_origins
        )
        return cls(
            home=home_path,
            base_url=url,
            ingest_url=ingest,
            token=token,
            token_note=_frozen(note),
            device=_frozen(device),
            enrolled=bool(device.get("surface_id")),
            private_dir=_expand(
                snapshot.get("SYNAPSE_PRIVATE_DIR") or "~/.synapse/private", environ
            ),
            device_origins=frozenset(allowed),
            environ=_frozen(snapshot),
            config=_frozen(config),
            extras=tuple((source, _frozen(values)) for source, values in extras),
        )
