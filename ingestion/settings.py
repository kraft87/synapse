"""Process configuration, resolved one way everywhere: the environment, then one .env file.

compose injects every variable, so inside the containers the file is never consulted; it is
what a bare ``python -m ingestion``, a dream run or a script falls back to. An empty
environment variable counts as unset, so the file can still supply it. ``SYNAPSE_ENV_FILE``
moves the file; tests point it at a nonexistent path so a developer's repo-root ``.env``
cannot leak into an assertion about unset config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: The tenant every store scopes its rows under. One owner today; the whole system reads it
#: from this one constant so a multi-tenant split threads through a single place.
KG_OWNER_ID = os.environ.get("SYNAPSE_KG_OWNER_ID", "default")


def env_file_path() -> Path:
    """``SYNAPSE_ENV_FILE``, else the ``.env`` at the repo root (``/app/.env`` in the image)."""
    override = os.environ.get("SYNAPSE_ENV_FILE")
    return Path(override) if override else Path(__file__).resolve().parent.parent / ".env"


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value.split(" #", 1)[0].rstrip()


def env_file_values() -> dict[str, str]:
    """The ``KEY=value`` lines of the .env file; a missing file is empty. Later lines win,
    surrounding quotes and a trailing `` # comment`` are dropped, as compose does."""
    path = env_file_path()
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = _unquote(value.strip())
    return values


def setting(key: str, default: str = "") -> str:
    """The environment variable ``key``, else its .env line, else ``default``."""
    return os.environ.get(key) or env_file_values().get(key, default)


def require_setting(key: str) -> str:
    """:func:`setting`, or a ``RuntimeError`` naming ``key`` and the file consulted."""
    value = setting(key)
    if not value:
        raise RuntimeError(f"{key} is not set (environment, or {env_file_path()})")
    return value


@dataclass(frozen=True)
class Settings:
    """What the poller process needs to start."""

    db_url: str
    poll_interval_seconds: int
    voyage_api_key: str


def get_settings() -> Settings:
    raw = setting("POLL_INTERVAL_SECONDS", "300")
    try:
        interval = int(raw)
    except ValueError:
        raise ValueError(
            f"POLL_INTERVAL_SECONDS must be a whole number of seconds, got {raw!r}"
        ) from None
    return Settings(
        db_url=require_setting("SYNAPSE_DB_URL"),
        poll_interval_seconds=interval,
        voyage_api_key=setting("VOYAGE_API_KEY"),
    )
