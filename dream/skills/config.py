# mypy: ignore-errors
"""Server-side config for the dream→skills lane (runs inside the synapse image / dream container).

Unlike the plugin's client config.py, this reads everything from the process env (compose
`env_file`), has **no on-disk skill catalog** (the lane reads its catalog from the
`skills_lane.skill_registry` table — the dream container has no `~/.claude/skills`), and no
client credential / ingest-URL handling. `SYNAPSE_ENV_FILE` is honored only as an optional
fallback for the DSN / API keys; compose normally injects them directly.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

DISCORD_WEBHOOK = os.environ.get("SKILLS_DISCORD_WEBHOOK", "")
EXCLUDE_PROJECTS = tuple(
    p.strip() for p in os.environ.get("SKILLS_EXCLUDE_PROJECTS", "").split(",") if p.strip()
)

# Optional .env fallback (compose injects env directly, so this rarely fires server-side).
ENV_FILE = Path(os.environ.get("SYNAPSE_ENV_FILE", "/app/.env"))


def _from_env_file(key: str) -> str | None:
    try:
        if ENV_FILE.exists():
            m = re.search(rf"^{re.escape(key)}=(\S+)", ENV_FILE.read_text(), re.MULTILINE)
            return m.group(1) if m else None
    except OSError:
        return None
    return None


def db_url() -> str:
    """Postgres DSN for the ledger + episode source. Env wins, else the .env fallback."""
    return os.environ.get("SYNAPSE_DB_URL") or _from_env_file("SYNAPSE_DB_URL") or ""


def secret(key: str) -> str:
    """Resolve an API key (VOYAGE_API_KEY, ...): env then .env fallback."""
    return os.environ.get(key) or _from_env_file(key) or ""
