# mypy: ignore-errors
"""Server-side config for the dream→skills lane (runs inside the synapse image / dream container).

Unlike the plugin's client config.py, this reads everything from the process env (compose
`env_file`), has **no on-disk skill catalog** (the lane reads its catalog from the
`skills_lane.skill_registry` table — the dream container has no `~/.claude/skills`), and no
client credential / ingest-URL handling. The DSN and API keys come from
`ingestion.settings` (env, then the one .env fallback).
"""

from __future__ import annotations

import os

DISCORD_WEBHOOK = os.environ.get("SKILLS_DISCORD_WEBHOOK", "")
# The only project exclusion: every detector and the config lane skip these project ids, and no
# project is excluded by default.
EXCLUDE_PROJECTS = tuple(
    p.strip() for p in os.environ.get("SKILLS_EXCLUDE_PROJECTS", "").split(",") if p.strip()
)
