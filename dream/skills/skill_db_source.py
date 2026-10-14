# mypy: ignore-errors
"""DB source for the dream->skills lanes: skill fires and explicit skill requests in a
lane window, read from the Synapse `episodes` table instead of raw JSONL transcripts.

Why: the episodes table already holds what the lane needs as parseable markers in
`content` ([user]/[tool:NAME]/[result]/[assistant]) plus the `human_turn` column (full user
text, up to 3000 chars). One indexed query per window beats globbing 75k+ JSONL files,
gives richer user phrasings, and is the single canonical source. Session views (and the
window and cursor every stage reads) live in dream/lane.py.

KNOWN LOSS vs JSONL: ingestion drops `<command-name>` slash-command records as
machinery, so SLASH-invoked skill fires are NOT visible here (only Skill-*tool*
invocations survive, as `[tool:Skill] {'skill': 'name'}`). Fine for DERIVE (the
guard is the catalog comparison, not fire-detection); matters for RETUNE counting —
fix later with a `[skill:]` ingest marker or a tiny JSONL fire-scan. Thinking blocks
are also not stored. Pass source="jsonl" to skill-derive when you need either.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from dream import lane

from . import config


def _dsn() -> str:
    dsn = config.db_url()
    if not dsn:
        raise RuntimeError(f"SYNAPSE_DB_URL not configured (env or {config.ENV_FILE})")
    return dsn


def _connect(dsn: str):
    try:
        import psycopg

        return psycopg.connect(dsn, connect_timeout=8)
    except ImportError:
        import psycopg2

        return psycopg2.connect(dsn, connect_timeout=8)


def db_session_views(
    days: float,
    max_sessions: int,
    platform: str = "claude_code",
    exclude_projects: tuple[str, ...] = config.EXCLUDE_PROJECTS,
) -> list[dict]:
    """Session views for every session with a turn in the last `days` days (skill_derive's
    dev CLI; the nightly loads only the sessions its window touched, dream/lane.py).

    exclude_projects: drop sessions from foreign skill-environments (e.g. transcribe_ai
    has its own ~/.claude/skills, so its procedures aren't gaps in this catalog).
    """
    conn = _connect(_dsn())
    cur = conn.cursor()
    cur.execute(
        """
        SELECT session_id, sequence, human_turn, content
        FROM episodes
        WHERE platform = %s
          AND created_at > now() - (%s || ' days')::interval
          AND (project IS NULL OR project <> ALL(%s))
        ORDER BY session_id, sequence
        """,
        (platform, str(days), list(exclude_projects)),
    )
    views = lane.build_views(cur.fetchall())
    conn.close()
    views.sort(key=lambda s: s["user_turns"], reverse=True)
    return views[: max_sessions * 4]  # caller re-filters to substantive + caps


# DISMISSAL = a fire immediately followed by a SHORT, STRONG, leading negation. Strict on purpose:
# this is a GROUNDED signal that gates apply, so precision >> recall (the loose version mislabeled 73%).
_CORRECTION = re.compile(
    r"^\s*(no\b|nope\b|wrong\b|stop\b|undo\b|revert\b|that'?s not right|that'?s wrong|"
    r"not what i (wanted|asked|meant)|actually,? no|don'?t do that)\b",
    re.I,
)
_DISMISS_MAX_WORDS = 12  # a real "no, that's wrong" correction is short, not a fresh long request
# explicit "make this a skill" — must literally invoke a skill in a make/create/save context
_EXPLICIT_SKILL = re.compile(
    r"\b(make (this|that|it) (in)?to a skill|turn (this|that) into a skill|"
    r"save (this|that) as a skill|create a skill (for|to)|should be a skill|"
    r"add a skill (for|to)|write a skill (for|to))\b",
    re.I,
)
# transcript machinery / banners that must never count as a user request
_BANNER = re.compile(
    r"^\s*(<document|<summary|<command|<system-reminder|this session is being continued|caveat:)",
    re.I,
)

_SKILL_MARK = re.compile(r"^\[tool:Skill\]\s*(.*)$")


def _event_time(meta_ts, created_at):
    """The episode's event time: the transcript's own timestamp when the parser recorded one
    (metadata.ts; one without an offset is UTC), else created_at. Guards fired_at against
    ingest-wall-clock stamping."""
    if meta_ts:
        try:
            ts = datetime.fromisoformat(str(meta_ts).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return created_at
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)
    return created_at


def fire_events(
    conn,
    window: lane.Window,
    platform: str = "claude_code",
    exclude_projects: tuple[str, ...] = config.EXCLUDE_PROJECTS,
) -> list[dict]:
    """Skill-fire events from the [tool:Skill] markers of the episodes in `window`, with
    dismissal detection from the NEXT user turn. Returns [{skill, session_id, fired_at, via,
    dismissed}]. Powers skill_usage, the grounded dismissals and post_fire.

    fired_at is the fire-position episode's EVENT time: the transcript's own timestamp
    (metadata->>'ts', stamped by the JSONL parser on the episode carrying the skill-invocation
    marker), falling back to created_at. created_at alone lied for backfilled sessions — it was
    the ingest wall clock, which stamped week-old fires with import day."""
    rows = conn.execute(
        "SELECT session_id, sequence, created_at, human_turn, content, metadata->>'ts' "
        "FROM episodes "
        "WHERE platform=%s AND id > %s AND id <= %s AND (project IS NULL OR project <> ALL(%s)) "
        "ORDER BY session_id, sequence",
        (platform, window.after_id, window.until_id, list(exclude_projects)),
    ).fetchall()
    events = []
    for i, (sid, _seq, created_at, _human, content, meta_ts) in enumerate(rows):
        ts = _event_time(meta_ts, created_at)
        for line in (content or "").splitlines():
            m = _SKILL_MARK.match(line)
            if not m:
                continue
            nm = lane.SKILL_NAME.search(m.group(1))
            if not nm:
                continue
            # dismissal: next turn is a SHORT, strong, leading negation (precision over recall)
            dismissed = False
            if i + 1 < len(rows) and str(rows[i + 1][0]) == str(sid):
                nxt = (rows[i + 1][3] or "").strip()
                dismissed = bool(
                    nxt and len(nxt.split()) <= _DISMISS_MAX_WORDS and _CORRECTION.search(nxt)
                )
            events.append(
                {
                    "skill": nm.group(1),
                    "session_id": str(sid),
                    "fired_at": ts,
                    "via": "tool",
                    "dismissed": dismissed,
                }
            )
    return events


def explicit_skill_requests(
    conn,
    window: lane.Window,
    platform: str = "claude_code",
    exclude_projects: tuple[str, ...] = config.EXCLUDE_PROJECTS,
) -> list[dict]:
    """User turns in `window` that explicitly ask for a skill — a strong GROUNDED derive
    signal. Returns [{session_id, ts, phrasing}]."""
    rows = conn.execute(
        "SELECT session_id, created_at, human_turn FROM episodes "
        "WHERE platform=%s AND human_turn IS NOT NULL AND id > %s AND id <= %s "
        "AND (project IS NULL OR project <> ALL(%s)) ORDER BY id",
        (platform, window.after_id, window.until_id, list(exclude_projects)),
    ).fetchall()
    return [
        {"session_id": str(sid), "ts": ts, "phrasing": human[:200]}
        for sid, ts, human in rows
        if human and not _BANNER.match(human) and _EXPLICIT_SKILL.search(human)
    ]
