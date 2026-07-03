"""DB source for the dream->skills lanes: reconstruct per-session views from the
# mypy: ignore-errors
Synapse `episodes` table instead of globbing raw JSONL transcripts.

Why: the episodes table already holds what gap-finding needs as parseable markers
in `content` ([user]/[tool:NAME]/[result]/[assistant]) plus the `human_turn` column
(full user text, up to 3000 chars). One indexed query by created_at/platform beats
globbing 75k+ JSONL files, gives richer user phrasings, and is the single canonical
source.

KNOWN LOSS vs JSONL: ingestion drops `<command-name>` slash-command records as
machinery, so SLASH-invoked skill fires are NOT visible here (only Skill-*tool*
invocations survive, as `[tool:Skill] {'skill': 'name'}`). Fine for DERIVE (the
guard is the catalog comparison, not fire-detection); matters for RETUNE counting —
fix later with a `[skill:]` ingest marker or a tiny JSONL fire-scan. Thinking blocks
are also not stored. Pass source="jsonl" to skill-derive when you need either.

Returns the SAME dict shape as skill-measure.parse_session, so it's a drop-in:
    {session, first_user, user_turns, user_msgs, tools(dict), bash_heads, fired}
"""

from __future__ import annotations

import re
from collections import Counter

from . import config

_MARKER = re.compile(r"^\[tool:([\w-]+)\]\s*(.*)$")
_SKILL_IN = re.compile(r"skill'?\s*[:=]\s*'?([\w-]+)")


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
    """One query for the window, grouped into session views in Python.

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
    sessions: dict[str, dict] = {}
    for sid, _seq, human, content in cur.fetchall():
        sid = str(sid)
        s = sessions.get(sid)
        if s is None:
            s = sessions[sid] = {
                "session": sid,
                "first_user": "",
                "user_turns": 0,
                "user_msgs": [],
                "tools": Counter(),
                "bash_heads": [],
                "fired": set(),
            }
        if human:
            h = human.strip()
            if h:
                s["user_turns"] += 1
                if not s["first_user"]:
                    s["first_user"] = h[:600]
                s["user_msgs"].append(h[:200])
        for line in (content or "").splitlines():
            m = _MARKER.match(line)
            if not m:
                continue
            name, detail = m.group(1), m.group(2)
            s["tools"][name] += 1
            if name == "Bash" and detail:
                s["bash_heads"].append(detail[:80])
            elif name == "Skill":
                sk = _SKILL_IN.search(detail)
                if sk:
                    s["fired"].add(sk.group(1))
    conn.close()

    views = []
    for s in sessions.values():
        s["tools"] = dict(s["tools"])
        s["bash_heads"] = s["bash_heads"][:25]
        s["fired"] = sorted(s["fired"])
        views.append(s)
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
_SKILL_NAME = re.compile(r"skill'?\s*[:=]\s*'?([\w-]+)")


def sessions_since(
    last_scan_at,
    max_sessions: int = 80,
    platform: str = "claude_code",
    exclude_projects: tuple[str, ...] = config.EXCLUDE_PROJECTS,
) -> list[dict]:
    """Whole-session views for any session TOUCHED since last_scan_at (Oracle Q2: scan unit =
    whole session, not raw new turns). Pulls each touched session's FULL episode set so the
    signature_key is stable across runs. last_scan_at None -> behaves like a 30d window."""
    conn = _connect(_dsn())
    cur = conn.cursor()
    if last_scan_at is None:
        cur.execute(
            "SELECT DISTINCT session_id FROM episodes WHERE platform=%s "
            "AND created_at > now() - interval '30 days' AND (project IS NULL OR project <> ALL(%s))",
            (platform, list(exclude_projects)),
        )
    else:
        cur.execute(
            "SELECT session_id FROM episodes WHERE platform=%s AND (project IS NULL OR project <> ALL(%s)) "
            "GROUP BY session_id HAVING max(created_at) > %s",
            (platform, list(exclude_projects), last_scan_at),
        )
    sids = [str(r[0]) for r in cur.fetchall()]
    conn.close()
    if not sids:
        return []
    # reuse the full-window builder, then keep only touched sessions (cheap at our scale)
    allv = {s["session"]: s for s in db_session_views(3650, 100000, platform, exclude_projects)}
    out = [allv[s] for s in sids if s in allv]
    out.sort(key=lambda s: s["user_turns"], reverse=True)
    return out[:max_sessions]


def fire_events(
    last_scan_at=None,
    days: float = 30,
    platform: str = "claude_code",
    exclude_projects: tuple[str, ...] = config.EXCLUDE_PROJECTS,
) -> list[dict]:
    """Skill-fire events from episode [tool:Skill] markers, with dismissal detection from the
    NEXT user turn. Returns [{skill, session_id, fired_at, via, dismissed}]. Powers skill_usage."""
    conn = _connect(_dsn())
    cur = conn.cursor()
    if last_scan_at is not None:
        cur.execute(
            "SELECT session_id, sequence, created_at, human_turn, content FROM episodes "
            "WHERE platform=%s AND created_at > %s AND (project IS NULL OR project <> ALL(%s)) "
            "ORDER BY session_id, sequence",
            (platform, last_scan_at, list(exclude_projects)),
        )
    else:
        cur.execute(
            "SELECT session_id, sequence, created_at, human_turn, content FROM episodes "
            "WHERE platform=%s AND created_at > now() - (%s||' days')::interval "
            "AND (project IS NULL OR project <> ALL(%s)) ORDER BY session_id, sequence",
            (platform, str(days), list(exclude_projects)),
        )
    rows = cur.fetchall()
    conn.close()
    events = []
    for i, (sid, _seq, ts, _human, content) in enumerate(rows):
        for line in (content or "").splitlines():
            m = _SKILL_MARK.match(line)
            if not m:
                continue
            nm = _SKILL_NAME.search(m.group(1))
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
    last_scan_at=None,
    days: float = 30,
    platform: str = "claude_code",
    exclude_projects: tuple[str, ...] = config.EXCLUDE_PROJECTS,
) -> list[dict]:
    """User turns that explicitly ask for a skill — a strong GROUNDED derive signal.
    Returns [{session_id, ts, phrasing}]."""
    conn = _connect(_dsn())
    cur = conn.cursor()
    cur.execute(
        "SELECT session_id, created_at, human_turn FROM episodes "
        "WHERE platform=%s AND human_turn IS NOT NULL "
        + (
            "AND created_at > %s "
            if last_scan_at is not None
            else "AND created_at > now() - (%s||' days')::interval "
        )
        + "AND (project IS NULL OR project <> ALL(%s)) ORDER BY created_at",
        (platform, last_scan_at if last_scan_at is not None else str(days), list(exclude_projects)),
    )
    out = []
    for sid, ts, human in cur.fetchall():
        if human and not _BANNER.match(human) and _EXPLICIT_SKILL.search(human):
            out.append({"session_id": str(sid), "ts": ts, "phrasing": human[:200]})
    conn.close()
    return out
