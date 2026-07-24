#!/usr/bin/env python3
# mypy: ignore-errors
"""dream->skills v2 detector: STRUGGLE ARCS (nightly; entry: run()).

The v1 gap scan asks "what procedure has no skill?" — this asks the sharper question:
"where did the agent visibly STRUGGLE?" A struggle arc is a run of user-correction turns
("didn't work", "same error", "you sure") clustered in one session. Two stages, so the
LLM only ever reads the worst few windows:

  A. lexical prescreen (SQL + pure python, cheap): score correction markers per NEW
     episode (hard markers weigh 2, soft markers 1 and only count mid-session — a fresh
     request containing "wrong" is not a correction), drop transcript machinery
     (task notifications, session-continuation summaries — they quote error text and
     false-positive hard), group survivors into arcs by (session, sequence proximity),
     keep the top `limit` arcs by score.
  B. windowed LLM judge (arc +/- 3 episodes): genuine struggle? skill-worthy per the
     negative capture list? UPDATE-FIRST — the judge sees the full active skill catalog
     IN THE PROMPT (no embedding routing — at catalog scale that misroutes) and must
     prefer patching an existing skill (retune + proposed_patch) over minting a new one
     (derive). Verdicts land in the ledger via skill_ledger.merge_candidate with
     signal='struggle_arc' evidence carrying a short verbatim quote + scan_night.

Topic gate, two halves: an optional project ALLOWLIST (STRUGGLE_INCLUDE_PROJECTS env)
keeps personal/medical/untagged sessions out of Stage A entirely, and an arc window with
no tool-use evidence (no [tool:*] marker, no file path) never reaches the judge —
correction phrasing without tool work is conversation, not a procedure. Derives always
carry a signature (judge topic key, synthesized fallback): the ledger's identity resolver
jaccard-matches signature tokens and would wildcard-merge signatureless derives.

Evidence class is 'judge' (self-assessment, 0.5x discounted) unless the arc's window
contains an explicit "make this a skill" user ask -> 'grounded'. Propose-only, like the
rest of the lane: nothing here touches disk skills or sets status past the ledger gates.
"""

from __future__ import annotations

import os
import re

from . import config
from . import skill_db_source as DB
from . import skill_ledger as L
from . import skill_measure as SM
from .skill_measure import _clamp_salience, _excerpt, _extract_json

# ------------------------------------------------------------------ Stage A knobs
# Marker sets from the 2026-07-18 struggle-mining pass (hard markers precision-tested
# on a 21-day window). Matched on the apostrophe-stripped, lowercased USER turn.
HARD_MARKERS = (
    "didnt work",
    "doesnt work",
    "didnt fix",
    "still broken",
    "same error",
    "you sure",
    "thats not",
    "revert",
    "you gotta verify",
    "gotta verify",
    "still isnt",
)
SOFT_MARKERS = (
    "wrong",
    "not working",
    "not right",
    "nope",
    "no luck",
    "still seeing",
    "still getting",
    "didnt help",
    "try again",
    "not fixed",
)
HARD_WEIGHT = 2
SOFT_WEIGHT = 1
FLAG_SCORE = 2  # episode enters an arc at this score (1 hard, or 2 soft mid-session)
SOFT_MAX_WORDS = 80  # real corrections are terse; a long fresh request isn't one
ARC_GAP = 6  # max sequence gap between flagged episodes in one arc
WINDOW_PAD = 3  # judge reads the arc +/- this many episodes
MAX_WINDOW_EPISODES = 24

# Transcript machinery that must never count as a user correction: task-notification
# turns and session-continuation summaries quote error text verbatim ("same error",
# tracebacks) and were the dominant Stage A false-positive class.
_NOISE = re.compile(
    r"^\s*(<document|<summary|<command|<system-reminder|<task|this session is being continued"
    r"|caveat:|\[?task notification|background task |\[request interrupted)",
    re.I,
)

# transcribe_ai has its own skill environment; its sessions are never gaps in this catalog.
EXCLUDED_PROJECTS = tuple(sorted({*config.EXCLUDE_PROJECTS, "transcribe_ai"}))


# Topic gate half 1: an optional PROJECT ALLOWLIST (comma-separated env). When set, Stage A
# scans ONLY those projects — personal / medical / untagged-personal sessions never reach the
# prescreen (correction phrasing like "that's not the case" false-positives hard there).
# Integration sets this in the dream container env; unset = exclusion-based filtering only.
def _include_projects() -> tuple[str, ...]:
    return tuple(
        p.strip() for p in os.environ.get("STRUGGLE_INCLUDE_PROJECTS", "").split(",") if p.strip()
    )


# Topic gate half 2: an arc window must show TOOL-USE evidence (a [tool:*] marker or a
# file path). A correction arc with no tool work near it is conversation, not a procedure.
_TOOL_MARK = re.compile(r"^\[tool:[\w-]+\]", re.M)
_PATHISH = re.compile(r"(?:^|\s)(?:/|~/)[\w.\-]+/[\w.\-/]+", re.M)

# ------------------------------------------------------------------ Stage B prompt
# The negative capture list ships VERBATIM in every detector judge prompt (design §5):
# each category is a class of junk that, once skill-ified, pollutes every future session.
NEGATIVE_LIST = """Do NOT capture (each of these hardens into junk that pollutes every future session):
- environment-dependent failures
- negative claims about tools ("X is broken" hardens into self-citing refusals)
- transient errors that resolved
- one-off task narratives
- novel debugging of a new problem (not a repeatable procedure) — novel debugging is NOT a skill gap
- behavioral/preference corrections (those belong to the config lane, not skills)
- one-off architecture decisions"""

_JUDGE_PROMPT = """You review one WINDOW of an AI coding-agent transcript that a lexical prescreen flagged for
user-correction markers. Decide whether it shows a GENUINE STRUGGLE a skill (a reusable procedure
document) could prevent next time — and if so, whether to PATCH an existing skill or propose a new one.

EXISTING SKILLS (name: trigger description) — check these FIRST:
{catalog}

UPDATE-FIRST ladder, strict order:
1. If an existing skill NEARLY covers this work but was wrong, missing a step, or mis-scoped ->
   kind="retune" targeting that skill, with direction "extend" (add coverage), "fix" (correct a
   step), or "narrow" (stop it over-applying), and a concrete proposed_patch.
2. Only if nothing comes close -> kind="derive" (a new skill).

{negative_list}
Also NOT a struggle: the user changing their mind, ordinary iteration on a hard task, or a
personal / non-technical conversation that merely matches correction phrasing.

TRANSCRIPT WINDOW (sequence-numbered; [user]/[assistant]/[tool:*] markers):
{window}

Output ONLY a JSON object, no prose:
{{"struggle": true, "skill_worthy": true, "why": "one sentence",
"kind": "derive" or "retune", "name": "short-kebab-name (for retune: the existing skill's exact name)",
"direction": "extend"|"fix"|"narrow" for retune else null,
"salience": 1-5 STRICT rubric — 3 = a typical single-session correction arc; 4 ONLY if the problem
recurs across sessions OR has high blast radius OR blocked the user for over an hour; 5 = severe
and repeated; 2 = minor friction; 1 = trivial,
"signature": "3-6 lowercase topic keywords naming the procedure (e.g. 'vpn acl connectivity triage')",
"summary": "1-2 sentences: the gap and what the skill should say",
"quote": "SHORT verbatim user quote from the window showing the struggle",
"trigger_phrasings": ["how a user would ask for this", "..."],
"proposed_patch": "for retune: 2-4 concrete bulleted change-intent lines (plain bullets, never a
unified diff — the skill file may live on another machine), else null}}"""


# ------------------------------------------------------------------ Stage A: prescreen
def _normalize(text: str) -> str:
    return (text or "").lower().replace("\u2019", "").replace("'", "")


def score_turn(human_turn: str, sequence: int) -> int:
    """Correction-marker score for one user turn. Hard markers always count; soft
    markers only mid-session (sequence > 1) and only on terse turns — the
    continuation filter that keeps 'wrong' inside a long fresh request from flagging."""
    text = _normalize(human_turn)
    if not text.strip():
        return 0
    score = sum(HARD_WEIGHT for m in HARD_MARKERS if m in text)
    if sequence > 1 and len(text.split()) <= SOFT_MAX_WORDS:
        score += sum(SOFT_WEIGHT for m in SOFT_MARKERS if m in text)
    return score


def is_noise(human_turn: str, content_head: str) -> bool:
    """True for transcript machinery (task notifications, continuation summaries)."""
    return bool(_NOISE.match(human_turn or "") or _NOISE.match(content_head or ""))


def has_tool_evidence(window: list[dict]) -> bool:
    """Topic gate: True when the arc window shows actual tool work (a [tool:*] marker
    or a file path). Pure-conversation windows never reach the judge."""
    for ep in window:
        text = ep.get("content") or ""
        if _TOOL_MARK.search(text) or _PATHISH.search(text):
            return True
    return False


def group_arcs(flagged: list[dict], gap: int = ARC_GAP) -> list[dict]:
    """Cluster flagged episodes into arcs by (session_id, sequence proximity).
    flagged entries: {session_id, sequence, score, date}."""
    arcs: list[dict] = []
    for ep in sorted(flagged, key=lambda e: (e["session_id"], e["sequence"])):
        cur = arcs[-1] if arcs else None
        if cur and cur["session_id"] == ep["session_id"] and ep["sequence"] - cur["seq_max"] <= gap:
            cur["seq_max"] = ep["sequence"]
            cur["score"] += ep["score"]
            cur["episodes"].append(ep)
        else:
            arcs.append(
                {
                    "session_id": ep["session_id"],
                    "seq_min": ep["sequence"],
                    "seq_max": ep["sequence"],
                    "score": ep["score"],
                    "date": ep.get("date"),
                    "episodes": [ep],
                }
            )
    return arcs


def _fetch_new_episodes(conn, since) -> list[dict]:
    """Episodes newer than `since` (None -> 30d), claude_code only, foreign skill
    environments excluded. With STRUGGLE_INCLUDE_PROJECTS set, only allowlisted
    projects are scanned (untagged episodes drop too). Only prescreen fields."""
    cur = conn.cursor()
    where_time = (
        "created_at > %s" if since is not None else "created_at > now() - interval '30 days'"
    )
    params: list = [] if since is None else [since]
    include = _include_projects()
    if include:
        where_project = "project = ANY(%s)"
        params.append(list(include))
    else:
        where_project = "(project IS NULL OR project <> ALL(%s))"
        params.append(list(EXCLUDED_PROJECTS))
    cur.execute(
        f"""SELECT session_id, sequence, created_at, human_turn, LEFT(content, 400)
              FROM episodes
             WHERE platform = 'claude_code' AND {where_time} AND {where_project}
             ORDER BY session_id, sequence""",
        tuple(params),
    )
    return [
        {
            "session_id": str(sid),
            "sequence": seq,
            "created_at": ts,
            "human_turn": human or "",
            "content_head": head or "",
        }
        for sid, seq, ts, human, head in cur.fetchall()
    ]


# ------------------------------------------------------------------ Stage B: judge
def _fetch_window(conn, session_id: str, lo: int, hi: int) -> list[dict]:
    cur = conn.cursor()
    cur.execute(
        """SELECT sequence, human_turn, content FROM episodes
            WHERE session_id = %s AND sequence BETWEEN %s AND %s
            ORDER BY sequence LIMIT %s""",
        (session_id, lo, hi, MAX_WINDOW_EPISODES),
    )
    return [
        {"sequence": seq, "human_turn": human or "", "content": content or ""}
        for seq, human, content in cur.fetchall()
    ]


def _load_catalog(conn) -> list[tuple[str, str]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT name, COALESCE(description, '') FROM skills_lane.skill_registry "
        "WHERE status = 'active' ORDER BY name"
    )
    return [(n, d) for n, d in cur.fetchall()]


def _judge_call(prompt: str, model: str | None = None) -> str:
    """Thin delegator to the lane's shared judge dispatch (skill_measure.run_judge).
    Kept as a named seam so tests can stub the LLM out."""
    return SM.run_judge(prompt, model)


def judge_arc(window: list[dict], catalog_text: str, model: str | None = None) -> dict | None:
    """One judge call for one arc window. Lenient JSON extraction, no retry — matches
    skill_derive's parsing conventions."""
    prompt = _JUDGE_PROMPT.format(
        catalog=catalog_text or "(none)", negative_list=NEGATIVE_LIST, window=_excerpt(window)
    )
    try:
        raw = _judge_call(prompt, model)
    except Exception as e:
        print(f"  struggle judge failed: {e}")
        return None
    return _extract_json(raw)


# ------------------------------------------------------------------ emission
def _emit(
    conn, arc: dict, window: list[dict], verdict: dict, skill_names: set[str]
) -> tuple[dict, str]:
    """Merge one judged arc into the ledger. Returns (merge result, resolved kind)."""
    scan_night = SM.scan_night()
    date = arc.get("date")
    # explicit "make this a skill" in the window -> grounded evidence (design §1)
    grounded = any(DB._EXPLICIT_SKILL.search(ep.get("human_turn") or "") for ep in window)
    quote = (verdict.get("quote") or "").strip()[:300] or None
    ev = [
        {
            "class": "grounded" if grounded else "judge",
            "signal": "struggle_arc",
            "session_id": arc["session_id"],
            "quote": quote,
            "scan_night": scan_night,
            "date": date,
        }
    ]
    salience = _clamp_salience(verdict.get("salience"))
    summary = (verdict.get("summary") or "").strip()
    kind = verdict.get("kind")
    name = (verdict.get("name") or "").strip()
    # retune must target a real skill; a retune of a skill that doesn't exist is a derive
    if kind == "retune" and name in skill_names:
        direction = verdict.get("direction")
        if direction not in ("extend", "fix", "narrow"):
            direction = "fix"
        patch = (verdict.get("proposed_patch") or "").strip() or None
        res = L.merge_candidate(
            conn,
            "retune",
            name,
            ev,
            target_skills=[name],
            direction=direction,
            summary=summary,
            salience=salience,
            source_detector="struggle_arc",
            proposed_patch=patch,
            do_embed=False,
        )
        return res, "retune"
    kebab = re.sub(r"[^a-z0-9-]", "", name.lower()) or "unnamed"
    # A derive must NEVER go in without a signature: the ledger resolver jaccard-matches
    # signature tokens, and empty-vs-empty compares as identical — unrelated no-signature
    # derives would wildcard-merge into one row. Judge-provided topic key, else synthesized.
    signature = (verdict.get("signature") or "").strip().lower()
    if not signature:
        signature = " ".join(sorted(L._tokens(kebab.replace("-", " "), summary))) or kebab
    res = L.merge_candidate(
        conn,
        "derive",
        kebab,
        ev,
        signature=signature,
        tools=_window_tools(window),
        summary=summary,
        trigger_phrasings=verdict.get("trigger_phrasings") or [],
        salience=salience,
        source_detector="struggle_arc",
    )
    return res, "derive"


def _window_tools(window: list[dict]) -> list[str]:
    """Tool names observed in the arc window ([tool:NAME] markers) — secondary identity
    signal for the derive resolver."""
    tools: set[str] = set()
    for ep in window:
        for line in (ep.get("content") or "").splitlines():
            m = DB._MARKER.match(line)
            if m:
                tools.add(m.group(1))
    return sorted(tools)[:8]


# ------------------------------------------------------------------ entry
def run(conn, *, since, limit: int = 10, model: str | None = None) -> dict:
    """One nightly pass: prescreen episodes newer than `since`, judge the top `limit`
    arcs, merge verdicts into the ledger. The caller owns cursor state."""
    stats = {
        "scanned": 0,
        "flagged": 0,
        "arcs": 0,
        "judged": 0,
        "judge_failures": 0,
        "skipped": 0,
        "skipped_no_tools": 0,
        "retunes": 0,
        "derives": 0,
    }
    episodes = _fetch_new_episodes(conn, since)
    stats["scanned"] = len(episodes)

    flagged = []
    for ep in episodes:
        if is_noise(ep["human_turn"], ep["content_head"]):
            continue
        score = score_turn(ep["human_turn"], ep["sequence"])
        if score >= FLAG_SCORE:
            ts = ep.get("created_at")
            flagged.append(
                {
                    "session_id": ep["session_id"],
                    "sequence": ep["sequence"],
                    "score": score,
                    "date": ts.date().isoformat() if ts else None,
                }
            )
    stats["flagged"] = len(flagged)

    arcs = group_arcs(flagged)
    stats["arcs"] = len(arcs)
    arcs.sort(key=lambda a: -a["score"])
    arcs = arcs[:limit]
    if not arcs:
        return stats

    catalog = _load_catalog(conn)
    catalog_text = "\n".join(f"- {n}: {d}" for n, d in catalog)
    skill_names = {n for n, _ in catalog}

    for arc in arcs:
        window = _fetch_window(
            conn,
            arc["session_id"],
            max(0, arc["seq_min"] - WINDOW_PAD),
            arc["seq_max"] + WINDOW_PAD,
        )
        if not has_tool_evidence(window):
            # topic gate half 2: correction phrasing with zero tool work = conversation
            stats["skipped_no_tools"] += 1
            continue
        verdict = judge_arc(window, catalog_text, model)
        if verdict is None:
            stats["judge_failures"] += 1
            continue
        stats["judged"] += 1
        if not (verdict.get("struggle") and verdict.get("skill_worthy")):
            stats["skipped"] += 1
            continue
        res, kind = _emit(conn, arc, window, verdict, skill_names)
        stats[kind + "s"] += 1
        print(f"  struggle arc -> {kind} cand#{res['id']} ({res['status']})")
    return stats
