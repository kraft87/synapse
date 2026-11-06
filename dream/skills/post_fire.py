#!/usr/bin/env python3
# mypy: ignore-errors
"""dream->skills v2 detector: POST-FIRE OUTCOMES (nightly; entry: run()).

Every skill fire is a free experiment: the aftermath says whether the skill's procedure
actually held. This detector watermarks on skill_usage.fired_at (NEW fires only — the
caller owns cursor state), reads the aftermath window that FOLLOWED each fire (up to ~10
episodes, stopping before the session's NEXT skill fire — later corrections belong to
that fire, not this one) alongside the skill's own registry description/body, and judges
the outcome:

  clean        — the procedure held; nothing to do
  deviation    — a step was wrong, missing, or mis-scoped and the user corrected it
  fight        — repeated pushback / outright override of the skill
  unassessable — no usable aftermath (session-tail fire, thin window, judge failure)

deviation/fight -> a RETUNE candidate via skill_ledger.merge_candidate with
signal='post_fire_deviation', a short verbatim quote, and a drafted proposed_patch
(2-4 concrete bullets) — the modification stream the review CLI renders.

Two data-quality rules learned the hard way (2026-07-18 audit):
  * NEVER trust fired_at for position — backfilled rows carry ingest-time stamps hours
    or days after the episodes. Resolve the fire's position by locating the skill
    invocation marker ([tool:Skill] / "Base directory for this skill") in episode
    content, using fired_at only to break ties between multiple fires of the same skill.
  * A fire on the session's LAST episode has no aftermath — count it unassessable and
    skip, never judge garbage.

And one learned 2026-11-06: the skill loader's own output (the "Base directory for this
skill: .../<skill>" banner and the skill body) is stored as the episode right after the
[tool:Skill] one. Read as the next fire, it emptied every window from 07-18 on, when
fired_at became the event time and the tie-break stopped landing on the banner by luck. The
fire's position is now the loader episode when there is one, and only a [tool:Skill] line
(any skill) or a DIFFERENT skill's banner ends the aftermath. A window that still comes out
empty is counted as empty_window, apart from the judge's unassessable.
"""

from __future__ import annotations

from . import skill_db_source as DB
from . import skill_ledger as L
from . import skill_measure as SM
from .skill_measure import _clamp_salience, _excerpt, _extract_json
from .struggle_arc import NEGATIVE_LIST

WINDOW_EPISODES = 10  # aftermath the judge reads
BODY_CHARS = 4000  # skill-body cap in the judge prompt

OUTCOMES = ("clean", "deviation", "fight", "unassessable")

_JUDGE_PROMPT = """You audit ONE firing of an agent SKILL (a procedure document injected into the agent's
context when it fired). Below: the skill's trigger description and body, then the transcript window
that FOLLOWED the fire. Judge how the fire went:

- "clean": the skill's procedure held; no user correction traceable to the skill.
- "deviation": the work deviated from the skill — a step was wrong, missing, or mis-scoped, and the
  user corrected it or worked around it. The skill needs a patch.
- "fight": repeated pushback — the user corrected the same skill-driven behavior two or more times,
  or overrode the skill outright.
- "unassessable": the window is too thin or interrupted, or the aftermath never exercises the skill.

ATTRIBUTION FIRST: for each correction in the window, decide — is it about THIS skill's output, or
about unrelated later work in the same session? Corrections about unrelated work do NOT count: a
window whose corrections all target other work is "clean" (or "unassessable"), never a deviation.
A hard task going badly for reasons the skill never claimed to cover is also NOT a deviation. When
drafting a patch, prefer updating this skill over describing a new one, and keep it out of these
categories:

{negative_list}

SKILL "{name}"
description: {description}
body (may be truncated):
{body}

POST-FIRE WINDOW ([user]/[assistant]/[tool:*] markers):
{window}

Output ONLY a JSON object, no prose:
{{"outcome": "clean"|"deviation"|"fight"|"unassessable", "why": "one sentence",
"direction": "extend"|"fix"|"narrow" for deviation/fight else null,
"salience": 1-5 (pain: wasted turns, user frustration, recurrence risk),
"quote": "SHORT verbatim user quote showing the deviation (deviation/fight only, else null)",
"summary": "one sentence: what the skill got wrong",
"proposed_patch": ["2-4 concrete change-intent bullets for the skill body/description — plain
bullets, never a unified diff (the skill file may live on another machine)"] for
deviation/fight else []}}"""


# ------------------------------------------------------------------ fire position
def _tool_fire(content: str, skill: str | None = None) -> bool:
    """True if the content carries a [tool:Skill] call (naming `skill`, when given)."""
    for line in (content or "").splitlines():
        m = DB._SKILL_MARK.match(line)
        if m and skill is None:
            return True
        if m:
            nm = DB._SKILL_NAME.search(m.group(1))
            if nm and nm.group(1) == skill:
                return True
    return False


def _mentions_fire(content: str, skill: str) -> bool:
    """True if this episode's content carries the invocation marker for `skill`:
    a [tool:Skill] call naming it, or a loader banner naming the skill's folder."""
    if _tool_fire(content, skill):
        return True
    folder = DB.skill_folder(skill)
    return any(DB.banner_skill(line) == folder for line in (content or "").splitlines())


def _ends_window(content: str, skill: str | None) -> bool:
    """True if this episode starts another fire: any [tool:Skill] line, or a loader banner
    for a skill other than `skill` (any banner at all when `skill` is None)."""
    if _tool_fire(content):
        return True
    folder = DB.skill_folder(skill) if skill else None
    for line in (content or "").splitlines():
        named = DB.banner_skill(line)
        if named is not None and named != folder:
            return True
    return False


def _is_loader(ep: dict, skill: str, body: str | None) -> bool:
    return DB.is_loader_episode(ep.get("content") or "", ep.get("human_turn") or "", skill, body)


def bound_window(
    episodes: list[dict],
    fire_seq: int,
    cap: int = WINDOW_EPISODES,
    *,
    skill: str | None = None,
    body: str | None = None,
) -> list[dict]:
    """The aftermath the judge may see: up to `cap` episodes after the fire, stopping
    BEFORE the next skill fire in the session — corrections past that point belong to
    the next fire (or to unrelated work), not this one. With `skill`, the fire's own loader
    episode right after `fire_seq` is skipped, and a later banner for the same skill does
    not end the window."""
    after = [ep for ep in episodes if ep["sequence"] > fire_seq]
    if skill and after and _is_loader(after[0], skill, body):
        after = after[1:]
    out: list[dict] = []
    for ep in after:
        if _ends_window(ep.get("content") or "", skill):
            break
        out.append(ep)
        if len(out) >= cap:
            break
    return out


def resolve_fire_position(
    episodes: list[dict], skill: str, fired_at, body: str | None = None
) -> int | None:
    """Sequence of the episode where this fire happened, resolved from content markers
    (fired_at only breaks ties between multiple fires of the same skill). A [tool:Skill]
    call beats a banner-only sighting, and when the loader's episode follows the call the
    fire sits there: the aftermath starts after it. None if no marker can be found — the
    caller counts that unassessable rather than guessing."""
    hits = [ep for ep in episodes if _tool_fire(ep.get("content") or "", skill)]
    hits = hits or [ep for ep in episodes if _mentions_fire(ep.get("content") or "", skill)]
    if not hits:
        return None
    if len(hits) == 1 or fired_at is None:
        seq = hits[0]["sequence"]
    else:

        def _dist(ep):
            ts = ep.get("created_at")
            return abs((ts - fired_at).total_seconds()) if ts is not None else float("inf")

        seq = min(hits, key=_dist)["sequence"]
    nxt = next((ep for ep in episodes if ep["sequence"] > seq), None)
    if nxt is not None and _is_loader(nxt, skill, body):
        return nxt["sequence"]
    return seq


# ------------------------------------------------------------------ DB fetches
def _fetch_fires(conn, since) -> list[dict]:
    cur = conn.cursor()
    if since is not None:
        cur.execute(
            "SELECT skill, session_id, fired_at FROM skills_lane.skill_usage "
            "WHERE fired_at > %s ORDER BY fired_at",
            (since,),
        )
    else:
        cur.execute(
            "SELECT skill, session_id, fired_at FROM skills_lane.skill_usage ORDER BY fired_at"
        )
    return [
        {"skill": s, "session_id": str(sid) if sid else None, "fired_at": ts}
        for s, sid, ts in cur.fetchall()
    ]


def _registered_skills(conn) -> set[str]:
    """Names of the skills in the registry (not retired). A fire of anything else, such
    as a skill built into the agent or shipped by a third-party plugin, has no registry
    row to retune, so judging it can only produce a proposal nobody can apply."""
    cur = conn.cursor()
    cur.execute("SELECT name FROM skills_lane.skill_registry WHERE status <> 'retired'")
    return {name for (name,) in cur.fetchall()}


def _fetch_session_episodes(conn, session_id: str) -> list[dict]:
    cur = conn.cursor()
    cur.execute(
        "SELECT sequence, created_at, human_turn, content FROM episodes "
        "WHERE session_id = %s ORDER BY sequence",
        (session_id,),
    )
    return [
        {"sequence": seq, "created_at": ts, "human_turn": human or "", "content": content or ""}
        for seq, ts, human, content in cur.fetchall()
    ]


def _load_skill_doc(conn, name: str) -> tuple[str, str]:
    """(description, body) from the registry; falls back to the bundled SKILL.md in
    skill_files when the registry body is empty."""
    cur = conn.cursor()
    cur.execute(
        "SELECT COALESCE(description, ''), COALESCE(body, '') "
        "FROM skills_lane.skill_registry WHERE name = %s",
        (name,),
    )
    row = cur.fetchone()
    desc, body = (row[0], row[1]) if row else ("", "")
    if not body:
        cur.execute(
            "SELECT content FROM skills_lane.skill_files WHERE skill_name = %s AND path = 'SKILL.md'",
            (name,),
        )
        r = cur.fetchone()
        if r and r[0]:
            body = bytes(r[0]).decode("utf-8", "replace")
    return desc, body[:BODY_CHARS]


# ------------------------------------------------------------------ judge
def _judge_call(prompt: str, model: str | None = None) -> str:
    """Thin delegator to the lane's shared judge dispatch (skill_measure.run_judge).
    Kept as a named seam so tests can stub the LLM out."""
    return SM.run_judge(prompt, model)


def judge_fire(
    skill: str, desc: str, body: str, window: list[dict], model: str | None = None
) -> dict | None:
    prompt = _JUDGE_PROMPT.format(
        negative_list=NEGATIVE_LIST,
        name=skill,
        description=desc or "(none)",
        body=body or "(no body in registry)",
        window=_excerpt(window),
    )
    try:
        raw = _judge_call(prompt, model)
    except Exception as e:
        print(f"  post-fire judge failed for {skill}: {e}")
        return None
    return _extract_json(raw)


def _patch_text(patch) -> str | None:
    """Judge output -> rendered proposed_patch: a list becomes markdown bullets."""
    if isinstance(patch, list):
        bullets = [str(b).strip() for b in patch if str(b).strip()]
        return "\n".join(f"- {b}" for b in bullets) or None
    if isinstance(patch, str) and patch.strip():
        return patch.strip()
    return None


# ------------------------------------------------------------------ entry
def run(conn, *, since, model: str | None = None) -> dict:
    """Judge every skill fire recorded after `since`; deviations/fights become retune
    candidates. Returns the per-outcome tally. empty_window (no aftermath between the fire
    and the next one) is counted on its own, outside unassessable."""
    stats = {
        "fires": 0,
        "clean": 0,
        "deviation": 0,
        "fight": 0,
        "unassessable": 0,
        "last_episode": 0,
        "unlocated": 0,
        "empty_window": 0,
        "candidates": 0,
        "judge_failures": 0,
        "unregistered": 0,
    }
    fires = _fetch_fires(conn, since)
    stats["fires"] = len(fires)
    if not fires:
        return stats

    scan_night = SM.scan_night()
    registered = _registered_skills(conn)
    sessions: dict[str, list[dict]] = {}
    docs: dict[str, tuple[str, str]] = {}
    for fire in fires:
        # Only skills the lane owns can be retuned. The other detectors already gate on
        # the registry (struggle_arc: name in skill_names; procedure_miner: extends in
        # registry); without this, a fire of a built-in skill became an unappliable
        # retune proposal.
        if fire["skill"] not in registered:
            stats["unregistered"] += 1
            continue
        sid = fire["session_id"]
        if not sid:
            stats["unassessable"] += 1
            stats["unlocated"] += 1
            continue
        if sid not in sessions:
            sessions[sid] = _fetch_session_episodes(conn, sid)
        episodes = sessions[sid]
        if fire["skill"] not in docs:
            docs[fire["skill"]] = _load_skill_doc(conn, fire["skill"])
        desc, body = docs[fire["skill"]]

        seq = resolve_fire_position(episodes, fire["skill"], fire["fired_at"], body)
        if seq is None:
            stats["unassessable"] += 1
            stats["unlocated"] += 1
            continue
        if not any(ep["sequence"] > seq for ep in episodes):  # session-tail fire
            stats["unassessable"] += 1
            stats["last_episode"] += 1
            continue
        window = bound_window(episodes, seq, skill=fire["skill"], body=body)
        if not window:  # next fire is immediate: no aftermath attributable to this one
            stats["empty_window"] += 1
            continue

        verdict = judge_fire(fire["skill"], desc, body, window, model)
        if verdict is None:
            stats["judge_failures"] += 1
            stats["unassessable"] += 1
            continue
        outcome = (verdict.get("outcome") or "").strip().lower()
        if outcome not in OUTCOMES:
            outcome = "unassessable"
        stats[outcome] += 1
        if outcome not in ("deviation", "fight"):
            continue

        direction = verdict.get("direction")
        if direction not in ("extend", "fix", "narrow"):
            direction = "fix"
        fire_ep = next((ep for ep in episodes if ep["sequence"] == seq), None)
        ts = fire_ep.get("created_at") if fire_ep else None
        ev = [
            {
                "class": "judge",
                "signal": "post_fire_deviation",
                "session_id": sid,
                "quote": (verdict.get("quote") or "").strip()[:300] or None,
                "scan_night": scan_night,
                "date": ts.date().isoformat() if ts else None,
            }
        ]
        res = L.merge_candidate(
            conn,
            "retune",
            fire["skill"],
            ev,
            target_skills=[fire["skill"]],
            direction=direction,
            summary=(verdict.get("summary") or "").strip(),
            salience=_clamp_salience(verdict.get("salience")),
            source_detector="post_fire",
            proposed_patch=_patch_text(verdict.get("proposed_patch")),
            do_embed=False,
        )
        stats["candidates"] += 1
        print(
            f"  post-fire {outcome} -> retune cand#{res['id']} ({res['status']}) for {fire['skill']}"
        )
    return stats
