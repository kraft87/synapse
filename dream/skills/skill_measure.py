#!/usr/bin/env python3
# mypy: ignore-errors
"""dream->skills, step 1: READ-ONLY measurement lane.

The prerequisite both Oracle and Gemini insisted on: before any skill edits,
measure — per skill — how often it FIRED vs how often it was APPLICABLE-but-
didn't-fire (under-trigger / recall gap) and FIRED-then-likely-dismissed
(false-fire / precision gap). No edits, no proposals; just a baseline + it
validates whether the "would-have-helped" LLM judge is trustworthy.

Source = raw Claude Code transcripts (~/.claude/projects/**/*.jsonl): tool_use
calls (incl. which skills fired) + the user requests. Skills live in
~/.claude/skills/*/SKILL.md (name + frontmatter description = the triggers).

Run dry first (no LLM) to see the landscape, then add the judge:
    skill-measure.py --days 2 --dry-run
    skill-measure.py --days 1 --max-sessions 40            # nightly, Opus judge

The judge model is SYNAPSE_SKILLS_JUDGE_MODEL (llm_client.stage_model; default Opus).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from dream import lane
from ingestion.llm_client import stage_model

from . import config

# No on-disk skill catalog server-side (the dream container has no ~/.claude/skills). The
# catalog comes from the skill_registry table; the client's skills_sync owns disk<->registry.
PROJECTS_DIR = config.PROJECTS_DIR  # dev-only --source jsonl path; the nightly reads PG
OUT_DIR = config.DATA_DIR


def load_catalog(conn) -> list[tuple[str, str]]:
    """(name, description) for ACTIVE skills, ordered by name — the lane-wide registry read
    every detector shares. Server-side the registry IS the catalog (the dream container has no
    ~/.claude/skills); the client's skills_sync owns the disk<->registry publish. `description`
    is the trigger surface."""
    cur = conn.cursor()
    cur.execute(
        "SELECT name, COALESCE(description, '') FROM skills_lane.skill_registry "
        "WHERE status = 'active' ORDER BY name"
    )
    return [(name, desc) for name, desc in cur.fetchall()]


def format_catalog(items) -> str:
    """Render a skill catalog for a judge prompt: one `- {name}: {desc}` line per entry,
    "(none)" when the catalog is empty."""
    return "\n".join(f"- {n}: {d}" for n, d in items) or "(none)"


def load_skills() -> dict[str, str]:
    """name -> description for ACTIVE skills — the connection-owning wrapper over load_catalog.

    Server-side the catalog IS the registry (the dream container has no ~/.claude/skills); the
    client's skills_sync owns the disk<->registry publish. `description` is the trigger surface."""
    import psycopg

    with psycopg.connect(config.db_url(), connect_timeout=10) as conn:
        return dict(load_catalog(conn))


def _content_blocks(rec: dict) -> list:
    msg = rec.get("message") or {}
    c = msg.get("content")
    return c if isinstance(c, list) else []


def parse_session(path: Path, skill_names: set[str]) -> dict | None:
    """Extract a compact session view: first user request, tool names used,
    bash command heads, and which skills fired. None if unreadable."""
    first_user = ""
    user_turns = 0
    user_msgs: list[str] = []  # the trigger surface: every user phrasing in the session
    tools: Counter = Counter()
    bash_heads: list[str] = []
    fired: set[str] = set()
    try:
        with path.open(errors="ignore") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                rtype = rec.get("type")
                if rtype == "user" and not rec.get("isSidechain"):
                    msg = rec.get("message") or {}
                    c = msg.get("content")
                    txt = (
                        c
                        if isinstance(c, str)
                        else " ".join(
                            b.get("text", "")
                            for b in (c or [])
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    )
                    txt = (txt or "").strip()
                    # skip tool_result-only user records + slash-command echoes
                    if (
                        txt
                        and not txt.startswith("<")
                        and "tool_use_id" not in (c[0] if isinstance(c, list) and c else {})
                    ):
                        user_turns += 1
                        if not first_user:
                            first_user = txt[:600]
                        user_msgs.append(txt[:200])
                    # slash-command skill invocation: <command-name>/<skill>
                    for sm in re.findall(r"<command-name>/?([a-z0-9_-]+)", txt):
                        if sm in skill_names:
                            fired.add(sm)
                elif rtype == "assistant":
                    for b in _content_blocks(rec):
                        if isinstance(b, dict) and b.get("type") == "tool_use":
                            nm = b.get("name", "")
                            tools[nm] += 1
                            inp = b.get("input") or {}
                            if nm == "Skill" and isinstance(inp, dict):
                                sk = inp.get("skill")
                                if sk:
                                    fired.add(sk)
                            if nm == "Bash" and isinstance(inp, dict):
                                cmd = (inp.get("command") or "").strip().splitlines()
                                if cmd:
                                    bash_heads.append(cmd[0][:80])
    except Exception:
        return None
    return {
        "session": path.stem,
        "first_user": first_user,
        "user_turns": user_turns,
        "user_msgs": user_msgs,
        "tools": dict(tools),
        "bash_heads": bash_heads[:25],
        "fired": sorted(fired),
    }


def is_substantive(s: dict) -> bool:
    """Filter out cron one-shots / trivial sessions. Needs real back-and-forth
    AND real tool work — that's where a skill could plausibly have helped."""
    if s["user_turns"] < 2:
        return False
    if sum(s["tools"].values()) < 3:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=1.0, help="look-back window (days)")
    ap.add_argument("--max-sessions", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true", help="parse+filter+fired stats, NO LLM judge")
    args = ap.parse_args()

    skills = load_skills()
    skill_names = set(skills)
    cutoff = time.time() - args.days * 86400

    paths = [p for p in PROJECTS_DIR.glob("**/*.jsonl") if p.stat().st_mtime >= cutoff]
    print(f"skills: {len(skills)} | jsonl in window ({args.days}d): {len(paths)}")

    sessions = []
    for p in paths:
        s = parse_session(p, skill_names)
        if s and is_substantive(s):
            sessions.append(s)
    sessions.sort(key=lambda s: s["user_turns"], reverse=True)
    print(f"substantive sessions (>=2 user turns, >=3 tools): {len(sessions)}")

    fired_hist: Counter = Counter()
    for s in sessions:
        for sk in s["fired"]:
            fired_hist[sk] += 1
    print("\n=== skills FIRED (across substantive sessions) ===")
    for sk in sorted(skills):
        print(f"  {sk:28s} fired in {fired_hist.get(sk, 0):3d} / {len(sessions)} sessions")

    if args.dry_run:
        print(
            f"\n[dry-run] would judge under-trigger on up to {min(len(sessions), args.max_sessions)} sessions. No LLM called."
        )
        # show a couple sample session views so I can sanity-check parsing
        print("\n=== sample substantive sessions ===")
        for s in sessions[:3]:
            print(
                f"  [{s['session'][:8]}] turns={s['user_turns']} tools={sum(s['tools'].values())} fired={s['fired']}"
            )
            print(f"      req: {s['first_user'][:120]!r}")
            print(f"      tools: {dict(sorted(s['tools'].items(), key=lambda x: -x[1]))}")
        return

    # --- judge phase: one call per session -> under-trigger signal ---
    catalog = "\n".join(f"- {n}: {d}" for n, d in sorted(skills.items()))
    under = Counter()  # skill -> applicable-but-didn't-fire count
    per_session = []
    judged = 0
    for s in sessions[: args.max_sessions]:
        verdict = analyze_session(s, catalog)
        if verdict is None:
            continue
        judged += 1
        miss = [
            m
            for m in verdict.get("would_have_helped", [])
            if m.get("skill") in skill_names and m.get("skill") not in s["fired"]
        ]
        for m in miss:
            under[m["skill"]] += 1
        per_session.append({"session": s["session"], "fired": s["fired"], "missed": miss})

    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "window_days": args.days,
        "substantive_sessions": len(sessions),
        "judged": judged,
        "model": stage_model(lane.JUDGE_STAGE, lane.OPUS),
        "per_skill": {
            n: {
                "fired": fired_hist.get(n, 0),
                "under_trigger": under.get(n, 0),  # recall gap
            }
            for n in sorted(skills)
        },
        "per_session": per_session,
    }
    out_path = OUT_DIR / f"report-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(f"\n=== baseline (judged {judged}/{len(sessions)} sessions, model={report['model']}) ===")
    print(f"{'skill':28s} {'fired':>6} {'under':>6}")
    for n in sorted(skills):
        ps = report["per_skill"][n]
        print(f"{n:28s} {ps['fired']:>6} {ps['under_trigger']:>6}")
    print(f"\nwrote {out_path}")
    # surface the actual missed-trigger evidence (this is the RETUNE fuel)
    print("\n=== under-trigger evidence (would've helped but didn't fire) ===")
    for p in per_session:
        for m in p["missed"]:
            print(f"  [{p['session'][:8]}] {m['skill']}: {m.get('why', '')[:140]}")


_ANALYZE_PROMPT = """You audit AI-agent skill routing and skill coverage. Below are the full skill catalog and one coding-session summary.
Skills auto-fire when the user's phrasing matches a skill's description. Judge two things STRICTLY, each on its own:

1. would_have_helped: skills that did NOT fire this session but CLEARLY should have (the work squarely matches the skill's purpose). Be conservative — empty list if none obviously apply. Do NOT list a skill that did fire.
   For EACH would_have_helped skill, compare the user's wording against that skill's CURRENT catalog description (the text after its name in the catalog — its trigger surface):
   - description_covers: true when the description ALREADY names what the user asked for (the same words, tool, command, product, file or concept). The skill was a match and simply was not invoked; adding trigger words would change nothing.
   - description_covers: false ONLY when there is a real vocabulary gap: the user used wording, a file, a tool, a setting name or a concept that the description does not mention.
   - missing_phrasing: when description_covers is false, the specific user wording or concept the description lacks (a few words, quoted from the user where possible). Use "" when description_covers is true.

2. gaps: a "gap" = a RECURRING, GENERALIZABLE, multi-step PROCEDURE the agent carried out (visible in the tools / bash commands / user requests) that NO catalog skill covers and that a skill SHOULD exist for.
   STRICT — false positives are expensive (a bad skill pollutes every future session):
   - It must be a repeatable PROCEDURE with an OBSERVABLE tool/command signature, not a one-off and not pure deliberation/chat.
   - It must NOT be covered by any catalog skill. If a skill NEARLY covers it, that's a retune of that skill (item 1), NOT a gap -> skip it.
   - It must generalize beyond this exact task (would plausibly recur on a different day or topic).
   - Most sessions have NO gap. Return an empty list rather than reaching.

SKILL CATALOG (name: current description):
{catalog}

SESSION:
- skills that FIRED: {fired}
- tools used (name: count): {tools}
- sample bash commands: {bash}
- USER MESSAGES across the session (the trigger surface — judge under-trigger against THESE phrasings, not just the first):
{user_msgs}

Output ONLY a JSON object, no prose:
{{"would_have_helped": [{{"skill": "name", "why": "one sentence", "description_covers": true or false, "missing_phrasing": "the user wording the description lacks; empty string when covered"}}],
"gaps": [{{"procedure": "short-kebab-name", "what": "one sentence describing the procedure",
"trigger_phrasings": ["how a user would ask for it", "another phrasing"],
"signature": "the tool/command pattern observed (e.g. rg -> Read -> Edit -> Bash pytest)",
"why_generalizable": "one sentence"}}]}}"""


# --- under-trigger gate: does a would_have_helped item show a real description gap? ---
# Whole words dropped before the deterministic coverage match, so filler ("run THIS sql")
# can't break an otherwise literal match. Only standalone words: the parts of a compound
# identifier (pipeline_a.yml -> pipeline, a, yml) are always kept.
_COVER_STOP = frozenset(
    "a an the this that these those it its to of in on at by for from with into and or "
    "is are be me my i we our you your please can could would should some any".split()
)


def _cover_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens in order. Punctuation, quotes and identifier
    separators (_ . - /) all split, so `SYNAPSE_RECALL_FLOOR` -> synapse recall floor and
    `pipeline_c.yml` -> pipeline c yml. Standalone stopwords are dropped."""
    out: list[str] = []
    for word in (text or "").lower().split():
        parts = re.findall(r"[a-z0-9]+", word)
        if len(parts) == 1 and parts[0] in _COVER_STOP:
            continue
        out.extend(parts)
    return out


def description_covers_phrasing(phrasing: str, description: str) -> bool:
    """Deterministic backstop for the under-trigger judge: True when `phrasing` is already
    literally present in the skill `description` — its normalized token sequence appears
    contiguously in the description's (the `Triggers on:` list is part of the description
    text, so a trigger phrase counts). Conservative by design: a phrasing with any word the
    description lacks (`pipeline_c.yml` against a description naming only pipeline_a.yml /
    pipeline_b.yml) is NOT covered, and an empty or all-stopword phrasing is never covered."""
    p = _cover_tokens(phrasing)
    if not p:
        return False
    return f" {' '.join(p)} " in f" {' '.join(_cover_tokens(description))} "


def _judge_bool(v) -> bool | None:
    """A judge-emitted boolean: real bools, or the strings "true"/"false" (LLMs drift).
    None when absent or unparseable."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    return None


def under_trigger_verdict(item: dict, description: str, phrasing: str = "") -> str:
    """Classify one would_have_helped item against the skill's CURRENT description:
      "gap"      — judge says description_covers=false AND names a non-empty
                   missing_phrasing that the description really lacks -> widen fuel;
      "covered"  — judge says description_covers=true: the skill matched but wasn't invoked;
      "backstop" — judge claims a gap, but its missing_phrasing (or the session `phrasing`)
                   is already literally in the description -> treated as covered;
      "no_gap"   — no stated gap: legacy output without the fields, description_covers
                   missing/unparseable, or false with an empty missing_phrasing.
    Safe default: only "gap" may widen a skill; everything else never does."""
    covers = _judge_bool(item.get("description_covers"))
    if covers is True:
        return "covered"
    missing = str(item.get("missing_phrasing") or "").strip()
    if covers is None or not missing:
        return "no_gap"
    if description_covers_phrasing(missing, description) or description_covers_phrasing(
        phrasing, description
    ):
        return "backstop"
    return "gap"


def _extract_json(raw: str) -> dict | None:
    """Lenient JSON extraction from a judge response: the first `{` .. last `}` slice.
    None on no-brace or parse failure. Shared by every dream->skills detector."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except Exception:
        return None


def _excerpt(episodes: list[dict], excerpt_chars: int = 700) -> str:
    """Render a transcript window for a judge prompt: one `[seq N]` block per episode
    (content, else human_turn), each capped at `excerpt_chars`. Shared leaf helper."""
    lines = []
    for ep in episodes:
        text = (ep.get("content") or "").strip() or (ep.get("human_turn") or "").strip()
        if len(text) > excerpt_chars:
            text = text[:excerpt_chars] + " …"
        lines.append(f"[seq {ep['sequence']}]\n{text}")
    return "\n\n".join(lines)


def _clamp_salience(v) -> int | None:
    """Coerce a judge-provided salience to 1..5; None when unparseable. Shared leaf helper."""
    try:
        return max(1, min(5, int(v)))
    except (TypeError, ValueError):
        return None


def scan_night() -> str:
    """The UTC scan-night stamp (YYYY-MM-DD) every detector tags its evidence with."""
    return datetime.now(UTC).date().isoformat()


def _sample(msgs: list[str], cap: int = 40) -> str:
    """All user phrasings if few; an evenly-spaced sample if many (bounds prompt size)."""
    if len(msgs) > cap:
        step = len(msgs) / cap
        msgs = [msgs[int(i * step)] for i in range(cap)]
    return "\n".join(f"  - {m}" for m in msgs)


def analyze_session(s: dict, catalog: str) -> dict | None:
    """The lane's one LLM read of a session: {would_have_helped, gaps} against the skill
    `catalog` (format_catalog). None when the reply carries no JSON object. Raises when the
    model cannot be reached (lane.judge)."""
    prompt = _ANALYZE_PROMPT.format(
        catalog=catalog,
        fired=s["fired"] or "(none)",
        tools=dict(sorted(s["tools"].items(), key=lambda x: -x[1])),
        bash=" | ".join(s["bash_heads"][:12]),
        user_msgs=_sample(s.get("user_msgs", [])),
    )
    return _extract_json(lane.judge(prompt))


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
