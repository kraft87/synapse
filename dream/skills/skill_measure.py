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
    SKILL_MEASURE_MODEL=deepseek skill-measure.py --days 30 # cheap backfill
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

from . import config

# No on-disk skill catalog server-side (the dream container has no ~/.claude/skills). The
# catalog comes from the skill_registry table; the client's skills_sync owns disk<->registry.
PROJECTS_DIR = config.PROJECTS_DIR  # dev-only --source jsonl path; the nightly reads PG
OUT_DIR = config.DATA_DIR


def skill_description(text: str) -> str:
    """Parse the frontmatter `description`, handling YAML block scalars (`>` / `|`),
    plain, and quoted forms. The naive `^description:\\s*(.+)$` grabbed just ">" for
    multi-line block-scalar descriptions — which silently fed the judge garbage."""
    m = re.search(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return ""
    lines = m.group(1).split("\n")
    for i, line in enumerate(lines):
        dm = re.match(r"^(\s*)description:\s*(.*)$", line)
        if not dm:
            continue
        base_indent, val = len(dm.group(1)), dm.group(2).strip()
        if val[:1] in ("|", ">"):  # block scalar — collect more-indented following lines
            collected = []
            for nxt in lines[i + 1 :]:
                if not nxt.strip():
                    continue
                if len(nxt) - len(nxt.lstrip()) <= base_indent:
                    break  # dedent = next key
                collected.append(nxt.strip())
            return " ".join(collected).strip()
        return val.strip("\"'")
    return ""


def load_skills() -> dict[str, str]:
    """name -> description for ACTIVE skills, read from the skill_registry table.

    Server-side the catalog IS the registry (the dream container has no ~/.claude/skills); the
    client's skills_sync owns the disk<->registry publish. `description` is the trigger surface."""
    import psycopg

    skills: dict[str, str] = {}
    with psycopg.connect(config.db_url(), connect_timeout=10) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT name, COALESCE(description, '') FROM skills_lane.skill_registry "
            "WHERE status = 'active'"
        )
        for name, desc in cur.fetchall():
            skills[name] = desc
    return skills


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

    # --- judge phase: one call per session -> under-trigger + dismissal signals ---
    catalog = "\n".join(f"- {n}: {d}" for n, d in sorted(skills.items()))
    under = Counter()  # skill -> applicable-but-didn't-fire count
    dismissed = Counter()  # skill -> fired-but-looked-like-a-mismatch count
    per_session = []
    judged = 0
    for s in sessions[: args.max_sessions]:
        verdict = judge_session(s, catalog)
        if verdict is None:
            continue
        judged += 1
        miss = [
            m
            for m in verdict.get("would_have_helped", [])
            if m.get("skill") in skill_names and m.get("skill") not in s["fired"]
        ]
        dis = [m for m in verdict.get("dismissed", []) if m.get("skill") in s["fired"]]
        for m in miss:
            under[m["skill"]] += 1
        for m in dis:
            dismissed[m["skill"]] += 1
        per_session.append(
            {"session": s["session"], "fired": s["fired"], "missed": miss, "dismissed": dis}
        )
        time.sleep(0.3)

    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "window_days": args.days,
        "substantive_sessions": len(sessions),
        "judged": judged,
        "model": os.environ.get("SKILL_MEASURE_MODEL", "opus"),
        "per_skill": {
            n: {
                "fired": fired_hist.get(n, 0),
                "under_trigger": under.get(n, 0),  # recall gap
                "dismissed": dismissed.get(n, 0),  # precision gap
            }
            for n in sorted(skills)
        },
        "per_session": per_session,
    }
    out_path = OUT_DIR / f"report-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(f"\n=== baseline (judged {judged}/{len(sessions)} sessions, model={report['model']}) ===")
    print(f"{'skill':28s} {'fired':>6} {'under':>6} {'dismiss':>8}")
    for n in sorted(skills):
        ps = report["per_skill"][n]
        print(f"{n:28s} {ps['fired']:>6} {ps['under_trigger']:>6} {ps['dismissed']:>8}")
    print(f"\nwrote {out_path}")
    # surface the actual missed-trigger evidence (this is the RETUNE fuel)
    print("\n=== under-trigger evidence (would've helped but didn't fire) ===")
    for p in per_session:
        for m in p["missed"]:
            print(f"  [{p['session'][:8]}] {m['skill']}: {m.get('why', '')[:140]}")


_JUDGE_PROMPT = """You audit AI-agent skill routing. Below is one coding-session summary and the full skill catalog.
Skills auto-fire when the user's phrasing matches a skill's description. Judge two things STRICTLY:

1. would_have_helped: skills that did NOT fire this session but CLEARLY should have (the work squarely matches the skill's purpose). Be conservative — empty list if none obviously apply. Do NOT list a skill that did fire.
2. dismissed: skills that DID fire but look like a MISMATCH (the work diverged from the skill's purpose / it shouldn't have fired).

SKILL CATALOG:
{catalog}

SESSION:
- skills that FIRED: {fired}
- tools used (name: count): {tools}
- sample bash commands: {bash}
- USER MESSAGES across the session (the trigger surface — judge under-trigger against THESE phrasings, not just the first):
{user_msgs}

Output ONLY a JSON object, no prose:
{{"would_have_helped": [{{"skill": "name", "why": "one sentence"}}], "dismissed": [{{"skill": "name", "why": "one sentence"}}]}}"""


def _run_judge(prompt: str) -> str:
    backend = os.environ.get("SKILL_MEASURE_MODEL", config.JUDGE_BACKEND)
    if backend in ("deepseek", "openrouter"):
        return _openrouter_judge(prompt)
    # default: claude CLI (no API key, runs on the Max subscription; nightly volume is tiny)
    r = subprocess.run(
        ["claude", "-p", prompt, "--model", config.JUDGE_MODEL],
        capture_output=True,
        text=True,
        timeout=240,
    )
    return r.stdout


def _openrouter_judge(prompt: str) -> str:
    import urllib.request

    key = config.secret("OPENROUTER_API_KEY")
    body = json.dumps(
        {"model": "deepseek/deepseek-chat", "messages": [{"role": "user", "content": prompt}]}
    ).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=240).read())["choices"][0]["message"][
        "content"
    ]


def _sample(msgs: list[str], cap: int = 40) -> str:
    """All user phrasings if few; an evenly-spaced sample if many (bounds prompt size)."""
    if len(msgs) > cap:
        step = len(msgs) / cap
        msgs = [msgs[int(i * step)] for i in range(cap)]
    return "\n".join(f"  - {m}" for m in msgs)


def judge_session(s: dict, catalog: str) -> dict | None:
    prompt = _JUDGE_PROMPT.format(
        catalog=catalog,
        fired=s["fired"] or "(none)",
        tools=dict(sorted(s["tools"].items(), key=lambda x: -x[1])),
        bash=" | ".join(s["bash_heads"][:12]),
        user_msgs=_sample(s.get("user_msgs", [])),
    )
    try:
        raw = _run_judge(prompt)
    except Exception as e:
        print(f"  judge failed for {s['session'][:8]}: {e}")
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except Exception:
        return None


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
