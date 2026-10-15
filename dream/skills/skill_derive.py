#!/usr/bin/env python3
# mypy: ignore-errors
"""dream->skills, step 2 (DERIVE): find skill GAPS in raw transcripts, draft new skills.

skill-measure.py audits EXISTING skills (under-trigger = RETUNE fuel). This is the
other half: spot recurring, generalizable multi-step PROCEDURES the
agent carried out by hand that NO existing skill covers, cluster the candidates ACROSS
sessions (frequency floor — don't overfit one utterance), then DRAFT a SKILL.md for
each survivor. Propose-only: writes to ~/data/skill-measure/proposals/, never touches
~/.claude/skills/. (The nightly lane stores its drafts on the candidate row instead; the
review accept applies them to the skill registry.)

Pipeline:
  1. parse raw transcripts (reuses skill-measure: same session view)
  2. per-session analysis (skill_measure.analyze_session, shared with the retune judge)
     -> candidate gaps with tool/command signature
  3. cross-session CLUSTER (1 call) -> merge dup candidates, count distinct sessions
  4. frequency floor (>=2 sessions by default) -> survivors
  5. per-survivor DRAFT (Opus, quality matters) -> SKILL.md + proposal.json

Models: the analysis and cluster calls use SYNAPSE_SKILLS_JUDGE_MODEL, drafting
SYNAPSE_SKILLS_DRAFT_MODEL (llm_client.stage_model; both default to Opus).

    skill-derive.py --days 30 --dry-run                      # parse only, no LLM
    skill-derive.py --days 7 --min-sessions 1                # show mechanism on a thin window
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict

from dream import lane
from ingestion.llm_client import stage_model

from . import config, skill_db_source, skill_measure

load_skills = skill_measure.load_skills
parse_session = skill_measure.parse_session
is_substantive = skill_measure.is_substantive
_sample = skill_measure._sample
_extract_json = skill_measure._extract_json

PROJECTS_DIR = config.PROJECTS_DIR
OUT_DIR = config.DATA_DIR
PROPOSALS_DIR = config.PROPOSALS_DIR


# --------------------------------------------------------------------------- gaps
def session_gaps(s: dict, analysis: dict | None) -> list[dict]:
    """The gaps in a session's analysis (skill_measure.analyze_session), each tagged with the
    session it came from."""
    gaps = (analysis or {}).get("gaps") or []
    for g in gaps:
        g["_session"] = s["session"]
        g["_bash_heads"] = s["bash_heads"][:12]
    return gaps


# ------------------------------------------------------------------------ cluster

_CLUSTER_PROMPT = """You are consolidating candidate "skill gaps" mined from many separate agent sessions.
Different sessions describe the SAME underlying procedure with different words. Merge them.

CANDIDATES (each has a source session id):
{candidates}

Group candidates that describe the same underlying repeatable procedure. For each group output a
single merged cluster. Drop singletons that look like one-off tasks. Be conservative — a cluster is
only worth a skill if the procedure is genuinely repeatable.

Output ONLY JSON, no prose:
{{"clusters": [{{"procedure": "short-kebab-name", "what": "one-sentence merged description",
"trigger_phrasings": ["merged, deduped phrasings"], "signature": "merged tool/command pattern",
"sessions": ["session-id", "..."]}}]}}"""


def cluster_gaps(gaps: list[dict]) -> list[dict]:
    if not gaps:
        return []
    lines = []
    for g in gaps:
        lines.append(
            f"- [{g.get('_session', '?')[:8]}] {g.get('procedure', '?')}: {g.get('what', '')} "
            f"| triggers={g.get('trigger_phrasings', [])} | sig={g.get('signature', '')}"
        )
    raw = lane.judge(_CLUSTER_PROMPT.format(candidates="\n".join(lines)))
    obj = _extract_json(raw) or {}
    clusters = obj.get("clusters", [])
    # attach distinct-session count + the observed bash heads from member sessions
    by_session = defaultdict(list)
    for g in gaps:
        by_session[g.get("_session")].append(g)
    for c in clusters:
        sids = {s for s in c.get("sessions", []) if s}
        # tolerate truncated ids from the model
        full = {g["_session"] for g in gaps for sid in sids if g["_session"].startswith(sid)}
        c["sessions"] = sorted(full or sids)
        c["n_sessions"] = len(c["sessions"])
        bash = []
        for sid in c["sessions"]:
            for g in by_session.get(sid, []):
                bash += g.get("_bash_heads", [])
        c["bash_evidence"] = bash[:20]
    return clusters


# -------------------------------------------------------------------------- draft

_DRAFT_PROMPT = """Write a Claude Code SKILL.md for the procedure below, derived from real agent sessions.

A SKILL.md has YAML frontmatter then a markdown body:
---
name: <kebab-name>
description: <ONE line. This is the TRIGGER — the model soft-matches user phrasing against it.
  Pack the natural user phrasings that should invoke this skill. Keep it under ~200 chars. No marketing.>
---
# <name>
<1-2 line purpose, then the numbered procedure: concrete steps with the actual tools/commands observed.>

PROCEDURE TO CAPTURE:
- what: {what}
- trigger phrasings observed: {triggers}
- tool/command signature: {signature}
- real bash commands seen across sessions: {bash}
- seen in {n} distinct sessions

Rules: steps must be concrete and reflect the observed signature; do not invent capabilities not in
evidence; the description must read like real user phrasings, not a feature list. Output ONLY the
SKILL.md file content (frontmatter + body), nothing else."""


def draft_skill(c: dict) -> str:
    raw = lane.draft(
        _DRAFT_PROMPT.format(
            what=c.get("what", ""),
            triggers=c.get("trigger_phrasings", []),
            signature=c.get("signature", ""),
            bash=" | ".join(c.get("bash_evidence", [])) or "(none captured)",
            n=c.get("n_sessions", 0),
        )
    ).strip()
    # strip an OUTER ```fence only if the WHOLE output is wrapped (don't eat inner ```bash blocks)
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
    return raw.strip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--max-sessions", type=int, default=60, help="cap on sessions scanned")
    ap.add_argument("--min-sessions", type=int, default=2, help="frequency floor for a draft")
    ap.add_argument(
        "--source",
        choices=["db", "jsonl"],
        default="db",
        help="db = Synapse episodes table (fast, single-source); jsonl = raw transcripts (has thinking + slash-fires)",
    )
    ap.add_argument("--dry-run", action="store_true", help="parse+filter, no LLM")
    args = ap.parse_args()

    skills = load_skills()
    skill_names = set(skills)
    catalog = "\n".join(f"- {n}: {d}" for n, d in sorted(skills.items()))

    if args.source == "db":
        raw_views = skill_db_source.db_session_views(args.days, args.max_sessions)
        sessions = [s for s in raw_views if is_substantive(s)]
        print(
            f"skills: {len(skills)} | source=db ({args.days}d): {len(raw_views)} sessions | substantive: {len(sessions)}"
        )
    else:
        cutoff = time.time() - args.days * 86400
        paths = [p for p in PROJECTS_DIR.glob("**/*.jsonl") if p.stat().st_mtime >= cutoff]
        sessions = []
        for p in paths:
            s = parse_session(p, skill_names)
            if s and is_substantive(s):
                sessions.append(s)
        sessions.sort(key=lambda s: s["user_turns"], reverse=True)
        print(
            f"skills: {len(skills)} | source=jsonl ({args.days}d): {len(paths)} files | substantive: {len(sessions)}"
        )

    if args.dry_run:
        print(
            f"[dry-run] would scan up to {min(len(sessions), args.max_sessions)} sessions for gaps. No LLM."
        )
        for s in sessions[:5]:
            print(
                f"  [{s['session'][:8]}] turns={s['user_turns']} tools={sum(s['tools'].values())} "
                f"fired={s['fired']} req={s['first_user'][:90]!r}"
            )
        return

    # 2. per-session gap scan
    print(
        f"\n=== scanning {min(len(sessions), args.max_sessions)} sessions for gaps (model={stage_model(lane.JUDGE_STAGE, lane.OPUS)}) ==="
    )
    all_gaps: list[dict] = []
    for s in sessions[: args.max_sessions]:
        g = session_gaps(s, skill_measure.analyze_session(s, catalog))
        if g:
            print(f"  [{s['session'][:8]}] +{len(g)}: {[x.get('procedure') for x in g]}")
            all_gaps += g
        time.sleep(0.2)
    print(f"raw gap candidates: {len(all_gaps)}")

    # 3. cluster across sessions
    clusters = cluster_gaps(all_gaps)
    clusters.sort(key=lambda c: c.get("n_sessions", 0), reverse=True)
    print(f"\n=== clusters ({len(clusters)}) ===")
    for c in clusters:
        flag = "DRAFT" if c.get("n_sessions", 0) >= args.min_sessions else "below-floor"
        print(
            f"  [{flag}] {c.get('procedure')}  (n={c.get('n_sessions')})  {c.get('what', '')[:90]}"
        )

    # 4+5. draft survivors
    survivors = [c for c in clusters if c.get("n_sessions", 0) >= args.min_sessions]
    print(
        f"\n=== drafting {len(survivors)} proposals (floor>={args.min_sessions}, model={stage_model(lane.DRAFT_STAGE, lane.OPUS)}) ==="
    )
    stamp = time.strftime("%Y%m%d-%H%M%S")
    written = []
    for c in survivors:
        name = re.sub(r"[^a-z0-9-]", "", c.get("procedure", "unnamed").lower()) or "unnamed"
        # don't propose a name that already exists as a skill
        if name in skill_names:
            print(
                f"  skip {name}: a skill with that name already exists (retune candidate, not a gap)"
            )
            continue
        body = draft_skill(c)
        d = PROPOSALS_DIR / f"{stamp}-{name}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(body)
        (d / "proposal.json").write_text(json.dumps(c, indent=2))
        written.append(str(d))
        print(f"  wrote {d}/SKILL.md  ({len(body)} chars, n_sessions={c.get('n_sessions')})")

    report = {
        "generated": stamp,
        "window_days": args.days,
        "substantive_sessions": len(sessions),
        "scanned": min(len(sessions), args.max_sessions),
        "raw_candidates": len(all_gaps),
        "clusters": clusters,
        "drafted": written,
        "min_sessions": args.min_sessions,
    }
    rp = OUT_DIR / f"derive-{stamp}.json"
    rp.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {rp}")
    if not written:
        print(
            "no proposals met the frequency floor. (--min-sessions 1 to see thin-window candidates.)"
        )


if __name__ == "__main__":
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    main()
