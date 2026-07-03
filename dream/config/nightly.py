#!/usr/bin/env python3
# mypy: ignore-errors
"""dream->config nightly lane (server-side in the dream container; entry: run_lane()).

Mines transcripts for BEHAVIORAL CORRECTIONS — moments the user corrected HOW the agent
writes/works/behaves (not task content) — and accumulates them into config_lane.config_proposals.
Mirrors dream/skills: scan sessions touched since the watermark -> LLM finds corrections -> union
evidence by session in the ledger -> gate observe->proposed at N distinct sessions.

PROPOSE-ONLY: writes config_proposals (the review CLI + apply phase land edits on disk). Proposals
target a dedicated `rules/learned.md` (global) so dream's contributions never rewrite a hand-kept
file. Gated CONFIG_LANE_ENABLED; ships inert.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime

import psycopg

from dream.skills import skill_db_source as DB
from dream.skills import skill_measure as SM
from dream.skills.config import db_url
from dream.skills.skill_derive import _extract_json

LEARNED_FILE = os.environ.get("CONFIG_LEARNED_FILE", "rules/learned.md")
PROPOSE_SESSIONS = int(
    os.environ.get("CONFIG_PROPOSE_SESSIONS", "2")
)  # distinct sessions -> proposed
IDENTITY_JACCARD = 0.5  # rule-token overlap above which two corrections are "the same rule".
# Rough dedup; the LLM tends to normalize phrasing for the same correction, and the review gate
# catches any duplicates. Embedding-based semantic identity is the robustness upgrade (later).

_STOP = {
    "the",
    "a",
    "an",
    "to",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "not",
    "do",
    "dont",
    "is",
    "be",
    "use",
    "using",
    "when",
    "with",
    "your",
    "you",
    "it",
    "that",
    "this",
    "always",
    "never",
    "should",
    "must",
    "agent",
    "claude",
}

_FIND_PROMPT = """You audit how well an AI agent follows its user's BEHAVIORAL preferences.
Below are the USER MESSAGES from one session. Find moments where the user CORRECTED the agent's
behavior, style, or working approach — a standing preference the agent should follow (it did the
wrong thing and got corrected, or the user repeated an instruction).

STRICT — false positives pollute the user's config, so under-report:
- Must be a GENERALIZABLE behavioral rule (would apply on a different day or topic), NOT a one-off
  task instruction ("rename this var") and NOT domain/work content.
- Must be about the agent's CONDUCT: tone, format, verbosity, what to do/avoid, process/workflow.
- Most sessions have NONE. Return an empty list rather than reaching.

USER MESSAGES:
{user_msgs}

Output ONLY a JSON object, no prose:
{{"corrections": [{{"rule": "imperative rule, e.g. 'Do not use em-dashes in prose'",
"quote": "the user's words showing the correction"}}]}}"""


# --------------------------------------------------------------------------- pure helpers
def _sig_tokens(rule: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9]+", (rule or "").lower()) if t not in _STOP and len(t) > 1
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _distinct_sessions(evidence: list[dict]) -> int:
    return len({e.get("session_id") for e in evidence if e.get("session_id")})


def _union_evidence(old: list[dict], new: list[dict]) -> list[dict]:
    """Append new entries, deduped by session_id — keeps distinct-session counting honest."""
    seen = {e.get("session_id") for e in old}
    out = list(old)
    for e in new:
        if e.get("session_id") not in seen:
            out.append(e)
            seen.add(e.get("session_id"))
    return out


def _best_match(rule_tokens: set[str], candidates: list[dict]) -> dict | None:
    """The active candidate whose rule is the same as rule_tokens (jaccard >= threshold), or None."""
    best, best_j = None, 0.0
    for c in candidates:
        j = _jaccard(rule_tokens, _sig_tokens(c["summary"]))
        if j >= IDENTITY_JACCARD and j > best_j:
            best, best_j = c, j
    return best


# --------------------------------------------------------------------------- LLM finder
def scan_corrections(s: dict) -> list[dict]:
    try:
        raw = SM._run_judge(_FIND_PROMPT.format(user_msgs=SM._sample(s.get("user_msgs", []))))
    except Exception as e:
        print(f"  config scan failed for {s['session'][:8]}: {e}")
        return []
    obj = _extract_json(raw) or {}
    out = []
    for c in obj.get("corrections", []):
        rule = (c.get("rule") or "").strip()
        if rule:
            out.append(
                {
                    "rule": rule,
                    "session_id": s["session"],
                    "quote": (c.get("quote") or "")[:300],
                }
            )
    return out


# --------------------------------------------------------------------------- ledger
def _upsert_correction(cur, c: dict) -> tuple[int, str, int]:
    """Resolve identity (rule-token jaccard) against active config proposals, union this session's
    evidence, recompute distinct sessions, gate observe->proposed. Returns (id, status, n_sessions)."""
    toks = _sig_tokens(c["rule"])
    ev_new = [
        {"session_id": c["session_id"], "ts": datetime.now(UTC).isoformat(), "quote": c["quote"]}
    ]
    cur.execute(
        "SELECT id, summary, evidence, status FROM config_lane.config_proposals "
        "WHERE file_key=%s AND status IN ('observe','proposed')",
        (LEARNED_FILE,),
    )
    match = _best_match(
        toks,
        [{"id": r[0], "summary": r[1], "evidence": r[2], "status": r[3]} for r in cur.fetchall()],
    )
    if match:
        evidence = _union_evidence(match["evidence"], ev_new)
        n = _distinct_sessions(evidence)
        status = (
            "proposed" if (match["status"] == "proposed" or n >= PROPOSE_SESSIONS) else "observe"
        )
        cur.execute(
            "UPDATE config_lane.config_proposals SET evidence=%s::jsonb, status=%s, updated_at=now() "
            "WHERE id=%s",
            (json.dumps(evidence), status, match["id"]),
        )
        return match["id"], status, n
    # new candidate at observe (or proposed if the gate is 1)
    n = 1
    status = "proposed" if n >= PROPOSE_SESSIONS else "observe"
    # scope='general': a behavioral rule applies across the user's surfaces (config_proposals.scope is
    # blast radius = local|general, NOT the registry's global|project axis). Review can flip to local.
    cur.execute(
        "INSERT INTO config_lane.config_proposals (kind, file_key, scope, diff, summary, evidence, status) "
        "VALUES ('add', %s, 'general', %s, %s, %s::jsonb, %s) RETURNING id",
        (LEARNED_FILE, c["rule"], c["rule"], json.dumps(ev_new), status),
    )
    return cur.fetchone()[0], status, n


# --------------------------------------------------------------------------- orchestrator
def run_lane(limit: int = 30, backfill: bool = False) -> dict:
    conn = psycopg.connect(db_url())
    try:
        cur = conn.cursor()
        cur.execute("SELECT last_scan_at FROM config_lane.config_scan_cursor WHERE id")
        row = cur.fetchone()
        last = None if backfill else (row[0] if row else None)
        scan_started = datetime.now(UTC)

        views = DB.sessions_since(last, max_sessions=limit * 4)
        print(f"[dream→config] last_scan_at={last}  sessions={len(views)}")
        found = proposed = 0
        for s in views:
            for c in scan_corrections(s):
                found += 1
                _id, status, n = _upsert_correction(cur, c)
                if status == "proposed":
                    proposed += 1
                print(f"  [{status} n={n}] {c['rule'][:70]}")
        conn.commit()

        cur.execute(
            "INSERT INTO config_lane.config_scan_cursor (id, last_scan_at) VALUES (true, %s) "
            "ON CONFLICT (id) DO UPDATE SET last_scan_at=EXCLUDED.last_scan_at",
            (scan_started,),
        )
        conn.commit()
        return {"sessions": len(views), "found": found, "proposed": proposed}
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--backfill", action="store_true")
    a = ap.parse_args()
    print(run_lane(limit=a.limit, backfill=a.backfill))
