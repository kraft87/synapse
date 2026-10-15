#!/usr/bin/env python3
# mypy: ignore-errors
"""dream->config nightly lane (server-side in the dream container; entry: run_lane()).

Mines transcripts for BEHAVIORAL CORRECTIONS — moments the user corrected HOW the agent
writes/works/behaves (not task content) — and accumulates them into config_lane.config_proposals.
Mirrors dream/skills: scan the sessions its episode window touched (dream/lane.py: oldest first,
overflow rolls to the next night) -> LLM finds corrections -> union evidence by session in the
ledger -> gate observe->proposed at N distinct sessions.

PROPOSE-ONLY: writes config_proposals (the review CLI + apply phase land edits on disk). Proposals
target a dedicated `rules/learned.md` (global) so dream's contributions never rewrite a hand-kept
file. ON by default; CONFIG_LANE_ENABLED=0 (read by dream/__main__) turns it off.

A row's status decides what a sighting of its rule does: observe and proposed rows gather
evidence toward (and wait at) review; accepted and applied rows gather it as a measure of
recurrence (an applied rule that is corrected again is a rule the agent does not follow); a
rejected row inside its cooling window does the same and raises nothing. A rule the user already
has in a mirrored config file or a live feedback note is not proposed at all.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from typing import NamedTuple

import psycopg

from dream import lane
from dream.skills import skill_measure as SM
from dream.skills.config import db_url
from dream.skills.skill_measure import _extract_json

LEARNED_FILE = os.environ.get("CONFIG_LEARNED_FILE", "rules/learned.md")
PROPOSE_SESSIONS = int(
    os.environ.get("CONFIG_PROPOSE_SESSIONS", "2")
)  # distinct sessions -> proposed
IDENTITY_JACCARD = 0.5  # rule-token overlap above which two corrections are "the same rule".
# Rough dedup; the LLM tends to normalize phrasing for the same correction, and the review gate
# catches any duplicates. Embedding-based semantic identity is the robustness upgrade (later).
REJECT_COOLING_DAYS = 30  # a rejected rule is not proposed again this long after the rejection
OBSERVE_TTL_DAYS = 28  # an observe row no session has re-sighted for this long is dropped

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


# not/never/dont are stop words, so "Run the tests" and "Do not run the tests" have identical
# tokens; the polarity is read from the raw words instead.
_NEGATION = re.compile(r"\b(?:not|never|no|avoid|stop|dont)\b|n['\u2019]t\b")


class _Rule(NamedTuple):
    tokens: set[str]
    negated: bool


def _rule(text: str) -> _Rule:
    return _Rule(_sig_tokens(text), bool(_NEGATION.search((text or "").lower())))


def _same_rule(a: _Rule, b: _Rule) -> bool:
    """Whether two rules say the same thing: one polarity (a rule and its reversal share every
    token) and rule-token jaccard at or above IDENTITY_JACCARD."""
    return a.negated == b.negated and _jaccard(a.tokens, b.tokens) >= IDENTITY_JACCARD


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


def _best_match(rule: _Rule, candidates: list[dict]) -> dict | None:
    """The candidate whose summary is the same rule as `rule` (closest by jaccard), or None."""
    best, best_j = None, 0.0
    for c in candidates:
        other = _rule(c["summary"])
        j = _jaccard(rule.tokens, other.tokens)
        if _same_rule(rule, other) and j > best_j:
            best, best_j = c, j
    return best


# --------------------------------------------------------------------------- LLM finder
def scan_corrections(s: dict) -> list[dict] | None:
    """The corrections the judge finds in one session, or None when its reply has no JSON
    object (told apart from "found nothing", which is an empty list). Raises when the model
    cannot be reached (lane.judge)."""
    raw = lane.judge(_FIND_PROMPT.format(user_msgs=SM._sample(s.get("user_msgs", []))))
    obj = _extract_json(raw)
    if obj is None:
        return None
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
class Upsert(NamedTuple):
    """What one correction did to the ledger. `sessions` is the distinct sessions in the row's
    evidence; `raised` that this call put the rule in front of review (inserted as proposed, or
    observe -> proposed); `recurred` that an applied rule was corrected again, in a session its
    evidence lacked."""

    id: int
    status: str
    sessions: int
    raised: bool
    recurred: bool


def _upsert_correction(cur, c: dict) -> Upsert:
    """Resolve identity (rule-token jaccard) against the config proposals, union this session's
    evidence, recompute distinct sessions, gate observe->proposed.

    Every status takes part in identity except a rejected row past its cooling window, which is
    history: a sighting then opens a fresh observe row. Only observe and proposed rows are still
    moving; the sighting of an accepted, applied or cooling rejected rule is added to its
    evidence and nothing else changes (updated_at dates a rejection, so it stays)."""
    rule = _rule(c["rule"])
    ev_new = [
        {"session_id": c["session_id"], "ts": datetime.now(UTC).isoformat(), "quote": c["quote"]}
    ]
    cur.execute(
        "SELECT id, summary, evidence, status FROM config_lane.config_proposals "
        "WHERE file_key=%s AND (status <> 'rejected' "
        "OR updated_at > now() - make_interval(days => %s))",
        (LEARNED_FILE, REJECT_COOLING_DAYS),
    )
    match = _best_match(
        rule,
        [{"id": r[0], "summary": r[1], "evidence": r[2], "status": r[3]} for r in cur.fetchall()],
    )
    if match:
        evidence = _union_evidence(match["evidence"], ev_new)
        n = _distinct_sessions(evidence)
        if match["status"] in ("observe", "proposed"):
            status = (
                "proposed"
                if (match["status"] == "proposed" or n >= PROPOSE_SESSIONS)
                else "observe"
            )
            cur.execute(
                "UPDATE config_lane.config_proposals SET evidence=%s::jsonb, status=%s, "
                "updated_at=now() WHERE id=%s",
                (json.dumps(evidence), status, match["id"]),
            )
            raised = match["status"] == "observe" and status == "proposed"
            return Upsert(match["id"], status, n, raised, False)
        new_session = len(evidence) > len(match["evidence"])
        if new_session:
            cur.execute(
                "UPDATE config_lane.config_proposals SET evidence=%s::jsonb WHERE id=%s",
                (json.dumps(evidence), match["id"]),
            )
        recurred = new_session and match["status"] == "applied"
        return Upsert(match["id"], match["status"], n, False, recurred)
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
    return Upsert(cur.fetchone()[0], status, n, status == "proposed", False)


def _expire_observed(cur) -> int:
    """Drop the observe rows no session has re-sighted for OBSERVE_TTL_DAYS. A rule corrected
    once and never again is not a standing preference, and the paraphrases nothing matched
    would otherwise pile up for every later correction to read. Proposed rows wait for a human
    and never expire. Returns how many went."""
    cur.execute(
        "DELETE FROM config_lane.config_proposals "
        "WHERE status='observe' AND updated_at < now() - make_interval(days => %s)",
        (OBSERVE_TTL_DAYS,),
    )
    return cur.rowcount


def _known_rules(cur) -> list[_Rule]:
    """The rules the user already has: every line of every mirrored config file (the lane's own
    learned file apart: a correction that matches an applied rule there is a recurrence, which
    the ledger counts) and the hook of every live feedback note. A correction that restates one
    is not a new rule. Headings, labels and lines under three tokens name a topic and state no
    rule, so they are too thin to match on."""
    cur.execute(
        "SELECT content FROM config_lane.config_registry WHERE file_key <> %s", (LEARNED_FILE,)
    )
    lines = [ln for (content,) in cur.fetchall() for ln in content.splitlines()]
    lines = [
        ln for ln in lines if not ln.lstrip().startswith("#") and not ln.rstrip().endswith(":")
    ]
    cur.execute("SELECT hook FROM notes WHERE type='feedback' AND superseded_by IS NULL")
    lines += [hook for (hook,) in cur.fetchall()]
    return [r for r in map(_rule, lines) if len(r.tokens) >= 3]


# --------------------------------------------------------------------------- orchestrator
def run_lane(limit: int = 30, backfill: bool = False) -> dict:
    # Autocommit: every write below stands alone, and a transaction opened by the first read
    # would sit idle (holding locks) through all of the judge calls.
    conn = psycopg.connect(db_url(), autocommit=True)
    try:
        cur = conn.cursor()
        cur.execute("SELECT last_scan_at FROM config_lane.config_scan_cursor WHERE id")
        row = cur.fetchone()
        scan_started = datetime.now(UTC)
        window = lane.open_window(
            conn, "config", legacy_since=row[0] if row else None, backfill=backfill
        )
        # A correction needs a turn to correct, so a single-turn session is not worth a call.
        views, carry = lane.sessions(
            conn, window, limit=limit, wanted=lambda v: v["user_turns"] >= 2
        )
        print(
            f"[dream→config] episodes {window.after_id}..{carry} (newest {window.until_id})  "
            f"sessions={len(views)}"
        )
        expired = _expire_observed(cur)
        known = _known_rules(cur)
        found = proposed = recurred = already_known = unparsed = 0
        for corrections in lane.run_parallel(scan_corrections, views):
            if corrections is None:
                unparsed += 1
                continue
            for c in corrections:
                found += 1
                rule = _rule(c["rule"])
                if any(_same_rule(rule, k) for k in known):
                    already_known += 1
                    print(f"  [known] {c['rule'][:70]}")
                    continue
                u = _upsert_correction(cur, c)
                proposed += u.raised
                recurred += u.recurred
                print(f"  [{u.status} n={u.sessions}] {c['rule'][:70]}")

        lane.mark_judged(conn, "config", views)
        lane.advance(conn, window, carry)
        # last_scan_at is the cursor a database without the dream lane cursor tables (and an
        # older image) reads; see dream/lane.py.
        cur.execute(
            "INSERT INTO config_lane.config_scan_cursor (id, last_scan_at) VALUES (true, %s) "
            "ON CONFLICT (id) DO UPDATE SET last_scan_at=EXCLUDED.last_scan_at",
            (scan_started,),
        )
        return {
            "sessions": len(views),
            "found": found,
            "proposed": proposed,
            "recurred": recurred,
            "already_known": already_known,
            "unparsed": unparsed,
            "expired": expired,
        }
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--backfill", action="store_true")
    a = ap.parse_args()
    print(run_lane(limit=a.limit, backfill=a.backfill))
