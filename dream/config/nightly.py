#!/usr/bin/env python3
# mypy: ignore-errors
"""dream->config nightly lane (server-side in the dream container; entry: run_lane()).

Mines transcripts for BEHAVIORAL CORRECTIONS — moments the user corrected HOW the agent
writes/works/behaves (not task content) — and accumulates them into config_lane.config_proposals.
Mirrors dream/skills: scan sessions touched since the watermark -> LLM finds corrections -> union
evidence by session in the ledger -> gate observe->proposed at N distinct sessions.

IDENTITY is by meaning. A correction is the same rule as an existing row when the cosine
between their embeddings clears IDENTITY_COSINE (the row's embedding is its founding rule text,
never moved, so a chain of near-paraphrases cannot drift away from the rule it started as). A
session that already has evidence on the row is held to the lower IDENTITY_COSINE_SAME_SESSION:
a long session is re-scanned whole each night and the finder words the same correction
differently each time, so a paraphrase from a session the row already counts is the row
repeating itself. The earlier token-Jaccard test never matched across sessions (best
cross-session pair 0.294 against a 0.5 bar), so no row ever reached two sessions.

BEFORE A ROW IS PROPOSED it is checked against the user's own rule files (config_registry,
scope 'global'): one judge call asks whether an existing rule already covers it. Covered rows
go to 'rejected' / 'auto:covered'. The review queue tends itself (review_queue.py): a stored
recommendation per proposal, a 21-day expiry, and a persisted report.

Rows that are not open still answer to a re-sighting: an accepted or applied rule, and one
rejected in the last COOLDOWN_DAYS (by a person or as covered), absorb the evidence without
coming back; an expired proposal reopens when a NEW session repeats it.

PROPOSE-ONLY: writes config_proposals (the review CLI + apply phase land edits on disk).
Proposals target a dedicated `rules/learned.md` (global) so dream's contributions never
rewrite a hand-kept file; they stay ask-only. Gated CONFIG_LANE_ENABLED. Needs schema 065 (one
probe, one log line, and the lane idles until it is applied).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import psycopg
from psycopg.rows import class_row
from psycopg.types.json import Json

from dream.config import identity_ready
from dream.skills import config
from dream.skills import skill_db_source as DB
from dream.skills import skill_ledger as L
from dream.skills import skill_measure as SM
from dream.skills.config import db_url
from dream.skills.review_queue import _plain
from dream.skills.skill_measure import _extract_json

LEARNED_FILE = os.environ.get("CONFIG_LEARNED_FILE", "rules/learned.md")
PROPOSE_SESSIONS = int(
    os.environ.get("CONFIG_PROPOSE_SESSIONS", "2")
)  # distinct sessions -> proposed
# Same rule when the cosine to the row's anchor reaches this. Chosen on the 208 rows the lane
# had collected by 2026-11-06 (see the PR): in that data the median cosine between two
# different rules is 0.52 and p99 is 0.75; hand-checked, every cross-session cluster at 0.82 is
# one rule, 0.80 starts joining rules that ask for opposite things (a targeted test vs a
# thorough one), and embeddings cannot see negation, so a bar any lower would merge them often.
IDENTITY_COSINE = float(os.environ.get("CONFIG_IDENTITY_COSINE", "0.82"))
# The bar for a candidate whose session already has evidence on the row (re-scan paraphrases).
IDENTITY_COSINE_SAME_SESSION = float(os.environ.get("CONFIG_IDENTITY_COSINE_SAME_SESSION", "0.75"))
COOLDOWN_DAYS = 30  # a rejected rule's re-sightings are absorbed for this long
REOPEN_EXPIRED_DAYS = 60  # an expired proposal reopens on a new session's sighting until then
EVIDENCE_CAP = 60  # bound the JSONB
COVERAGE_CAP = 10  # coverage judge calls per night (each is one call)
REGISTRY_CHARS = 30000  # the user's global rule files, as sent to the coverage judge

EXPIRED = "expired"
COVERED = "auto:covered"

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

_COVER_PROMPT = """You check whether a proposed standing rule for an AI agent is ALREADY covered by
the rules its user keeps in their own config files.

THE USER'S EXISTING RULES (their config files, verbatim):
{registry}

PROPOSED RULE: {rule}

Answer covered=true only when an existing rule already tells the agent to do (or to avoid) the
same thing, in any wording. A rule that is related, narrower, broader in a way that changes what
the agent would do, or about something else, is NOT covered.

Output ONLY a JSON object, no prose:
{{"covered": true | false, "by": "the existing line that covers it, quoted briefly; empty when not covered"}}"""


# --------------------------------------------------------------------------- seams
def embed(texts: list[str]) -> list[list[float]]:
    """The embedding backend (the same one recall and the skills lane use). A named seam so
    tests stub it."""
    return L.embed(texts, "document")


def _judge_call(prompt: str) -> str:
    """The lane's judge (claude CLI by default). A named seam so tests stub the LLM."""
    return SM.run_judge(prompt)


# --------------------------------------------------------------------------- pure helpers
def _sessions(evidence: list[dict]) -> set[str]:
    return {e["session_id"] for e in evidence or [] if e.get("session_id")}


def _distinct_sessions(evidence: list[dict]) -> int:
    return len(_sessions(evidence))


def _union_evidence(old: list[dict], new: list[dict]) -> list[dict]:
    """Append new entries, deduped by session_id — keeps distinct-session counting honest."""
    seen = {e.get("session_id") for e in old}
    out = list(old)
    for e in new:
        if e.get("session_id") not in seen:
            out.append(e)
            seen.add(e.get("session_id"))
    return out[-EVIDENCE_CAP:]


def row_class(row: dict) -> str | None:
    """How a stored row answers a re-sighting of its rule.

    open    observe / proposed: the evidence accumulates and the gate runs
    reopen  an expired proposal: a NEW session's sighting brings it back to observe
    held    accepted / applied, or rejected within COOLDOWN_DAYS: absorbs the evidence, stays put
    None    anything else (an old rejection, a row nobody could match): a fresh row starts
    `row['age_days']` is days since updated_at (a reject, an expiry and a cover all stamp it)."""
    status = row["status"]
    if status in ("observe", "proposed"):
        return "open"
    if status in ("accepted", "applied"):
        return "held"
    if status == "rejected":
        age = row.get("age_days") or 0
        if row.get("reject_reason") == EXPIRED:
            return "reopen" if age <= REOPEN_EXPIRED_DAYS else None
        return "held" if age <= COOLDOWN_DAYS else None
    return None


def pick_match(rows: list[dict], session_id: str | set[str]) -> tuple[dict, str] | None:
    """The stored row a correction from `session_id` (or, for a stored row being folded in, its
    set of sessions) is the same rule as, with its class (row_class), or None. `rows` carry
    `cos` (cosine to the row's anchor), `evidence`, `status`. Best cosine wins among the rows
    that clear their bar; a session the row already counts is held to
    IDENTITY_COSINE_SAME_SESSION, any other to IDENTITY_COSINE."""
    mine = {session_id} if isinstance(session_id, str) else set(session_id)
    best: tuple[dict, str] | None = None
    for r in rows:
        klass = row_class(r)
        if klass is None or r.get("cos") is None:
            continue
        bar = IDENTITY_COSINE_SAME_SESSION if mine & _sessions(r["evidence"]) else IDENTITY_COSINE
        if r["cos"] >= bar and (best is None or r["cos"] > best[0]["cos"]):
            best = (r, klass)
    return best


def _evidence_key(summary: str, evidence: list[dict]) -> str:
    """What a recommendation was made for: the rule and the set of sessions behind it."""
    raw = (summary or "") + "|" + ",".join(sorted(_sessions(evidence)))
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


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
_CANDIDATES = """
SELECT id, summary, evidence, status, reject_reason,
       extract(epoch FROM now() - updated_at) / 86400.0 AS age_days,
       1 - (embedding <=> %s::halfvec) AS cos
  FROM config_lane.config_proposals
 WHERE file_key = %s AND embedding IS NOT NULL"""


@dataclass(frozen=True, slots=True)
class _CandidateRow:
    """One stored proposal of the learned file, with its cosine to the new correction."""

    id: int
    summary: str
    evidence: list[dict[str, Any]]
    status: str
    reject_reason: str | None
    age_days: Decimal
    cos: float | None


def _candidate_rows(cur, vec: list[float]) -> list[dict]:
    rows = (
        cur.connection.cursor(row_factory=class_row(_CandidateRow))
        .execute(_CANDIDATES, (L.vec_literal(vec), LEARNED_FILE))
        .fetchall()
    )
    return [
        {
            "id": r.id,
            "summary": r.summary,
            "evidence": r.evidence or [],
            "status": r.status,
            "reject_reason": r.reject_reason,
            "age_days": float(r.age_days or 0),
            "cos": float(r.cos) if r.cos is not None else None,
        }
        for r in rows
    ]


def _upsert_correction(cur, c: dict) -> tuple[int, str, int]:
    """Resolve identity by embedding against the stored rows, union this session's evidence,
    and return (id, status, distinct sessions). `c` carries rule, session_id, quote and `vec`
    (the rule's embedding). The observe->proposed gate is NOT run here: regate() runs it once
    per night, after the coverage check. A repeat of a rule that is not open is absorbed
    (row_class), or reopens an expired proposal when the session is new."""
    ev_new = [
        {"session_id": c["session_id"], "ts": datetime.now(UTC).isoformat(), "quote": c["quote"]}
    ]
    hit = pick_match(_candidate_rows(cur, c["vec"]), c["session_id"])
    if hit:
        row, klass = hit
        evidence = _union_evidence(row["evidence"], ev_new)
        n = _distinct_sessions(evidence)
        changed = len(evidence) != len(row["evidence"])
        if klass == "held" or not changed:
            if changed:  # absorb quietly: updated_at is the rejection's clock, leave it
                cur.execute(
                    "UPDATE config_lane.config_proposals SET evidence=%s::jsonb WHERE id=%s",
                    (json.dumps(evidence), row["id"]),
                )
            return row["id"], row["status"], n
        if klass == "reopen":
            cur.execute(
                "UPDATE config_lane.config_proposals SET evidence=%s::jsonb, status='observe', "
                "reject_reason=NULL, recommendation=NULL, updated_at=now() WHERE id=%s",
                (json.dumps(evidence), row["id"]),
            )
            return row["id"], "observe", n
        cur.execute(
            "UPDATE config_lane.config_proposals SET evidence=%s::jsonb, updated_at=now() "
            "WHERE id=%s",
            (json.dumps(evidence), row["id"]),
        )
        return row["id"], row["status"], n
    # scope='general': a behavioral rule applies across the user's surfaces (config_proposals.scope is
    # blast radius = local|general, NOT the registry's global|project axis). Review can flip to local.
    cur.execute(
        "INSERT INTO config_lane.config_proposals "
        "(kind, file_key, scope, diff, summary, evidence, status, embedding) "
        "VALUES ('add', %s, 'general', %s, %s, %s::jsonb, 'observe', %s::halfvec) RETURNING id",
        (LEARNED_FILE, c["rule"], c["rule"], json.dumps(ev_new), L.vec_literal(c["vec"])),
    )
    return cur.fetchone()[0], "observe", 1


@dataclass(frozen=True, slots=True)
class _UnanchoredRow:
    """One proposal that has no embedding anchor yet."""

    id: int
    summary: str


def ensure_embeddings(conn) -> int:
    """Embed the rows that have no anchor yet (everything written before schema 065, or by an
    older image) and store it; returns how many. One batch call."""
    cur = conn.cursor()
    rows = (
        conn.cursor(row_factory=class_row(_UnanchoredRow))
        .execute(
            "SELECT id, summary FROM config_lane.config_proposals "
            "WHERE embedding IS NULL AND summary <> '' ORDER BY id"
        )
        .fetchall()
    )
    if not rows:
        return 0
    vecs = embed([r.summary for r in rows])
    if len(vecs) != len(rows):
        raise RuntimeError(f"embedder returned {len(vecs)} vectors for {len(rows)} rules")
    for r, v in zip(rows, vecs, strict=True):
        cur.execute(
            "UPDATE config_lane.config_proposals SET embedding=%s::halfvec WHERE id=%s",
            (L.vec_literal(v), r.id),
        )
    conn.commit()
    return len(rows)


# --------------------------------------------------------------------------- coverage
@dataclass(frozen=True, slots=True)
class _RegistryFileRow:
    """One distinct global rule file of the user's config registry."""

    file_key: str
    content: str


def registry_digest(cur) -> tuple[str, str]:
    """(text, hash) of the user's own global rule files: config_registry rows with scope
    'global', one copy of each distinct content (every machine mirrors its own copy), capped
    at REGISTRY_CHARS. ('', '') when there are none."""
    rows = (
        cur.connection.cursor(row_factory=class_row(_RegistryFileRow))
        .execute(
            "SELECT DISTINCT ON (content_hash) file_key, content FROM config_lane.config_registry "
            "WHERE scope = 'global' AND content <> '' ORDER BY content_hash, surface_id, file_key"
        )
        .fetchall()
    )
    files = sorted((r.file_key, r.content) for r in rows)
    text = "\n\n".join(f"### {k}\n{v.strip()}" for k, v in files)[:REGISTRY_CHARS]
    return text, (hashlib.sha1(text.encode()).hexdigest()[:16] if text else "")


def coverage_check(rule: str, digest: str, registry_hash: str) -> dict | None:
    """Is `rule` already covered by the user's rule files? {covered, by, registry_hash, model,
    at}, or None when the judge failed (the caller retries tomorrow). No files, no call."""
    stamp = {"registry_hash": registry_hash, "at": datetime.now(UTC).isoformat()}
    if not digest:
        return {"covered": False, "by": "", "model": None, **stamp}
    try:
        obj = _extract_json(_judge_call(_COVER_PROMPT.format(registry=digest, rule=rule)) or "")
    except Exception as e:
        print(f"  coverage check failed: {e}")
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("covered"), bool):
        return None
    return {
        "covered": obj["covered"],
        "by": _plain(obj.get("by") or "", 200) if obj["covered"] else "",
        "model": config.JUDGE_MODEL,
        **stamp,
    }


@dataclass(frozen=True, slots=True)
class _GateRow:
    """One observe or proposed row of the learned file, as the gate reads it."""

    id: int
    summary: str
    evidence: list[dict[str, Any]]
    status: str
    coverage: dict[str, Any] | None
    recommendation: dict[str, Any] | None


def regate(conn, cap: int = COVERAGE_CAP) -> dict:
    """The observe->proposed gate, run once per night after the scan: every observe row with
    PROPOSE_SESSIONS distinct sessions is checked against the user's rule files (one judge
    call) and goes to 'proposed' (proposed_at stamped), or to 'rejected' / 'auto:covered'.
    A failed call leaves the row at observe for tomorrow. Proposed rows are re-checked too,
    when they were never checked or the rule files changed since (the user may have written
    the rule by hand while it waited). Returns {proposed: [...], covered: [...], failed, deferred}
    with brief items (id, summary, evidence, recommendation, covered_by)."""
    cur = conn.cursor()
    digest, rhash = registry_digest(cur)
    rows = (
        conn.cursor(row_factory=class_row(_GateRow))
        .execute(
            "SELECT id, summary, evidence, status, coverage, recommendation "
            "FROM config_lane.config_proposals "
            "WHERE file_key=%s AND status IN ('observe','proposed') "
            "ORDER BY (status = 'observe') DESC, id",
            (LEARNED_FILE,),
        )
        .fetchall()
    )
    todo = []
    for r in rows:
        if r.status == "observe" and _distinct_sessions(r.evidence or []) >= PROPOSE_SESSIONS:
            todo.append(r)
        elif r.status == "proposed" and (
            not isinstance(r.coverage, dict) or r.coverage.get("registry_hash") != rhash
        ):
            todo.append(r)
    out = {"proposed": [], "covered": [], "failed": 0, "deferred": max(0, len(todo) - cap)}
    for r in todo[:cap]:
        cov = r.coverage
        ev = r.evidence or []
        same = isinstance(cov, dict) and cov.get("registry_hash") == rhash
        verdict = cov if same and isinstance(cov.get("covered"), bool) else None
        verdict = verdict or coverage_check(r.summary, digest, rhash)
        if verdict is None:
            out["failed"] += 1
            continue
        item = {
            "id": r.id,
            "summary": r.summary,
            "evidence": ev,
            "recommendation": r.recommendation,
        }
        if verdict["covered"]:
            cur.execute(
                "UPDATE config_lane.config_proposals SET status='rejected', reject_reason=%s, "
                "coverage=%s, updated_at=now() WHERE id=%s",
                (COVERED, Json(verdict), r.id),
            )
            out["covered"].append({**item, "covered_by": verdict.get("by") or ""})
        elif r.status == "observe":
            cur.execute(
                "UPDATE config_lane.config_proposals SET status='proposed', proposed_at=now(), "
                "coverage=%s, updated_at=now() WHERE id=%s",
                (Json(verdict), r.id),
            )
            out["proposed"].append(item)
        else:  # a proposed row, still not covered: remember what it was checked against
            cur.execute(
                "UPDATE config_lane.config_proposals SET coverage=%s WHERE id=%s",
                (Json(verdict), r.id),
            )
    conn.commit()
    return out


# --------------------------------------------------------------------------- orchestrator
def run_lane(
    limit: int = 30, backfill: bool = False, no_discord: bool = False, report: bool = True
) -> dict:
    """Run the dream->config lane once; returns flat counts for dream_runs. Idles (and leaves
    the scan watermark where it is, so nothing is skipped) until schema 065 is applied."""
    from dream.config import review_queue as RQ

    conn = psycopg.connect(db_url())
    try:
        if not identity_ready(conn):
            print("[dream→config] schema 065 not applied; the lane idles until it is")
            return {"skipped": "schema 065 not applied"}
        cur = conn.cursor()
        cur.execute("SELECT now()")
        run_started = cur.fetchone()[0]
        cur.execute("SELECT last_scan_at FROM config_lane.config_scan_cursor WHERE id")
        row = cur.fetchone()
        last = None if backfill else (row[0] if row else None)
        scan_started = datetime.now(UTC)

        anchored = ensure_embeddings(conn)
        views = DB.sessions_since(last, max_sessions=limit * 4)
        print(f"[dream→config] last_scan_at={last}  sessions={len(views)}  anchored={anchored}")
        found = 0
        for s in views:
            corrections = scan_corrections(s)
            if not corrections:
                continue
            # an embedding failure aborts the run: nothing is committed, the watermark stays,
            # and tomorrow's scan starts from the same place
            vecs = embed([c["rule"] for c in corrections])
            if len(vecs) != len(corrections):
                raise RuntimeError(f"embedder returned {len(vecs)} vectors for {len(corrections)}")
            for c, v in zip(corrections, vecs, strict=True):
                found += 1
                c["vec"] = v
                _id, status, n = _upsert_correction(cur, c)
                print(f"  [{status} n={n}] {c['rule'][:70]}")
        conn.commit()

        gate = regate(conn)
        print(
            f"regate: proposed={len(gate['proposed'])} covered={len(gate['covered'])} "
            f"failed={gate['failed']} deferred={gate['deferred']}"
        )
        expired = RQ.expire(conn)
        print(f"expire: {expired['expired']}")
        recommended = RQ.recommend(conn)
        print("recommend:", recommended)

        cur.execute(
            "INSERT INTO config_lane.config_scan_cursor (id, last_scan_at) VALUES (true, %s) "
            "ON CONFLICT (id) DO UPDATE SET last_scan_at=EXCLUDED.last_scan_at",
            (scan_started,),
        )
        conn.commit()
        sent = None
        if report:
            sent = RQ.report(
                conn,
                run_started,
                {"covered": gate["covered"], "expired": expired["items"]},
                no_discord,
            )
            print("report:", sent)
        return {
            "sessions": len(views),
            "found": found,
            "proposed": len(gate["proposed"]),
            "covered": len(gate["covered"]),
            "expired": expired["expired"],
            "recommended": recommended.get("recommended", 0),
            "digest_sent": (sent or {}).get("sent", 0),
        }
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--no-discord", action="store_true")
    a = ap.parse_args()
    print(run_lane(limit=a.limit, backfill=a.backfill, no_discord=a.no_discord))
