#!/usr/bin/env python3
# mypy: ignore-errors
"""dream->skills nightly lane (runs server-side in the dream container; entry: run_lane()).

Sequences the whole lane against the skills_lane ledger, reading its catalog from the
skill_registry table (the client's skills_sync owns the disk<->registry publish):
  1. consolidate    — embed any active skills missing a description embedding; overlap ->
                      consolidate candidates (no disk access, no auto-merge)
  2. sync_usage     — episode [tool:Skill] markers -> skill_usage (+ dismissal); rollup into registry
  3. DERIVE         — gap scan -> cluster -> ledger (judge evidence); draft SKILL.md for proposed
  4. RETUNE         — under-trigger judge -> ledger retune/widen (judge evidence), each
                      session judged only against the skills served to it
  5. grounded       — dismissals -> retune/narrow; explicit "make a skill" -> derive grounded
     (then the v2 detectors, and a re-gate of open derive rows)
  6. draft          — every proposed derive/retune without a body (or with a stale retune
                      draft) gets a complete SKILL.md + base_body_hash (skill_draft.py)
  7. review queue   — auto-reject undeliverable retunes, expire unanswered proposals, store
                      a recommendation per draft (review_queue.py)
  7b. autopilot     — autoreview.run: roll back auto-applies that went wrong, and (when this
                      container has the reviewer) judge the queue and apply/reject the routine
                      rows; SKILLS_AUTOREVIEW = off | shadow | live
  8. decay + report — retire stale 'observe' candidates (proposed rows wait for review);
                      the verdict-first report when the queue changed, weekly otherwise
Incremental via skill_scan_cursor (whole-session scan since last_scan_at). --backfill ignores the watermark.

The LLM finders honor SKILL_MEASURE_MODEL (deepseek for cheap backfill); drafting uses Opus.
Nothing here writes skill_registry or sets status='promoted' except through
skill_sync_routes.apply_candidate: a person's review accept, or the autopilot's own apply
(dream/skills/autoreview.py, a recorded review row each).
"""

from __future__ import annotations

import argparse
import os
import re

from psycopg.rows import dict_row

from . import (
    autoreview,
    config,
    post_fire,
    procedure_miner,
    review_queue,
    skill_draft,
    struggle_arc,
)
from . import skill_db_source as DB
from . import skill_derive as SD
from . import skill_doc as D
from . import skill_ledger as L

SM = SD.skill_measure

PROPOSALS_DIR = config.PROPOSALS_DIR
OVERLAP_COS = 0.90  # description-embedding cosine above this = consolidate candidate
OVERLAP_TOP_K = 5  # cap consolidate nominations per run (avoid flooding the review queue)


def _cosine(a, b):
    import math

    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ------------------------------------------------- 1. consolidate pass (registry-based)
def consolidate_pass(conn, seen_ids):
    """Server-side: embed any active-skill descriptions that lack an embedding (Voyage), then
    nominate near-duplicate skill pairs as consolidate candidates for human review. Reads the
    skill_registry (published by the client's skills_sync); no disk access, no auto-merge."""
    cur = conn.cursor()

    # (1) backfill description embeddings for active skills missing one.
    cur.execute(
        "SELECT name, description FROM skills_lane.skill_registry "
        "WHERE status='active' AND description IS NOT NULL AND description <> '' "
        "AND description_embedding IS NULL"
    )
    missing = cur.fetchall()
    if missing:
        vecs = L.embed([d for _, d in missing], "document")
        for (name, _), v in zip(missing, vecs, strict=False):
            cur.execute(
                "UPDATE skills_lane.skill_registry SET description_embedding=%s::halfvec, "
                "updated_at=now() WHERE name=%s",
                (L.vec_literal(v), name),
            )
        conn.commit()

    # (2) overlap detection -> consolidate candidates (top-K most-similar pairs above threshold).
    cur.execute(
        "SELECT name, description_embedding FROM skills_lane.skill_registry "
        "WHERE status='active' AND description_embedding IS NOT NULL"
    )
    rows = [(n, [float(x) for x in str(e).strip("[]").split(",")]) for n, e in cur.fetchall()]
    cand = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            c = _cosine(rows[i][1], rows[j][1])
            if c >= OVERLAP_COS:
                cand.append((c, *sorted([rows[i][0], rows[j][0]])))
    cand.sort(reverse=True)
    for c, a, b in cand[:OVERLAP_TOP_K]:
        res = L.merge_candidate(
            conn,
            "consolidate",
            f"{a}+{b}",
            [{"session_id": None, "class": "judge", "signal": "overlap"}],
            target_skills=[a, b],
            summary=f"{a} and {b} have near-duplicate trigger descriptions (cos={c:.2f}); review for merge.",
            do_embed=False,
        )
        seen_ids.add(res["id"])
    return {"embedded": len(missing), "overlap_pairs": len(cand[:OVERLAP_TOP_K])}


# -------------------------------------------------------------------- 2. usage sync
def sync_usage(conn, last_scan_at):
    cur = conn.cursor()
    events = DB.fire_events(last_scan_at)
    for e in events:
        cur.execute(
            """INSERT INTO skills_lane.skill_usage (skill, fired_at, session_id, via, dismissed)
                 VALUES (%s,%s,%s,%s,%s)
               ON CONFLICT (skill, session_id, fired_at) DO UPDATE SET dismissed=EXCLUDED.dismissed""",
            (e["skill"], e["fired_at"], e["session_id"], e["via"], e["dismissed"]),
        )
    # rollup into registry
    cur.execute(
        """UPDATE skills_lane.skill_registry r SET
             fire_count = u.cnt, last_fired = u.last, updated_at = now()
           FROM (SELECT skill, count(*) cnt, max(fired_at) last FROM skills_lane.skill_usage GROUP BY skill) u
           WHERE r.name = u.skill"""
    )
    conn.commit()
    return {"fire_events": len(events), "dismissals": sum(1 for e in events if e["dismissed"])}


# --------------------------------------------------------------- 3+4+5 candidate gen
def run_derive(conn, substantive, catalog, seen_ids):
    gaps = []
    for s in substantive:
        gaps += SD.scan_session(s, catalog)
    clusters = SD.cluster_gaps(gaps)
    n_proposed = 0
    night = SM.scan_night()
    dates = {s["session"]: s.get("date") for s in substantive}
    for c in clusters:
        sessions = c.get("sessions") or []
        ev = [
            {
                "session_id": sid,
                "class": "judge",
                "signal": "gap_scan",
                "scan_night": night,
                "date": dates.get(sid),
            }
            for sid in sessions
        ]
        if not ev:
            continue
        res = L.merge_candidate(
            conn,
            "derive",
            c.get("procedure", "unnamed"),
            ev,
            signature=c.get("signature"),
            tools=_tools_from_bash(c.get("bash_evidence", [])),
            summary=c.get("what", ""),
            trigger_phrasings=c.get("trigger_phrasings", []),
        )
        seen_ids.add(res["id"])
        if res["status"] == "proposed":
            _draft_if_needed(conn, res["id"], c)
            n_proposed += 1
    return {"gaps": len(gaps), "clusters": len(clusters), "proposed": n_proposed}


def run_retune(conn, substantive, catalog, seen_ids, descriptions, scopes=None):
    """Under-trigger judge -> retune/widen candidates, gated on a REAL description gap.

    A would_have_helped skill whose current description already covers the request was a
    match the model simply didn't invoke; more trigger words change nothing, so it must not
    become a widen proposal. Only SM.under_trigger_verdict() == "gap" merges evidence. The
    rest is counted, never merged: covered / backstop-covered items are tallied per skill
    (matched-but-not-invoked telemetry), and items with no stated gap (incl. legacy judge
    output without description_covers / missing_phrasing) are dropped as `no_gap_stated`.
    `descriptions` is name -> current description (the catalog the judge saw).

    `scopes` (name -> registry scope): each session is judged only against the skills
    served to it (its view's `scope`, see _session_scopes; global when unset). A project
    skill can't fire in a session of another project whatever its description says, so
    widening it there is noise: #76 asked to widen docx (project:transcribe_ai) on two
    jobs/neuron sessions."""
    stats = {"under_trigger": 0, "covered_not_fired": 0, "backstop_covered": 0, "no_gap_stated": 0}
    covered_by_skill: dict[str, int] = {}
    night = SM.scan_night()
    for s in substantive:
        served, cat = descriptions, catalog
        if scopes is not None:
            served = SM.served_skills(descriptions, scopes, s.get("scope", "global"))
            cat = SM.format_catalog(sorted(served.items()))
        v = SM.judge_session(s, cat)
        if not v:
            continue
        for m in v.get("would_have_helped", []):
            sk = m.get("skill")
            # a skill missing from the catalog has no description to widen
            if not sk or sk in s["fired"] or sk not in served:
                continue
            phrasing = s.get("first_user", "")[:160]
            verdict = SM.under_trigger_verdict(m, descriptions[sk], phrasing)
            if verdict in ("covered", "backstop"):
                stats["covered_not_fired"] += 1
                if verdict == "backstop":  # judge claimed a gap the description already has
                    stats["backstop_covered"] += 1
                covered_by_skill[sk] = covered_by_skill.get(sk, 0) + 1
                continue
            if verdict != "gap":
                stats["no_gap_stated"] += 1
                continue
            missing = str(m["missing_phrasing"]).strip()[:160]
            ev = [
                {
                    "session_id": s["session"],
                    "class": "judge",
                    "signal": "under_trigger",
                    "skill": sk,
                    "phrasing": phrasing,
                    "missing_phrasing": missing,
                    "why": m.get("why", ""),
                    "scan_night": night,
                    "date": s.get("date"),
                }
            ]
            res = L.merge_candidate(
                conn,
                "retune",
                sk,
                ev,
                direction="widen",
                target_skills=[sk],
                summary=f"under-fires; description lacks {missing[:60]!r}: {m.get('why', '')[:120]}",
                do_embed=False,
            )
            seen_ids.add(res["id"])
            stats["under_trigger"] += 1
    if covered_by_skill:
        stats["covered_by_skill"] = dict(sorted(covered_by_skill.items()))
    return stats


def capture_grounded(conn, last_scan_at, seen_ids):
    # dismissals -> retune/narrow (grounded)
    dis = [e for e in DB.fire_events(last_scan_at) if e["dismissed"]]
    for e in dis:
        ev = [
            {
                "session_id": e["session_id"],
                "class": "grounded",
                "signal": "dismissal",
                "skill": e["skill"],
            }
        ]
        res = L.merge_candidate(
            conn,
            "retune",
            e["skill"],
            ev,
            direction="narrow",
            target_skills=[e["skill"]],
            summary=f"{e['skill']} fired then was overridden — consider narrowing its trigger.",
            do_embed=False,
        )
        seen_ids.add(res["id"])
    # explicit "make this a skill" -> derive (grounded). Attach session so identity can resolve.
    reqs = DB.explicit_skill_requests(last_scan_at)
    for r in reqs:
        ev = [
            {
                "session_id": r["session_id"],
                "class": "grounded",
                "signal": "explicit_request",
                "phrasing": r["phrasing"],
            }
        ]
        res = L.merge_candidate(
            conn,
            "derive",
            f"explicit:{r['session_id'][:8]}",
            ev,
            summary=r["phrasing"],
            do_embed=False,
        )
        seen_ids.add(res["id"])
    return {"dismissals": len(dis), "explicit_requests": len(reqs)}


def _tools_from_bash(bash_heads):
    return sorted({h.split()[0] for h in bash_heads if h and h.split()})[:8]


def _draft_if_needed(conn, cid, cluster):
    cur = conn.cursor()
    cur.execute("SELECT proposal_body FROM skills_lane.skill_gap_candidates WHERE id=%s", (cid,))
    if (cur.fetchone() or [None])[0]:
        return
    name = re.sub(r"[^a-z0-9-]", "", cluster.get("procedure", "unnamed").lower()) or "unnamed"
    body = SD.draft_skill(cluster)
    if D.new_skill_problem(body):
        return  # unusable draft (no frontmatter name/description): the draft pass retries
    d = PROPOSALS_DIR / f"cand{cid}-{name}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(body)
    # Store the draft body in the row too (not just the path): the review path runs over HTTP
    # via mcp-server, which can't read this container's disk. The body makes it self-describing.
    cur.execute(
        "UPDATE skills_lane.skill_gap_candidates SET proposal_path=%s, proposal_body=%s WHERE id=%s",
        (str(d / "SKILL.md"), body, cid),
    )
    conn.commit()


# ------------------------------------------------------------- 7. counts + scopes
def _tally(counts: dict, stage: str, stats: dict) -> dict:
    """Copy a stage's integer stats into the run's flat counts as `<stage>_<key>`, and hand
    the stats back for the log line."""
    for k, v in (stats or {}).items():
        if isinstance(v, int) and not isinstance(v, bool):
            counts[f"{stage}_{k}"] = v
    return stats


def _session_scopes(conn, views) -> None:
    """Stamp each view's `scope`: where its skills were served from, by the accept path's
    own rule (mcp_server.skill_sync_routes._evidence_scope; one project is
    project:<label>, a session that spans projects or can't be placed is global)."""
    from mcp_server.skill_sync_routes import _evidence_scope

    cur = conn.cursor(row_factory=dict_row)
    for v in views:
        v["scope"] = _evidence_scope(cur, [{"session_id": v["session"]}])


def _ledger_counts(conn, since) -> dict:
    """Candidate rows the run created, merged into, and raised to proposed (schema 062's
    proposed_at; absent before it is applied)."""
    cur = conn.cursor()
    cur.execute(
        "SELECT count(*) FILTER (WHERE created_at >= %s), "
        "count(*) FILTER (WHERE created_at < %s AND last_seen >= %s) "
        "FROM skills_lane.skill_gap_candidates",
        (since, since, since),
    )
    new, merged = cur.fetchone()
    out = {"candidates_new": new, "candidates_merged": merged}
    if L.queue_ready(conn):
        cur.execute(
            "SELECT count(*) FROM skills_lane.skill_gap_candidates WHERE proposed_at >= %s",
            (since,),
        )
        out["to_proposed"] = cur.fetchone()[0]
    return out


def run_lane(limit: int = 40, backfill: bool = False, no_discord: bool = False) -> dict:
    """Run the whole dream→skills lane once. No argparse, so the dream container's run_once
    can call it directly (the catalog comes from the registry, not disk). Returns the run's
    flat counts (`<stage>_<key>` per stage, plus sessions and the candidate tallies), which
    dream/__main__ records in dream_runs.counts as skills_*."""
    conn = L.connect()
    cur = conn.cursor()
    cur.execute("SELECT now()")
    run_started = cur.fetchone()[0]
    cur_state = L.get_cursor(conn)
    last = None if backfill else cur_state.get("last_scan_at")

    skills = SM.load_skills()
    scopes = SM.load_scopes(conn)
    catalog = "\n".join(f"- {n}: {d}" for n, d in sorted(skills.items()))
    seen: set[int] = set()
    counts: dict = {}

    print(
        f"[dream→skills] last_scan_at={last}  model={os.environ.get('SKILL_MEASURE_MODEL', 'opus')}"
    )
    print("consolidate:", _tally(counts, "consolidate", consolidate_pass(conn, seen)))
    print("usage:", _tally(counts, "usage", sync_usage(conn, last)))

    views = DB.sessions_since(last, max_sessions=limit * 4)
    substantive = [v for v in views if SM.is_substantive(v)][:limit]
    _session_scopes(conn, substantive)
    counts["sessions_touched"], counts["sessions"] = len(views), len(substantive)
    print(f"substantive sessions to judge: {len(substantive)}")

    print("derive:", _tally(counts, "derive", run_derive(conn, substantive, catalog, seen)))
    print(
        "retune:",
        _tally(counts, "retune", run_retune(conn, substantive, catalog, seen, skills, scopes)),
    )
    print("grounded:", _tally(counts, "grounded", capture_grounded(conn, last, seen)))

    # v2 detectors — on by default; SYNAPSE_SKILLS_DETECTORS is the kill switch
    # (comma list to run a subset, empty string to disable all).
    detectors = {
        d.strip()
        for d in os.environ.get(
            "SYNAPSE_SKILLS_DETECTORS", "struggle_arc,post_fire,procedure_miner"
        ).split(",")
        if d.strip()
    }
    if "struggle_arc" in detectors:
        print("struggle_arc:", _tally(counts, "struggle_arc", struggle_arc.run(conn, since=last)))
    if "post_fire" in detectors:
        print("post_fire:", _tally(counts, "post_fire", post_fire.run(conn, since=last)))
    if "procedure_miner" in detectors and procedure_miner.due(conn):
        print("procedure_miner:", _tally(counts, "procedure_miner", procedure_miner.run(conn)))

    print("regate:", _tally(counts, "regate", L.regate_observe(conn)))
    print("drafts:", _tally(counts, "drafts", skill_draft.draft_pending(conn)))
    rejected = review_queue.auto_reject(conn)
    expired = review_queue.expire(conn)
    counts["auto_rejected"], counts["expired"] = rejected["auto_rejected"], expired["expired"]
    print(f"review: auto_rejected={rejected['auto_rejected']} expired={expired['expired']}")
    # The autopilot (autoreview): always reconciles what it applied before; judges and acts on
    # the queue here only when SKILLS_AUTOREVIEW_REVIEW=1 (the reviewer is a logged-in claude
    # CLI; the usual deployment runs it from the host, scripts/skills_autopilot.py).
    auto = autoreview.run(
        conn, run_started, review=os.environ.get("SKILLS_AUTOREVIEW_REVIEW", "") == "1"
    )
    print("autopilot:", _tally(counts, "autoreview", autoreview.flat_counts(auto)))
    print("recommend:", _tally(counts, "recommend", review_queue.recommend(conn)))
    decay = L.decay_stale(conn)
    counts["retired"] = decay["retired"]
    print("decay:", decay)
    changes = {
        "auto_rejected": rejected["items"],
        "expired": expired["items"],
        "autopilot": auto,
    }
    rep = review_queue.report(conn, run_started, changes, no_discord)
    counts["digest_sent"] = rep["sent"]
    print("report:", rep)
    counts.update(_ledger_counts(conn, run_started))

    L.update_cursor(
        conn,
        run_started,
        config={
            "excluded_projects": ["transcribe_ai"],
            "propose_score": L.PROPOSE_SCORE,
            "session_jaccard": L.SESSION_JACCARD,
            "limit": limit,
        },
    )
    conn.close()
    print("done.")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true", help="ignore watermark; scan 30d")
    ap.add_argument("--limit", type=int, default=40, help="max substantive sessions to judge")
    ap.add_argument("--no-discord", action="store_true")
    args = ap.parse_args()
    run_lane(limit=args.limit, backfill=args.backfill, no_discord=args.no_discord)


if __name__ == "__main__":
    main()
