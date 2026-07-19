#!/usr/bin/env python3
# mypy: ignore-errors
"""dream->skills nightly lane (runs server-side in the dream container; entry: run_lane()).

Sequences the whole lane against the skills_lane ledger, reading its catalog from the
skill_registry table (the client's skills_sync owns the disk<->registry publish):
  1. consolidate    — embed any active skills missing a description embedding; overlap ->
                      consolidate candidates (no disk access, no auto-merge)
  2. sync_usage     — episode [tool:Skill] markers -> skill_usage (+ dismissal); rollup into registry
  3. DERIVE         — gap scan -> cluster -> ledger (judge evidence); draft SKILL.md for proposed
  4. RETUNE         — under-trigger judge -> ledger retune/widen (judge evidence)
  5. grounded       — dismissals -> retune/narrow; explicit "make a skill" -> derive grounded
  6. decay + digest — retire stale 'observe' candidates (proposed rows wait for review);
                      throttled Discord digest (top 5, pings only on NEW proposals)
Incremental via skill_scan_cursor (whole-session scan since last_scan_at). --backfill ignores the watermark.

The LLM finders honor SKILL_MEASURE_MODEL (deepseek for cheap backfill); drafting uses Opus.
Propose-only: nothing here ever writes ~/.claude/skills or sets status='promoted'.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request

from . import config, post_fire, procedure_miner, struggle_arc
from . import skill_db_source as DB
from . import skill_derive as SD
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
    for c in clusters:
        sessions = c.get("sessions") or []
        ev = [{"session_id": sid, "class": "judge", "signal": "gap_scan"} for sid in sessions]
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


def run_retune(conn, substantive, catalog, seen_ids):
    n = 0
    for s in substantive:
        v = SM.judge_session(s, catalog)
        if not v:
            continue
        for m in v.get("would_have_helped", []):
            sk = m.get("skill")
            if not sk or sk in s["fired"]:
                continue
            ev = [
                {
                    "session_id": s["session"],
                    "class": "judge",
                    "signal": "under_trigger",
                    "skill": sk,
                    "phrasing": s.get("first_user", "")[:160],
                    "why": m.get("why", ""),
                }
            ]
            res = L.merge_candidate(
                conn,
                "retune",
                sk,
                ev,
                direction="widen",
                target_skills=[sk],
                summary=f"under-fires: {m.get('why', '')[:120]}",
                do_embed=False,
            )
            seen_ids.add(res["id"])
            n += 1
    return {"under_trigger": n}


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
    cur.execute("SELECT proposal_path FROM skills_lane.skill_gap_candidates WHERE id=%s", (cid,))
    if (cur.fetchone() or [None])[0]:
        return
    name = re.sub(r"[^a-z0-9-]", "", cluster.get("procedure", "unnamed").lower()) or "unnamed"
    body = SD.draft_skill(cluster)
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


# --------------------------------------------------------------------- 6. digest
DIGEST_CAP = 5  # proposals shown per digest — flood control, the rest is a "+N pending" footer


def discord_digest(conn, no_discord):
    """Throttled review digest. Shows the top-DIGEST_CAP proposals (salience DESC, then
    score), and PINGS only when there are proposals not covered by the previous digest —
    a nightly re-wall of the same pending list is how review queues rot. The seen-ids
    watermark lives in skill_scan_cursor.config['digest_ids'] (update_cursor merges,
    never clobbers)."""
    cur = conn.cursor()
    cur.execute(
        """SELECT id, kind, name, score, grounded_sessions, judge_sessions, salience, source_detector
             FROM skills_lane.skill_gap_candidates WHERE status='proposed'
             ORDER BY salience DESC NULLS LAST, score DESC, id"""
    )
    rows = cur.fetchall()
    if not rows:
        return {"proposed": 0, "new": 0}
    ids = [r[0] for r in rows]
    prev = set((L.get_cursor(conn).get("config") or {}).get("digest_ids") or [])
    new = [i for i in ids if i not in prev]

    lines = [f"**dream→skills — {len(rows)} proposal(s) awaiting review**"]
    for _id, kind, name, score, gs, js, sal, det in rows[:DIGEST_CAP]:
        g = f" · {gs} grounded" if gs else ""
        s = f" · sal {sal}" if sal is not None else ""
        v = f" · {det}" if det else ""
        lines.append(f"- `{kind}` **{name}** (score {score:.1f}, {js} judge{g}{s}{v})")
    if len(rows) > DIGEST_CAP:
        lines.append(f"+{len(rows) - DIGEST_CAP} more pending (skill_review list)")
    lines.append("\nReview: `/skill-review` (or skill_review.py list) · accept/reject/promote <id>")
    msg = "\n".join(lines)

    cur.execute(
        "UPDATE skills_lane.skill_scan_cursor SET config = COALESCE(config,'{}'::jsonb) || %s::jsonb, "
        "updated_at=now() WHERE id=1",
        (json.dumps({"digest_ids": ids}),),
    )
    conn.commit()

    if not new:
        print(f"(digest: {len(rows)} pending, none new since last digest — not pinging)")
        return {"proposed": len(rows), "new": 0}
    # Notifier: POST to a Discord webhook if configured; otherwise just print (default).
    if not no_discord and config.DISCORD_WEBHOOK:
        try:
            body = json.dumps({"content": msg[:1900]}).encode()
            req = urllib.request.Request(
                config.DISCORD_WEBHOOK, data=body, headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            print(f"(notifier failed: {e})\n{msg}")
    else:
        print(msg)
    return {"proposed": len(rows), "new": len(new)}


def run_lane(limit: int = 40, backfill: bool = False, no_discord: bool = False) -> None:
    """Run the whole dream→skills lane once. No argparse, so the dream container's run_once
    can call it directly (the catalog comes from the registry, not disk)."""
    conn = L.connect()
    cur = conn.cursor()
    cur.execute("SELECT now()")
    run_started = cur.fetchone()[0]
    cur_state = L.get_cursor(conn)
    last = None if backfill else cur_state.get("last_scan_at")

    catalog = "\n".join(f"- {n}: {d}" for n, d in sorted(SM.load_skills().items()))
    seen: set[int] = set()

    print(
        f"[dream→skills] last_scan_at={last}  model={os.environ.get('SKILL_MEASURE_MODEL', 'opus')}"
    )
    print("consolidate:", consolidate_pass(conn, seen))
    print("usage:", sync_usage(conn, last))

    views = DB.sessions_since(last, max_sessions=limit * 4)
    substantive = [v for v in views if SM.is_substantive(v)][:limit]
    print(f"substantive sessions to judge: {len(substantive)}")

    print("derive:", run_derive(conn, substantive, catalog, seen))
    print("retune:", run_retune(conn, substantive, catalog, seen))
    print("grounded:", capture_grounded(conn, last, seen))

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
        print("struggle_arc:", struggle_arc.run(conn, since=last))
    if "post_fire" in detectors:
        print("post_fire:", post_fire.run(conn, since=last))
    if "procedure_miner" in detectors and procedure_miner.due(conn):
        print("procedure_miner:", procedure_miner.run(conn))

    print("decay:", L.decay_stale(conn))
    print("digest:", discord_digest(conn, no_discord))

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true", help="ignore watermark; scan 30d")
    ap.add_argument("--limit", type=int, default=40, help="max substantive sessions to judge")
    ap.add_argument("--no-discord", action="store_true")
    args = ap.parse_args()
    run_lane(limit=args.limit, backfill=args.backfill, no_discord=args.no_discord)


if __name__ == "__main__":
    main()
