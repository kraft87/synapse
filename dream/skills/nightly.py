#!/usr/bin/env python3
# mypy: ignore-errors
"""dream->skills nightly lane (runs server-side in the dream container; entry: run_lane()).

Sequences the whole lane against the skills_lane ledger, reading its catalog from the
skill_registry table (the client's skills_sync owns the disk<->registry publish):
  1. sync_usage     — episode [tool:Skill] markers -> skill_usage (+ dismissal)
  2. judge          — ONE analysis call per session returns its gaps and its would-have-helped
                      skills (a few calls in flight at once):
                      DERIVE: gaps -> cluster -> ledger (judge evidence)
                      RETUNE: under-trigger judge -> ledger retune/widen (judge evidence)
  3. grounded       — dismissals -> retune/narrow; explicit "make a skill" -> derive grounded
  4. detectors      — struggle_arc, post_fire, and the weekly procedure_miner (each its own stage)
  5. draft          — every proposed derive/retune without a body (or with a stale retune
                      draft) gets a complete SKILL.md + base_body_hash (skill_draft.py)
  6. decay, effects, digest
                    — retire stale 'observe' candidates (proposed rows wait for review);
                      measure what retunes accepted 7-30 days ago did to their skill's fires,
                      dismissals and deviations; throttled Discord digest (top 5, pings only
                      on NEW proposals)
Incremental over an episode-id window (dream/lane.py): every stage covers the same episodes,
and the cursor advances only past the sessions the run got to (at most `limit` are judged a
night, oldest first; the rest roll to the next night). --backfill starts 30 days back.

Every stage runs on its own: one that fails is reported and the rest still run. A model that
cannot be reached (usage limit, outage) stops the lane instead, and a failed stage that reads
the window holds the cursor, so the next run covers the same episodes again.

The judges use SYNAPSE_SKILLS_JUDGE_MODEL and the drafters SYNAPSE_SKILLS_DRAFT_MODEL, both
through ingestion.llm_client (default Opus).
Propose-only: nothing here writes skill_registry or sets status='promoted' — the review
accept does both (mcp_server/skill_sync_routes.py).
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from dataclasses import replace
from typing import Any

from dream import lane
from ingestion.llm_client import LLM_TRANSPORT_ERRORS, stage_model

from . import config, post_fire, procedure_miner, skill_draft, struggle_arc
from . import skill_db_source as DB
from . import skill_derive as SD
from . import skill_ledger as L
from . import skill_measure as SM

logger = logging.getLogger(__name__)

BASH_HEADS_KEPT = 8  # shell commands kept per session on a derive's evidence, for its draft


# -------------------------------------------------------------------- 1. usage sync
def sync_usage(conn, events):
    """Record the window's skill fire events in skill_usage. The events (DB.fire_events) feed
    the dismissal and post-fire stages too, which add their verdicts to the same rows."""
    cur = conn.cursor()
    for e in events:
        cur.execute(
            """INSERT INTO skills_lane.skill_usage (skill, fired_at, session_id, via, dismissed)
                 VALUES (%s,%s,%s,%s,%s)
               ON CONFLICT (skill, session_id, fired_at) DO UPDATE SET dismissed=EXCLUDED.dismissed""",
            (e["skill"], e["fired_at"], e["session_id"], e["via"], e["dismissed"]),
        )
    conn.commit()
    return {"fire_events": len(events), "dismissals": sum(1 for e in events if e["dismissed"])}


# --------------------------------------------------------------- 3+4+5 candidate gen
def judge_sessions(conn, substantive, catalog, scopes, seen_ids):
    """The lane's per-session LLM pass. One analysis call per session (lane.LLM_WORKERS in
    flight at once) returns both its gaps and its would-have-helped skills; they feed DERIVE
    and RETUNE. `unparsed` counts replies with no JSON object (their session adds nothing).

    A session is judged against the skills served in its project (global plus its own
    `project:<p>`; SM.served_catalog): a skill its folder never receives is neither something
    it should have fired nor a reason to call a gap covered."""
    served = {
        p: dict(SM.served_catalog(catalog, scopes, p))
        for p in {s.get("project") for s in substantive}
    }
    analyses = lane.run_parallel(
        lambda s: SM.analyze_session(s, SM.format_catalog(served[s.get("project")].items())),
        substantive,
    )
    return {
        "sessions": len(substantive),
        "unparsed": sum(1 for a in analyses if a is None),
        "derive": run_derive(conn, substantive, analyses, seen_ids),
        "retune": run_retune(conn, substantive, analyses, seen_ids, served),
    }


def run_derive(conn, substantive, analyses, seen_ids):
    """Gaps -> clusters -> derive candidates. A cluster that turns 'proposed' is drafted by the
    draft stage (skill_draft.draft_pending), which reads the bash heads each session's evidence
    entry carries."""
    gaps = [g for s, a in zip(substantive, analyses, strict=True) for g in SD.session_gaps(s, a)]
    heads = {s["session"]: s["bash_heads"][:BASH_HEADS_KEPT] for s in substantive}
    clusters = SD.cluster_gaps(gaps)
    n_proposed = 0
    for c in clusters:
        sessions = c.get("sessions") or []
        ev = [
            {
                "session_id": sid,
                "class": "judge",
                "signal": "gap_scan",
                "bash": heads.get(sid, []),
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
            n_proposed += 1
    return {"gaps": len(gaps), "clusters": len(clusters), "proposed": n_proposed}


def run_retune(conn, substantive, analyses, seen_ids, served):
    """Under-trigger judge -> retune/widen candidates, gated on a REAL description gap.

    A would_have_helped skill whose current description already covers the request was a
    match the model simply didn't invoke; more trigger words change nothing, so it must not
    become a widen proposal. Only SM.under_trigger_verdict() == "gap" merges evidence. The
    rest is counted, never merged: covered / backstop-covered items are tallied per skill
    (matched-but-not-invoked telemetry), and items with no stated gap (incl. legacy judge
    output without description_covers / missing_phrasing) are dropped as `no_gap_stated`.
    `analyses` holds each session's analysis (SM.analyze_session; None = no usable reply);
    `served` maps a project to name -> current description of the skills served there (the
    catalog the judge saw for its sessions)."""
    stats = {"under_trigger": 0, "covered_not_fired": 0, "backstop_covered": 0, "no_gap_stated": 0}
    covered_by_skill: dict[str, int] = {}
    for s, v in zip(substantive, analyses, strict=True):
        if not v:
            continue
        descriptions = served[s.get("project")]
        for m in v.get("would_have_helped") or []:
            sk = m.get("skill")
            # a skill missing from the session's catalog (unknown, or not served in its
            # project) has no description to widen
            if not sk or sk in s["fired"] or sk not in descriptions:
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


def capture_grounded(conn, window, fires, seen_ids, *, catalog):
    # dismissals -> retune/narrow (grounded); `fires` is sync_usage's read of the same window.
    # Only a skill the lane owns can be narrowed: a dismissal of one the registry doesn't carry
    # (a built-in) would be a proposal with nothing to draft against.
    registered = {name for name, _ in catalog}
    dis = [e for e in fires if e["dismissed"] and e["skill"] in registered]
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
    reqs = DB.explicit_skill_requests(conn, window)
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


# ------------------------------------------------------------- 5. effect of accepted retunes
EFFECT_MIN_DAYS = 7  # a retune is measured from this long after its accept...
EFFECT_MAX_DAYS = 30  # ...until this long after it

# One row per accepted retune: the target skill's fires, dismissals, judged fires (post_fire
# kept a verdict) and deviations (partial / failed), before the accept and since it. The span
# before is as long as the span since.
_EFFECTS_SQL = """
SELECT c.name, c.direction, extract(day FROM now() - c.updated_at)::int,
       count(u.id) FILTER (WHERE u.fired_at < c.updated_at),
       count(u.id) FILTER (WHERE u.fired_at < c.updated_at AND u.dismissed),
       count(u.id) FILTER (WHERE u.fired_at < c.updated_at
                             AND u.outcome IN ('worked', 'partial', 'failed')),
       count(u.id) FILTER (WHERE u.fired_at < c.updated_at AND u.outcome IN ('partial', 'failed')),
       count(u.id) FILTER (WHERE u.fired_at >= c.updated_at),
       count(u.id) FILTER (WHERE u.fired_at >= c.updated_at AND u.dismissed),
       count(u.id) FILTER (WHERE u.fired_at >= c.updated_at
                             AND u.outcome IN ('worked', 'partial', 'failed')),
       count(u.id) FILTER (WHERE u.fired_at >= c.updated_at AND u.outcome IN ('partial', 'failed'))
  FROM skills_lane.skill_gap_candidates c
  LEFT JOIN skills_lane.skill_usage u
         ON u.skill = COALESCE(c.target_skills[1], c.name)
        AND u.fired_at >= c.updated_at - (now() - c.updated_at)
 WHERE c.kind = 'retune' AND c.status = 'promoted'
   AND c.updated_at <= now() - make_interval(days => %s)
   AND c.updated_at > now() - make_interval(days => %s)
 GROUP BY c.id ORDER BY c.updated_at DESC"""


def skill_effects(conn):
    """What each accepted retune did to its skill, in SQL only: for the retunes promoted
    EFFECT_MIN_DAYS..EFFECT_MAX_DAYS ago (a promoted row's updated_at is its accept), the
    skill's fires, dismissals and judged deviations (post_fire's verdicts) over the span since
    the accept against the span of the same length before it. Information for the digest, not a
    gate: a handful of fires says little, and fires from before post_fire kept its verdicts
    show as "-"."""

    def frac(n, d):
        return f"{n}/{d}" if d else "-"

    lines = []
    for name, direction, days, *counts in conn.execute(
        _EFFECTS_SQL, (EFFECT_MIN_DAYS, EFFECT_MAX_DAYS)
    ).fetchall():
        (bf, bd, bj, bx), (af, ad, aj, ax) = counts[:4], counts[4:]
        lines.append(
            f"{name} ({direction}, {days}d ago): fires {bf} -> {af}"
            f" · dismissed {frac(bd, bf)} -> {frac(ad, af)}"
            f" · deviated {frac(bx, bj)} -> {frac(ax, aj)}"
        )
    return {"measured": len(lines), "lines": lines}


def _tools_from_bash(bash_heads):
    return sorted({h.split()[0] for h in bash_heads if h and h.split()})[:8]


def run_procedure_miner(conn, catalog):
    """The weekly procedure miner, when it is due."""
    if not procedure_miner.due(conn):
        return {"due": False}
    return procedure_miner.run(conn, catalog=catalog)


# --------------------------------------------------------------------- 6. digest
DIGEST_CAP = 5  # proposals shown per digest — flood control, the rest is a "+N pending" footer


def discord_digest(conn, no_discord, effects=()):
    """Throttled review digest. Shows the top-DIGEST_CAP proposals (salience DESC, then
    score), and PINGS only when there are proposals not covered by the previous digest —
    a nightly re-wall of the same pending list is how review queues rot. The seen-ids
    watermark lives in skill_scan_cursor.config['digest_ids'] (update_cursor merges,
    never clobbers); it is written only once the digest was delivered, and a failed Discord
    POST raises. `effects` are skill_effects' lines, shown under the proposals."""
    cur = conn.cursor()
    # Claude Code puts every served skill's description in each session's context: the total of
    # the global ones, which every session pays for, shows whether widens are inflating it.
    cur.execute(
        "SELECT count(*), COALESCE(sum(length(description)), 0) FROM skills_lane.skill_registry "
        "WHERE status='active' AND scope='global'"
    )
    n_skills, chars = cur.fetchone()
    size = {"global_skills": n_skills, "global_description_chars": int(chars)}
    cur.execute(
        """SELECT id, kind, name, score, grounded_sessions, judge_sessions, salience, source_detector
             FROM skills_lane.skill_gap_candidates WHERE status='proposed'
             ORDER BY salience DESC NULLS LAST, score DESC, id"""
    )
    rows = cur.fetchall()
    if not rows:
        return {"proposed": 0, "new": 0, **size}
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
    if effects:
        lines.append("\nAccepted changes (the span since the accept, against the span before it):")
        lines.extend(f"- {e}" for e in effects[:DIGEST_CAP])
    lines.append(
        f"\nGlobal skill descriptions: {n_skills} skills, {int(chars)} chars in every session"
    )
    lines.append("Review: `/skill-review` (or skill_review.py list) · accept/reject <id>")
    msg = "\n".join(lines)

    if not new:
        print(f"(digest: {len(rows)} pending, none new since last digest — not pinging)")
        return {"proposed": len(rows), "new": 0, **size}
    # Notifier: POST to a Discord webhook if configured; otherwise just print (default).
    if not no_discord and config.DISCORD_WEBHOOK:
        try:
            body = json.dumps({"content": msg[:1900]}).encode()
            req = urllib.request.Request(
                config.DISCORD_WEBHOOK, data=body, headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            # Keep the digest in the log. digest_ids stays unwritten, so the next run pings
            # again, and the stage error marks tonight's run not ok.
            print(msg)
            raise
    else:
        print(msg)
    # Only a delivered digest counts as seen.
    cur.execute(
        "UPDATE skills_lane.skill_scan_cursor SET config = COALESCE(config,'{}'::jsonb) || %s::jsonb, "
        "updated_at=now() WHERE id=1",
        (json.dumps({"digest_ids": ids}),),
    )
    conn.commit()
    return {"proposed": len(rows), "new": len(new), **size}


#: The stages that consume the run's episode window: when one fails, the cursor holds.
WINDOW_STAGES = ("usage", "judge", "grounded", "struggle_arc", "post_fire")


def run_lane(limit: int = 40, backfill: bool = False, no_discord: bool = False) -> dict:
    """Run the whole dream→skills lane once. No argparse, so the dream container's run_once
    can call it directly (the catalog comes from the registry, not disk).

    Returns {"stages": {stage: stats}, "errors": ["stage: ExcType: message"], "advanced": bool}.
    A stage that raises lands in "errors" and the stages after it still run. A model that
    cannot be reached (llm_client.LLM_TRANSPORT_ERRORS: usage limit, outage, no login) stops
    the lane instead, since every later judge call would fail the same way: the exception
    propagates and the cursor holds. The cursor also holds when a WINDOW_STAGES stage failed;
    it moves only past work that finished."""
    stages: dict[str, Any] = {}
    errors: dict[str, str] = {}

    def stage(name, fn, *args, **kwargs):
        try:
            stages[name] = fn(*args, **kwargs)
        except LLM_TRANSPORT_ERRORS:
            raise
        except Exception as e:
            logger.error("dream→skills %s failed: %s", name, e, exc_info=True)
            errors[name] = f"{type(e).__name__}: {e}"
        else:
            print(f"{name}:", stages[name])

    with L.connect() as conn:
        # Autocommit: every write below commits on its own, and a transaction opened by a
        # read would sit idle through the minutes of judge calls, holding locks on episodes
        # and the skill tables (and blocking a migration's ALTER TABLE queued behind them).
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT now()")
        run_started = cur.fetchone()[0]
        window = lane.open_window(
            conn, "skills", legacy_since=L.get_cursor(conn).get("last_scan_at"), backfill=backfill
        )
        substantive, carry = lane.sessions(conn, window, limit=limit, wanted=SM.is_substantive)
        # Every stage covers the episodes up to where the session walk stopped, so the stages
        # agree on what this run consumed and the next run starts right after it.
        covered = replace(window, until_id=carry)

        catalog = SM.load_catalog(conn)
        scopes = SM.load_scopes(conn)
        fires = DB.fire_events(conn, covered)
        seen: set[int] = set()

        print(
            f"[dream→skills] episodes {covered.after_id}..{covered.until_id} "
            f"(newest {window.until_id})  judge={stage_model(lane.JUDGE_STAGE, lane.OPUS)} "
            f"draft={stage_model(lane.DRAFT_STAGE, lane.OPUS)}"
        )
        print(f"substantive sessions to judge: {len(substantive)}")
        stage("usage", sync_usage, conn, fires)
        stage("judge", judge_sessions, conn, substantive, catalog, scopes, seen)
        if "judge" not in errors:
            lane.mark_judged(conn, "skills", substantive)
        stage("grounded", capture_grounded, conn, covered, fires, seen, catalog=catalog)
        stage("struggle_arc", struggle_arc.run, conn, window=covered, catalog=catalog)
        stage("post_fire", post_fire.run, conn, fires=fires, catalog=catalog)
        stage("procedure_miner", run_procedure_miner, conn, catalog)
        stage("drafts", skill_draft.draft_pending, conn)
        stage("decay", L.decay_stale, conn)
        stage("effects", skill_effects, conn)
        stage(
            "digest", discord_digest, conn, no_discord, stages.get("effects", {}).get("lines", ())
        )

        held = [name for name in WINDOW_STAGES if name in errors]
        if held:
            print(f"cursor held at episode {window.after_id}: {', '.join(held)} failed")
        else:
            lane.advance(conn, window, carry)
            # last_scan_at is the cursor a database without the dream lane cursor tables (and
            # an older image) reads; see dream/lane.py.
            L.update_cursor(
                conn,
                run_started,
                config={
                    "propose_score": L.PROPOSE_SCORE,
                    "session_jaccard": L.SESSION_JACCARD,
                    "limit": limit,
                },
            )
    print("done.")
    return {
        "stages": stages,
        "errors": [f"{name}: {msg}" for name, msg in errors.items()],
        "advanced": not held,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true", help="ignore the cursor; scan 30d")
    ap.add_argument("--limit", type=int, default=40, help="max substantive sessions to judge")
    ap.add_argument("--no-discord", action="store_true")
    args = ap.parse_args()
    res = run_lane(limit=args.limit, backfill=args.backfill, no_discord=args.no_discord)
    if res["errors"]:
        raise SystemExit("stages failed: " + "; ".join(res["errors"]))


if __name__ == "__main__":
    main()
