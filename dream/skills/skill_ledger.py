#!/usr/bin/env python3
# mypy: ignore-errors
"""skills_lane ledger writer — the dream->skills "writer contract" from schema/022.

Run under the synapse venv (needs psycopg + voyageai):
    ~/services/synapse/.venv/bin/python ~/scripts/skill_ledger.py --selftest

Owns: identity resolution (derive = signature_key + session-id Jaccard, NOT name, over
open rows plus rejected ones and ones retired in the last REOPEN_RETIRED_DAYS;
retune/consolidate = (name,direction)/target), evidence accumulation (dedup by
(session, signal, class, scan_night), recompute weights from the FULL evidence every
merge — never incrementally; a judge signal votes once per session however many nights
re-scan it), the 0.5x judge discount (the score column does it),
observe->proposed classification (v2 gates: retune proposes on one quote-carrying
instance; derive on recurrence — PROPOSE_SESSIONS distinct sessions on as many distinct
days — high salience, or the legacy score), and decay of unseen 'observe' candidates
('proposed' rows wait for human review). Grounded signals advance; the LLM judge only
nominates. promoted is NEVER set here (only the review accept, which applies the
proposal to skill_registry).

Evidence entry shape (all fields beyond class/signal optional, backward compatible):
    {"class": "judge"|"grounded", "signal": str, "session_id": str|None,
     "quote": str|None, "scan_night": "YYYY-MM-DD"|None, "date": "YYYY-MM-DD"|None}
"""

from __future__ import annotations

import argparse
import json
import re

from ingestion.schema_check import ColumnProbe

from . import config

#: Schema 062 adds skill_gap_candidates.proposed_at + recommendation. The image ships before
#: it is applied, so every statement that names either column asks this probe first; until
#: both exist the lane keeps its pre-062 behaviour (no proposed_at stamp, no expiry, no
#: stored recommendations).
QUEUE_PROBE = ColumnProbe(
    "SELECT 1 FROM pg_attribute "
    "WHERE attrelid = to_regclass('skills_lane.skill_gap_candidates') "
    "  AND attname IN ('proposed_at', 'recommendation') AND NOT attisdropped "
    "HAVING count(*) = 2",
    "schema 062 (skill_gap_candidates.proposed_at, recommendation)",
)


def queue_ready(conn) -> bool:
    return QUEUE_PROBE.ready(conn, conn.info.dsn)


#: Schema 064 adds the autopilot's audit table (skill_auto_reviews) and the columns that say
#: who applied a row (applied_by, auto_review_id) and which body the lane wrote
#: (skill_registry.lane_body_hash). Until all of them exist the autopilot does not run and
#: a human accept leaves applied_by unset.
AUTOPILOT_PROBE = ColumnProbe(
    "SELECT 1 FROM pg_attribute "
    "WHERE NOT attisdropped AND ("
    "  (attrelid = to_regclass('skills_lane.skill_gap_candidates') "
    "     AND attname IN ('applied_by', 'auto_review_id')) "
    "  OR (attrelid = to_regclass('skills_lane.skill_registry') AND attname = 'lane_body_hash') "
    "  OR (attrelid = to_regclass('skills_lane.skill_auto_reviews') "
    "     AND attname = 'rolled_back_at')) "
    "HAVING count(*) = 4",
    "schema 064 (skill_auto_reviews, applied_by, auto_review_id, lane_body_hash)",
)


def autopilot_ready(conn) -> bool:
    return AUTOPILOT_PROBE.ready(conn, conn.info.dsn)


# identity / classification knobs (Oracle-reviewed defaults)
SESSION_JACCARD = 0.4  # derive identity: evidence-session overlap (ground truth, primary)
SIGNATURE_JACCARD = 0.6  # derive identity: signature-token overlap (secondary)
SIGNATURE_VETO_FLOOR = 0.15  # session-overlap match vetoed when both signatures exist below this
PROPOSE_SCORE = 1.5  # legacy observe -> proposed gate (3 judge sessions, or 1 grounded + 1 judge)
PROPOSE_SALIENCE = 4  # derive: propose when detector-rated pain/severity reaches this (1-5)
# derive recurrence: this many distinct sessions, started on this many distinct days. Scan
# nights don't count: one long session re-scanned the next night is not a repeat (6 of the
# 8 re-seen derive rows of the 60 days to 2026-11-06 were exactly that).
PROPOSE_SESSIONS = 2
DECAY_TTL_DAYS = 28  # last_seen older than this -> retired (time-based; no nightly mass-bump)
# A derive retired this recently still takes a re-sighting (it reopens) instead of a fresh
# row starting from zero.
REOPEN_RETIRED_DAYS = 60
EVIDENCE_CAP = 60  # bound the JSONB

# grounded signals advance toward apply; judge signals only nominate (and get 0.5x'd by `score`).
_GROUNDED_W = {
    "explicit_request": 3.0,
    "accept": 3.0,
    "reject": -3.0,
    "revive": 3.0,  # a person bringing a rejected/retired proposal back undoes their reject
    "post_change_fired": 2.0,
    "dismissal": 1.0,
    "user_correction": 1.0,
}
_JUDGE_W = {"gap_scan": 1.0, "under_trigger": 1.0, "overlap": 1.0}

_STOP = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "into",
    "your",
    "you",
    "run",
    "use",
    "get",
    "set",
    "via",
    "all",
    "any",
    "out",
    "not",
}


def _env(key: str) -> str:
    v = config.db_url() if key == "SYNAPSE_DB_URL" else config.secret(key)
    if not v:
        raise RuntimeError(f"{key} not configured (set env or {config.ENV_FILE})")
    return v


def connect():
    import psycopg

    return psycopg.connect(_env("SYNAPSE_DB_URL"), connect_timeout=10)


# --------------------------------------------------------------------- embeddings
_embedder = None


def embed(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Same embedding backend as Synapse recall (ingestion.embedding factory;
    Voyage voyage-4-large @ 2048 dims by default)."""
    global _embedder
    texts = [t for t in texts if t]
    if not texts:
        return []
    if _embedder is None:
        from ingestion.embedding import create_embedder, embed_provider

        # Keep the loud missing-key error on the default (Voyage) backend;
        # other providers resolve their own SYNAPSE_EMBED_* env.
        key = _env("VOYAGE_API_KEY") if embed_provider() == "voyage" else None
        _embedder = create_embedder(voyage_api_key=key)
    task = "query" if input_type == "query" else "document"
    return _embedder.embed(texts, task=task)


def vec_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


# ----------------------------------------------------------------------- identity
def _tokens(*parts: str) -> set[str]:
    text = " ".join(p for p in parts if p).lower()
    return {t for t in re.findall(r"[a-z0-9]{3,}", text) if t not in _STOP}


def signature_key(signature: str | None, tools: list[str] | None) -> str:
    toks = _tokens(signature or "") | {t.lower() for t in (tools or [])}
    return " ".join(sorted(toks))


def _jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _ev_sessions(evidence: list[dict]) -> set[str]:
    return {e["session_id"] for e in evidence if e.get("session_id")}


def _union_evidence(old: list[dict], new: list[dict]) -> list[dict]:
    """Append new, dedup by (session_id, signal, class, scan_night): a re-sighting on a LATER
    scan night counts as fresh recurrence evidence, while same-night duplicates still collapse.
    Legacy entries without scan_night all key to one bucket (scan_night=None) — same collapse
    behavior they had before scan nights existed. A review act (a session-less grounded
    entry: accept / reject / revive) is never a duplicate: reject, revive, reject is three acts."""
    out, seen = [], set()
    for e in list(old) + list(new):
        k = (e.get("session_id"), e.get("signal"), e.get("class"), e.get("scan_night"))
        if k in seen and not (e.get("class") == "grounded" and not e.get("session_id")):
            continue
        seen.add(k)
        out.append(e)
    return out[-EVIDENCE_CAP:]


def _session_dates(evidence: list[dict]) -> dict[str, str | None]:
    """session_id -> the earliest `date` its entries carry (None when none carries one)."""
    out: dict[str, str | None] = {}
    for e in evidence:
        sid = e.get("session_id")
        if not sid:
            continue
        d, prev = e.get("date") or None, out.get(sid)
        out[sid] = min(prev, d) if prev and d else (prev or d)
    return out


def _recurrence(evidence: list[dict]) -> bool:
    """The derive recurrence gate: PROPOSE_SESSIONS distinct sessions on as many distinct
    session days. An undated session (its episodes are gone) can't make up a day."""
    return len({d for d in _session_dates(evidence).values() if d}) >= PROPOSE_SESSIONS


def _standing_reject(evidence: list[dict]) -> bool:
    """True when a person's reject is the last word in the evidence (no revive or accept
    after it)."""
    last = None
    for e in evidence:
        if e.get("class") == "grounded" and e.get("signal") in ("reject", "revive", "accept"):
            last = e["signal"]
    return last == "reject"


def _fill_session_dates(cur, evidence: list[dict]) -> None:
    """Stamp `date` in place on the entries of every session with no dated entry yet: the
    session's first episode day, min(created_at), the same clock the rest of the lane
    dates by. Rows written before run_derive stamped dates get theirs at their next merge."""
    sids = sorted(s for s, d in _session_dates(evidence).items() if not d)
    if not sids:
        return
    cur.execute(
        "SELECT session_id, min(created_at)::date FROM episodes "
        "WHERE session_id = ANY(%s) GROUP BY session_id",
        (sids,),
    )
    found = {sid: d.isoformat() for sid, d in cur.fetchall() if d}
    for e in evidence:
        if e.get("session_id") in found and not e.get("date"):
            e["date"] = found[e["session_id"]]


def _rollup(evidence: list[dict]) -> tuple[int, int, float, float]:
    """Recompute (judge_sessions, grounded_sessions, judge_weight, grounded_weight) from FULL
    evidence. A judge signal votes once per session: entries are kept per scan night, and a
    session re-scanned on three nights is still one sighting, not a 1.5 score on its own."""
    j_sess, g_sess = set(), set()
    j_votes: set[tuple] = set()
    jw = gw = 0.0
    for e in evidence:
        sid, sig, cls = e.get("session_id"), e.get("signal"), e.get("class")
        if cls == "grounded":
            if sid:
                g_sess.add(sid)
            gw += _GROUNDED_W.get(sig, 1.0)
        else:
            if sid:
                j_sess.add(sid)
                if (sid, sig) in j_votes:
                    continue
                j_votes.add((sid, sig))
            jw += _JUDGE_W.get(sig, 1.0)
    return len(j_sess), len(g_sess), jw, gw


def _resolve_id(cur, kind, name, direction, target_skills, sigkey, new_sessions):
    """Return (id, evidence, cooling) of the matching candidate, or None. `cooling` is True
    when the row is rejected inside its 30d suppression window (rejected_until)."""
    if kind in ("retune", "consolidate"):
        # Match the skill_gap_named_idx unique key EXACTLY — no status filter. A row in a
        # terminal state (promoted/rejected/retired) still occupies (kind, name, direction);
        # filtering it out here sent merge_candidate down the INSERT path straight into a
        # UniqueViolation every night the lane re-derived the same candidate.
        cur.execute(
            "SELECT id, evidence, "
            "(status='rejected' AND COALESCE(rejected_until > now(), false)) AS cooling "
            "FROM skills_lane.skill_gap_candidates "
            "WHERE kind=%s AND name=%s AND COALESCE(direction,'-')=COALESCE(%s,'-') LIMIT 1",
            (kind, name, direction),
        )
        r = cur.fetchone()
        return (r[0], r[1], r[2]) if r else None
    # derive: semantic identity over open rows, rejected ones (a revival inherits the
    # cooldown and the human reject instead of coming back as a fresh row) and ones retired
    # in the last REOPEN_RETIRED_DAYS (a repeat after decay reopens the row it repeats).
    # Empty-vs-empty is NOT identity: a candidate submitted without session ids / signature
    # must never wildcard-match (and clobber) an existing empty-keyed row — each leg only
    # counts when BOTH sides are non-empty. Open rows come first, so on a tie the sighting
    # joins the open row rather than an old one's cooldown.
    cur.execute(
        "SELECT id, evidence, signature_key, "
        "(status='rejected' AND COALESCE(rejected_until > now(), false)) AS cooling "
        "FROM skills_lane.skill_gap_candidates "
        "WHERE kind='derive' AND (status IN ('observe','proposed','rejected') "
        "  OR (status='retired' AND updated_at > now() - (%s || ' days')::interval)) "
        "ORDER BY (status IN ('observe','proposed')) DESC, id",
        (str(REOPEN_RETIRED_DAYS),),
    )
    best, best_score = None, 0.0
    new_toks = set((sigkey or "").split())
    for cid, ev, rk, cooling in cur.fetchall():
        old_sessions = _ev_sessions(ev or [])
        old_toks = set((rk or "").split())
        sj = _jaccard(new_sessions, old_sessions) if new_sessions and old_sessions else 0.0
        tj = _jaccard(new_toks, old_toks) if new_toks and old_toks else 0.0
        # Signature disagreement vetoes session-overlap identity: one long session
        # routinely contains several distinct gaps, so a shared session alone is not
        # identity when both sides carry signatures that don't resemble each other.
        session_match = sj >= SESSION_JACCARD and not (
            new_toks and old_toks and tj < SIGNATURE_VETO_FLOOR
        )
        if session_match or tj >= SIGNATURE_JACCARD:
            rank = max(sj, tj * 0.9)  # session overlap weighted above signature tokens
            if rank > best_score:
                best, best_score = (cid, ev or [], bool(cooling)), rank
    return best


def _passes_gate(kind: str, evidence: list[dict], score, salience) -> bool:
    """v2 observe -> proposed gates (design 00: §3). The legacy score bar stays for every
    kind (compat with existing rows/config); the new per-kind gates are ADDITIVE looseners:
      - retune: ONE strong instance — an evidence entry carrying a verbatim quote (the
        deviation/correction moment). Cheap to review, high prior.
      - derive: recurrence (PROPOSE_SESSIONS distinct sessions on as many distinct days;
        entries need their `date`, see _fill_session_dates) OR high detector salience.
        Neither applies while a person's reject stands: a rejected derive reopened past
        its cooldown still has the salience and session days it was rejected with, so
        only the score, which carries the -3, can re-propose it.
      - consolidate: legacy score path only (overlap nominations carry no quotes; one
        cosine sighting is not review-worthy on its own).
    Junk is controlled downstream (candidate decay + human review), not at this gate."""
    if score is not None and score >= PROPOSE_SCORE:
        return True
    if kind == "retune":
        return any((e.get("quote") or "").strip() for e in evidence)
    if kind == "derive":
        if _standing_reject(evidence):
            return False
        if salience is not None and salience >= PROPOSE_SALIENCE:
            return True
        return _recurrence(evidence)
    return False


def merge_candidate(
    conn,
    kind,
    name,
    evidence_entries,
    *,
    signature=None,
    tools=None,
    summary="",
    trigger_phrasings=None,
    target_skills=None,
    direction=None,
    salience=None,
    source_detector=None,
    proposed_patch=None,
    do_embed=True,
) -> dict:
    """Resolve identity, union evidence, recompute rollups, upsert. Returns {id, status, score, merged}.

    Re-sighting a row in a terminal state (retune/consolidate share one unique key per
    (kind, name, direction) regardless of status; a derive matches rejected rows and rows
    retired in the last REOPEN_RETIRED_DAYS):
      - rejected inside its 30d cooldown: skipped silently, no write;
      - rejected past cooldown / retired: evidence merges and the row REOPENS to 'observe';
      - accepted / promoted: evidence merges (the sighting is recorded) but status never
        changes here — the observe->proposed gate below only fires on 'observe'.

    salience persists as max(existing, new) — a candidate's pain rating only ratchets up.
    source_detector / proposed_patch: latest non-null wins (a fresh patch draft supersedes)."""
    cur = conn.cursor()
    sigkey = signature_key(signature, tools) if kind == "derive" else None
    new_sessions = _ev_sessions(evidence_entries)
    match = _resolve_id(cur, kind, name, direction, target_skills, sigkey, new_sessions)

    if match:
        cid, old_ev, cooling = match
        if cooling:
            # rejected within the 30d suppression window: respect the cooldown — no
            # evidence merge, no write, and no fresh INSERT against the occupied key.
            conn.commit()
            return {"id": cid, "status": "rejected", "score": None, "merged": False}
        evidence = _union_evidence(old_ev, evidence_entries)
        merged = True
    else:
        cid, evidence, merged = None, _union_evidence([], evidence_entries), False
    if kind == "derive":
        _fill_session_dates(cur, evidence)  # the recurrence gate counts session days

    js, gs, jw, gw = _rollup(evidence)
    emb_lit = None
    if do_embed and kind == "derive" and summary:
        try:
            emb_lit = vec_literal(embed([summary], "document")[0])
        except Exception:
            emb_lit = None
    phr = json.dumps(trigger_phrasings or [])
    ev_json = json.dumps(evidence)
    tgt = list(target_skills or [])

    if cid:
        cur.execute(
            """UPDATE skills_lane.skill_gap_candidates SET
                 evidence=%s::jsonb, judge_sessions=%s, grounded_sessions=%s,
                 judge_weight=%s, grounded_weight=%s,
                 summary=COALESCE(NULLIF(%s,''), summary),
                 signature=COALESCE(%s, signature), signature_key=COALESCE(%s, signature_key),
                 trigger_phrasings=%s::jsonb, target_skills=%s,
                 summary_embedding=COALESCE(%s::halfvec, summary_embedding),
                 salience=GREATEST(salience, %s::smallint),
                 source_detector=COALESCE(%s, source_detector),
                 proposed_patch=COALESCE(%s, proposed_patch),
                 last_seen=now(), runs_since_seen=0, updated_at=now()
               WHERE id=%s
               RETURNING id, status, score, salience""",
            (
                ev_json,
                js,
                gs,
                jw,
                gw,
                summary,
                signature,
                sigkey,
                phr,
                tgt,
                emb_lit,
                salience,
                source_detector,
                proposed_patch,
                cid,
            ),
        )
    else:
        cur.execute(
            """INSERT INTO skills_lane.skill_gap_candidates
                 (kind, name, signature_key, target_skills, direction, summary, signature,
                  trigger_phrasings, summary_embedding, evidence,
                  judge_sessions, grounded_sessions, judge_weight, grounded_weight,
                  salience, source_detector, proposed_patch)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::halfvec,%s::jsonb,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (kind, name, COALESCE(direction, '-'))
                 WHERE kind IN ('retune', 'consolidate') DO NOTHING
               RETURNING id, status, score, salience""",
            (
                kind,
                name,
                sigkey,
                tgt,
                direction,
                summary,
                signature,
                phr,
                emb_lit,
                ev_json,
                js,
                gs,
                jw,
                gw,
                salience,
                source_detector,
                proposed_patch,
            ),
        )
    row = cur.fetchone()
    if row is None:
        # ON CONFLICT backstop fired: the unique key is occupied by a row the resolver
        # didn't see (e.g. a concurrent writer landed it between resolve and insert).
        # Never crash the lane — report the occupying row, untouched.
        conn.commit()
        cur.execute(
            "SELECT id, status, score FROM skills_lane.skill_gap_candidates "
            "WHERE kind=%s AND name=%s AND COALESCE(direction,'-')=COALESCE(%s,'-')",
            (kind, name, direction),
        )
        r = cur.fetchone()
        return {
            "id": r[0] if r else None,
            "status": r[1] if r else None,
            "score": r[2] if r else None,
            "merged": False,
        }
    rid, status, score, row_salience = row

    # A re-sighted terminal row REOPENS (rejected past its cooldown, or retired-stale):
    # back to 'observe' so the gates below re-decide. The old reject evidence merged in
    # above still weighs against it (-3 grounded; for a derive it also keeps the salience
    # and recurrence gates shut, see _passes_gate), so reopening stays conservative.
    if status in ("rejected", "retired"):
        cur.execute(
            "UPDATE skills_lane.skill_gap_candidates SET status='observe', "
            "reject_reason=NULL, rejected_until=NULL, updated_at=now() WHERE id=%s",
            (rid,),
        )
        status = "observe"

    # observe -> proposed via the v2 gates; grounded->accepted is the review path's job
    if status == "observe" and _passes_gate(kind, evidence, score, row_salience):
        _mark_proposed(conn, cur, rid)
        status = "proposed"
    conn.commit()
    return {"id": rid, "status": status, "score": score, "merged": merged}


def _mark_proposed(conn, cur, rid) -> None:
    """status -> 'proposed', stamping proposed_at once schema 062 is applied."""
    stamp = ", proposed_at=now()" if queue_ready(conn) else ""
    cur.execute(
        f"UPDATE skills_lane.skill_gap_candidates SET status='proposed'{stamp}, "
        "updated_at=now() WHERE id=%s",
        (rid,),
    )


def regate_observe(conn) -> dict:
    """Re-run the derive gate over 'observe' derive rows, so the recurrence gate (and the
    session days of rows written before the nightly stamped them) takes effect without
    waiting for a re-sighting. Only rows that now pass are written: their evidence with the
    days filled in, and status 'proposed'."""
    cur = conn.cursor()
    cur.execute(
        "SELECT id, evidence, salience FROM skills_lane.skill_gap_candidates "
        "WHERE kind='derive' AND status='observe'"
    )
    raised = []
    for rid, ev, salience in cur.fetchall():
        ev = list(ev or [])
        _fill_session_dates(cur, ev)
        _js, _gs, jw, gw = _rollup(ev)
        if _passes_gate("derive", ev, gw + 0.5 * jw, salience):
            cur.execute(
                "UPDATE skills_lane.skill_gap_candidates SET evidence=%s::jsonb WHERE id=%s",
                (json.dumps(ev), rid),
            )
            _mark_proposed(conn, cur, rid)
            raised.append(rid)
    conn.commit()
    return {"regated": len(raised), "ids": raised}


def decay_stale(conn, ttl_days: int = DECAY_TTL_DAYS) -> dict:
    """Retire 'observe' candidates whose last_seen aged past the TTL (time-based; no nightly
    mass-bump of inactive rows). Re-seen candidates get last_seen=now() in merge_candidate, so
    only the genuinely stale age out — only the actually-retiring rows are written (no
    dead-tuple bloat). 'proposed' rows are EXEMPT: they wait for human review and never
    silently decay (v2 — proposals were being lost to this)."""
    cur = conn.cursor()
    cur.execute(
        """UPDATE skills_lane.skill_gap_candidates
             SET status='retired', reject_reason='stale', updated_at=now()
           WHERE status='observe' AND last_seen < now() - (%s || ' days')::interval""",
        (str(ttl_days),),
    )
    retired = cur.rowcount
    conn.commit()
    return {"retired": retired}


def get_cursor(conn) -> dict:
    cur = conn.cursor()
    cur.execute(
        "SELECT last_scan_at, last_run_at, runs, config FROM skills_lane.skill_scan_cursor WHERE id=1"
    )
    r = cur.fetchone()
    return {"last_scan_at": r[0], "last_run_at": r[1], "runs": r[2], "config": r[3]} if r else {}


def update_cursor(conn, last_scan_at, config=None) -> None:
    """Advance the scan watermark. config keys MERGE into the stored JSONB (never replace
    wholesale): other stages park their own state there — e.g. the digest's seen-ids —
    and a run-config write must not clobber it."""
    cur = conn.cursor()
    cur.execute(
        """UPDATE skills_lane.skill_scan_cursor
             SET last_scan_at=GREATEST(COALESCE(last_scan_at, %s), %s),
                 last_run_at=now(), runs=runs+1,
                 config=COALESCE(config, '{}'::jsonb) || COALESCE(%s::jsonb, '{}'::jsonb),
                 updated_at=now()
           WHERE id=1""",
        (last_scan_at, last_scan_at, json.dumps(config) if config is not None else None),
    )
    conn.commit()


# ------------------------------------------------------------------------ selftest
def _selftest() -> None:
    """Insert two derive candidates for the same procedure under DIFFERENT names (the drift case),
    confirm they MERGE into one accumulating row, then clean up. Verifies the Oracle Q1 fix."""
    conn = connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM skills_lane.skill_gap_candidates WHERE name LIKE 'selftest-%'")
    conn.commit()
    ev1 = [
        {
            "session_id": "sess-A",
            "ts": "2026-06-21T00:00:00",
            "class": "judge",
            "signal": "gap_scan",
            "tools": ["Bash", "Read"],
        }
    ]
    r1 = merge_candidate(
        conn,
        "derive",
        "selftest-restic-backup",
        ev1,
        signature="restic backup to nas rclone",
        tools=["Bash", "Read"],
        summary="back up to NAS via restic then rclone to proton",
        do_embed=False,
    )
    # same procedure, DIFFERENT llm name, overlapping session + signature tokens
    ev2 = [
        {
            "session_id": "sess-A",
            "ts": "2026-06-22T00:00:00",
            "class": "judge",
            "signal": "gap_scan",
            "tools": ["Bash"],
        },
        {
            "session_id": "sess-B",
            "ts": "2026-06-22T00:00:00",
            "class": "judge",
            "signal": "gap_scan",
        },
    ]
    r2 = merge_candidate(
        conn,
        "derive",
        "selftest-nas-backup-workflow",
        ev2,
        signature="restic nas backup rclone proton",
        tools=["Bash"],
        summary="nightly restic backup workflow",
        do_embed=False,
    )
    cur.execute(
        "SELECT count(*), max(judge_sessions), max(score) FROM skills_lane.skill_gap_candidates WHERE name LIKE 'selftest-%'"
    )
    n, jsess, score = cur.fetchone()
    print(f"r1={r1}\nr2={r2}")
    print(
        f"rows={n} (want 1 = merged despite name drift)  judge_sessions={jsess} (want 2)  score={score}"
    )
    # grounded signal advances
    r3 = merge_candidate(
        conn,
        "derive",
        "selftest-restic-backup",
        [{"session_id": "sess-C", "class": "grounded", "signal": "explicit_request"}],
        do_embed=False,
    )
    print(
        f"after explicit_request: status={r3['status']} score={r3['score']} (grounded should push score up)"
    )
    # decay: age ONLY the selftest rows, run the normal 28d decay (fresh real rows stay safe),
    # confirm retire. Reset to 'observe' first — 'proposed' rows are decay-exempt by design.
    cur.execute(
        "UPDATE skills_lane.skill_gap_candidates SET status='observe', "
        "last_seen = now() - interval '60 days' WHERE name LIKE 'selftest-%'"
    )
    conn.commit()
    d = decay_stale(conn)
    cur.execute(
        "SELECT count(*) FROM skills_lane.skill_gap_candidates WHERE name LIKE 'selftest-%' AND status='retired'"
    )
    print(
        f"decay retired {cur.fetchone()[0]} aged selftest row(s) (want >=1); total retired this call={d['retired']}"
    )
    cur.execute("DELETE FROM skills_lane.skill_gap_candidates WHERE name LIKE 'selftest-%'")
    conn.commit()
    print("cleaned up.")
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
