#!/usr/bin/env python3
# mypy: ignore-errors
"""skills_lane ledger writer — the dream->skills "writer contract" from schema/022.

Run under the synapse venv (needs psycopg + voyageai):
    ~/services/synapse/.venv/bin/python ~/scripts/skill_ledger.py --selftest

Owns: identity resolution (derive = signature_key + session-id Jaccard, NOT name;
retune/consolidate = (name,direction)/target), evidence accumulation (dedup by
session+signal, recompute weights from the FULL evidence every merge — never
incrementally), the 0.5x judge discount (the score column does it), observe->proposed
classification on score, and decay of unseen candidates. Grounded signals advance;
the LLM judge only nominates. promoted is NEVER set here (filesystem-accept path only).
"""

from __future__ import annotations

import argparse
import json
import re

from . import config

# identity / classification knobs (Oracle-reviewed defaults)
SESSION_JACCARD = 0.4  # derive identity: evidence-session overlap (ground truth, primary)
SIGNATURE_JACCARD = 0.6  # derive identity: signature-token overlap (secondary)
PROPOSE_SCORE = 1.5  # observe -> proposed gate (3 judge sessions, or 1 grounded + 1 judge)
DECAY_TTL_DAYS = 28  # last_seen older than this -> retired (time-based; no nightly mass-bump)
EVIDENCE_CAP = 60  # bound the JSONB

# grounded signals advance toward apply; judge signals only nominate (and get 0.5x'd by `score`).
_GROUNDED_W = {
    "explicit_request": 3.0,
    "accept": 3.0,
    "reject": -3.0,
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
    """Append new, dedup by (session_id, signal, class) — keeps distinct-session counting honest."""
    out, seen = [], set()
    for e in list(old) + list(new):
        k = (e.get("session_id"), e.get("signal"), e.get("class"))
        if k in seen:
            continue
        seen.add(k)
        out.append(e)
    return out[-EVIDENCE_CAP:]


def _rollup(evidence: list[dict]) -> tuple[int, int, float, float]:
    """Recompute (judge_sessions, grounded_sessions, judge_weight, grounded_weight) from FULL evidence."""
    j_sess, g_sess = set(), set()
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
            jw += _JUDGE_W.get(sig, 1.0)
    return len(j_sess), len(g_sess), jw, gw


def _resolve_id(cur, kind, name, direction, target_skills, sigkey, new_sessions):
    """Return (id, evidence) of the matching active candidate, or None."""
    if kind in ("retune", "consolidate"):
        cur.execute(
            "SELECT id, evidence FROM skills_lane.skill_gap_candidates "
            "WHERE kind=%s AND name=%s AND COALESCE(direction,'-')=COALESCE(%s,'-') "
            "AND status IN ('observe','proposed','accepted') LIMIT 1",
            (kind, name, direction),
        )
        r = cur.fetchone()
        return (r[0], r[1]) if r else None
    # derive: semantic identity over active rows
    cur.execute(
        "SELECT id, evidence, signature_key FROM skills_lane.skill_gap_candidates "
        "WHERE kind='derive' AND status IN ('observe','proposed')"
    )
    best, best_score = None, 0.0
    for cid, ev, rk in cur.fetchall():
        sj = _jaccard(new_sessions, _ev_sessions(ev or []))
        tj = _jaccard((sigkey or "").split(), (rk or "").split())
        if sj >= SESSION_JACCARD or tj >= SIGNATURE_JACCARD:
            rank = max(sj, tj * 0.9)  # session overlap weighted above signature tokens
            if rank > best_score:
                best, best_score = (cid, ev or []), rank
    return best


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
    do_embed=True,
) -> dict:
    """Resolve identity, union evidence, recompute rollups, upsert. Returns {id, status, score, merged}."""
    cur = conn.cursor()
    sigkey = signature_key(signature, tools) if kind == "derive" else None
    new_sessions = _ev_sessions(evidence_entries)
    match = _resolve_id(cur, kind, name, direction, target_skills, sigkey, new_sessions)

    if match:
        cid, old_ev = match
        evidence = _union_evidence(old_ev, evidence_entries)
        merged = True
    else:
        cid, evidence, merged = None, _union_evidence([], evidence_entries), False

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
                 last_seen=now(), runs_since_seen=0, updated_at=now()
               WHERE id=%s
               RETURNING id, status, score""",
            (ev_json, js, gs, jw, gw, summary, signature, sigkey, phr, tgt, emb_lit, cid),
        )
    else:
        cur.execute(
            """INSERT INTO skills_lane.skill_gap_candidates
                 (kind, name, signature_key, target_skills, direction, summary, signature,
                  trigger_phrasings, summary_embedding, evidence,
                  judge_sessions, grounded_sessions, judge_weight, grounded_weight)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::halfvec,%s::jsonb,%s,%s,%s,%s)
               RETURNING id, status, score""",
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
            ),
        )
    rid, status, score = cur.fetchone()

    # observe -> proposed once the (discounted) score clears the bar; grounded->accepted is the review path's job
    if status == "observe" and score is not None and score >= PROPOSE_SCORE:
        cur.execute(
            "UPDATE skills_lane.skill_gap_candidates SET status='proposed', updated_at=now() WHERE id=%s",
            (rid,),
        )
        status = "proposed"
    conn.commit()
    return {"id": rid, "status": status, "score": score, "merged": merged}


def decay_stale(conn, ttl_days: int = DECAY_TTL_DAYS) -> dict:
    """Retire candidates whose last_seen aged past the TTL (time-based; no nightly mass-bump of
    inactive rows). Re-seen candidates get last_seen=now() in merge_candidate, so only the
    genuinely stale age out — only the actually-retiring rows are written (no dead-tuple bloat)."""
    cur = conn.cursor()
    cur.execute(
        """UPDATE skills_lane.skill_gap_candidates
             SET status='retired', reject_reason='stale', updated_at=now()
           WHERE status IN ('observe','proposed') AND last_seen < now() - (%s || ' days')::interval""",
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
    cur = conn.cursor()
    cur.execute(
        """UPDATE skills_lane.skill_scan_cursor
             SET last_scan_at=GREATEST(COALESCE(last_scan_at, %s), %s),
                 last_run_at=now(), runs=runs+1,
                 config=COALESCE(%s::jsonb, config), updated_at=now()
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
    # decay: age ONLY the selftest rows, run the normal 28d decay (fresh real rows stay safe), confirm retire
    cur.execute(
        "UPDATE skills_lane.skill_gap_candidates SET last_seen = now() - interval '60 days' WHERE name LIKE 'selftest-%'"
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
