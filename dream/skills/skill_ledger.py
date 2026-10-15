# mypy: ignore-errors
"""skills_lane ledger writer — the dream->skills "writer contract" from schema/022.

Owns: identity resolution (derive = signature_key + session-id Jaccard, NOT name;
retune = (name,direction)), evidence accumulation (dedup by
(session, signal, class, work date), recompute weights from the FULL evidence every
merge — never incrementally), the 0.5x judge discount (the score column does it),
observe->proposed classification (v2 gates: retune proposes on one quote-carrying
instance; derive on recurrence across distinct days, high salience, or the legacy
score), and decay of unseen 'observe' candidates ('proposed' rows wait for human
review). Grounded signals advance; the LLM judge only nominates. promoted is NEVER
set here (only the review accept, which applies the proposal to skill_registry).

Evidence entry shape (all fields beyond class/signal optional, backward compatible):
    {"class": "judge"|"grounded", "signal": str, "session_id": str|None,
     "quote": str|None, "scan_night": "YYYY-MM-DD"|None, "date": "YYYY-MM-DD"|None}
"""

from __future__ import annotations

import json
import re

from ingestion.schema_check import ColumnProbe

from . import config

# identity / classification knobs (Oracle-reviewed defaults)
SESSION_JACCARD = 0.4  # derive identity: evidence-session overlap (ground truth, primary)
SIGNATURE_JACCARD = 0.6  # derive identity: signature-token overlap (secondary)
SIGNATURE_VETO_FLOOR = 0.15  # session-overlap match vetoed when both signatures exist below this
PROPOSE_SCORE = 1.5  # legacy observe -> proposed gate (3 judge sessions, or 1 grounded + 1 judge)
PROPOSE_SALIENCE = 4  # derive: propose when detector-rated pain/severity reaches this (1-5)
PROPOSE_DAYS = 2  # derive: propose when its evidence dates span this many distinct days
DECAY_TTL_DAYS = 28  # last_seen older than this -> retired (time-based; no nightly mass-bump)
EVIDENCE_CAP = 60  # bound the JSONB
PHRASING_CAP = 12  # trigger phrasings kept on a derive candidate
# The states a retune's (name, direction) key is unique among (the open-identity migration's
# index predicate): the ones still waiting on a decision. A promoted or retired row is history.
_OPEN = "status IN ('observe', 'proposed', 'rejected')"

#: The draft-attempts migration adds the draft-attempt columns. The image ships before it is
#: applied, so every statement that names them asks this probe first.
DRAFT_STATE = ColumnProbe(
    "SELECT 1 FROM pg_attribute "
    "WHERE attrelid = to_regclass('skills_lane.skill_gap_candidates') "
    "  AND attname = 'draft_attempts' AND NOT attisdropped",
    "the draft-attempts migration (skill_gap_candidates.draft_attempts)",
)

# grounded signals advance toward apply; judge signals only nominate (and get 0.5x'd by `score`).
_GROUNDED_W = {
    "explicit_request": 3.0,
    "accept": 3.0,
    "reject": -3.0,
    "dismissal": 1.0,
}
_JUDGE_W = {"gap_scan": 1.0, "under_trigger": 1.0}

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
    """Append new, dedup by (session_id, signal, class, work date): a scanner seeing the same
    work again (a re-run after a held cursor, the miner's overlapping window) adds nothing,
    while the same session on another day is a fresh sighting. Entries without a date
    (gap_scan, explicit requests, legacy rows) key on scan_night instead, so a later night's
    sighting still counts; entries with neither all key to one bucket.

    The cap drops the oldest judge entries. A grounded entry is a human decision or a user
    signal (an accept is what lets a row stand as promoted: grounded_required_to_apply) and
    is never dropped."""
    out, seen = [], set()
    for e in list(old) + list(new):
        k = (
            e.get("session_id"),
            e.get("signal"),
            e.get("class"),
            e.get("date") or e.get("scan_night"),
        )
        if k in seen:
            continue
        seen.add(k)
        out.append(e)
    excess = len(out) - EVIDENCE_CAP
    if excess > 0:
        oldest = {id(e) for e in [e for e in out if e.get("class") != "grounded"][:excess]}
        out = [e for e in out if id(e) not in oldest]
    return out


def _union_phrasings(old, new) -> list[str]:
    """Trigger phrasings of a candidate: the stored ones, then the new ones, deduped. A merge
    that brings none leaves them as they were."""
    both = [p for p in [*(old or []), *(new or [])] if isinstance(p, str) and p.strip()]
    return list(dict.fromkeys(both))[:PHRASING_CAP]


def _recurrence_days(evidence: list[dict]) -> int:
    """DISTINCT evidence dates (the day the sighted work happened) — the derive recurrence
    unit. Not scan nights: the scanner seeing the same work again (procedure_miner's weekly
    35-day window, a re-run after a crash) is not the work recurring. Entries without a
    date (gap_scan, explicit requests, legacy rows) never count."""
    return len({e["date"] for e in evidence if e.get("date")})


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
    """Return (id, evidence, trigger_phrasings, cooling) of the matching open candidate, or
    None. `cooling` is True when the row is rejected inside its 30d suppression window
    (rejected_until)."""
    if kind == "retune":
        # Match skill_gap_named_idx's unique key among the open states (the open-identity
        # migration): a promoted or retired row is history, so a new sighting opens a fresh
        # candidate. On a database still on the old index (unique across every status) the
        # INSERT then conflicts with the closed row; merge_candidate reports that row untouched.
        cur.execute(
            "SELECT id, evidence, trigger_phrasings, "
            "(status='rejected' AND COALESCE(rejected_until > now(), false)) AS cooling "
            "FROM skills_lane.skill_gap_candidates "
            f"WHERE kind=%s AND name=%s AND COALESCE(direction,'-')=COALESCE(%s,'-') AND {_OPEN} "
            "LIMIT 1",
            (kind, name, direction),
        )
        r = cur.fetchone()
        return (r[0], r[1], r[2], r[3]) if r else None
    # derive: semantic identity over active rows. Empty-vs-empty is NOT identity: a candidate
    # submitted without session ids / signature must never wildcard-match (and clobber) an
    # existing empty-keyed row — each leg only counts when BOTH sides are non-empty.
    cur.execute(
        "SELECT id, evidence, signature_key, trigger_phrasings, name "
        "FROM skills_lane.skill_gap_candidates "
        "WHERE kind='derive' AND status IN ('observe','proposed')"
    )
    rows = cur.fetchall()
    new_toks = set((sigkey or "").split())
    if not new_toks:
        # No signature to compare (an explicit "make this a skill" request): a shared session
        # alone says nothing about the topic, and would rewrite another gap's candidate. Only
        # the row of the same name is this one.
        for cid, ev, _rk, phr, row_name in rows:
            if row_name == name:
                return cid, ev or [], phr or [], False
        return None
    best, best_score = None, 0.0
    for cid, ev, rk, phr, _row_name in rows:
        old_sessions = _ev_sessions(ev or [])
        old_toks = set((rk or "").split())
        sj = _jaccard(new_sessions, old_sessions) if new_sessions and old_sessions else 0.0
        tj = _jaccard(new_toks, old_toks) if old_toks else 0.0
        # Signature disagreement vetoes session-overlap identity: one long session
        # routinely contains several distinct gaps, so a shared session alone is not
        # identity when both sides carry signatures that don't resemble each other.
        session_match = sj >= SESSION_JACCARD and not (old_toks and tj < SIGNATURE_VETO_FLOOR)
        if session_match or tj >= SIGNATURE_JACCARD:
            rank = max(sj, tj * 0.9)  # session overlap weighted above signature tokens
            if rank > best_score:
                best, best_score = (cid, ev or [], phr or [], False), rank
    return best


def _passes_gate(kind: str, evidence: list[dict], score, salience) -> bool:
    """v2 observe -> proposed gates (design 00: §3). The legacy score bar stays for both
    kinds (compat with existing rows/config); the new per-kind gates are ADDITIVE looseners:
      - retune: ONE strong instance — an evidence entry carrying a verbatim quote (the
        deviation/correction moment). Cheap to review, high prior.
      - derive: recurrence (distinct evidence dates) OR high detector salience.
    Junk is controlled downstream (candidate decay + human review), not at this gate."""
    if score is not None and score >= PROPOSE_SCORE:
        return True
    if kind == "retune":
        return any((e.get("quote") or "").strip() for e in evidence)
    if kind == "derive":
        if salience is not None and salience >= PROPOSE_SALIENCE:
            return True
        return _recurrence_days(evidence) >= PROPOSE_DAYS
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

    A retune's (name, direction) is one row among the OPEN candidates (observe / proposed /
    rejected; the open-identity migration). A sighting of a skill whose earlier row is promoted or retired
    therefore opens a fresh candidate holding only the new evidence; the closed row is
    history. A re-sighted rejected row:
      - inside its 30d cooldown: skipped silently, no write;
      - past the cooldown: evidence merges and the row REOPENS to 'observe'.
    Before that migration the closed row still occupies the key: the INSERT does nothing and the
    row is reported untouched.

    salience persists as max(existing, new) — a candidate's pain rating only ratchets up.
    source_detector / proposed_patch: latest non-null wins (a fresh patch draft supersedes)."""
    cur = conn.cursor()
    sigkey = signature_key(signature, tools) if kind == "derive" else None
    new_sessions = _ev_sessions(evidence_entries)
    match = _resolve_id(cur, kind, name, direction, target_skills, sigkey, new_sessions)

    if match:
        cid, old_ev, old_phr, cooling = match
        if cooling:
            # rejected within the 30d suppression window: respect the cooldown — no
            # evidence merge, no write, and no fresh INSERT against the occupied key.
            conn.commit()
            return {"id": cid, "status": "rejected", "score": None, "merged": False}
        evidence = _union_evidence(old_ev, evidence_entries)
        phrasings = _union_phrasings(old_phr, trigger_phrasings)
        merged = True
    else:
        cid, merged = None, False
        evidence = _union_evidence([], evidence_entries)
        phrasings = _union_phrasings([], trigger_phrasings)

    js, gs, jw, gw = _rollup(evidence)
    emb_lit = None
    if do_embed and kind == "derive" and summary:
        try:
            emb_lit = vec_literal(embed([summary], "document")[0])
        except Exception:
            emb_lit = None
    phr = json.dumps(phrasings)
    ev_json = json.dumps(evidence)
    tgt = list(target_skills or [])

    if cid:
        cur.execute(
            """UPDATE skills_lane.skill_gap_candidates SET
                 evidence=%s::jsonb, judge_sessions=%s, grounded_sessions=%s,
                 judge_weight=%s, grounded_weight=%s,
                 summary=COALESCE(NULLIF(%s,''), summary),
                 signature=COALESCE(%s, signature), signature_key=COALESCE(NULLIF(%s,''), signature_key),
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
            f"""INSERT INTO skills_lane.skill_gap_candidates
                 (kind, name, signature_key, target_skills, direction, summary, signature,
                  trigger_phrasings, summary_embedding, evidence,
                  judge_sessions, grounded_sessions, judge_weight, grounded_weight,
                  salience, source_detector, proposed_patch)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::halfvec,%s::jsonb,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (kind, name, COALESCE(direction, '-'))
                 WHERE kind = 'retune' AND {_OPEN} DO NOTHING
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
        # didn't see (a concurrent writer landed it between resolve and insert, or, before the
        # open-identity migration, a closed row). Never crash the lane — report the occupying
        # row, untouched; an open one first.
        conn.commit()
        cur.execute(
            "SELECT id, status, score FROM skills_lane.skill_gap_candidates "
            "WHERE kind=%s AND name=%s AND COALESCE(direction,'-')=COALESCE(%s,'-') "
            f"ORDER BY ({_OPEN}) DESC, id DESC LIMIT 1",
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

    # A re-sighted rejected row past its cooldown REOPENS: back to 'observe' so the gates
    # below re-decide. The old reject evidence merged in above still weighs against it (-3
    # grounded), so reopening stays conservative.
    if status == "rejected":
        cur.execute(
            "UPDATE skills_lane.skill_gap_candidates SET status='observe', "
            "reject_reason=NULL, rejected_until=NULL, updated_at=now() WHERE id=%s",
            (rid,),
        )
        status = "observe"

    # observe -> proposed via the v2 gates; grounded->accepted is the review path's job
    if status == "observe" and _passes_gate(kind, evidence, score, row_salience):
        cur.execute(
            "UPDATE skills_lane.skill_gap_candidates SET status='proposed', updated_at=now() WHERE id=%s",
            (rid,),
        )
        status = "proposed"
    conn.commit()
    return {"id": rid, "status": status, "score": score, "merged": merged}


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
