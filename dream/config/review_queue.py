# mypy: ignore-errors
"""dream->config: the review queue looks after itself (nightly, from nightly.run_lane after the
gate; entries: expire, recommend, report). The same treatment the skills lane's queue gets
(dream/skills/review_queue.py). Nothing here applies a change: a config proposal edits the
user's own global rules, so accept stays a person's act (ask-only, no auto-apply).

  expire     a proposal unanswered for EXPIRE_DAYS goes to 'rejected', reject_reason 'expired'.
             It comes back by itself when a NEW session repeats the correction (nightly's
             reopen), so an answer is never forced and nothing is lost.
  recommend  the lane's judge reads each proposal once per rule + evidence and stores
             {verdict, reason} in recommendation. Evidence from different sessions is matched
             by meaning, and meaning can't see negation, so the prompt asks the judge to reject
             a rule whose quotes ask for opposite things.
  report     one plain-text post when tonight changed the queue (a new proposal, one dropped as
             already covered, an expiry), else a weekly reminder of what is still pending (or
             that nothing is). Every report that is built is stored in
             skills_lane.skill_scan_cursor.config['config_last_report'] = {at, text}, next to
             the skills lane's own last_report: delivery is whoever reads it. When
             SKILLS_DISCORD_WEBHOOK is set it is also posted there, else printed.

Everything here needs schema 065 (proposed_at, recommendation); run_lane probes first.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timedelta

from psycopg.rows import dict_row
from psycopg.types.json import Json

from dream.config import nightly as N
from dream.skills import config
from dream.skills import skill_measure as SM
from dream.skills.review_queue import EXPIRE_DAYS, REMIND_DAYS, REPORT_CHARS, _plain

RECOMMEND_CAP = 5  # recommendation calls per night (each is one judge call)
QUOTES = 8  # evidence quotes shown to the judge
REPORT_KEY = "config_last_report"
SENT_KEY = "config_report_sent_at"

_RECOMMEND_PROMPT = """You review one proposed standing rule for an AI agent. A nightly job mined
it from the user's own corrections across their sessions; if the user accepts it, it is written
into their global rules and applies everywhere.

Accept when the rule is a generalizable behavioral preference (it would apply on a different day
or topic), is about the agent's conduct, and the quotes below really ask for it. Reject when it is
a one-off task instruction, domain content, too vague to act on, or when the quotes are not
about the same thing. The quotes come from different sessions and were grouped by similarity of
meaning, which cannot tell a rule from its opposite: if two quotes ask for opposite things
(for example "be brief" and "be thorough"), reject and say they conflict.

PROPOSED RULE #{id}: {rule}
THE USER'S WORDS, ONE PER SESSION ({n} sessions):
{quotes}

Output ONLY a JSON object, no prose:
{{"verdict": "accept" | "reject", "reason": "one short sentence in plain words, no markdown"}}"""


# ------------------------------------------------------------------ expiry
def expire(conn, days: int = EXPIRE_DAYS) -> dict:
    """Reject proposals unanswered for `days` (reject_reason 'expired'; they reopen on a new
    session's sighting)."""
    c = conn.cursor(row_factory=dict_row)
    rows = c.execute(
        "UPDATE config_lane.config_proposals SET status='rejected', reject_reason=%s, "
        "updated_at=now() "
        "WHERE status='proposed' AND proposed_at < now() - make_interval(days => %s) "
        "RETURNING id, summary, recommendation, (now()::date - proposed_at::date) AS days",
        (N.EXPIRED, days),
    ).fetchall()
    conn.commit()
    return {"expired": len(rows), "items": sorted(rows, key=lambda r: r["id"])}


# ------------------------------------------------------------------ recommendation
def _quotes(evidence, cap: int = QUOTES) -> str:
    seen, lines = set(), []
    for e in evidence or []:
        sid, q = e.get("session_id"), " ".join(str(e.get("quote") or "").split())
        if not q or sid in seen:
            continue
        seen.add(sid)
        lines.append(f'- "{q[:300]}"')
    return "\n".join(lines[-cap:]) or "(no quotes recorded)"


def needs_recommendation(row: dict) -> bool:
    rec = row.get("recommendation")
    return not isinstance(rec, dict) or rec.get("key") != N._evidence_key(
        row["summary"], row["evidence"]
    )


def recommend(conn, cap: int = RECOMMEND_CAP) -> dict:
    """Store a {verdict, reason} recommendation for up to `cap` proposals that have none for
    their current rule and evidence. A failed or unparseable judge call is retried tomorrow."""
    c = conn.cursor(row_factory=dict_row)
    rows = [
        r
        for r in c.execute(
            "SELECT id, summary, evidence, recommendation FROM config_lane.config_proposals "
            "WHERE status='proposed' ORDER BY proposed_at DESC NULLS LAST, id"
        ).fetchall()
        if needs_recommendation(r)
    ]
    stats = {"pending": len(rows), "recommended": 0, "failed": 0}
    for row in rows[:cap]:
        ev = row["evidence"] or []
        prompt = _RECOMMEND_PROMPT.format(
            id=row["id"],
            rule=row["summary"],
            n=N._distinct_sessions(ev),
            quotes=_quotes(ev),
        )
        try:
            obj = SM._extract_json(N._judge_call(prompt) or "") or {}
        except Exception as e:
            print(f"  recommendation failed for config#{row['id']}: {e}")
            obj = {}
        verdict = str(obj.get("verdict") or "").strip().lower()
        reason = _plain(obj.get("reason") or "")
        if verdict not in ("accept", "reject") or not reason:
            stats["failed"] += 1
            continue
        rec = {
            "verdict": verdict,
            "reason": reason,
            "key": N._evidence_key(row["summary"], ev),
            "model": config.JUDGE_MODEL,
        }
        c.execute(
            "UPDATE config_lane.config_proposals SET recommendation = %s::jsonb || "
            "jsonb_build_object('at', now()) WHERE id=%s AND status='proposed'",
            (Json(rec), row["id"]),
        )
        conn.commit()
        stats["recommended"] += 1
    stats["deferred"] = max(0, len(rows) - cap)
    return stats


# ------------------------------------------------------------------ report
def _verdict(rec) -> str:
    if isinstance(rec, dict) and rec.get("verdict") in ("accept", "reject"):
        return rec["verdict"]
    return "no verdict yet"


def _pending(conn, run_started) -> list[dict]:
    c = conn.cursor(row_factory=dict_row)
    return c.execute(
        "SELECT id, summary, evidence, recommendation, "
        "(now()::date - proposed_at::date) AS days, "
        "COALESCE(proposed_at >= %s, false) AS new "
        "FROM config_lane.config_proposals WHERE status='proposed' "
        "ORDER BY proposed_at DESC NULLS LAST, id",
        (run_started,),
    ).fetchall()


def _pending_line(p: dict) -> str:
    rec = p.get("recommendation")
    sessions = N._distinct_sessions(p.get("evidence") or [])
    when = (
        "new tonight"
        if p["new"]
        else f"{p['days']}d pending"
        if p.get("days") is not None
        else "pending"
    )
    why = f": {_plain(rec['reason'])}" if isinstance(rec, dict) and rec.get("reason") else ""
    return (
        f"{_verdict(rec)} #{p['id']} {_plain(p['summary'], 110)} ({when}, {sessions} sessions){why}"
    )


def build_report(day: str, pending: list[dict], covered: list, expired: list) -> str:
    """The plain-text report, at most REPORT_CHARS. A header, then one line per item, each
    proposal starting with its verdict: tonight's new proposals, the ones dropped as already
    covered and the expiries, then the older pending ones; past the cap the oldest lines give
    way to a '+N more' line."""
    new = [p for p in pending if p["new"]]
    if new or covered or expired:
        parts = [f"{len(new)} new"] if new else []
        parts += [f"{len(covered)} already covered by your rules"] if covered else []
        parts += [f"{len(expired)} expired"] if expired else []
        head = f"Config lane {day}: {', '.join(parts)}, {len(pending)} pending."
    elif pending:
        head = f"Config lane weekly reminder {day}: {len(pending)} pending."
    else:
        head = f"Config lane weekly reminder {day}: nothing pending."
    body = [_pending_line(p) for p in new]
    for x in covered:
        by = f" ({_plain(x['covered_by'], 90)})" if x.get("covered_by") else ""
        body.append(f"dropped #{x['id']} {_plain(x['summary'], 110)}: already in your rules{by}.")
    for x in expired:
        rec = x.get("recommendation")
        was = f" (recommendation was {_verdict(rec)})" if isinstance(rec, dict) else ""
        body.append(
            f"expired #{x['id']} {_plain(x['summary'], 110)} after {x['days']}d unanswered{was}. "
            "It comes back if you correct it again."
        )
    body += [_pending_line(p) for p in pending if not p["new"]]
    tail = (
        ["Ask-only: nothing changes until you reply 'approve N', 'reject N' or 'tell me about N'."]
        if pending
        else []
    )

    def render(keep: int) -> str:
        more = [f"+{len(body) - keep} more: config_review.py list"] if keep < len(body) else []
        return "\n".join([head, *body[:keep], *more, *tail])

    keep = len(body)
    msg = render(keep)
    while len(msg) > REPORT_CHARS and keep > 0:
        keep -= 1
        msg = render(keep)
    return msg[:REPORT_CHARS]


def _cursor_config(conn) -> dict:
    row = conn.execute("SELECT config FROM skills_lane.skill_scan_cursor WHERE id=1").fetchone()
    return (row[0] if row else None) or {}


def _reminder_due(cursor_config: dict, now) -> bool:
    try:
        last = datetime.fromisoformat(str(cursor_config[SENT_KEY]))
    except (KeyError, TypeError, ValueError):
        return True
    return now - last >= timedelta(days=REMIND_DAYS)


def _merge_cursor_config(conn, patch: dict) -> None:
    conn.execute(
        "UPDATE skills_lane.skill_scan_cursor SET config = COALESCE(config,'{}'::jsonb) || %s::jsonb, "
        "updated_at=now() WHERE id=1",
        (json.dumps(patch),),
    )
    conn.commit()


def report(conn, run_started, changes: dict, no_discord: bool = False) -> dict:
    """Build the report when tonight changed the queue, or when REMIND_DAYS passed since the
    last one; store it in skill_scan_cursor.config['config_last_report'] = {at, text}; post
    it to Discord when SKILLS_DISCORD_WEBHOOK is set, else print it, and stamp
    config['config_report_sent_at']. changes: covered, expired (items from regate/expire)."""
    pending = _pending(conn, run_started)
    covered = changes.get("covered") or []
    expired = changes.get("expired") or []
    changed = bool(any(p["new"] for p in pending) or covered or expired)
    due = _reminder_due(_cursor_config(conn), run_started)
    stats = {"pending": len(pending), "sent": 0, "reminder": int(due and not changed)}
    if not changed and not due:
        print(f"(report: {len(pending)} pending, nothing changed tonight, reminder not due)")
        return stats
    msg = build_report(run_started.strftime("%m-%d"), pending, covered, expired)
    _merge_cursor_config(conn, {REPORT_KEY: {"at": run_started.isoformat(), "text": msg}})
    if not no_discord and config.DISCORD_WEBHOOK:
        try:
            req = urllib.request.Request(
                config.DISCORD_WEBHOOK,
                data=json.dumps({"content": msg}).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
            stats["sent"] = 1
        except Exception as e:
            print(f"(notifier failed: {e})\n{msg}")
            return stats
    else:
        print(msg)
        stats["sent"] = 1
    _merge_cursor_config(conn, {SENT_KEY: run_started.isoformat()})
    return stats
