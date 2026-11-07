# mypy: ignore-errors
"""dream->skills: the review queue looks after itself (nightly, from nightly.run_lane after
drafting; entries: auto_reject, expire, recommend, report).

Nothing here applies a change. Accept stays a person's act (schema 023's
grounded_required_to_apply); this module only keeps the queue short and says what to do:

  auto_reject   a proposed retune that can't be applied as drafted goes to 'rejected',
                reject_reason 'auto:<code>', with the normal 30-day cooldown but WITHOUT the
                grounded -3 a human reject writes, so new evidence or a human revive can
                still bring it back:
                  no_target       the target skill is not in the registry
                  retired         the target skill was retired
                  renamed         the draft renames the skill
                  scope_mismatch  an evidence session ran where the skill is not served
                                  (_evidence_scope against the registry scope). The
                                  out-of-scope evidence and the draft made from it are
                                  dropped, so a revival is judged on the rest. #76 (docx,
                                  project:transcribe_ai, evidence from jobs and neuron).
                  no_change       a widen whose draft leaves the description unchanged
                  covered         a widen whose every missing phrase the current
                                  description already has
                A stale draft is left alone: draft_pending re-drafts it against the new body
                the same night, and a cap-deferred re-draft is not a reason to reject.
  expire        a proposal unanswered for EXPIRE_DAYS goes to 'retired', reject_reason
                'expired'. A re-sighting reopens it (skill_ledger), so does
                `skill_review.py revive <id>`.
  recommend     the lane's judge (skill_measure.run_judge, the claude CLI) reads each
                drafted proposal once per draft and stores {verdict, reason} in
                recommendation; a changed draft is asked again.
  report        one plain-text Discord post when tonight changed the queue (a new
                proposal, an auto-reject, an expiry, anything the autopilot did), else a
                weekly reminder of what is still pending (or that nothing is). The
                autopilot's lines come first (auto-applied ones end with 'undo: revert N'),
                then each item starts with its verdict. Every report that is built is also
                stored in skill_scan_cursor.config.last_report = {at, text}: delivery is
                whoever reads it, not this module.

Everything that needs schema 062 (proposed_at, recommendation) checks skill_ledger's probe
first and skips with one log line until the migration is applied.
"""

from __future__ import annotations

import difflib
import json
import re
import urllib.request
from datetime import datetime, timedelta

from psycopg.rows import dict_row
from psycopg.types.json import Json

from . import config
from . import skill_doc as D
from . import skill_ledger as L
from . import skill_measure as SM
from .autoreview import ROLLBACK_DAYS
from .skill_draft import missing_phrasings, target_skill
from .struggle_arc import NEGATIVE_LIST

EXPIRE_DAYS = 21  # a proposal nobody answered in this long retires (revivable)
COOLDOWN_DAYS = 30  # auto-rejects keep the same cooldown a human reject gets
RECOMMEND_CAP = 5  # recommendation calls per night (each is one judge call)
REMIND_DAYS = 7  # a quiet queue still reports this often
REPORT_CHARS = 1500
DIFF_CHARS = 4000

_ROLLBACK_REASONS = {
    "dismissed": "the skill was dismissed after the change",
    "post_fire_deviation": "a session deviated from the skill after the change",
    "owner_edit": "the owner edited the change away",
    "owner_retired": "the owner retired the new skill",
    "skill_gone": "the skill is gone",
    "manual": "undone by hand",
}

_REASONS = {
    "no_target": "the skill is no longer in the registry",
    "retired": "the skill was retired",
    "renamed": "the draft renames the skill",
    "scope_mismatch": "its evidence came from sessions where the skill is not installed",
    "no_change": "the draft leaves the description as it is",
    "covered": "the description already covers every missing phrase",
}

_RECOMMEND_PROMPT = """You review one proposed change to a Claude Code skill (a SKILL.md procedure
document) that a nightly job mined from real sessions. Say whether the skill's owner should
accept it.

Accept when the change is grounded in the evidence, reusable beyond one task, and the draft
does what the evidence asks without inventing tools, commands or paths. Reject when it is
not. Never propose these:
{negative_list}

PROPOSAL #{id}: {what}
SUMMARY: {summary}
EVIDENCE FROM THE SESSIONS:
{evidence}

{change}

Output ONLY a JSON object, no prose:
{{"verdict": "accept" | "reject", "reason": "one short sentence in plain words, no markdown"}}"""


# ------------------------------------------------------------------ helpers
def _plain(text: str, cap: int = 240) -> str:
    """Report-safe text: one line, no markdown emphasis, no em/en dashes, capped."""
    t = " ".join(str(text or "").split())
    t = re.sub(r"\s*\N{EM DASH}\s*", ", ", t).replace("\N{EN DASH}", "-")
    t = t.replace("**", "").replace("`", "'")
    return t if len(t) <= cap else t[: cap - 3].rstrip() + "..."


def _what(row: dict) -> str:
    if row["kind"] == "derive":
        return f"new skill {row['name']}"
    if row["kind"] == "consolidate":
        return f"merge {row['name']}"
    return f"{target_skill(row)} {row.get('direction') or 'retune'}"


def _routes():
    """The accept path's own checks. Lazy: the dream image ships mcp_server/, and importing
    it at module load would drag the HTTP stack into every lane import."""
    from mcp_server import skill_sync_routes as R

    return R


def _evidence_lines(evidence, cap: int = 8) -> str:
    lines: list[str] = []
    for e in evidence or []:
        if not isinstance(e, dict) or e.get("class") == "grounded":
            continue
        bits = [e.get("signal") or "?"]
        for key in ("quote", "missing_phrasing", "phrasing", "why"):
            if e.get(key):
                bits.append(f"{key}: {str(e[key])[:200]}")
        line = "- " + "; ".join(bits)
        if line not in lines:
            lines.append(line)
    return "\n".join(lines[-cap:]) or "(no detail recorded)"


# ------------------------------------------------------------------ auto-reject
_PROPOSED_RETUNES = """
SELECT c.id, c.kind, c.name, c.direction, c.target_skills, c.status, c.evidence,
       c.proposal_body, c.base_body_hash
  FROM skills_lane.skill_gap_candidates c
 WHERE c.status = 'proposed' AND c.kind = 'retune'
 ORDER BY c.id"""


def _check(c, R, row: dict) -> tuple[str, str, list] | None:
    """(code, detail, kept_evidence) when `row` can't be applied as drafted, else None.
    kept_evidence is what stays on the row (only scope_mismatch drops any)."""
    body = row["proposal_body"] or ""
    plan = R._plan_apply(c, row, body, None)
    for issue in plan["issues"]:
        if issue["code"] in ("no_target", "retired", "renamed"):
            return issue["code"], issue["detail"], row["evidence"]
    if not plan["exists"]:
        return None
    ev = row["evidence"] or []
    out_of_scope = set()
    for sid in sorted({e["session_id"] for e in ev if isinstance(e, dict) and e.get("session_id")}):
        session_scope = R._evidence_scope(c, [{"session_id": sid}])
        if not SM.skill_served(plan["scope"], session_scope):
            out_of_scope.add(sid)
    if out_of_scope:
        kept = [e for e in ev if not (isinstance(e, dict) and e.get("session_id") in out_of_scope)]
        detail = (
            f"'{plan['skill']}' is {plan['scope']}, but {len(out_of_scope)} of its evidence "
            "sessions ran outside it"
        )
        return "scope_mismatch", detail, kept
    if row.get("direction") == "widen" and body and plan["registry_body"]:
        old = SM.skill_description(plan["registry_body"])
        if SM.skill_description(body) == old:
            return "no_change", "the drafted description equals the current one", ev
        phrases = missing_phrasings(ev)
        if phrases and all(SM.description_covers_phrasing(p, old) for p in phrases):
            return "covered", f"the description already has: {', '.join(phrases)}", ev
    return None


def auto_reject(conn) -> dict:
    """Reject the proposed retunes that can't be applied as drafted (codes in the module
    docstring). No grounded entry is written: this is the lane's call, not a person's."""
    R = _routes()
    c = conn.cursor(row_factory=dict_row)
    rows = c.execute(_PROPOSED_RETUNES).fetchall()
    items = []
    for row in rows:
        hit = _check(c, R, row)
        if hit is None:
            continue
        code, detail, kept = hit
        js, gs, jw, gw = L._rollup(kept)
        drop_draft = code == "scope_mismatch"
        c.execute(
            "UPDATE skills_lane.skill_gap_candidates SET status='rejected', reject_reason=%s, "
            "rejected_until=now() + make_interval(days => %s), evidence=%s::jsonb, "
            "judge_sessions=%s, grounded_sessions=%s, judge_weight=%s, grounded_weight=%s, "
            "proposal_body=CASE WHEN %s THEN NULL ELSE proposal_body END, "
            "base_body_hash=CASE WHEN %s THEN NULL ELSE base_body_hash END, updated_at=now() "
            "WHERE id=%s AND status='proposed'",
            (
                f"auto:{code}",
                COOLDOWN_DAYS,
                Json(kept),
                js,
                gs,
                jw,
                gw,
                drop_draft,
                drop_draft,
                row["id"],
            ),
        )
        items.append({**_brief(row), "code": code, "detail": detail})
        print(f"  auto-rejected cand#{row['id']} ({row['name']}): {code}: {detail}")
    conn.commit()
    return {"auto_rejected": len(items), "items": items}


def _brief(row: dict) -> dict:
    return {
        "id": row["id"],
        "kind": row["kind"],
        "name": row["name"],
        "direction": row.get("direction"),
        "target_skills": row.get("target_skills"),
    }


# ------------------------------------------------------------------ expiry
def expire(conn, days: int = EXPIRE_DAYS) -> dict:
    """Retire proposals unanswered for `days` (reject_reason 'expired'; revivable)."""
    if not L.queue_ready(conn):
        return {"expired": 0, "skipped": "schema 062 not applied", "items": []}
    c = conn.cursor(row_factory=dict_row)
    rows = c.execute(
        "UPDATE skills_lane.skill_gap_candidates SET status='retired', reject_reason='expired', "
        "updated_at=now() "
        "WHERE status='proposed' AND proposed_at < now() - make_interval(days => %s) "
        "RETURNING id, kind, name, direction, target_skills, recommendation, "
        "(now()::date - proposed_at::date) AS days",
        (days,),
    ).fetchall()
    conn.commit()
    items = [{**_brief(r), "days": r["days"], "recommendation": r["recommendation"]} for r in rows]
    return {"expired": len(items), "items": sorted(items, key=lambda i: i["id"])}


# ------------------------------------------------------------------ recommendation
_DRAFTED = """
SELECT c.id, c.kind, c.name, c.direction, c.target_skills, c.summary, c.evidence,
       c.proposal_body, c.recommendation, r.body AS registry_body
  FROM skills_lane.skill_gap_candidates c
  LEFT JOIN skills_lane.skill_registry r
         ON c.kind = 'retune' AND r.name = COALESCE(c.target_skills[1], c.name)
 WHERE c.status = 'proposed' AND c.proposal_body IS NOT NULL
 ORDER BY c.salience DESC NULLS LAST, c.score DESC, c.id"""


def _judge_call(prompt: str) -> str:
    """The lane's judge (claude CLI by default). A named seam so tests stub the LLM."""
    return SM.run_judge(prompt)


def _change_block(row: dict) -> str:
    body = row["proposal_body"] or ""
    if row["kind"] == "retune" and row.get("registry_body"):
        diff = "".join(
            difflib.unified_diff(
                row["registry_body"].splitlines(keepends=True),
                body.splitlines(keepends=True),
                fromfile="current/SKILL.md",
                tofile="proposed/SKILL.md",
            )
        )
        return "THE CHANGE (unified diff against the current skill):\n" + diff[:DIFF_CHARS]
    return "THE DRAFTED SKILL.md:\n" + body[:DIFF_CHARS]


def needs_recommendation(row: dict) -> bool:
    rec = row.get("recommendation")
    return not isinstance(rec, dict) or rec.get("draft_hash") != D.body_hash(row["proposal_body"])


def recommend(conn, cap: int = RECOMMEND_CAP) -> dict:
    """Store a {verdict, reason} recommendation for up to `cap` drafted proposals that have
    none for their current draft. A failed or unparseable judge call is retried tomorrow."""
    if not L.queue_ready(conn):
        return {"recommended": 0, "skipped": "schema 062 not applied"}
    c = conn.cursor(row_factory=dict_row)
    rows = [r for r in c.execute(_DRAFTED).fetchall() if needs_recommendation(r)]
    stats = {"pending": len(rows), "recommended": 0, "failed": 0}
    for row in rows[:cap]:
        prompt = _RECOMMEND_PROMPT.format(
            negative_list=NEGATIVE_LIST,
            id=row["id"],
            what=_what(row),
            summary=row.get("summary") or "(none)",
            evidence=_evidence_lines(row["evidence"]),
            change=_change_block(row),
        )
        try:
            obj = SM._extract_json(_judge_call(prompt) or "") or {}
        except Exception as e:
            print(f"  recommendation failed for cand#{row['id']}: {e}")
            obj = {}
        verdict = str(obj.get("verdict") or "").strip().lower()
        reason = _plain(obj.get("reason") or "")
        if verdict not in ("accept", "reject") or not reason:
            stats["failed"] += 1
            continue
        rec = {
            "verdict": verdict,
            "reason": reason,
            "draft_hash": D.body_hash(row["proposal_body"]),
            "model": config.JUDGE_MODEL,
        }
        c.execute(
            "UPDATE skills_lane.skill_gap_candidates SET recommendation = %s::jsonb || "
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


def _pending(conn, run_started, ready: bool) -> list[dict]:
    c = conn.cursor(row_factory=dict_row)
    extra = (
        ", recommendation, (now()::date - proposed_at::date) AS days, "
        "COALESCE(proposed_at >= %s, false) AS new"
        if ready
        else ", NULL::jsonb AS recommendation, NULL::int AS days, (created_at >= %s) AS new"
    )
    return c.execute(
        "SELECT id, kind, name, direction, target_skills" + extra + " "
        "FROM skills_lane.skill_gap_candidates WHERE status='proposed' "
        "ORDER BY salience DESC NULLS LAST, score DESC, id",
        (run_started,),
    ).fetchall()


def _pending_line(p: dict) -> str:
    rec = p.get("recommendation")
    if p["new"]:
        when = "new tonight"
    else:
        when = f"{p['days']}d pending" if p.get("days") is not None else "pending"
    why = f": {_plain(rec['reason'])}" if isinstance(rec, dict) and rec.get("reason") else ""
    return f"{_verdict(rec)} #{p['id']} {_what(p)} ({when}){why}"


def _auto_lines(auto: dict | None) -> tuple[list[str], list[str]]:
    """(header parts, body lines) for what the autopilot did tonight, applied ones first."""
    if not auto:
        return [], []
    parts: list[str] = []
    lines: list[str] = []
    applied = auto.get("applied") or []
    if applied:
        parts.append(f"{len(applied)} auto-applied")
    for a in applied:
        lines.append(
            f"auto-applied #{a['id']} {a['what']} (undo: revert {a['id']}): "
            f"{_plain(a.get('reason'), 160)}"
        )
    rolled = auto.get("rolled_back") or []
    if rolled:
        parts.append(f"{len(rolled)} rolled back")
    for r in rolled:
        lines.append(
            f"rolled back #{r['id']} {r['what']}: {_ROLLBACK_REASONS.get(r['reason'], r['reason'])}."
        )
    for x in auto.get("conflicts") or []:
        if x["what"] == "merged":
            lines.append(
                f"kept both #{x['id']} {x['skill']}: a local edit was pulled over, the new "
                "description was written onto it."
            )
        else:
            lines.append(
                f"restored a local edit of {x['skill']} (#{x['id']} is back in the queue)."
            )
    rejected = auto.get("rejected") or []
    if rejected:
        parts.append(f"{len(rejected)} reviewer-rejected")
    for r in rejected:
        lines.append(
            f"auto-rejected #{r['id']} {r['what']}: {_plain(r.get('reason'), 160)} "
            f"Reply 'revive {r['id']}' to bring it back."
        )
    br = auto.get("breaker")
    if br:
        lines.insert(
            0,
            f"autopilot paused: {br['rollbacks']} rollbacks in {ROLLBACK_DAYS} days "
            "({}); it only records what it would do until you run "
            "'skill_review.py autopilot resume'.".format(", ".join(br["skills"])),
        )
    would = auto.get("would") or {}
    if would and auto.get("effective_mode", auto.get("mode")) != "live":
        lines.append(
            f"autopilot (shadow): would have applied {would.get('would_apply', 0)}, "
            f"rejected {would.get('would_reject', 0)}."
        )
    return parts, lines


def build_report(
    day: str, pending: list[dict], rejected: list, expired: list, auto: dict | None = None
) -> str:
    """The plain-text report, at most REPORT_CHARS. A header, then one line per item: the
    autopilot's first (auto-applied, rolled back, paused), then each item starting with its
    verdict: tonight's new proposals, auto-rejects and expiries, then the older pending
    ones; past the cap the oldest lines give way to a '+N more' line."""
    new = [p for p in pending if p["new"]]
    auto_parts, auto_lines = _auto_lines(auto)
    if new or rejected or expired or auto_parts or auto_lines:
        parts = list(auto_parts)
        parts += [f"{len(new)} new"] if new else []
        parts += [f"{len(rejected)} auto-rejected"] if rejected else []
        parts += [f"{len(expired)} expired"] if expired else []
        head = (
            f"Skills lane {day}: {', '.join(parts) or 'autopilot notes'}, {len(pending)} pending."
        )
    elif pending:
        head = f"Skills lane weekly reminder {day}: {len(pending)} pending."
    else:
        head = f"Skills lane weekly reminder {day}: nothing pending."
    body = list(auto_lines)
    body += [_pending_line(p) for p in new]
    for r in rejected:
        body.append(f"auto-rejected #{r['id']} {_what(r)}: {_REASONS.get(r['code'], r['code'])}.")
    for x in expired:
        rec = x.get("recommendation")
        was = f" (recommendation was {_verdict(rec)})" if isinstance(rec, dict) else ""
        body.append(
            f"expired #{x['id']} {_what(x)} after {x['days']}d unanswered{was}. "
            f"Reply 'revive {x['id']}' to bring it back."
        )
    body += [_pending_line(p) for p in pending if not p["new"]]
    tail = ["Reply 'approve N', 'reject N' or 'tell me about N'."] if pending else []

    def render(keep: int) -> str:
        more = [f"+{len(body) - keep} more: skill_review.py list"] if keep < len(body) else []
        return "\n".join([head, *body[:keep], *more, *tail])

    keep = len(body)
    msg = render(keep)
    while len(msg) > REPORT_CHARS and keep > 0:
        keep -= 1
        msg = render(keep)
    return msg[:REPORT_CHARS]


def _reminder_due(cursor_config: dict, now) -> bool:
    try:
        last = datetime.fromisoformat(str(cursor_config["report_sent_at"]))
    except (KeyError, TypeError, ValueError):
        return True
    return now - last >= timedelta(days=REMIND_DAYS)


def report(
    conn, run_started, changes: dict, no_discord: bool = False, deliver: bool = True
) -> dict:
    """Build the report when tonight changed the queue (or the autopilot did something), or
    when REMIND_DAYS passed since the last one; store it in skill_scan_cursor.config
    ['last_report'] = {at, text}. deliver=True also posts it: Discord when
    SKILLS_DISCORD_WEBHOOK is set, stdout otherwise, and stamps config['report_sent_at']
    (update_cursor merges). deliver=False leaves delivery to the caller, which stamps it
    with mark_sent. changes: auto_rejected, expired, autopilot (autoreview.run's result)."""
    ready = L.queue_ready(conn)
    pending = _pending(conn, run_started, ready)
    rejected = changes.get("auto_rejected") or []
    expired = changes.get("expired") or []
    auto = changes.get("autopilot") or {}
    auto_changed = bool(
        any(
            auto.get(k)
            for k in ("applied", "rolled_back", "conflicts", "rejected", "breaker", "capped")
        )
    )
    changed = bool(any(p["new"] for p in pending) or rejected or expired or auto_changed)
    due = _reminder_due(L.get_cursor(conn).get("config") or {}, run_started)
    stats = {"pending": len(pending), "sent": 0, "reminder": int(due and not changed)}
    if not changed and not due:
        print(f"(report: {len(pending)} pending, nothing changed tonight, reminder not due)")
        return stats
    msg = build_report(run_started.strftime("%m-%d"), pending, rejected, expired, auto)
    stats["text"] = msg
    c = conn.cursor()
    c.execute(
        "UPDATE skills_lane.skill_scan_cursor SET config = COALESCE(config,'{}'::jsonb) || %s::jsonb, "
        "updated_at=now() WHERE id=1",
        (json.dumps({"last_report": {"at": run_started.isoformat(), "text": msg}}),),
    )
    conn.commit()
    if not deliver:
        return stats
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
    mark_sent(conn, run_started)
    return stats


def mark_sent(conn, run_started) -> None:
    """Stamp the report as delivered (the weekly reminder counts from here)."""
    c = conn.cursor()
    c.execute(
        "UPDATE skills_lane.skill_scan_cursor SET config = COALESCE(config,'{}'::jsonb) || %s::jsonb, "
        "updated_at=now() WHERE id=1",
        (json.dumps({"report_sent_at": run_started.isoformat()}),),
    )
    conn.commit()
