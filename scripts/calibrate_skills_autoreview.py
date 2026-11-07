#!/usr/bin/env python3
# mypy: ignore-errors
"""Calibrate the dream->skills autopilot (dream/skills/autoreview.py) against the proposals
a person already decided. Read-only: it reads a JSON dump, never the database.

    psql ... -At -f dump.sql > dump.json          # the SELECT in DUMP_SQL below
    SYNAPSE_LLM_PROVIDER=openai SYNAPSE_LLM_MODEL=<verifier> SYNAPSE_LLM_API_KEY=... \\
      uv run python scripts/calibrate_skills_autoreview.py dump.json --out DIR

For every candidate a person promoted or rejected (and every one still proposed, with no
human verdict to compare), it rebuilds the proposal as it stood when it was decided:

  * evidence: entries dated after the decision are dropped, and so are the person's own
    accept/reject/revive acts (the autopilot never decides a row a person touched; at
    decision time there was no act yet);
  * the target skill's body: the registry version in force at the decision time
    (skill_history rows are the bodies each write replaced). A promoted row's decision
    time is the first write to its target after the row was created when that is earlier
    than its last update (a promoted row keeps merging evidence afterwards);
  * the catalog: the skills that existed then, minus the one this row created.

Then it runs the production preflight (#229's auto-reject checks: no_target, retired,
scope_mismatch, covered), the production tiering, reviewer and verifier (both evidence
orders), and decide(). Most decided rows predate stored drafts, so the change is judged
from the stored patch / summary / missing phrasings (autoreview.change_block's no-draft
form); the nightly stage judges a drafted body, i.e. more information, not less.

Writes DIR/results.jsonl (one row per candidate; LLM answers are cached there, so a rerun
resumes) and prints the summary table. The headline: proposals a person REJECTED that the
autopilot would have APPLIED (the go-live gate requires 0).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dream.skills import autoreview as A
from dream.skills import skill_measure as SM
from dream.skills.skill_draft import missing_phrasings

DUMP_SQL = """
SELECT json_build_object(
 'candidates', (SELECT json_agg(row_to_json(c) ORDER BY c.id) FROM (
   SELECT id, kind, name, direction, target_skills, status, reject_reason, summary, signature,
          trigger_phrasings, evidence, proposal_body, base_body_hash, proposed_patch,
          salience, source_detector, score, created_at, updated_at
     FROM skills_lane.skill_gap_candidates
    WHERE status IN ('promoted','accepted','rejected','proposed')) c),
 'registry', (SELECT json_agg(row_to_json(r)) FROM (
   SELECT name, scope, description, body, pinned, status, first_seen
     FROM skills_lane.skill_registry) r),
 'history', (SELECT json_agg(row_to_json(h) ORDER BY h.id) FROM (
   SELECT id, name, scope, body, op, recorded_at
     FROM skills_lane.skill_history WHERE op <> 'disk_overwrite') h),
 'sessions', (SELECT json_agg(row_to_json(s)) FROM (
   SELECT e.session_id, min(e.created_at)::date AS day,
          array_agg(DISTINCT e.project) AS projects,
          array_agg(DISTINCT e.metadata->>'cwd') AS cwds
     FROM episodes e
    WHERE e.session_id IN (
      SELECT DISTINCT x->>'session_id'
        FROM skills_lane.skill_gap_candidates c, jsonb_array_elements(c.evidence) x
       WHERE c.status IN ('promoted','accepted','rejected','proposed')
         AND x->>'session_id' IS NOT NULL)
    GROUP BY e.session_id) s));
"""


def ts(v) -> datetime:
    return datetime.fromisoformat(str(v))


class FakeCursor:
    """Answers skill_sync_routes._evidence_scope's one query from the dump."""

    def __init__(self, sessions: dict):
        self.sessions = sessions
        self._rows: list = []

    def execute(self, _sql, params):
        out = []
        for sid in params[0]:
            s = self.sessions.get(sid)
            if not s:
                continue
            for p in s.get("projects") or [None]:
                for c in s.get("cwds") or [None]:
                    out.append({"session_id": sid, "project": p, "cwd": c})
        self._rows = out
        return self

    def fetchall(self):
        return self._rows


class World:
    def __init__(self, dump: dict):
        self.registry = {r["name"]: r for r in dump["registry"] or []}
        self.history: dict[str, list] = {}
        for h in dump["history"] or []:
            self.history.setdefault(h["name"], []).append(h)
        self.sessions = {s["session_id"]: s for s in dump["sessions"] or []}

    def version_at(self, name: str, t: datetime) -> dict | None:
        """The registry row of `name` as it was at `t` (body, scope, status), or None when
        the skill did not exist yet."""
        reg = self.registry.get(name)
        if reg is None or ts(reg["first_seen"]) > t:
            return None
        for h in self.history.get(name, []):
            if h["op"] in ("superseded", "rescoped", "deleted") and ts(h["recorded_at"]) >= t:
                # today's scope: labels were corrected since (home-folder labels became
                # global), and the preflight reads the registry scope as it is now
                return {"body": h["body"] or "", "scope": reg["scope"], "status": "active"}
        return {"body": reg["body"] or "", "scope": reg["scope"], "status": reg["status"]}

    def catalog_at(self, t: datetime, exclude: set[str]) -> list[tuple[str, str]]:
        out = []
        for name, reg in self.registry.items():
            if name in exclude or ts(reg["first_seen"]) >= t:
                continue
            v = self.version_at(name, t)
            if v and v["status"] == "active":
                out.append((name, SM.skill_description(v["body"]) or reg["description"] or ""))
        return out


def decision_time(row: dict, world: World) -> datetime:
    t = ts(row["updated_at"])
    if row["status"] != "promoted":
        return t
    if row["kind"] == "derive":
        reg = world.registry.get(row["name"])
        return min(t, ts(reg["first_seen"])) if reg else t
    target = (row["target_skills"] or [row["name"]])[0]
    later = [
        ts(h["recorded_at"])
        for h in world.history.get(target, [])
        if h["op"] == "superseded" and ts(h["recorded_at"]) > ts(row["created_at"])
    ]
    return min([t, *later])


def evidence_at(evidence, t: datetime) -> list[dict]:
    day = t.date().isoformat()
    out = []
    for e in evidence or []:
        if not isinstance(e, dict):
            continue
        if e.get("class") == "grounded" and not e.get("session_id"):
            continue  # a person's act: did not exist before the decision
        when = e.get("scan_night") or e.get("date")
        if when and str(when) > day:
            continue
        out.append(e)
    return out


def preflight(row: dict, world: World, t: datetime) -> tuple[str | None, str, dict | None]:
    """(auto_reject_code or None, tier, target version) by the production rules."""
    from mcp_server import skill_sync_routes as R

    if row["kind"] == "derive":
        tier, _ = A.tier_for({**row, "proposal_body": None}, {}, False)
        # no draft to check: the tier is the recurrence gate's answer alone
        return None, ("new_skill" if tier != "derive_no_recurrence" else tier), None
    target = (row["target_skills"] or [row["name"]])[0]
    pinned = bool(world.registry.get(target, {}).get("pinned"))
    # a widen draft changes only the description (skill_draft.draft_widen guarantees it);
    # extend/fix/narrow drafts are body revisions
    tier = "pinned" if pinned else "description" if row.get("direction") == "widen" else "body_edit"
    v = world.version_at(target, t)
    if v is None:
        return "no_target", tier, None
    if v["status"] == "retired":
        return "retired", tier, v
    cur = FakeCursor(world.sessions)
    for sid in sorted({e["session_id"] for e in row["evidence"] if e.get("session_id")}):
        if not SM.skill_served(v["scope"], R._evidence_scope(cur, [{"session_id": sid}])):
            return "scope_mismatch", tier, v
    if row.get("direction") == "widen":
        old = SM.skill_description(v["body"])
        phrases = missing_phrasings(row["evidence"])
        if phrases and all(SM.description_covers_phrasing(p, old) for p in phrases):
            return "covered", tier, v
    return None, tier, v


def run(dump_path: str, out_dir: str, only: set[int] | None) -> None:
    dump = json.loads(Path(dump_path).read_text())
    world = World(dump)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.jsonl"
    done = {}
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    for row in dump["candidates"]:
        if only and row["id"] not in only:
            continue
        if row["id"] in done and done[row["id"]].get("judged") is not None:
            continue
        human = {"promoted": "accept", "accepted": "accept", "rejected": "reject"}.get(
            row["status"]
        )
        t = decision_time(row, world)
        r = {**row, "evidence": evidence_at(row["evidence"], t), "proposal_body": None}
        code, tier, v = preflight(r, world, t)
        exclude = (
            {row["name"]} if row["kind"] == "derive" and row["status"] == "promoted" else set()
        )
        catalog = world.catalog_at(t, exclude)
        rec = {
            "id": row["id"],
            "kind": row["kind"],
            "direction": row.get("direction"),
            "human": human,
            "decided_at": t.isoformat(),
            "preflight": code,
            "tier": tier,
            "catalog_size": len(catalog),
        }
        if code:
            rec.update(final="reject", judged=False)
        else:
            j = A.judge(r, v["body"] if v else None, catalog)
            rec.update(
                reviewer=j["reviewer"],
                verifier=j["verifier"],
                final=A.decide(
                    tier,
                    (j["reviewer"] or {}).get("verdict"),
                    j["verifier"]["result"],
                ),
                judged=True,
            )
        print(
            f"#{rec['id']} {rec['kind']}/{rec['direction'] or '-'} human={human} tier={tier} "
            f"pre={code} rev={(rec.get('reviewer') or {}).get('verdict')} "
            f"ver={(rec.get('verifier') or {}).get('result')} -> {rec['final']}",
            flush=True,
        )
        done[row["id"]] = rec
        res_path.write_text("".join(json.dumps(x, default=str) + "\n" for x in done.values()))
    summarize(list(done.values()))


def summarize(rows: list[dict]) -> None:
    decided = [r for r in rows if r["human"]]
    applied_rejected = [r for r in decided if r["final"] == "apply" and r["human"] == "reject"]
    rejected_accepted = [r for r in decided if r["final"] == "reject" and r["human"] == "accept"]
    agree = [
        r
        for r in decided
        if (r["final"], r["human"]) in (("apply", "accept"), ("reject", "reject"))
    ]
    asks = [r for r in decided if r["final"] == "ask"]
    rev_agree = [r for r in decided if (r.get("reviewer") or {}).get("verdict") == r["human"]]
    judged = [r for r in decided if r.get("judged")]
    print("\n== calibration ==")
    print(
        f"decided rows: {len(decided)}  (accepted {sum(r['human'] == 'accept' for r in decided)},"
        f" rejected {sum(r['human'] == 'reject' for r in decided)})"
    )
    print(
        f"HEADLINE rejected by a person but auto-APPLIED: {len(applied_rejected)} "
        f"{[r['id'] for r in applied_rejected]}"
    )
    print(
        f"accepted by a person but auto-REJECTED: {len(rejected_accepted)} "
        f"{[r['id'] for r in rejected_accepted]}"
    )
    print(f"autopilot acted and agreed: {len(agree)}/{len(decided)}; asked: {len(asks)}")
    print(f"reviewer verdict = person's: {len(rev_agree)}/{len(judged)} judged")
    print(
        f"  of those, by #229's deterministic preflight: "
        f"{[r['id'] for r in rejected_accepted if r.get('preflight')]}"
    )
    by = {}
    for r in decided:
        via = f"pre:{r['preflight']}" if r.get("preflight") else "llm"
        key = (r["tier"], r["human"], r["final"], via)
        by[key] = by.get(key, 0) + 1
    for k in sorted(by):
        print(f"  tier={k[0]:<22} human={k[1]:<6} -> {k[2]:<6} via {k[3]:<20} {by[k]}")
    pending = [r for r in rows if not r["human"]]
    for r in pending:
        print(f"  (undecided #{r['id']}: {r['tier']} -> {r['final']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="comma-separated candidate ids")
    ap.add_argument("--summary", action="store_true", help="summarize results.jsonl only")
    ap.add_argument(
        "--model", help="reviewer model (default: SKILLS_AUTOREVIEW_MODEL or the lane's)"
    )
    a = ap.parse_args()
    if a.summary:
        rows = [json.loads(x) for x in (Path(a.out) / "results.jsonl").read_text().splitlines()]
        summarize(rows)
        return
    only = {int(x) for x in a.only.split(",")} if a.only else None
    if a.model:
        os.environ["SKILLS_AUTOREVIEW_MODEL"] = a.model
    run(a.dump, a.out, only)


if __name__ == "__main__":
    main()
