#!/usr/bin/env python3
"""One-time re-cluster of config_lane.config_proposals: merge the rows the old identity test
could not, union their evidence, and apply the gate.

Until the lane matched by embedding (dream/config/nightly.py), identity was a token-overlap
test that never matched across sessions, and a long session re-scanned whole each night added
near-duplicate rows. By 2026-11-06 the table held 208 rows, every one 'observe' with one
evidence entry. This script walks the open rows (observe / proposed) in id order, exactly the
way the nightly would have met them, with the lane's own identity rule (``pick_match``: cosine
to the row's founding embedding, a lower bar for a session the row already counts), and:

1. folds each row into the row it is the same rule as: the LOWEST id of a cluster survives
   (its summary and anchor stay), takes the union of the members' evidence (deduped by
   session, so distinct sessions are counted honestly), and the other rows are deleted;
2. runs the gate on every survivor: PROPOSE_SESSIONS distinct sessions, then the coverage
   check (one judge call asks whether the user's own global rule files already cover the
   rule; covered -> 'rejected' / 'auto:covered'), else 'proposed'. ``--no-llm`` skips the
   check and leaves the graduates at 'observe' for the nightly run to gate.

It needs NO schema change: with schema 065 applied it also writes the embedding (and
proposed_at, coverage); without it, the nightly run fills those once 065 is there (065's own
UPDATE stamps proposed_at for rows already proposed).

Writes (``--apply``): embeddings and judge calls happen first, outside any transaction. Then
one transaction takes a SHARE ROW EXCLUSIVE lock on config_proposals (the nightly run cannot
write meanwhile), re-reads the rows, re-derives the clusters from the same embeddings and
refuses if they differ from the plan (a row appeared since: re-run), copies every row it will
touch into

    config_lane.config_proposals_bak_<YYYYMMDD>     full rows, plus backed_up_at

and then updates survivors and deletes the folded rows. A re-run finds nothing to merge.
``--restore YYYYMMDD`` puts the backed-up rows back as they were (dry run unless --apply):
it deletes the rows the backup holds and re-inserts them from it, and leaves every other row
alone.

Usage:
    scripts/recluster_config_proposals.py [--db-url DSN] [--no-llm] [--plan-out FILE]
    scripts/recluster_config_proposals.py --apply [--no-llm] [--date YYYYMMDD] [--plan-out FILE]
    scripts/recluster_config_proposals.py --restore YYYYMMDD [--apply]

The DSN comes from --db-url, else $SYNAPSE_DB_URL; the dry run uses a READ ONLY connection.
Embedding needs VOYAGE_API_KEY (or the configured embed provider's env). ``--plan-out`` holds
ids and counts only (never rule text) and must live outside the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import psycopg  # noqa: E402
from psycopg.types.json import Json  # noqa: E402

from dream.config import identity_ready  # noqa: E402
from dream.config import nightly as N  # noqa: E402
from dream.skills import skill_ledger as L  # noqa: E402

_SELECT_OPEN = (
    "SELECT id, summary, evidence, status FROM config_lane.config_proposals "
    "WHERE file_key = %s AND status IN ('observe', 'proposed') ORDER BY id"
)


# ------------------------------------------------------------------- pure clustering
def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _unit(v: list[float]) -> list[float]:
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v


def cluster(rows: list[dict], vecs: dict[int, list[float]]) -> list[dict]:
    """Fold `rows` (id order; id, evidence, status) into clusters with the lane's identity
    rule. Each cluster: {id (the survivor = its lowest id), members (ids, survivor first),
    evidence (union), status ('proposed' if any member was)}. `vecs` maps id -> embedding."""
    out: list[dict] = []
    for r in rows:
        v = _unit(vecs[r["id"]])
        cands = [
            {
                "id": c["id"],
                "cos": _dot(v, c["unit"]),
                "evidence": c["evidence"],
                "status": "observe",
                "_c": c,
            }
            for c in out
        ]
        hit = N.pick_match(cands, N._sessions(r["evidence"]))
        if hit is None:
            out.append(
                {
                    "id": r["id"],
                    "unit": v,
                    "members": [r["id"]],
                    "evidence": list(r["evidence"]),
                    "status": r["status"],
                    "was": r["status"],
                }
            )
            continue
        c = hit[0]["_c"]
        c["members"].append(r["id"])
        c["evidence"] = N._union_evidence(c["evidence"], r["evidence"])
        if r["status"] == "proposed":
            c["status"] = "proposed"
    return out


def signature(clusters: list[dict]) -> list[list[int]]:
    return sorted(c["members"] for c in clusters)


def gate(c: dict) -> bool:
    """A cluster that still needs the gate: observe with enough distinct sessions."""
    return c["status"] == "observe" and N._distinct_sessions(c["evidence"]) >= N.PROPOSE_SESSIONS


# ------------------------------------------------------------------- the judge, on a workstation
_STRIP_ENV = {
    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
    "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
}


def stripped_judge(prompt: str) -> str:
    """The coverage judge for a run on a workstation: the lane's model through a bare
    `claude -p` (no tools, settings, MCP, hooks or memory files, run from an empty directory),
    so the workstation's own CLAUDE.md can neither colour the verdict nor ride along as
    ~16k tokens a call. The nightly run in the dream container uses the lane's default judge."""
    import subprocess
    import tempfile

    from dream.skills import config as skills_config

    with tempfile.TemporaryDirectory(prefix="config-recluster-") as scratch:
        r = subprocess.run(
            [
                "claude",
                "-p",
                prompt,
                "--model",
                skills_config.JUDGE_MODEL,
                "--no-session-persistence",
                "--setting-sources",
                "",
                "--strict-mcp-config",
                "--tools",
                "",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, **_STRIP_ENV},
            cwd=scratch,
        )
    return r.stdout or ""


# ------------------------------------------------------------------- planning
def plan(conn, use_llm: bool) -> tuple[dict, list[dict], dict[int, list[float]]]:
    """(plan, clusters, vecs) from the live rows. Embeds every open row; with `use_llm` runs
    the coverage check on each graduate."""
    rows = [
        {"id": i, "summary": s, "evidence": e or [], "status": st}
        for i, s, e, st in conn.execute(_SELECT_OPEN, (N.LEARNED_FILE,)).fetchall()
    ]
    texts = [r["summary"] for r in rows]
    embedded = N.embed(texts) if texts else []
    if len(embedded) != len(rows):
        raise RuntimeError(f"embedder returned {len(embedded)} vectors for {len(rows)} rows")
    vecs = {r["id"]: v for r, v in zip(rows, embedded, strict=True)}
    clusters = cluster(rows, vecs)
    digest, rhash = N.registry_digest(conn.cursor())
    out: list[dict] = []
    for c in clusters:
        n = N._distinct_sessions(c["evidence"])
        item = {
            "survivor": c["id"],
            "members": c["members"],
            "sessions": n,
            "status_before": c["was"],
            "outcome": c["status"],
        }
        c["coverage"] = None
        if gate(c) and use_llm:
            verdict = N.coverage_check(
                next(r["summary"] for r in rows if r["id"] == c["id"]), digest, rhash
            )
            c["coverage"] = verdict
            if verdict is None:
                item["outcome"] = "observe (coverage check failed: the nightly run retries)"
            elif verdict["covered"]:
                c["status"] = "rejected"
                item["outcome"] = "rejected:auto:covered"
            else:
                c["status"] = "proposed"
                item["outcome"] = "proposed"
        elif gate(c):
            item["outcome"] = "observe (graduate; --no-llm leaves the gate to the nightly run)"
        out.append(item)
    merged = sum(len(c["members"]) - 1 for c in clusters)
    graduates = [i for i in out if i["sessions"] >= N.PROPOSE_SESSIONS]
    summary = {
        "rows_before": len(rows),
        "clusters": len(clusters),
        "rows_merged_away": merged,
        "clusters_with_2plus_sessions": len(graduates),
        "proposed": sum(1 for i in out if i["outcome"] == "proposed"),
        "dropped_as_covered": sum(1 for i in out if i["outcome"] == "rejected:auto:covered"),
        "left_at_observe_to_regate": sum(
            1 for i in graduates if i["outcome"].startswith("observe")
        ),
        "thresholds": {
            "cross_session": N.IDENTITY_COSINE,
            "same_session": N.IDENTITY_COSINE_SAME_SESSION,
        },
    }
    return (
        {
            "summary": summary,
            "clusters": [i for i in out if len(i["members"]) > 1 or i in graduates],
        },
        clusters,
        vecs,
    )


def _check_path(path: str | None) -> None:
    if path and _REPO in Path(path).resolve().parents:
        sys.exit(f"--plan-out must live outside the repository ({_REPO}); got {path}")


def _print_plan(p: dict) -> None:
    s = p["summary"]
    print("re-cluster plan (ids and counts only):")
    for k, v in s.items():
        print(f"  {k}: {v}")
    for c in p["clusters"]:
        print(
            f"  cluster {c['survivor']} <- {c['members'][1:] or '-'}  "
            f"sessions={c['sessions']}  {c['outcome']}"
        )


# ------------------------------------------------------------------- apply / restore
def _backup_name(day: str) -> str:
    if not re.fullmatch(r"\d{8}", day):
        sys.exit(f"--date/--restore take YYYYMMDD, got {day!r}")
    return f"config_lane.config_proposals_bak_{day}"


def apply(dsn: str, use_llm: bool, day: str, plan_out: str | None) -> int:
    bak = _backup_name(day)
    with psycopg.connect(dsn) as conn:
        conn.read_only = True
        p, clusters, vecs = plan(conn, use_llm)
        conn.rollback()
    _print_plan(p)
    if plan_out:
        Path(plan_out).write_text(json.dumps(p, indent=1))
        print(f"plan written to {plan_out}")
    work = [c for c in clusters if len(c["members"]) > 1 or c["status"] != c["was"]]
    if not work:
        print("nothing to do.")
        return 0
    with psycopg.connect(dsn) as conn:
        cur = conn.cursor()
        cur.execute("LOCK TABLE config_lane.config_proposals IN SHARE ROW EXCLUSIVE MODE")
        rows = [
            {"id": i, "summary": s, "evidence": e or [], "status": st}
            for i, s, e, st in cur.execute(_SELECT_OPEN, (N.LEARNED_FILE,)).fetchall()
        ]
        if any(r["id"] not in vecs for r in rows):
            sys.exit("a row appeared since the plan was made; nothing written. Re-run.")
        again = cluster(rows, vecs)
        if signature(again) != signature(clusters):
            sys.exit("the rows changed since the plan was made; nothing written. Re-run.")
        ready = identity_ready(conn)
        touched = sorted({m for c in work for m in c["members"]})
        cur.execute(f"SELECT to_regclass('{bak}')")
        if cur.fetchone()[0] is not None:
            sys.exit(f"{bak} already exists: pass another --date, or --restore {day} first")
        cur.execute(
            f"CREATE TABLE {bak} AS SELECT *, now() AS backed_up_at "
            "FROM config_lane.config_proposals WHERE id = ANY(%s)",
            (touched,),
        )
        merged_away = 0
        for c in work:
            sets = [
                "evidence = %s::jsonb",
                "status = %s",
                "reject_reason = %s",
                "updated_at = now()",
            ]
            args: list = [
                json.dumps(c["evidence"]),
                c["status"],
                N.COVERED if c["status"] == "rejected" else None,
            ]
            if ready:
                sets.append("embedding = %s::halfvec")
                args.append(L.vec_literal(vecs[c["id"]]))
                if c["status"] == "proposed":
                    sets.append("proposed_at = COALESCE(proposed_at, now())")
                if c["coverage"]:
                    sets.append("coverage = %s")
                    args.append(Json(c["coverage"]))
            cur.execute(
                f"UPDATE config_lane.config_proposals SET {', '.join(sets)} WHERE id = %s",
                [*args, c["id"]],
            )
            gone = c["members"][1:]
            if gone:
                cur.execute("DELETE FROM config_lane.config_proposals WHERE id = ANY(%s)", (gone,))
                merged_away += cur.rowcount
        conn.commit()
    print(
        f"applied: {len(work)} cluster(s) updated, {merged_away} row(s) folded away, "
        f"backup {bak} ({len(touched)} rows). Undo: --restore {day} --apply"
    )
    return 0


def restore(dsn: str, day: str, do_apply: bool) -> int:
    bak = _backup_name(day)
    with psycopg.connect(dsn) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT to_regclass('{bak}')")
        if cur.fetchone()[0] is None:
            sys.exit(f"{bak} does not exist")
        cols = [
            r[0]
            for r in cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='config_lane' AND table_name=%s AND column_name <> 'backed_up_at' "
                "ORDER BY ordinal_position",
                (bak.split(".")[1],),
            ).fetchall()
        ]
        cur.execute(f"SELECT id FROM {bak} ORDER BY id")
        ids = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT count(*) FROM config_lane.config_proposals WHERE id = ANY(%s)", (ids,))
        present = cur.fetchone()[0]
        print(
            f"restore {day}: {len(ids)} backed-up row(s) ({present} still present, "
            f"{len(ids) - present} folded away); other rows are left alone"
        )
        if not do_apply:
            conn.rollback()
            print("dry run: nothing written. Add --apply to restore.")
            return 0
        cur.execute("LOCK TABLE config_lane.config_proposals IN SHARE ROW EXCLUSIVE MODE")
        cur.execute("DELETE FROM config_lane.config_proposals WHERE id = ANY(%s)", (ids,))
        col_list = ", ".join(cols)
        cur.execute(
            f"INSERT INTO config_lane.config_proposals ({col_list}) SELECT {col_list} FROM {bak}"
        )
        conn.commit()
        print(f"restored {len(ids)} row(s) from {bak} (the backup table is kept).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--db-url", default=os.environ.get("SYNAPSE_DB_URL"))
    ap.add_argument("--apply", action="store_true", help="write (default is a read-only dry run)")
    ap.add_argument("--no-llm", action="store_true", help="skip the coverage check")
    ap.add_argument("--date", default=date.today().strftime("%Y%m%d"))
    ap.add_argument("--plan-out", help="write the plan (ids and counts) here; outside the repo")
    ap.add_argument("--restore", metavar="YYYYMMDD")
    a = ap.parse_args()
    if not a.db_url:
        sys.exit("no DSN: pass --db-url or set SYNAPSE_DB_URL")
    _check_path(a.plan_out)
    N._judge_call = stripped_judge
    if a.restore:
        return restore(a.db_url, a.restore, a.apply)
    if a.apply:
        return apply(a.db_url, not a.no_llm, a.date, a.plan_out)
    with psycopg.connect(a.db_url) as conn:
        conn.read_only = True
        p, _, _ = plan(conn, not a.no_llm)
        conn.rollback()
    _print_plan(p)
    if a.plan_out:
        Path(a.plan_out).write_text(json.dumps(p, indent=1))
        print(f"plan written to {a.plan_out}")
    print("dry run: nothing written. Add --apply to write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
