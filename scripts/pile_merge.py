#!/usr/bin/env python3
"""Preview, check and undo the one-memory merge (schema/067_kg_one_memory.sql).

067 folds the 'personal' knowledge-graph pile into 'technical' and stamps itself. This
script is its companion; it never prints fact text or entity names, only counts and uuids.

    --preview   What 067 would do right now, as counts: the twin map (losers, survivors,
                regrouped, and the folded uuids a writer minted again), the facts it
                re-points, the self-loops and missing endpoints it would leave, the twins
                whose supertypes differ (row 10b: by pair, and how many involve 'Person'),
                the stragglers it would stamp, and how many entries of the optional
                exclusion table exist, are personal twins, still fold (owner flag) or
                outrank their twin. It runs the exact SQL text 067 runs (the regions marked
                pm_ghost, pm_map, pm_fx and pm_counts in that file) on a READ ONLY
                connection, so the numbers are the ones an apply would record in
                kg_pile_merge_067_runs.counts (a run that finds only notes or preferences
                left records just their counts).
    --health [--just-merged]
                After an apply: exits non-zero on any failed check (see ``health``). With
                --just-merged, for the check right after the merge, two more must hold: no
                survivor has left 'technical', and the fact count equals the one before the
                merge. Both are exact only while nothing else writes to the graph (the
                pollers are stopped, no dashboard fact or episode delete has run), so leave
                the flag off once they have; they are then reported and never fail the run.
    --undo [--run N|all] [--apply]
                Reverse the recorded runs from their before images. Revert the code first: a
                build that requires 067 exits on a database stamped 066, which is what a
                complete undo stamps. It refuses, changing nothing, when the stamp is above
                067: a later migration is applied, and stamping 066 would put the version
                below objects that migration made. A dry run (no --apply) does everything a
                real run does, including taking SHARE ROW EXCLUSIVE on kg_entities and
                kg_relationships, and rolls back at the end: writers wait for it as they
                would for the real undo, so run it with the pollers stopped too. Rows written
                since the merge are never overwritten: a row that no longer has the value the
                run gave it is reported as drift and left alone. A regrouped entity is moved
                back even when edges the merge never touched point at it now; those entities
                are reported as drift. --run all (the default) undoes every open run in one
                transaction, latest first, each by its own twin map and images; a twin that
                two runs folded comes back from the earlier run's image. --run N undoes the
                latest open run only, and undoing runs one at a time reports a twin that a
                later run just gave back as drift. Leaves the owner-only indexes and the
                column defaults (both are additive).
    --exclude FILE [--apply]
                Keep some personal twins from being folded: create public.kg_pile_merge_067_
                exclude IF NOT EXISTS and insert the (owner_id, uuid) pairs of FILE, one
                "owner_id<TAB>uuid" per line (blank lines and '#' comments skipped). 067 reads the
                table only if it exists and regroups a listed entity as it is. Dry run unless
                --apply. This keeps existing edges apart, with two limits that are reported by
                uuid, here and in --preview: a flagged entity (is_self) still folds into the
                owner's flagged technical one, listed or not, and a later mention of the same
                name resolves to the first row by SURVIVOR_ORDER, which is the listed row when
                its id is lower than its twin's.

The DSN comes from --db-url, else $SYNAPSE_DB_URL.

Restores that write group_id, src_uuid or tgt_uuid from an older backup must not run after
the merge unless it is undone first: repair_numbered_entity_magnets.py --restore 20261106 and
a restore from kg_fact_tzfix_20261011.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from ingestion.kg_pg_read import SURVIVOR_ORDER  # noqa: E402

SCHEMA_DIR = _REPO / "schema"
SQL_066 = SCHEMA_DIR / "066_kg_fact_audience.sql"
SQL_067 = SCHEMA_DIR / "067_kg_one_memory.sql"

LEDGER = "kg_pile_merge_067"
EXCLUDE_TABLE = f"public.{LEDGER}_exclude"
RETIRED_RESTORES = (
    "repair_numbered_entity_magnets.py --restore 20261106",
    "restore from kg_fact_tzfix_20261011",
)
INDEXES = ("kg_rel_pair_any", "kg_rel_tgt_any", "kg_entities_norm_any")
KG_TABLES = ("kg_entities", "kg_relationships", "notes", "preferences")
#: The two checks ``--health --just-merged`` adds (otherwise these figures are only reported).
_JM_SURVIVORS = "no survivor has left technical"
_JM_FACTS = "fact count equals the count before the merge"
#: Uuids per drift list in the printed output; the counts are always printed in full.
_LIST_CAP = 200
#: What an undo counts and what it reports as drift, listed so a run that finds nothing
#: still prints every key.
_UNDO_COUNTS = (
    "flags moved back",
    "folded entities re-created",
    "entities regrouped back",
    "survivors restored",
    "facts restored",
    "notes restored",
    "preferences restored",
)
_UNDO_DRIFT = (
    "folded entity uuid already exists",
    "regrouped entity gone or changed group",
    "regrouped entity moved back has edges the merge never touched",
    "survivor edited since the merge",
    "fact changed since the merge, or gone",
    "notes gone or changed group (ids)",
    "preferences gone or changed group (ids)",
)

# ------------------------------------------------------------- shared SQL text


def _region(text: str, name: str) -> str:
    """The text between ``-- pm_<name>:begin`` and ``-- pm_<name>:end`` lines."""
    m = re.search(
        rf"^[ \t]*-- pm_{name}:begin[ \t]*\n(.*?)^[ \t]*-- pm_{name}:end[ \t]*$", text, re.S | re.M
    )
    if m is None:
        raise RuntimeError(f"{SQL_067.name} has no pm_{name} region")
    return m.group(1).strip()


@functools.cache
def shared_sql() -> dict[str, str]:
    """The ghost, map, fact and count texts, read from 067 itself so --preview and the apply
    cannot differ. Fails loudly if 067's survivor pick is not ``ingestion.kg_pg_read.SURVIVOR_ORDER``."""
    text = SQL_067.read_text()
    out = {name: _region(text, name) for name in ("ghost", "map", "fx", "counts")}
    if f"ORDER BY owner_id, k, {SURVIVOR_ORDER}" not in out["map"]:
        raise RuntimeError(
            f"{SQL_067.name} does not pick survivors by SURVIVOR_ORDER ({SURVIVOR_ORDER!r})"
        )
    return out


def preview_sql(*, with_exclude_table: bool, with_ledger: bool = False) -> str:
    """One statement returning the counts jsonb, from CTEs named as 067's temp tables."""
    parts = shared_sql()
    excl = (
        f"SELECT owner_id, uuid FROM {EXCLUDE_TABLE}"
        if with_exclude_table
        else "SELECT NULL::text AS owner_id, NULL::text AS uuid WHERE false"
    )
    ghost = (
        parts["ghost"]
        if with_ledger
        else "SELECT NULL::text AS owner_id, NULL::text AS uuid, NULL::text AS survivor_uuid WHERE false"
    )
    return (
        f"WITH _pm_ghost AS MATERIALIZED ({ghost}),\n"
        f"     _pm_excl AS MATERIALIZED ({excl}),\n"
        f"     _pm_map AS MATERIALIZED ({parts['map']}),\n"
        f"     _pm_fx AS MATERIALIZED ({parts['fx']})\n"
        f"SELECT {parts['counts']}"
    )


# ------------------------------------------------------------------ connection


def _connect(url: str, *, read_only: bool) -> Any:
    import psycopg

    conn = psycopg.connect(url, autocommit=True)
    if read_only:
        conn.execute("SET default_transaction_read_only = on")
    return conn


def _exists(conn: Any, name: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s) IS NOT NULL", (name,)).fetchone()
    return bool(row and row[0])


def _capped(uuids: list[str]) -> dict[str, Any]:
    return {"count": len(uuids), "uuids": sorted(uuids)[:_LIST_CAP]}


# --------------------------------------------------------------------- preview


def preview(conn: Any) -> dict[str, Any]:
    counts = conn.execute(
        preview_sql(
            with_exclude_table=_exists(conn, EXCLUDE_TABLE),
            with_ledger=_exists(conn, f"public.{LEDGER}_entity_map")
            and _exists(conn, f"public.{LEDGER}_runs"),
        )
    ).fetchone()[0]
    gates = {
        "self_loops_created is 0": counts["self_loops_created"] == 0,
        "missing_endpoints_after is 0": counts["missing_endpoints_after"] == 0,
        "label_disagreements is 0": counts["label_disagreements"] == 0,
        "exclude_owner_flag is 0 (every listed entity is kept apart)": counts["exclude_owner_flag"]
        == 0,
        "exclude_outranks_twin is 0 (no listed entity becomes first for its name)": counts[
            "exclude_outranks_twin"
        ]
        == 0,
    }
    return {"counts": counts, "gates": gates}


# ---------------------------------------------------------------------- health


def health(conn: Any, *, just_merged: bool = False) -> dict[str, Any]:
    """Every check, with its numbers. ``ok`` is false if any check fails.

    * no 'personal' row in kg_entities, kg_relationships, notes or preferences;
    * for the runs not undone: every folded entity is gone (a failure is a folded uuid that a
      writer minted again after the run; the report says which of them applying 067 again
      folds, and which it leaves alone: 067 folds a returned uuid only into a live technical
      survivor of the same owner, and not a flagged entity), no edge references
      one, no more edges with a missing endpoint than before the first run that merged
      knowledge-graph rows, no more dangling invalidated_by than before that run;
    * at most one flagged (is_self) entity per owner;
    * the 066 insert trigger and its function are gone, group_id defaults to 'technical' on
      the four tables, the three owner-only indexes exist and are valid;
    * schema_version is at least 067.
    Reported under ``reported_not_checked``, never failing ``ok``: the survivors that are no
    longer in 'technical', and the fact count against the one before the first run. Both
    move for good once the graph does: an episode delete removes facts, and the orphan sweep
    removes a survivor whose edges went with them. With ``just_merged`` they are checks
    instead: no survivor has left 'technical' and the fact count equals the one before the
    first run, which holds exactly only right after the merge, with the pollers stopped and
    no delete made through the dashboard (the merge itself deletes no fact). Also reported:
    live same-pair, same-text fact duplicates among the merged facts.
    """
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, ok: bool, **detail: Any) -> None:
        checks[name] = {"ok": bool(ok), **detail}

    personal = {
        t: conn.execute(f"SELECT count(*) FROM {t} WHERE group_id = 'personal'").fetchone()[0]  # nosec B608
        for t in KG_TABLES
    }
    detail: dict[str, Any] = {"personal": personal}
    if any(personal.values()):
        detail["remedy"] = "apply 067 again, then restart the pollers"
    check("no personal rows", not any(personal.values()), **detail)

    multi = [
        r[0]
        for r in conn.execute(
            "SELECT owner_id FROM kg_entities WHERE is_self GROUP BY owner_id HAVING count(*) > 1"
        ).fetchall()
    ]
    check("at most one flagged entity per owner", not multi, owners=len(multi))

    check(
        "066 insert trigger absent",
        not conn.execute(
            "SELECT 1 FROM pg_trigger WHERE tgname = 'kg_rel_personal_audience' AND NOT tgisinternal"
        ).fetchone(),
    )
    check(
        "066 trigger function absent",
        not conn.execute(
            "SELECT 1 FROM pg_proc WHERE proname = 'kg_relationships_personal_audience_trg'"
        ).fetchone(),
    )
    no_default = [
        t
        for t in KG_TABLES
        if not conn.execute(
            "SELECT 1 FROM pg_attrdef d JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum "
            "WHERE d.adrelid = %s::regclass AND a.attname = 'group_id' "
            "AND pg_get_expr(d.adbin, d.adrelid) = %s",
            (f"public.{t}", "'technical'::text"),
        ).fetchone()
    ]
    check("group_id defaults to technical", not no_default, missing=no_default)

    index_state = {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT c.relname, i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "WHERE c.relnamespace = 'public'::regnamespace AND c.relname = ANY(%s)",
            (list(INDEXES),),
        ).fetchall()
    }
    bad_idx = [n for n in INDEXES if not index_state.get(n)]
    check("owner-only indexes exist and are valid", not bad_idx, missing_or_invalid=bad_idx)

    row = conn.execute("SELECT value FROM synapse_meta WHERE key = 'schema_version'").fetchone()
    stamp = row[0] if row else None
    check("schema_version is at least 067", stamp is not None and stamp >= "067", stamp=stamp)

    runs: list[tuple[int, dict[str, Any]]] = []
    if _exists(conn, f"public.{LEDGER}_runs"):
        runs = [
            (r[0], r[1] or {})
            for r in conn.execute(
                f"SELECT run_id, counts FROM {LEDGER}_runs WHERE undone_at IS NULL ORDER BY run_id"  # nosec B608
            ).fetchall()
        ]
    info: dict[str, Any] = {"runs_not_undone": [r[0] for r in runs]}
    reported: dict[str, Any] = {}
    if runs:
        ids = [r[0] for r in runs]
        # A run that only flipped notes or preferences records no graph counts.
        first = next((c for _, c in runs if "facts_total" in c), None)
        present = conn.execute(
            f"SELECT e.uuid, e.group_id FROM kg_entities e JOIN {LEDGER}_entity_map m ON m.loser_uuid = e.uuid "  # nosec B608
            "WHERE m.run_id = ANY(%s)",
            (ids,),
        ).fetchall()
        # 067 folds a returned uuid into its survivor only in the cases the ghost text it
        # runs accepts, so this cannot drift from it. A returned uuid that is still 'personal'
        # and not accepted is not listed here: a re-run regroups it (or folds it into a
        # same-name twin), and the "no personal rows" check says to apply again.
        foldable = {uuid for _owner, uuid, _survivor in conn.execute(shared_sql()["ghost"])}
        again = [u for u, _g in present if u in foldable]
        stranded = [u for u, g in present if u not in foldable and g != "personal"]
        gone_detail: dict[str, Any] = {"still_present": _capped([u for u, _ in present])}
        remedy = []
        if again or any(g == "personal" and u not in foldable for u, g in present):
            remedy.append("apply 067 again, then restart the pollers")
        if again:
            gone_detail["apply_again_folds"] = _capped(again)
        if stranded:
            gone_detail["apply_again_leaves_alone"] = _capped(stranded)
            remedy.append(
                "applying 067 again leaves the uuids in apply_again_leaves_alone as they are: it "
                "folds a returned uuid only into a live technical survivor of the same owner, and "
                "never a flagged entity. Undo the run (--undo) if the merge is to be reversed; "
                "otherwise each is an ordinary entity now: remove it together with its edges, or "
                "re-point the edges first (a removal that leaves edges behind fails the check of "
                "edges with a missing endpoint)"
            )
        if remedy:
            gone_detail["remedy"] = "; ".join(remedy)
        check("every folded entity is gone", not present, **gone_detail)
        on_loser = conn.execute(
            f"SELECT count(*) FROM kg_relationships r WHERE r.src_uuid IN (SELECT loser_uuid FROM {LEDGER}_entity_map "  # nosec B608
            f"WHERE run_id = ANY(%s)) OR r.tgt_uuid IN (SELECT loser_uuid FROM {LEDGER}_entity_map WHERE run_id = ANY(%s))",
            (ids, ids),
        ).fetchone()[0]
        check("no edge references a folded entity", on_loser == 0, edges=on_loser)
        gone = [
            r[0]
            for r in conn.execute(
                f"SELECT DISTINCT m.survivor_uuid FROM {LEDGER}_entity_map m "  # nosec B608
                "LEFT JOIN kg_entities e ON e.uuid = m.survivor_uuid "
                "WHERE m.run_id = ANY(%s) AND (e.uuid IS NULL OR e.group_id <> 'technical')",
                (ids,),
            ).fetchall()
        ]
        if just_merged:
            check(_JM_SURVIVORS, not gone, **_capped(gone))
        else:
            reported["survivors_no_longer_in_technical"] = _capped(gone)
        if first is not None:
            missing = conn.execute(
                "SELECT count(*) FROM kg_relationships r WHERE NOT EXISTS (SELECT 1 FROM kg_entities e WHERE e.uuid = r.src_uuid) "
                "OR NOT EXISTS (SELECT 1 FROM kg_entities e WHERE e.uuid = r.tgt_uuid)"
            ).fetchone()[0]
            base = first.get("missing_endpoints_before", 0)
            check(
                "no new edges with a missing endpoint",
                missing <= base,
                now=missing,
                before_merge=base,
            )
            facts = {
                "now": conn.execute("SELECT count(*) FROM kg_relationships").fetchone()[0],
                "before_merge": first.get("facts_total"),
            }
            if just_merged:
                check(_JM_FACTS, facts["now"] == facts["before_merge"], **facts)
            else:
                reported["fact_count"] = facts
            dangling = conn.execute(
                "SELECT count(*) FROM kg_relationships r WHERE r.invalidated_by IS NOT NULL "
                "AND NOT EXISTS (SELECT 1 FROM kg_relationships o WHERE o.uuid = r.invalidated_by)"
            ).fetchone()[0]
            base = first.get("dangling_invalidated_by", 0)
            check(
                "no new dangling invalidated_by",
                dangling <= base,
                now=dangling,
                before_merge=base,
            )
        info["live_duplicate_pairs_among_merged_facts"] = conn.execute(
            f"""
            SELECT count(*) FROM kg_relationships a
              JOIN kg_relationships b
                ON b.owner_id = a.owner_id AND b.src_uuid = a.src_uuid AND b.tgt_uuid = a.tgt_uuid
               AND b.id > a.id AND b.t_invalid IS NULL
               AND lower(btrim(regexp_replace(COALESCE(b.fact, ''), '\\s+', ' ', 'g')))
                 = lower(btrim(regexp_replace(COALESCE(a.fact, ''), '\\s+', ' ', 'g')))
             WHERE a.t_invalid IS NULL
               AND (a.uuid IN (SELECT uuid FROM {LEDGER}_relationships WHERE run_id = ANY(%s) AND image = 'before')
                 OR b.uuid IN (SELECT uuid FROM {LEDGER}_relationships WHERE run_id = ANY(%s) AND image = 'before'))
            """,  # nosec B608
            (ids, ids),
        ).fetchone()[0]
    else:
        info["note"] = "no run recorded (nothing was merged, or every run is undone)"
    return {
        "ok": all(c["ok"] for c in checks.values()),
        "checks": checks,
        "reported_not_checked": reported,
        **info,
        "retired_restores (do not run unless the merge is undone first)": list(RETIRED_RESTORES),
    }


# ------------------------------------------------------------------------ undo


def _segment_066(pattern: str) -> str:
    m = re.search(pattern, SQL_066.read_text(), re.S)
    if m is None:
        raise RuntimeError(f"{SQL_066.name}: {pattern} not found")
    return m.group(0)


def _common_cols(conn: Any, live: str, ledger: str) -> str:
    """Quoted column list present in both tables (the ledger's own run_id/backed_up_at/image
    are not live columns)."""
    rows = conn.execute(
        "SELECT a.attname FROM pg_attribute a WHERE a.attrelid = %s::regclass AND a.attnum > 0 AND NOT a.attisdropped "
        "AND EXISTS (SELECT 1 FROM pg_attribute b WHERE b.attrelid = %s::regclass AND b.attname = a.attname "
        "AND b.attnum > 0 AND NOT b.attisdropped) ORDER BY a.attnum",
        (f"public.{live}", f"public.{ledger}"),
    ).fetchall()
    return ", ".join(f'"{r[0]}"' for r in rows)


def _undo_run(
    conn: Any, run_id: int, earlier: list[int], counts: dict[str, int], drift: dict[str, list[Any]]
) -> None:
    """Reverse one run from its own map and its own before and after images.

    ``earlier`` are the runs undone after this one in the same transaction. A folded uuid
    that one of them folded too is left for that run to re-create: its before image is the
    original row, where this run's is a copy minted again under the uuid."""
    for t in ("_u_map", "_u_before", "_u_after", "_u_fb"):
        conn.execute(f"DROP TABLE IF EXISTS {t}")  # nosec B608
    conn.execute(
        f"CREATE TEMP TABLE _u_map ON COMMIT DROP AS SELECT * FROM {LEDGER}_entity_map WHERE run_id = %s",
        (run_id,),
    )  # nosec B608
    for name, table, image in (
        ("_u_before", "entities", "before"),
        ("_u_after", "entities", "after"),
        ("_u_fb", "relationships", "before"),
    ):
        conn.execute(
            f"CREATE TEMP TABLE {name} ON COMMIT DROP AS SELECT * FROM {LEDGER}_{table} "  # nosec B608
            "WHERE run_id = %s AND image = %s",
            (run_id, image),
        )

    def count(key: str, n: int) -> None:
        counts[key] = counts.get(key, 0) + n

    def report(key: str, found: list[Any]) -> None:
        drift.setdefault(key, []).extend(found)

    # --- entities
    count(
        "flags moved back",
        conn.execute(
            "UPDATE kg_entities e SET is_self = false FROM _u_map m, _u_after a "
            "WHERE m.flag_moved AND e.uuid = m.survivor_uuid AND a.uuid = e.uuid AND e.is_self AND a.is_self"
        ).rowcount,
    )

    report(
        "folded entity uuid already exists",
        [
            r[0]
            for r in conn.execute(
                "SELECT m.loser_uuid FROM _u_map m JOIN kg_entities e ON e.uuid = m.loser_uuid"
            )
        ],
    )
    cols = _common_cols(conn, "kg_entities", f"{LEDGER}_entities")
    count(
        "folded entities re-created",
        conn.execute(
            f"INSERT INTO kg_entities ({cols}) SELECT {cols} FROM _u_before b "  # nosec B608
            "WHERE b.uuid IN (SELECT loser_uuid FROM _u_map) "
            f"AND b.uuid NOT IN (SELECT loser_uuid FROM {LEDGER}_entity_map WHERE run_id = ANY(%s)) "
            "AND NOT EXISTS (SELECT 1 FROM kg_entities e WHERE e.uuid = b.uuid OR e.id = b.id)",
            (earlier,),
        ).rowcount,
    )

    report(
        "regrouped entity gone or changed group",
        [
            r[0]
            for r in conn.execute(
                "SELECT b.uuid FROM _u_before b LEFT JOIN kg_entities e ON e.uuid = b.uuid "
                "WHERE b.group_id = 'personal' AND b.uuid NOT IN (SELECT loser_uuid FROM _u_map) "
                "AND (e.uuid IS NULL OR e.group_id <> 'technical')"
            )
        ],
    )
    # Edges the merge never touched (written since, or already across the two groups) that
    # point at an entity about to move back: they stay in 'technical' beside it. One EXISTS
    # per end, each with the owner, so the owner-only indexes (kg_rel_pair_any leads with
    # owner and source, kg_rel_tgt_any with owner and target) serve them: an OR over both
    # ends, with no owner, scans every edge once per regrouped entity.
    never_touched = (
        "EXISTS (SELECT 1 FROM kg_relationships r WHERE r.owner_id = e.owner_id "
        "AND r.{end}_uuid = e.uuid AND r.group_id <> 'personal' "
        "AND r.uuid NOT IN (SELECT uuid FROM _u_fb))"
    )
    report(
        "regrouped entity moved back has edges the merge never touched",
        [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT e.uuid FROM kg_entities e JOIN _u_before b ON b.uuid = e.uuid "
                "WHERE b.group_id = 'personal' AND e.group_id = 'technical' "
                "AND b.uuid NOT IN (SELECT loser_uuid FROM _u_map) "
                f"AND ({never_touched.format(end='src')} OR {never_touched.format(end='tgt')})"
            )
        ],
    )
    count(
        "entities regrouped back",
        conn.execute(
            "UPDATE kg_entities e SET group_id = 'personal' FROM _u_before b "
            "WHERE e.uuid = b.uuid AND b.group_id = 'personal' AND e.group_id = 'technical' "
            "AND b.uuid NOT IN (SELECT loser_uuid FROM _u_map)"
        ).rowcount,
    )

    same = "md5(ROW({0}.summary, {0}.embedding::text, {0}.entity_supertype, {0}.entity_subtype, {0}.degree)::text)"
    report(
        "survivor edited since the merge",
        [
            r[0]
            for r in conn.execute(
                "SELECT a.uuid FROM _u_after a JOIN _u_map m ON m.survivor_uuid = a.uuid "
                "LEFT JOIN kg_entities e ON e.uuid = a.uuid "
                f"WHERE e.uuid IS NULL OR {same.format('e')} <> {same.format('a')} "
                "GROUP BY a.uuid"
            )
        ],
    )
    count(
        "survivors restored",
        conn.execute(
            "UPDATE kg_entities e SET summary = b.summary, embedding = b.embedding, "
            "entity_supertype = b.entity_supertype, entity_subtype = b.entity_subtype, degree = b.degree "
            "FROM _u_before b, _u_after a "
            "WHERE e.uuid = b.uuid AND a.uuid = b.uuid AND b.uuid IN (SELECT survivor_uuid FROM _u_map) "
            f"AND {same.format('e')} = {same.format('a')}"
        ).rowcount,
    )

    # --- facts: group and endpoints, and the straggler stamp, only where still as the merge left them
    restorable = (
        "FROM _u_fb b "
        "LEFT JOIN _u_map ms ON ms.loser_uuid = b.src_uuid "
        "LEFT JOIN _u_map mt ON mt.loser_uuid = b.tgt_uuid "
        "WHERE r.uuid = b.uuid "
        "AND r.group_id = CASE WHEN b.group_id = 'personal' THEN 'technical' ELSE b.group_id END "
        "AND r.src_uuid = COALESCE(ms.survivor_uuid, b.src_uuid) "
        "AND r.tgt_uuid = COALESCE(mt.survivor_uuid, b.tgt_uuid)"
    )
    count(
        "facts restored",
        conn.execute(
            "UPDATE kg_relationships r SET group_id = b.group_id, src_uuid = b.src_uuid, tgt_uuid = b.tgt_uuid, "
            "audience = CASE WHEN b.group_id = 'personal' AND b.audience IS NULL AND r.audience = 'personal' "
            "AND r.label_version = 0 THEN NULL ELSE r.audience END, "
            "label_version = CASE WHEN b.group_id = 'personal' AND b.audience IS NULL AND r.audience = 'personal' "
            "AND r.label_version = 0 THEN NULL ELSE r.label_version END " + restorable
        ).rowcount,
    )
    report(
        "fact changed since the merge, or gone",
        [
            r[0]
            for r in conn.execute(
                "SELECT b.uuid FROM _u_fb b LEFT JOIN kg_relationships r ON r.uuid = b.uuid "
                "WHERE r.uuid IS NULL OR r.group_id <> b.group_id OR r.src_uuid <> b.src_uuid OR r.tgt_uuid <> b.tgt_uuid"
            )
        ],
    )

    # --- notes and preferences
    for table in ("notes", "preferences"):
        count(
            f"{table} restored",
            conn.execute(
                f"UPDATE {table} t SET group_id = 'personal' FROM {LEDGER}_{table} b "  # nosec B608
                "WHERE b.run_id = %s AND b.image = 'before' AND t.id = b.id AND t.group_id = 'technical'",
                (run_id,),
            ).rowcount,
        )
        report(
            f"{table} gone or changed group (ids)",
            [
                r[0]
                for r in conn.execute(
                    f"SELECT b.id FROM {LEDGER}_{table} b LEFT JOIN {table} t ON t.id = b.id "  # nosec B608
                    "WHERE b.run_id = %s AND b.image = 'before' AND (t.id IS NULL OR t.group_id <> 'personal')",
                    (run_id,),
                )
            ],
        )


def undo(conn: Any, *, run: str, write: bool) -> dict[str, Any]:
    """Reverse the chosen runs in one transaction (rolled back unless ``write``).

    Latest run first, each from its own entity map and before and after images, so a uuid
    that two runs folded into different survivors, and a survivor two runs wrote, are each
    judged against the run that last wrote them. Entities come back from the earliest
    before image of each uuid; a survivor's merged columns only where it still equals the
    after image of the run being undone; a fact only where it still has the group and
    endpoints the merge gave it; notes and preferences only where still 'technical'.
    Everything else is reported by uuid as drift and kept as it is.

    The open runs are read after SHARE ROW EXCLUSIVE is taken on the two KG tables, which
    conflicts with itself: a 067 apply that was running has committed its run by then, and
    an apply that has graph rows to merge and starts later waits for this transaction. One
    with none returns without that lock; its tail reads the stamp row FOR UPDATE, so it waits
    for this transaction's stamp write, or this one waits for it. A dry run takes the same
    lock and runs every statement before it rolls back, so it holds writers for as long as a
    real undo.
    Refuses, with nothing changed, when the stamp is above 067: the stamp this writes is
    066, below the objects of a later migration."""
    import psycopg

    if not _exists(conn, f"public.{LEDGER}_runs"):
        raise SystemExit("no ledger: nothing to undo")

    out: dict[str, Any] = {"written": write}
    counts: dict[str, int] = dict.fromkeys(_UNDO_COUNTS, 0)
    found: dict[str, list[Any]] = {key: [] for key in _UNDO_DRIFT}
    conn.autocommit = False
    try:
        with conn.transaction():
            conn.execute("SET LOCAL lock_timeout = '15s'")
            # A statement_timeout set on the role must not cut off a statement that scans the
            # graph (067 sets it to 0 for its own transactions the same way).
            conn.execute("SET LOCAL statement_timeout = 0")
            conn.execute("LOCK TABLE kg_entities, kg_relationships IN SHARE ROW EXCLUSIVE MODE")

            row = conn.execute(
                "SELECT value FROM synapse_meta WHERE key = 'schema_version'"
            ).fetchone()
            if row and row[0] > "067":
                raise SystemExit(
                    f"schema_version is {row[0]}, above 067: a later migration is applied, and "
                    "an undo would stamp 066 below its objects. Nothing was changed."
                )
            open_runs = [
                r[0]
                for r in conn.execute(
                    f"SELECT run_id FROM {LEDGER}_runs WHERE undone_at IS NULL ORDER BY run_id"  # nosec B608
                )
            ]
            if not open_runs:
                raise SystemExit("every run is already undone")
            if run == "all":
                ids = open_runs
            else:
                n = int(run)
                if n not in open_runs:
                    raise SystemExit(
                        f"run {n} is not a run that can be undone (open runs: {open_runs})"
                    )
                if any(r > n for r in open_runs):
                    raise SystemExit(
                        f"run {n} has later runs ({[r for r in open_runs if r > n]}): undo those first, or pass --run all"
                    )
                ids = [n]
            stamp_066 = set(ids) == set(open_runs)
            out["runs"] = ids

            # The 066 insert trigger and its function, verbatim from 066.
            conn.execute(
                _segment_066(
                    r"CREATE OR REPLACE FUNCTION kg_relationships_personal_audience_trg\(\).*?\$\$;"
                )
            )
            if not conn.execute(
                "SELECT 1 FROM pg_trigger WHERE tgname = 'kg_rel_personal_audience' AND NOT tgisinternal"
            ).fetchone():
                conn.execute(_segment_066(r"CREATE TRIGGER kg_rel_personal_audience\b.*?;"))

            for n in sorted(ids, reverse=True):
                _undo_run(conn, n, [i for i in ids if i < n], counts, found)

            conn.execute(
                f"UPDATE {LEDGER}_runs SET undone_at = now() WHERE run_id = ANY(%s)", (ids,)
            )  # nosec B608
            if stamp_066:
                conn.execute(
                    "INSERT INTO synapse_meta (key, value) VALUES ('schema_version', '066') "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
                )
            if not write:
                raise psycopg.Rollback()
    finally:
        conn.autocommit = True
    out.update(counts)
    out["stamped 066"] = stamp_066
    out["drift"] = {
        key: (
            {"ids": sorted(set(vals))[:_LIST_CAP]}
            if key.endswith("(ids)")
            else _capped(sorted(set(vals)))
        )
        for key, vals in found.items()
    }
    return out


# --------------------------------------------------------------------- exclude

_EXCLUDE_DDL = (
    f"CREATE TABLE IF NOT EXISTS {EXCLUDE_TABLE} "
    "(owner_id text NOT NULL, uuid text NOT NULL, PRIMARY KEY (owner_id, uuid))"
)


def read_exclude_file(path: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.rstrip("\r\n").split("\t")
        if len(fields) != 2 or not all(f.strip() for f in fields):
            raise SystemExit(f"{path}:{n}: expected 'owner_id<TAB>uuid'")
        pairs.append((fields[0].strip(), fields[1].strip()))
    return list(dict.fromkeys(pairs))


_KEY = "COALESCE(NULLIF({0}.normalized_name, ''), lower(trim({0}.name)))"

#: The personal entities of a file, and what 067 would do with each. Never prints a name.
_EXCLUDE_ROWS = (
    "WITH x AS (SELECT * FROM unnest(%s::text[], %s::text[]) AS t(owner_id, uuid)) "
    "SELECT e.uuid, "
    # a technical row has this entity's key (the twin the listing is meant to keep apart)
    f"       {_KEY.format('e')} <> '' AND EXISTS (SELECT 1 FROM kg_entities t "
    "          WHERE t.owner_id = e.owner_id AND t.group_id = 'technical' "
    f"          AND {_KEY.format('t')} = {_KEY.format('e')}) AS twin, "
    # rule (a) folds a flagged entity into the owner's flagged technical one, listed or not
    "       e.is_self AND EXISTS (SELECT 1 FROM kg_entities t WHERE t.owner_id = e.owner_id "
    "          AND t.group_id = 'technical' AND t.is_self) AS owner_flag, "
    # no technical row of its key ranks before it by SURVIVOR_ORDER (is_self DESC, id)
    f"       {_KEY.format('e')} <> '' AND EXISTS (SELECT 1 FROM kg_entities t "
    "          WHERE t.owner_id = e.owner_id AND t.group_id = 'technical' "
    f"          AND {_KEY.format('t')} = {_KEY.format('e')}) "
    "       AND NOT EXISTS (SELECT 1 FROM kg_entities t "
    "          WHERE t.owner_id = e.owner_id AND t.group_id = 'technical' "
    f"          AND {_KEY.format('t')} = {_KEY.format('e')} "
    "          AND (t.is_self AND NOT e.is_self OR t.is_self = e.is_self AND t.id < e.id)) "
    "       AS first_of_key "
    "FROM x JOIN kg_entities e ON e.owner_id = x.owner_id AND e.uuid = x.uuid "
    "AND e.group_id = 'personal'"
)


def exclude(conn: Any, pairs: list[tuple[str, str]], *, write: bool) -> dict[str, Any]:
    """List the pairs; report what 067 will and will not do with them, by uuid.

    Two listings do not do what the owner may expect: a flagged entity folds into the owner's
    flagged technical one whatever is listed (two flagged rows cannot share a group), and a
    listed entity with a lower id than its technical twin becomes the row every later mention
    of that name resolves to (SURVIVOR_ORDER). Both are reported, never refused."""
    owners = [p[0] for p in pairs]
    uuids = [p[1] for p in pairs]
    found = conn.execute(_EXCLUDE_ROWS, (owners, uuids)).fetchall()  # nosec B608
    folded = [r[0] for r in found if r[2]]
    ranks_first = [r[0] for r in found if r[3] and not r[2]]
    out: dict[str, Any] = {
        "pairs in file": len(pairs),
        "exist as personal entities": len(found),
        "of those, personal twins (would have been folded)": sum(1 for r in found if r[1]),
        "of those, still folded: the owner-flag rule ignores the listing": _capped(folded),
        "of those, kept apart but first for their name by SURVIVOR_ORDER": _capped(ranks_first),
        "written": write,
    }
    warnings = []
    if folded:
        warnings.append(
            f"{len(folded)} listed entit{'y' if len(folded) == 1 else 'ies'} will still fold "
            "into the owner's flagged technical entity (rule a)"
        )
    if ranks_first:
        warnings.append(
            f"{len(ranks_first)} listed entit{'y' if len(ranks_first) == 1 else 'ies'} will "
            "outrank the technical twin: every later mention of the name resolves to it"
        )
    if warnings:
        out["warnings"] = warnings
        for w in warnings:
            print(f"warning: {w}", file=sys.stderr)
    if write:
        conn.execute(_EXCLUDE_DDL)
        cur = conn.execute(
            f"INSERT INTO {EXCLUDE_TABLE} (owner_id, uuid) SELECT * FROM unnest(%s::text[], %s::text[]) "  # nosec B608
            "ON CONFLICT DO NOTHING",
            (owners, uuids),
        )
        out["newly listed"] = cur.rowcount
    return out


# ------------------------------------------------------------------------ main


def _just_merged_failures(report: dict[str, Any]) -> list[tuple[str, str]]:
    """The failed checks that ``--just-merged`` adds, each with a line saying why."""
    out: list[tuple[str, str]] = []
    survivors = report["checks"].get(_JM_SURVIVORS)
    if survivors and not survivors["ok"]:
        out.append(
            (
                _JM_SURVIVORS,
                f"survivors_no_longer_in_technical.count is {survivors['count']}, not 0",
            )
        )
    facts = report["checks"].get(_JM_FACTS)
    if facts and not facts["ok"]:
        out.append(
            (
                _JM_FACTS,
                f"fact_count.now is {facts['now']}, before_merge is {facts['before_merge']}",
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db-url", default=os.environ.get("SYNAPSE_DB_URL", ""))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preview", action="store_true", help="counts of what 067 would do (read-only)"
    )
    mode.add_argument(
        "--health", action="store_true", help="post-merge checks (read-only; non-zero on failure)"
    )
    mode.add_argument(
        "--undo",
        action="store_true",
        help="reverse the recorded runs (dry run unless --apply; either way it holds writers "
        "off the KG tables while it runs, so stop the pollers first)",
    )
    mode.add_argument("--exclude", metavar="FILE", help="list personal twins 067 must not fold")
    ap.add_argument("--run", default="all", help="with --undo: a run id, or 'all' (default)")
    ap.add_argument("--apply", action="store_true", help="with --undo / --exclude: write")
    ap.add_argument(
        "--just-merged",
        action="store_true",
        help="with --health, right after the merge with the pollers stopped: also fail unless no "
        "survivor has left technical and the fact count equals the one before the merge",
    )
    args = ap.parse_args(argv)
    if not args.db_url:
        ap.error("no DSN: pass --db-url or set SYNAPSE_DB_URL")
    if args.run != "all" and not args.run.isdigit():
        ap.error("--run takes a run id or 'all'")
    if args.just_merged and not args.health:
        ap.error("--just-merged goes with --health")

    if args.preview:
        print(json.dumps(preview(_connect(args.db_url, read_only=True)), indent=1, sort_keys=True))
        return 0
    if args.health:
        report = health(_connect(args.db_url, read_only=True), just_merged=args.just_merged)
        print(json.dumps(report, indent=1, sort_keys=True))
        for name, why in _just_merged_failures(report):
            print(f"--just-merged: {name}: {why}", file=sys.stderr)
        return 0 if report["ok"] else 1
    if args.undo:
        print(
            json.dumps(
                undo(_connect(args.db_url, read_only=False), run=args.run, write=args.apply),
                indent=1,
                sort_keys=True,
            )
        )
        return 0
    pairs = read_exclude_file(args.exclude)
    conn = _connect(args.db_url, read_only=not args.apply)
    print(json.dumps(exclude(conn, pairs, write=args.apply), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
