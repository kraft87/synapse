#!/usr/bin/env python3
"""Map every stored project label onto one canonical name per project, then (with
--apply) rewrite the stored labels.

Until plugin 0.21 a project label was the last "/"-separated piece of the session's
working directory, derived four slightly different ways, so one project could be stored
under many names: a whole Windows path (the server split on "/" only), the same path with
a lowercase drive letter, several casings of the folder name, one label per subfolder a
session was started in, one per git worktree, and a hyphen/underscore twin when two
machines named the checkout differently. Clients now resolve one label (a
``.synapse-project`` marker, else the git toplevel's name, else the folder's name; see
``plugin/scripts/project_label.py``); this script brings the stored history in line.

How a label is mapped (the dry run prints the reason for every change):

1. **A path becomes its last component.** Labels are compared as paths with either
   separator, ignoring trailing separators; Windows paths compare case-insensitively.
2. **A subfolder joins its project.** Every directory a label was derived from (an
   episode's ``metadata.cwd`` whose basename IS the label, or a label that is itself a
   path) is a candidate project root, except home directories, filesystem and temp
   roots, and folders that hold projects rather than being one (``services``,
   ``workspace``, ``projects``, ``.claude`` ... and any ``container:`` line in the alias
   file). A candidate with a candidate above it is not a root; every label then maps to
   the longest root at or above its directory. So ``<repo>/tickets`` and
   ``<repo>/.claude/worktrees/agent-x`` both become ``<repo>``'s label. A label set
   explicitly (it differs from its cwd's basename) is never re-derived from the cwd.
3. **Aliases.** ``old = new`` lines in the alias file (matched case-insensitively) cover
   what no rule can see, e.g. two machines' checkouts named differently.
4. **One casing per name.** Labels equal under case-folding merge into the casing an
   alias names, else the one with the most episodes.

What is rewritten: ``episodes.project`` (per label and cwd; the
``episodes_kg_source_projects_relabel`` trigger recomputes the facts that cite each
episode), the ``project`` column of every other table that has one (found in the
catalog, so prod-only tables are included), each surface's ``allowed_projects``
(mapped, de-duplicated case-insensitively), and ``project:<label>`` skill scopes in
``skills_lane.skill_registry``.

The alias file stays OUTSIDE the repository, like the dry run's output: both name
private projects and paths, and this repository is public. Format, one per line:

    # comments and blank lines are ignored
    transcribe-ai = transcribe_ai
    container: /home/me/code

Usage:
    scripts/relabel_projects.py --dry-run [--aliases FILE] [--db-url DSN] [--out FILE]
    scripts/relabel_projects.py --apply   [--aliases FILE] [--db-url DSN]
    scripts/relabel_projects.py --inventory-sql
    scripts/relabel_projects.py --dry-run --inventory FILE [--aliases FILE] [--out FILE]

The DSN comes from --db-url, else $SYNAPSE_DB_URL. --dry-run runs in a READ ONLY
transaction and writes nothing. --apply re-plans from the live data and commits one
label at a time, so it can be re-run after an interruption.

Without a DSN on hand (the database only reachable through ``docker exec ... psql``), dry
run in two steps: ``--inventory-sql`` prints ONE read-only statement whose JSON cell is
everything the plan reads; run it with ``psql -tA`` and pass the output to ``--dry-run
--inventory FILE``, which plans without connecting anywhere:

    scripts/relabel_projects.py --inventory-sql > inv.sql
    docker exec -i <postgres> psql -U synapse -d synapse -tA < inv.sql > inv.json
    scripts/relabel_projects.py --dry-run --inventory inv.json --aliases FILE
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.project_label import label_from_path

# ------------------------------------------------------------------------- planning

#: Folder names that hold projects rather than being one. A container is never a root,
#: so it never absorbs the projects inside it; its own label (if any) is left alone.
DEFAULT_CONTAINERS = frozenset(
    {
        ".agents",
        ".cache",
        ".claude",
        ".codex",
        ".config",
        ".local",
        "cache",
        "code",
        "data",
        "desktop",
        "dev",
        "documents",
        "downloads",
        "git",
        "github",
        "plugins",
        "projects",
        "repos",
        "scratch",
        "scratchpad",
        "services",
        "skills",
        "src",
        "temp",
        "tmp",
        "workspace",
        "workspaces",
        "worktrees",
    }
)
_HOME = re.compile(r"^(?:/home/[^/]+|/Users/[^/]+|/root|[A-Za-z]:/Users/[^/]+)$", re.IGNORECASE)
_TEMP_ROOT = re.compile(r"^(?:/tmp|/var/tmp|/private/tmp|[A-Za-z]:/Windows/Temp)(?:/|$)", re.I)
_ONEDRIVE = re.compile(r"^onedrive(?: - .*)?$", re.IGNORECASE)
_DRIVE_PATH = re.compile(r"^/?([A-Za-z]):(?=/|$)")


def is_path_label(label: str) -> bool:
    return "/" in label or "\\" in label


def norm_path(path: str | None) -> str | None:
    """``path`` with forward slashes, no repeated or trailing separators, and an
    upper-case drive (``/c:/x`` and ``c:\\x`` both become ``C:/x``)."""
    if not path:
        return None
    p = re.sub(r"/+", "/", str(path).replace("\\", "/"))
    m = _DRIVE_PATH.match(p)
    if m:
        p = m.group(1).upper() + ":" + p[m.end() :]
    p = p.rstrip("/")
    if not p:
        return "/"
    return p + "/" if re.fullmatch(r"[A-Z]:", p) else p


def path_key(p: str) -> str:
    """The comparison key of a normalised path: Windows paths ignore case."""
    return p.casefold() if re.match(r"^[A-Z]:", p) else p


def _parent(key: str) -> str | None:
    if "/" not in key.rstrip("/"):
        return None
    head = key.rsplit("/", 1)[0]
    if not head:
        return "/" if key != "/" else None
    return head + "/" if re.fullmatch(r"[a-z]:|[A-Z]:", head) else head


def _ancestors_or_self(key: str) -> Iterable[str]:
    k: str | None = key
    while k:
        yield k
        k = _parent(k)


@dataclass
class AliasConfig:
    aliases: dict[str, str] = field(default_factory=dict)  # casefolded old -> new
    containers: set[str] = field(default_factory=set)  # path keys

    @classmethod
    def parse(cls, text: str) -> AliasConfig:
        cfg = cls()
        for n, raw in enumerate(text.splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("container:"):
                p = norm_path(line.split(":", 1)[1].strip())
                if p:
                    cfg.containers.add(path_key(p))
                continue
            if "=" not in line:
                raise ValueError(f"alias file line {n}: expected 'old = new' or 'container: PATH'")
            old, new = (s.strip() for s in line.split("=", 1))
            if not old or not new or is_path_label(new):
                raise ValueError(f"alias file line {n}: both sides needed, no path on the right")
            cfg.aliases[old.casefold()] = new
        return cfg


@dataclass(frozen=True)
class Change:
    label: str
    cwd_root: str | None  # the folder that decided it, for display
    canonical: str
    rows: int
    reason: str


@dataclass
class Plan:
    #: (label, cwd) -> canonical, for every episode group (unchanged ones included).
    groups: dict[tuple[str, str | None], str]
    #: Every change, for the report: (label, cwd-root) aggregated.
    changes: list[Change]
    #: label -> canonical for tables without a cwd (and allowlists, skill scopes).
    labels: dict[str, str]
    roots: dict[str, str]  # root key -> display path

    def map_label(self, label: str | None) -> str | None:
        if label is None:
            return None
        return self.labels.get(label, label)


class Planner:
    def __init__(self, cfg: AliasConfig | None = None) -> None:
        self.cfg = cfg or AliasConfig()

    def _excluded(self, p: str) -> bool:
        key = path_key(p)
        base = (label_from_path(p) or "").casefold()
        return (
            p == "/"
            or bool(re.fullmatch(r"[A-Z]:/", p))
            or bool(_HOME.match(p))
            or bool(_TEMP_ROOT.match(p) and p.count("/") <= 2)
            or base in DEFAULT_CONTAINERS
            or bool(_ONEDRIVE.match(base))
            or key in self.cfg.containers
        )

    @staticmethod
    def _derived_path(label: str, cwd: str | None) -> str | None:
        """The directory ``label`` was derived from, or None when it was set explicitly."""
        if is_path_label(label):
            return norm_path(label)
        if cwd:
            base = label_from_path(cwd)
            if base and base.casefold() == label.casefold():
                return norm_path(cwd)
        return None

    def plan(
        self,
        episode_groups: dict[tuple[str, str | None], int],
        other_labels: Iterable[str] = (),
    ) -> Plan:
        # 1. Candidate roots: every directory a label was derived from.
        candidates: dict[str, str] = {}
        for label, cwd in episode_groups:
            p = self._derived_path(label, cwd)
            if p and not self._excluded(p):
                candidates.setdefault(path_key(p), p)
        for label in other_labels:
            p = self._derived_path(label, None)
            if p and not self._excluded(p):
                candidates.setdefault(path_key(p), p)
        # 2. Roots: candidates with no candidate above them.
        roots = {
            k: p
            for k, p in candidates.items()
            if not any(a in candidates for a in list(_ancestors_or_self(k))[1:])
        }

        def base_of(label: str, cwd: str | None) -> tuple[str, str | None, str]:
            p = self._derived_path(label, cwd)
            if p is None:
                return label, None, "kept"
            for a in _ancestors_or_self(path_key(p)):
                if a in roots:
                    name = label_from_path(roots[a]) or label
                    why = "path → its name" if a == path_key(p) else "subfolder → project root"
                    if a == path_key(p) and not is_path_label(label):
                        why = "kept"
                    return name, roots[a], why
            return (
                label_from_path(p) or label,
                None,
                ("path → its name" if is_path_label(label) else "kept"),
            )

        # 3. Aliases, then 4. one casing per name.
        def alias(name: str) -> tuple[str, bool]:
            new = self.cfg.aliases.get(name.casefold())
            return (new, True) if new else (name, False)

        pre: dict[tuple[str, str | None], tuple[str, str | None, str]] = {}
        totals: dict[str, int] = defaultdict(int)
        for (label, cwd), n in episode_groups.items():
            name, root, why = base_of(label, cwd)
            name2, aliased = alias(name)
            if aliased and name2 != name:
                why = "alias" if why == "kept" else f"{why}, alias"
            pre[(label, cwd)] = (name2, root, why)
            totals[name2] += n
        label_pre: dict[str, tuple[str, str]] = {}
        for label in {lb for lb, _ in episode_groups} | set(other_labels):
            name, _root, why = base_of(label, None)
            name2, aliased = alias(name)
            if aliased and name2 != name:
                why = "alias" if why == "kept" else f"{why}, alias"
            label_pre[label] = (name2, why)
            totals.setdefault(name2, 0)

        casing: dict[str, str] = {}
        forced = {v.casefold(): v for v in self.cfg.aliases.values()}
        by_fold: dict[str, list[str]] = defaultdict(list)
        for name in totals:
            by_fold[name.casefold()].append(name)
        for fold, names in by_fold.items():
            pick = forced.get(fold) or sorted(names, key=lambda s: (-totals[s], s))[0]
            for name in names:
                casing[name] = pick

        def finish(name: str, why: str) -> tuple[str, str]:
            final = casing.get(name, name)
            if final != name:
                why = "case merge" if why == "kept" else f"{why}, case merge"
            return final, why

        groups: dict[tuple[str, str | None], str] = {}
        agg: dict[tuple[str, str | None, str, str], int] = defaultdict(int)
        for key, n in episode_groups.items():
            name, root, why = pre[key]
            final, why = finish(name, why)
            groups[key] = final
            if final != key[0]:
                agg[(key[0], root, final, why)] += n
        labels: dict[str, str] = {}
        for label, (name, why) in label_pre.items():
            final, _why = finish(name, why)
            labels[label] = final
        changes = [
            Change(label, root, canonical, rows, why)
            for (label, root, canonical, why), rows in agg.items()
        ]
        changes.sort(key=lambda c: (c.canonical.casefold(), c.label, c.cwd_root or ""))
        return Plan(groups=groups, changes=changes, labels=labels, roots=roots)


def map_allowlist(plan: Plan, entries: Iterable[str]) -> list[str]:
    """Each entry mapped, first spelling of each name kept, case-insensitively unique."""
    out: list[str] = []
    seen: set[str] = set()
    for e in entries:
        m = plan.map_label(e) or e
        if m.casefold() not in seen:
            seen.add(m.casefold())
            out.append(m)
    return out


def map_scope(plan: Plan, scope: str) -> str:
    if scope.startswith("project:"):
        return "project:" + (plan.map_label(scope[len("project:") :]) or scope[len("project:") :])
    return scope


# ------------------------------------------------------------------------- database

_PROJECT_TABLES = """
SELECT c.table_schema, c.table_name
  FROM information_schema.columns c
  JOIN information_schema.tables t
    ON t.table_schema = c.table_schema AND t.table_name = c.table_name
 WHERE c.column_name = 'project' AND c.data_type = 'text' AND t.table_type = 'BASE TABLE'
   AND c.table_schema NOT IN ('pg_catalog', 'information_schema')
 ORDER BY 1, 2
"""


def _qual(schema: str, table: str) -> str:
    return f'"{schema}"."{table}"'


@dataclass
class Inventory:
    episode_groups: dict[tuple[str, str | None], int]
    tables: dict[str, dict[str, int]]  # qualified table -> label -> rows (non-episodes)
    surfaces: list[dict[str, Any]]
    skills: list[dict[str, Any]]
    config_scopes: dict[str, int] = field(default_factory=dict)  # 'project:<label>' -> rows

    def other_labels(self) -> set[str]:
        labels = {lb for counts in self.tables.values() for lb in counts}
        labels |= {s[len("project:") :] for s in self.config_scopes}
        return labels


def read_inventory(conn: Any) -> Inventory:
    groups: dict[tuple[str, str | None], int] = {}
    for project, cwd, n in conn.execute(
        "SELECT project, metadata->>'cwd', count(*) FROM episodes "
        "WHERE project IS NOT NULL AND project <> '' GROUP BY 1, 2"
    ).fetchall():
        groups[(project, cwd)] = n
    tables: dict[str, dict[str, int]] = {}
    for schema, table in conn.execute(_PROJECT_TABLES).fetchall():
        if (schema, table) == ("public", "episodes"):
            continue
        q = _qual(schema, table)
        rows = conn.execute(
            f"SELECT project, count(*) FROM {q} "  # nosec B608 - catalog names, quoted
            "WHERE project IS NOT NULL AND project <> '' GROUP BY 1"
        ).fetchall()
        tables[q] = {p: n for p, n in rows}
    surfaces = []
    if conn.execute("SELECT to_regclass('surfaces')").fetchone()[0]:
        surfaces = [
            {"surface_id": s, "allowed_projects": list(a or [])}
            for s, a in conn.execute(
                "SELECT surface_id, allowed_projects FROM surfaces ORDER BY surface_id"
            ).fetchall()
        ]
    skills = []
    if conn.execute("SELECT to_regclass('skills_lane.skill_registry')").fetchone()[0]:
        skills = [
            {"name": n, "scope": s}
            for n, s in conn.execute(
                "SELECT name, scope FROM skills_lane.skill_registry "
                "WHERE scope LIKE 'project:%' ORDER BY name"
            ).fetchall()
        ]
    config_scopes: dict[str, int] = {}
    if conn.execute("SELECT to_regclass('config_lane.config_registry')").fetchone()[0]:
        config_scopes = {
            s: n
            for s, n in conn.execute(
                "SELECT scope, count(*) FROM config_lane.config_registry "
                "WHERE scope LIKE 'project:%' GROUP BY 1"
            ).fetchall()
        }
    return Inventory(groups, tables, surfaces, skills, config_scopes)


#: The read-only inventory as one JSON cell (for ``--inventory``): what read_inventory
#: reads, in one statement psql can run. Tables with a ``project`` column are found in the
#: catalog and counted through query_to_xml, so no dynamic SQL needs a function.
INVENTORY_SQL = r"""
BEGIN READ ONLY;
WITH tabs AS (
  SELECT c.table_schema AS s, c.table_name AS t
    FROM information_schema.columns c
    JOIN information_schema.tables tt
      ON tt.table_schema = c.table_schema AND tt.table_name = c.table_name
   WHERE c.column_name = 'project' AND c.data_type = 'text' AND tt.table_type = 'BASE TABLE'
     AND c.table_schema NOT IN ('pg_catalog', 'information_schema')
     AND NOT (c.table_schema = 'public' AND c.table_name = 'episodes')
), counts AS (
  SELECT s, t, x.project, x.n
    FROM tabs,
         XMLTABLE('/table/row' PASSING query_to_xml(
           format('SELECT project, count(*) AS n FROM %I.%I '
                  'WHERE project IS NOT NULL AND project <> %L GROUP BY 1', s, t, ''),
           true, false, '')  -- nulls, NOT a forest: an empty table is <table/>
         COLUMNS project text PATH 'project', n bigint PATH 'n') AS x
)
SELECT json_build_object(
  'episodes', (SELECT coalesce(json_agg(json_build_array(project, cwd, n)), '[]')
                 FROM (SELECT project, metadata->>'cwd' AS cwd, count(*) AS n FROM episodes
                        WHERE project IS NOT NULL AND project <> '' GROUP BY 1, 2) e),
  'tables', (SELECT coalesce(json_object_agg(q, m), '{}')
               FROM (SELECT format('"%s"."%s"', tabs.s, tabs.t) AS q,
                            coalesce(json_object_agg(c.project, c.n)
                                       FILTER (WHERE c.project IS NOT NULL), '{}') AS m
                       FROM tabs LEFT JOIN counts c ON c.s = tabs.s AND c.t = tabs.t
                      GROUP BY tabs.s, tabs.t) z),
  'surfaces', (SELECT coalesce(json_agg(json_build_object(
                   'surface_id', surface_id, 'allowed_projects', allowed_projects)
                 ORDER BY surface_id), '[]') FROM surfaces),
  'skills', (SELECT coalesce(json_agg(json_build_object('name', name, 'scope', scope)
               ORDER BY name), '[]')
               FROM skills_lane.skill_registry WHERE scope LIKE 'project:%'),
  'config_scopes', (SELECT coalesce(json_object_agg(scope, n), '{}')
                      FROM (SELECT scope, count(*) AS n FROM config_lane.config_registry
                             WHERE scope LIKE 'project:%' GROUP BY 1) c));
ROLLBACK;
"""


def inventory_from_json(text: str) -> Inventory:
    """An Inventory from INVENTORY_SQL's JSON (psql -tA output; the BEGIN/ROLLBACK lines
    around it are ignored)."""
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("{"))
    data = json.loads(line)
    groups = {(p, c): int(n) for p, c, n in data.get("episodes") or []}
    tables = {
        q: {lb: int(n) for lb, n in (m or {}).items()}
        for q, m in (data.get("tables") or {}).items()
    }
    surfaces = [
        {"surface_id": s["surface_id"], "allowed_projects": list(s.get("allowed_projects") or [])}
        for s in data.get("surfaces") or []
    ]
    skills = [{"name": s["name"], "scope": s["scope"]} for s in data.get("skills") or []]
    config = {s: int(n) for s, n in (data.get("config_scopes") or {}).items()}
    return Inventory(groups, tables, surfaces, skills, config)


def _unique_project_indexes(conn: Any, qualified: str) -> list[str]:
    """Unique indexes that include ``project``: a relabel there could collide."""
    return [
        r[0]
        for r in conn.execute(
            "SELECT i.indexrelid::regclass::text FROM pg_index i "
            "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
            "WHERE i.indrelid = %s::regclass AND i.indisunique AND a.attname = 'project'",
            (qualified,),
        ).fetchall()
    ]


def report(inv: Inventory, plan: Plan) -> str:
    out: list[str] = []
    before = {lb for lb, _ in inv.episode_groups}
    after = set(plan.groups.values())
    moved = sum(c.rows for c in plan.changes)
    total = sum(inv.episode_groups.values())
    out.append(
        f"episodes: {len(before)} labels -> {len(after)}; {moved} of {total} labelled rows change\n"
    )
    out.append("| label | decided by folder | canonical | rows | why |")
    out.append("| --- | --- | --- | ---: | --- |")
    for c in plan.changes:
        out.append(f"| {c.label} | {c.cwd_root or ''} | {c.canonical} | {c.rows} | {c.reason} |")
    out.append("\ncanonical labels after (episodes):")
    counts: dict[str, int] = defaultdict(int)
    for key, n in inv.episode_groups.items():
        counts[plan.groups[key]] += n
    for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        out.append(f"  {n:>7}  {name}")
    for q, labels in inv.tables.items():
        ch = {lb: plan.map_label(lb) for lb in labels if plan.map_label(lb) != lb}
        if ch:
            out.append(f"\n{q}: {sum(labels[lb] for lb in ch)} rows change")
            for lb, new in sorted(ch.items()):
                out.append(f"  {labels[lb]:>7}  {lb} -> {new}")
        else:
            out.append(f"\n{q}: no change ({sum(labels.values())} labelled rows)")
    out.append("\nsurfaces.allowed_projects:")
    for s in inv.surfaces:
        new = map_allowlist(plan, s["allowed_projects"])
        mark = "" if new != s["allowed_projects"] else "  (unchanged)"
        out.append(f"  {s['surface_id']}: {s['allowed_projects']} -> {new}{mark}")
    out.append("\nconfig_lane.config_registry.scope:")
    for scope, n in sorted(inv.config_scopes.items()):
        new = map_scope(plan, scope)
        out.append(f"  {n:>7}  {scope} -> {new}" + ("" if new != scope else "  (unchanged)"))
    out.append("\nskills_lane.skill_registry.scope:")
    for s in inv.skills:
        new = map_scope(plan, s["scope"])
        out.append(
            f"  {s['name']}: {s['scope']} -> {new}" + ("" if new != s["scope"] else "  (unchanged)")
        )
    return "\n".join(out) + "\n"


def apply(conn: Any, inv: Inventory, plan: Plan, log=print) -> None:
    """Write the plan, one committed transaction per label (and one for the small
    tables), so an interrupted run resumes by planning again."""
    by_label: dict[str, list[tuple[str | None, str]]] = defaultdict(list)
    for (label, cwd), canonical in plan.groups.items():
        by_label[label].append((cwd, canonical))
    for label, targets in sorted(by_label.items()):
        changed = [(cwd, c) for cwd, c in targets if c != label]
        if not changed:
            continue
        with conn.transaction():
            if len(changed) == len(targets) and len({c for _, c in changed}) == 1:
                n = conn.execute(
                    "UPDATE episodes SET project = %s WHERE project = %s",
                    (changed[0][1], label),
                ).rowcount
            else:
                n = 0
                for cwd, canonical in changed:
                    n += conn.execute(
                        "UPDATE episodes SET project = %s WHERE project = %s "
                        "AND (metadata->>'cwd') IS NOT DISTINCT FROM %s",
                        (canonical, label, cwd),
                    ).rowcount
        log(f"episodes: {label} -> {sorted({c for _, c in changed})}: {n} rows")
    for q, labels in inv.tables.items():
        ch = {lb: plan.map_label(lb) for lb in labels if plan.map_label(lb) != lb}
        if not ch:
            continue
        clash = _unique_project_indexes(conn, q)
        if clash:
            log(f"{q}: SKIPPED, unique index on project ({', '.join(clash)}); relabel by hand")
            continue
        with conn.transaction():
            for lb, new in ch.items():
                conn.execute(
                    f"UPDATE {q} SET project = %s WHERE project = %s",  # nosec B608
                    (new, lb),
                )
        log(f"{q}: {sum(labels[lb] for lb in ch)} rows")
    for scope in sorted(inv.config_scopes):
        new = map_scope(plan, scope)
        if new == scope:
            continue
        # (surface_id, scope, file_key) is the key: a file already mirrored under the new
        # scope keeps that row, and the old-scope one stays for its owner to drop.
        with conn.transaction():
            moved = conn.execute(
                "UPDATE config_lane.config_registry r SET scope = %s WHERE r.scope = %s "
                "AND NOT EXISTS (SELECT 1 FROM config_lane.config_registry x WHERE "
                "x.surface_id = r.surface_id AND x.scope = %s AND x.file_key = r.file_key)",
                (new, scope, new),
            ).rowcount
        left = inv.config_scopes[scope] - moved
        log(f"config_registry: {scope} -> {new}: {moved} rows" + (f", {left} kept" if left else ""))
    with conn.transaction():
        for s in inv.surfaces:
            new = map_allowlist(plan, s["allowed_projects"])
            if new != s["allowed_projects"]:
                conn.execute(
                    "UPDATE surfaces SET allowed_projects = %s, updated_at = now() "
                    "WHERE surface_id = %s",
                    (new, s["surface_id"]),
                )
                log(f"surfaces: {s['surface_id']} -> {new}")
        for s in inv.skills:
            new = map_scope(plan, s["scope"])
            if new != s["scope"]:
                conn.execute(
                    "UPDATE skills_lane.skill_registry SET scope = %s, updated_at = now() "
                    "WHERE name = %s",
                    (new, s["name"]),
                )
                log(f"skill_registry: {s['name']} -> {new}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="print the mapping, write nothing")
    mode.add_argument("--apply", action="store_true", help="rewrite the stored labels")
    mode.add_argument("--inventory-sql", action="store_true", help="print the inventory SQL")
    p.add_argument("--aliases", type=Path, help="alias file (keep it outside the repo)")
    p.add_argument("--out", type=Path, help="--dry-run: also write the report here")
    p.add_argument("--inventory", type=Path, help="--dry-run: plan from --inventory-sql output")
    p.add_argument("--db-url", default=os.environ.get("SYNAPSE_DB_URL", ""))
    args = p.parse_args(argv)
    if args.inventory_sql:
        print(INVENTORY_SQL.strip())
        return 0
    cfg = AliasConfig()
    if args.aliases:
        cfg = AliasConfig.parse(args.aliases.expanduser().read_text(encoding="utf-8"))
    if args.inventory is not None:
        if not args.dry_run:
            print("--inventory plans offline, so it only goes with --dry-run", file=sys.stderr)
            return 2
        inv = inventory_from_json(args.inventory.expanduser().read_text(encoding="utf-8"))
        text = report(inv, Planner(cfg).plan(inv.episode_groups, inv.other_labels()))
        print(text, end="")
        if args.out:
            args.out.expanduser().write_text(text, encoding="utf-8")
        return 0
    if not args.db_url:
        print("no database URL — pass --db-url or set SYNAPSE_DB_URL", file=sys.stderr)
        return 2

    import psycopg

    with psycopg.connect(args.db_url, autocommit=True) as conn:
        if args.dry_run:
            with conn.transaction():
                conn.execute("SET TRANSACTION READ ONLY")
                inv = read_inventory(conn)
        else:
            inv = read_inventory(conn)
        plan = Planner(cfg).plan(inv.episode_groups, inv.other_labels())
        text = report(inv, plan)
        if args.dry_run:
            print(text, end="")
            if args.out:
                args.out.expanduser().write_text(text, encoding="utf-8")
            return 0
        apply(conn, inv, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
