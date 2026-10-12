"""scripts/relabel_projects.py end to end against a real database.

The planner's rules are pinned in tests/test_relabel_projects.py. What only a database
can show is that the script reads what is really stored and writes it back faithfully:

* ``--dry-run`` reports the change table for the seeded labels (with and without an
  alias file) and writes NOTHING, the report file included in what it prints;
* ``--apply`` moves ``episodes.project`` per (label, cwd) group, every other table with a
  ``project`` column (found in the catalog, so a table outside ``public`` is included,
  and one with a unique index on ``project`` is skipped, not half-written), each
  surface's ``allowed_projects`` (mapped, de-duplicated) and ``project:`` skill scopes;
* the ``episodes_kg_source_projects_relabel`` trigger (schema 056) is observed
  recomputing ``kg_relationships.source_projects`` for the facts citing a moved episode;
* a second ``--apply`` is a no-op.

The database is shared with every other DB test, so every row seeded here carries a
per-test tag (labels, paths, session ids, surface ids, skill names) and is deleted
afterwards; nothing is truncated. The relabel itself plans over the whole database, so
the assertions only ever look at this test's rows.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from psycopg.types.json import Jsonb

from tests.helpers.surfaces import register_device

_SPEC = importlib.util.spec_from_file_location(
    "relabel_projects",
    Path(__file__).resolve().parent.parent / "scripts" / "relabel_projects.py",
)
rp = importlib.util.module_from_spec(_SPEC)
sys.modules["relabel_projects"] = rp
_SPEC.loader.exec_module(rp)


@dataclass
class World:
    """One deployment's worth of generic labels, unique to one test."""

    tag: str
    eps: dict[str, int] = field(default_factory=dict)  # seed key -> episode id

    @property
    def prefix(self) -> str:
        return f"rlbl-{self.tag}"

    @property
    def big(self) -> str:
        return f"BigProj-{self.tag}"

    @property
    def win(self) -> str:
        return rf"C:\Users\pat\OneDrive - Acme Ltd\projects\{self.big}"

    @property
    def win_lc(self) -> str:
        return "c" + self.win[1:]

    @property
    def win_root(self) -> str:
        return f"C:/Users/pat/OneDrive - Acme Ltd/projects/{self.big}"

    @property
    def home(self) -> str:
        return f"/home/dev-{self.tag}"

    @property
    def acme(self) -> str:
        return f"acme-{self.tag}"

    @property
    def repo(self) -> str:
        return f"{self.home}/code/{self.acme}"

    @property
    def twin(self) -> str:
        """Another machine's checkout of the same project, named differently."""
        return f"big-proj-{self.tag}"

    @property
    def alpha(self) -> str:
        """An explicit ``.synapse-project`` label: differs from its folder's name."""
        return f"alpha-{self.tag}"

    @property
    def schema(self) -> str:
        return f"rlbl_{self.tag}"

    # (key, label, cwd, episodes, canonical after --apply with the alias file)
    def episode_plan(self) -> list[tuple[str, str, str | None, int, str]]:
        w = self.win
        return [
            ("win", w, w, 2, self.big),
            ("win_lc", self.win_lc, self.win_lc, 1, self.big),
            ("tickets", w + r"\tickets", w + r"\tickets", 1, self.big),
            ("lab_out", w + r"\tickets\lab\out", w + r"\tickets\lab\out", 1, self.big),
            ("cursor", self.big, "/" + self.win_root, 1, self.big),
            ("case", self.big.lower(), None, 1, self.big),
            ("twin", self.twin, f"{self.home}/code/{self.twin}", 1, self.big),
            ("acme", self.acme, self.repo, 2, self.acme),
            ("scripts", "scripts", f"{self.repo}/scripts", 1, self.acme),
            ("agent", "agent-x", f"{self.repo}/.claude/worktrees/agent-x", 1, self.acme),
            ("alpha", self.alpha, f"{self.repo}/sub", 1, self.alpha),
            ("home", f"dev-{self.tag}", self.home, 1, f"dev-{self.tag}"),
        ]

    @property
    def restricted(self) -> str:
        return f"{self.prefix}-dev-laptop"

    @property
    def full(self) -> str:
        return f"{self.prefix}-dev-full"

    def allowlist_before(self) -> list[str]:
        return [
            self.win,
            self.acme,
            self.win_lc,
            self.big.lower(),
            self.twin,
            self.alpha,
            self.win + r"\tickets",
        ]

    def allowlist_after(self) -> list[str]:
        return [self.big, self.acme, self.alpha]

    def skills_before(self) -> dict[str, str]:
        p = self.prefix
        return {
            f"{p}-triage": f"project:{self.win}",
            f"{p}-notes": f"project:{self.big.lower()}",
            f"{p}-deploy": f"project:{self.acme}",
            f"{p}-global": "global",
        }

    def skills_after(self) -> dict[str, str]:
        p = self.prefix
        return {
            f"{p}-triage": f"project:{self.big}",
            f"{p}-notes": f"project:{self.big}",
            f"{p}-deploy": f"project:{self.acme}",
            f"{p}-global": "global",
        }

    def alias_text(self) -> str:
        return f"# generic test aliases\n{self.twin} = {self.big}\n"

    def config_before(self) -> list[tuple[str, str]]:
        """config_lane rows (scope, file_key) mirrored from the restricted device. The
        first one collides: CLAUDE.md is already mirrored under the canonical scope."""
        return [
            (f"project:{self.win}", "CLAUDE.md"),
            (f"project:{self.big}", "CLAUDE.md"),
            (f"project:{self.big.lower()}", "AGENTS.md"),
            (f"project:{self.acme}", "CLAUDE.md"),
            ("global", "CLAUDE.md"),
        ]

    def config_after(self) -> list[tuple[str, str]]:
        return sorted(
            [
                (f"project:{self.win}", "CLAUDE.md"),  # kept: the new key already exists
                (f"project:{self.big}", "CLAUDE.md"),
                (f"project:{self.big}", "AGENTS.md"),
                (f"project:{self.acme}", "CLAUDE.md"),
                ("global", "CLAUDE.md"),
            ]
        )


def _seed(conn, w: World) -> None:
    n = 0
    for key, label, cwd, count, _canonical in w.episode_plan():
        for i in range(count):
            n += 1
            meta = Jsonb({"cwd": cwd, "platform": "test"} if cwd else {"platform": "test"})
            eid = conn.execute(
                "INSERT INTO episodes (session_id, sequence, project, content, metadata) "
                "VALUES (%s, 1, %s, 'a turn', %s) RETURNING id",
                (f"{w.prefix}-{n}", label, meta),
            ).fetchone()[0]
            w.eps.setdefault(key, eid)
            w.eps[f"{key}#{i}"] = eid
    e = w.eps
    for uid, episodes in (
        ("one", [e["win"]]),
        ("mixed", [e["win_lc"], e["scripts"]]),
        ("explicit", [e["alpha"]]),
        ("strids", [str(e["tickets"]), str(e["acme"])]),
    ):
        conn.execute(
            "INSERT INTO kg_relationships (uuid, group_id, src_uuid, tgt_uuid, name, fact, "
            "  episodes, t_valid) VALUES (%s, %s, 'ent-x', 'ent-y', 'RELATES', 'a fact', %s, "
            "  '2026-01-01T00:00:00+00:00')",
            (f"{w.prefix}-{uid}", w.prefix, Jsonb(episodes)),
        )
    for uid, project in (("ent-1", w.win + r"\tickets"), ("ent-2", w.alpha)):
        conn.execute(
            "INSERT INTO kg_entities (uuid, group_id, project, name) VALUES (%s, %s, %s, 'x')",
            (f"{w.prefix}-{uid}", w.prefix, project),
        )
    conn.execute(
        "INSERT INTO notes (owner_id, group_id, project, type, hook, body) "
        "VALUES ('default', %s, %s, 'project', 'a hook', 'a body')",
        (w.prefix, w.big.lower()),
    )
    conn.execute(
        "INSERT INTO timeline_events (t_valid, fact, source, source_ref, project) "
        "VALUES (now(), 'an event', 'test', %s, %s)",
        (f"{w.prefix}-tl-1", w.win_lc),
    )
    conn.execute(
        "INSERT INTO preferences (owner_id, group_id, project, pref, polarity) "
        "VALUES ('default', %s, %s, 'prefers tabs', 'like')",
        (w.prefix, w.twin),
    )
    # A table outside public (the script finds tables by catalog, so prod-only tables are
    # covered) and one whose unique index on project would make a relabel collide.
    conn.execute(f'CREATE SCHEMA "{w.schema}"')
    conn.execute(f'CREATE TABLE "{w.schema}".jobs (id serial PRIMARY KEY, project text)')
    conn.execute(f'CREATE TABLE "{w.schema}".registry (project text UNIQUE)')
    conn.execute(f'INSERT INTO "{w.schema}".jobs (project) VALUES (%s), (%s)', (w.win, w.alpha))
    conn.execute(f'INSERT INTO "{w.schema}".registry (project) VALUES (%s)', (w.win,))

    register_device(
        conn,
        f"fake-test-token-{w.tag}",
        trust="restricted",
        projects=w.allowlist_before(),
        surface_id=w.restricted,
    )
    register_device(conn, f"fake-test-token-full-{w.tag}", trust="full", surface_id=w.full)
    for name, scope in w.skills_before().items():
        conn.execute(
            "INSERT INTO skills_lane.skill_registry (name, description, body, scope, status) "
            "VALUES (%s, 'a test skill', 'body', %s, 'active')",
            (name, scope),
        )
    for scope, file_key in w.config_before():
        conn.execute(
            "INSERT INTO config_lane.config_registry "
            "  (surface_id, file_key, abs_path, content, content_hash, scope) "
            "VALUES (%s, %s, %s, 'text', 'hash', %s)",
            (w.restricted, file_key, f"/x/{file_key}", scope),
        )


def _wipe(conn, w: World) -> None:
    like = f"{w.prefix}%"
    conn.execute("DELETE FROM kg_relationships WHERE uuid LIKE %s", (like,))
    conn.execute("DELETE FROM kg_entities WHERE uuid LIKE %s", (like,))
    conn.execute("DELETE FROM episodes WHERE session_id LIKE %s", (like,))
    conn.execute("DELETE FROM notes WHERE group_id = %s", (w.prefix,))
    conn.execute("DELETE FROM preferences WHERE group_id = %s", (w.prefix,))
    conn.execute("DELETE FROM timeline_events WHERE source_ref LIKE %s", (like,))
    conn.execute("DELETE FROM surfaces WHERE surface_id LIKE %s", (like,))
    conn.execute("DELETE FROM skills_lane.skill_registry WHERE name LIKE %s", (like,))
    conn.execute("DELETE FROM skills_lane.skill_history WHERE name LIKE %s", (like,))
    conn.execute("DELETE FROM config_lane.config_registry WHERE surface_id LIKE %s", (like,))
    conn.execute(f'DROP SCHEMA IF EXISTS "{w.schema}" CASCADE')


@pytest.fixture()
def world(conn):
    w = World(tag=uuid.uuid4().hex[:8])
    try:
        _seed(conn, w)
        yield w
    finally:
        _wipe(conn, w)


@pytest.fixture()
def aliases(tmp_path, world) -> Path:
    """The alias file lives outside the repository, as the script's docstring asks."""
    path = tmp_path / "relabel-aliases.txt"
    path.write_text(world.alias_text(), encoding="utf-8")
    return path


def _snapshot(conn, w: World) -> dict[str, list]:
    """Everything this test seeded that the relabel could touch, in a stable order."""
    like = f"{w.prefix}%"
    q = {
        "episodes": (
            "SELECT session_id, project, metadata->>'cwd' FROM episodes "
            "WHERE session_id LIKE %s ORDER BY id",
            (like,),
        ),
        "kg_relationships": (
            "SELECT uuid, source_projects FROM kg_relationships WHERE uuid LIKE %s ORDER BY uuid",
            (like,),
        ),
        "kg_entities": (
            "SELECT uuid, project FROM kg_entities WHERE uuid LIKE %s ORDER BY uuid",
            (like,),
        ),
        "notes": ("SELECT project FROM notes WHERE group_id = %s", (w.prefix,)),
        "timeline_events": (
            "SELECT project FROM timeline_events WHERE source_ref LIKE %s",
            (like,),
        ),
        "preferences": ("SELECT project FROM preferences WHERE group_id = %s", (w.prefix,)),
        "jobs": (f'SELECT id, project FROM "{w.schema}".jobs ORDER BY id', ()),
        "registry": (f'SELECT project FROM "{w.schema}".registry', ()),
        "surfaces": (
            "SELECT surface_id, allowed_projects, updated_at FROM surfaces "
            "WHERE surface_id LIKE %s ORDER BY surface_id",
            (like,),
        ),
        "skills": (
            "SELECT name, scope, updated_at FROM skills_lane.skill_registry "
            "WHERE name LIKE %s ORDER BY name",
            (like,),
        ),
        "skill_history": (
            "SELECT count(*) FROM skills_lane.skill_history WHERE name LIKE %s",
            (like,),
        ),
        "config": (
            "SELECT scope, file_key FROM config_lane.config_registry "
            "WHERE surface_id LIKE %s ORDER BY scope, file_key",
            (like,),
        ),
    }
    return {
        k: [tuple(r) for r in conn.execute(sql, args).fetchall()] for k, (sql, args) in q.items()
    }


def _source_projects(conn, w: World) -> dict[str, set[str] | None]:
    return {
        uid.removeprefix(f"{w.prefix}-"): (None if sp is None else set(sp))
        for uid, sp in conn.execute(
            "SELECT uuid, source_projects FROM kg_relationships WHERE uuid LIKE %s",
            (f"{w.prefix}%",),
        ).fetchall()
    }


def _labels_by_key(conn, w: World) -> dict[str, set[str]]:
    rows = dict(
        conn.execute(
            "SELECT id, project FROM episodes WHERE session_id LIKE %s", (f"{w.prefix}%",)
        ).fetchall()
    )
    out: dict[str, set[str]] = {}
    for key, *_rest in w.episode_plan():
        ids = [eid for k, eid in w.eps.items() if k.startswith(f"{key}#")]
        out[key] = {rows[i] for i in ids}
    return out


def _run(argv: list[str], capsys) -> str:
    assert rp.main(argv) == 0
    return capsys.readouterr().out


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------


def test_dry_run_reports_the_mapping_and_writes_nothing(conn, db_url, world, aliases, capsys):
    w = world
    before = _snapshot(conn, w)
    assert _source_projects(conn, w)["one"] == {w.win}

    plain = _run(["--dry-run", "--db-url", db_url], capsys)
    assert _snapshot(conn, w) == before
    # Without the alias file the other machine's checkout keeps its own name.
    assert f"| {w.twin} |" not in plain
    assert f"| {w.win} | {w.win_root} | {w.big} | 2 | path → its name |" in plain

    out = aliases.parent / "dry-run.md"
    text = _run(
        ["--dry-run", "--aliases", str(aliases), "--db-url", db_url, "--out", str(out)], capsys
    )
    assert _snapshot(conn, w) == before, "a dry run wrote to the database"
    assert out.read_text(encoding="utf-8") == text

    rows = text.splitlines()
    for line in (
        f"| {w.win} | {w.win_root} | {w.big} | 2 | path → its name |",
        f"| {w.win_lc} | {w.win_root} | {w.big} | 1 | path → its name |",
        f"| {w.win}\\tickets | {w.win_root} | {w.big} | 1 | subfolder → project root |",
        f"| {w.win}\\tickets\\lab\\out | {w.win_root} | {w.big} | 1 | subfolder → project root |",
        f"| {w.big.lower()} |  | {w.big} | 1 | case merge |",
        f"| {w.twin} | {w.home}/code/{w.twin} | {w.big} | 1 | alias |",
        f"| scripts | {w.repo} | {w.acme} | 1 | subfolder → project root |",
        f"| agent-x | {w.repo} | {w.acme} | 1 | subfolder → project root |",
    ):
        assert line in rows, line
    mine = [r for r in rows if w.tag in r and r.startswith("| ")]
    assert not any(
        r.startswith((f"| {w.alpha} |", f"| {w.big} |", f"| dev-{w.tag} |")) for r in mine
    )
    big_rows = sum(n for _k, _l, _c, n, canonical in w.episode_plan() if canonical == w.big)
    assert f"  {big_rows:>7}  {w.big}" in rows
    assert f"  {w.restricted}: {w.allowlist_before()} -> {w.allowlist_after()}" in rows
    assert f"  {w.full}: [] -> []  (unchanged)" in rows
    assert f"  {w.prefix}-triage: project:{w.win} -> project:{w.big}" in rows
    assert f"  {w.prefix}-deploy: project:{w.acme} -> project:{w.acme}  (unchanged)" in rows
    assert f'"{w.schema}"."jobs": ' in text
    config = rows[rows.index("config_lane.config_registry.scope:") + 1 :]
    for line in (
        f"        1  project:{w.win} -> project:{w.big}",
        f"        1  project:{w.big.lower()} -> project:{w.big}",
        f"        1  project:{w.big} -> project:{w.big}  (unchanged)",
        f"        1  project:{w.acme} -> project:{w.acme}  (unchanged)",
    ):
        assert line in config, line
    assert f"  {1:>7}  {w.win} -> {w.big}" in rows


# ---------------------------------------------------------------------------
# --apply, then again
# ---------------------------------------------------------------------------


def test_apply_moves_every_label_and_a_second_apply_is_a_no_op(
    conn, db_url, world, aliases, capsys
):
    w = world
    sp_before = _source_projects(conn, w)
    assert sp_before == {
        "one": {w.win},
        "mixed": {w.win_lc, "scripts"},
        "explicit": {w.alpha},
        "strids": {w.win + r"\tickets", w.acme},
    }
    first = _snapshot(conn, w)
    history_before = first["skill_history"]

    plan = aliases.parent / "plan.json"
    dry = _run(
        ["--dry-run", "--aliases", str(aliases), "--db-url", db_url, "--plan-out", str(plan)],
        capsys,
    )
    log = _run(["--apply", "--plan", str(plan), "--db-url", db_url], capsys)

    # episodes: every (label, cwd) group moved to its canonical name; cwd untouched.
    expected = {key: {canonical} for key, _l, _c, _n, canonical in w.episode_plan()}
    assert _labels_by_key(conn, w) == expected
    cwd_of = {sid: cwd for sid, _p, cwd in first["episodes"]}
    assert {sid: cwd for sid, _p, cwd in _snapshot(conn, w)["episodes"]} == cwd_of

    # The 056 relabel trigger recomputed every fact citing a moved episode.
    assert _source_projects(conn, w) == {
        "one": {w.big},
        "mixed": {w.big, w.acme},
        "explicit": {w.alpha},
        "strids": {w.big, w.acme},
    }

    snap = _snapshot(conn, w)
    assert snap["kg_entities"] == [(f"{w.prefix}-ent-1", w.big), (f"{w.prefix}-ent-2", w.alpha)]
    assert snap["notes"] == [(w.big,)]
    assert snap["timeline_events"] == [(w.big,)]
    assert snap["preferences"] == [(w.big,)]
    assert [p for _id, p in snap["jobs"]] == [w.big, w.alpha]
    # A unique index on project could collide: the dry run reports that table and the
    # plan leaves it out, so --apply never touches it.
    assert snap["registry"] == [(w.win,)]
    assert f'"{w.schema}"."registry": 1 rows WOULD change, but --apply SKIPS' in dry
    assert '"registry"' not in log

    allow = {sid: list(a) for sid, a, _u in snap["surfaces"]}
    assert allow == {w.full: [], w.restricted: w.allowlist_after()}
    # config_lane: (surface_id, scope, file_key) is the key, so a file already mirrored
    # under the canonical scope keeps that row and the old one stays for its owner.
    assert sorted(snap["config"]) == w.config_after()
    assert f"config_registry: project:{w.big.lower()} -> project:{w.big}: 1 rows" in log
    assert f"config_registry: project:{w.win} -> project:{w.big}: 0 rows, 1 kept" in log
    assert {n: s for n, s, _u in snap["skills"]} == w.skills_after()
    # A scope change is not a body change: the history trigger stays quiet.
    assert snap["skill_history"] == history_before

    # Running the same plan again writes nothing.
    again = _run(["--apply", "--plan", str(plan), "--db-url", db_url], capsys)
    mentions = [ln for ln in again.splitlines() if w.tag in ln]
    assert mentions
    for line in mentions:
        assert line.endswith((": 0 rows", ": 0 rows, 1 kept")) or ": already " in line, line
    assert _snapshot(conn, w) == snap

    dry = _run(["--dry-run", "--aliases", str(aliases), "--db-url", db_url], capsys)
    assert not [r for r in dry.splitlines() if r.startswith("| ") and w.tag in r]
    assert f"  {w.restricted}: {w.allowlist_after()} -> {w.allowlist_after()}  (unchanged)" in dry


def test_apply_writes_the_approved_plan_and_nothing_that_appeared_since(
    conn, db_url, world, aliases, capsys
):
    """Between the dry run and --apply, a backup copy of a table is made and new sessions
    arrive (a label the plan never saw, and more rows of a group it did). --apply writes
    exactly the approved plan: the copy is never touched, the new label stays, and the
    planned group moves whole."""
    w = world
    plan = aliases.parent / "plan.json"
    _run(
        ["--dry-run", "--aliases", str(aliases), "--db-url", db_url, "--plan-out", str(plan)],
        capsys,
    )
    conn.execute(
        f'CREATE TABLE "{w.schema}".jobs_backup AS SELECT id, project FROM "{w.schema}".jobs'
    )
    later = w.win + r"\later"
    for n, (label, cwd) in enumerate([(later, later), (w.win, w.win)]):
        conn.execute(
            "INSERT INTO episodes (session_id, sequence, project, content, metadata) "
            "VALUES (%s, 1, %s, 'a later turn', %s)",
            (f"{w.prefix}-later-{n}", label, Jsonb({"cwd": cwd})),
        )

    log = _run(["--apply", "--plan", str(plan), "--db-url", db_url], capsys)

    backup = conn.execute(f'SELECT project FROM "{w.schema}".jobs_backup ORDER BY id').fetchall()
    assert [p for (p,) in backup] == [w.win, w.alpha]
    assert "jobs_backup" not in log
    stored = dict(
        conn.execute(
            "SELECT session_id, project FROM episodes WHERE session_id LIKE %s",
            (f"{w.prefix}-later-%",),
        ).fetchall()
    )
    assert stored == {f"{w.prefix}-later-0": later, f"{w.prefix}-later-1": w.big}


def test_a_re_run_after_an_interruption_still_maps_allowlists_and_scopes(
    conn, db_url, world, aliases, capsys
):
    w = world
    plan = aliases.parent / "plan.json"
    _run(
        ["--dry-run", "--aliases", str(aliases), "--db-url", db_url, "--plan-out", str(plan)],
        capsys,
    )
    _run(["--apply", "--plan", str(plan), "--db-url", db_url], capsys)
    # Drop the two rows that still hold w.win (both collisions), so that no table names
    # the old labels any more.
    conn.execute(f'DELETE FROM "{w.schema}".registry')
    conn.execute(
        "DELETE FROM config_lane.config_registry WHERE surface_id = %s AND scope = %s",
        (w.restricted, f"project:{w.win}"),
    )
    conn.execute(
        "UPDATE config_lane.config_registry SET scope = %s WHERE surface_id = %s AND scope = %s",
        (f"project:{w.big.lower()}", w.restricted, f"project:{w.big}"),
    )
    # The state an interrupted run leaves: episodes and tables moved, the final
    # surfaces + skills transaction never committed.
    conn.execute(
        "UPDATE surfaces SET allowed_projects = %s WHERE surface_id = %s",
        (w.allowlist_before(), w.restricted),
    )
    for name, scope in w.skills_before().items():
        conn.execute(
            "UPDATE skills_lane.skill_registry SET scope = %s WHERE name = %s", (scope, name)
        )

    _run(["--apply", "--plan", str(plan), "--db-url", db_url], capsys)

    snap = _snapshot(conn, w)
    assert {sid: list(a) for sid, a, _u in snap["surfaces"]}[w.restricted] == w.allowlist_after()
    assert {n: s for n, s, _u in snap["skills"]} == w.skills_after()
    assert sorted(snap["config"]) == [c for c in w.config_after() if w.win not in c[0]]


def test_an_allowlist_or_scope_edited_since_the_dry_run_is_left_alone_and_apply_fails(
    conn, db_url, world, aliases, capsys
):
    """An allowlist edited between the dry run and --apply (here only reordered) is left
    as it is: the plan was decided for the old one. The labels still move, so its old
    path entries now match nothing: --apply says so and exits non-zero instead of 0, so
    an operator can't miss the device that just lost access."""
    w = world
    plan = aliases.parent / "plan.json"
    _run(
        ["--dry-run", "--aliases", str(aliases), "--db-url", db_url, "--plan-out", str(plan)],
        capsys,
    )
    edited = list(reversed(w.allowlist_before()))
    conn.execute(
        "UPDATE surfaces SET allowed_projects = %s WHERE surface_id = %s", (edited, w.restricted)
    )
    name, _scope = next(iter(w.skills_before().items()))
    conn.execute("UPDATE skills_lane.skill_registry SET scope = 'global' WHERE name = %s", (name,))

    assert rp.main(["--apply", "--plan", str(plan), "--db-url", db_url]) == 1
    out, err = capsys.readouterr()
    assert f"surfaces: {w.restricted}: changed since the dry run, left alone" in out
    assert w.restricted in err and name in err and "changed since the dry run" in err
    snap = _snapshot(conn, w)
    assert {sid: list(a) for sid, a, _u in snap["surfaces"]}[w.restricted] == edited
    assert {n: s for n, s, _u in snap["skills"]}[name] == "global"
    expected = {key: {canonical} for key, _l, _c, _n, canonical in w.episode_plan()}
    assert _labels_by_key(conn, w) == expected

    # Once the plan has run, re-running it is not drift: 'already' lines, exit 0.
    conn.execute(
        "UPDATE surfaces SET allowed_projects = %s WHERE surface_id = %s",
        (w.allowlist_after(), w.restricted),
    )
    conn.execute(
        "UPDATE skills_lane.skill_registry SET scope = %s WHERE name = %s",
        (w.skills_after()[name], name),
    )
    again = _run(["--apply", "--plan", str(plan), "--db-url", db_url], capsys)
    assert f"skill_registry: {name}: already {w.skills_after()[name]}" in again


# ---------------------------------------------------------------------------
# The offline inventory: --inventory-sql through psql, then --dry-run --inventory
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("psql") is None, reason="psql not installed")
def test_the_offline_inventory_is_the_live_one(conn, db_url, world, aliases, tmp_path, capsys):
    """Prod is reached through ``docker exec ... psql``, not a DSN: the one statement
    --inventory-sql prints must read exactly what read_inventory reads, and the dry run
    planned from its output must be the live dry run, byte for byte. It writes nothing."""
    w = world
    before = _snapshot(conn, w)
    sql = tmp_path / "inv.sql"
    sql.write_text(_run(["--inventory-sql"], capsys), encoding="utf-8")
    psql = subprocess.run(
        ["psql", db_url, "-X", "-tA", "-v", "ON_ERROR_STOP=1", "-f", str(sql)],
        capture_output=True,
        text=True,
        check=True,
    )
    inv_file = tmp_path / "inv.json"
    inv_file.write_text(psql.stdout, encoding="utf-8")
    assert _snapshot(conn, w) == before

    offline = rp.inventory_from_json(psql.stdout)
    live = rp.read_inventory(conn)
    assert offline.episode_groups == live.episode_groups
    assert offline.tables == live.tables
    assert offline.surfaces == live.surfaces
    assert offline.skills == live.skills
    assert offline.config_scopes == live.config_scopes
    assert offline.full_surfaces == live.full_surfaces
    assert w.full in live.full_surfaces and w.restricted not in live.full_surfaces
    assert offline.tables[f'"{w.schema}"."jobs"'] == {w.win: 1, w.alpha: 1}
    assert offline.episode_groups[(w.big.lower(), None)] == 1

    live_text = _run(["--dry-run", "--aliases", str(aliases), "--db-url", db_url], capsys)
    offline_text = _run(
        ["--dry-run", "--inventory", str(inv_file), "--aliases", str(aliases)], capsys
    )
    assert offline_text == live_text
    assert f"| {w.twin} | {w.home}/code/{w.twin} | {w.big} | 1 | alias |" in offline_text
