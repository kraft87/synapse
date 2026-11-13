"""scripts/apply_schema.sh as a runner: the ledger, what an upgrade applies, the markers.

The script under test is the real one with its FILES list swapped for eight stand-ins, each
a few lines that note the file ran. So a test says which files ran, in which order, on which
kind of database, without the real schema's extensions or its history. The real list on a
real database is CI's provisioning step, and tests/test_schema.py reads what it left.

The stand-ins run in one scratch database on the test server, emptied before every test.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "apply_schema.sh"

NAMES = [
    "001_base.sql",  # episodes, and the table the stand-ins note their run in
    "002_meta.sql",  # synapse_meta: no stamp can be written before it
    "003_plain.sql",
    "004_optional.sql",
    "005_plain.sql",
    "006_manual.sql",  # required and manual: an upgrade stops before it
    "007_plain.sql",
    "008_cleanup.sql",  # optional and manual: an upgrade leaves it alone
]
OPTIONAL = "-- schema-check: optional"
MANUAL = "-- schema-apply: manual"
MARKERS = {
    "004_optional.sql": (OPTIONAL,),
    "006_manual.sql": (MANUAL,),
    "008_cleanup.sql": (OPTIONAL, MANUAL),
}
ALL = [n[:3] for n in NAMES]


class Stand:
    """The script with a stand-in list, its schema directory, and the scratch database."""

    def __init__(self, tmp: Path, url: str, names: list[str] = NAMES) -> None:
        self.url = url
        self.schema = tmp / "schema"
        self.schema.mkdir()
        listed = "FILES=(\n" + "".join(f"  {n}\n" for n in names) + ")\n"
        text, swapped = re.subn(
            r"^FILES=\(\n.*?^\)\n", lambda _: listed, _SCRIPT.read_text(), flags=re.M | re.S
        )
        assert swapped == 1, "apply_schema.sh no longer holds one FILES=( ... ) list"
        self.script = tmp / "apply_schema.sh"
        self.script.write_text(text)
        for name in names:
            self.write(name, MARKERS.get(name, ()))

    def write(self, name: str, markers: tuple[str, ...] = (), body: str | None = None) -> None:
        lines = [f"-- stand-in for {name}", *markers]
        if name == NAMES[0]:
            # Plain CREATE, like the real 001: a second run of it fails.
            lines.append("CREATE TABLE episodes (id int);")
            lines.append("CREATE TABLE ran (n serial PRIMARY KEY, file text NOT NULL);")
        if name == NAMES[1]:
            lines.append("CREATE TABLE synapse_meta (key text PRIMARY KEY, value text NOT NULL);")
        lines.append(body or f"INSERT INTO ran (file) VALUES ('{name}');")
        (self.schema / name).write_text("\n".join(lines) + "\n")

    def run(self, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        base = {k: v for k, v in os.environ.items() if not k.startswith("SYNAPSE_")}
        return subprocess.run(
            ["bash", str(self.script), *args, self.url],
            capture_output=True,
            text=True,
            env={**base, "SCHEMA_DIR": str(self.schema), **(env or {})},
        )

    def sql(self, query: str) -> list[tuple]:
        with psycopg.connect(self.url, autocommit=True) as c:
            cur = c.execute(query)
            return cur.fetchall() if cur.description else []

    def ran(self) -> list[str]:
        return [f for (f,) in self.sql("SELECT file FROM ran ORDER BY n")]

    def forget_runs(self) -> None:
        self.sql("TRUNCATE ran")

    def ledger(self) -> dict[str, tuple[str, bool]]:
        """{number: (filename, was run by the script)}"""
        rows = self.sql("SELECT version, filename, applied_at IS NOT NULL FROM schema_migrations")
        return {v: (f, run) for v, f, run in rows}

    def has_ledger(self) -> bool:
        return self.sql("SELECT to_regclass('public.schema_migrations') IS NOT NULL")[0][0]

    def stamp(self) -> str | None:
        rows = self.sql("SELECT value FROM synapse_meta WHERE key = 'schema_version'")
        return rows[0][0] if rows else None

    def set_stamp(self, value: str) -> None:
        self.sql(f"UPDATE synapse_meta SET value = '{value}' WHERE key = 'schema_version'")

    def unrecord(self, *numbers: str) -> None:
        self.sql(
            f"DELETE FROM schema_migrations WHERE version IN ({', '.join(map(repr, numbers))})"
        )

    def provisioned(self, *, episodes: bool = True) -> Stand:
        """Every file applied, as a first boot leaves it; ``episodes`` puts data in it."""
        ran = self.run()
        assert ran.returncode == 0, ran.stderr
        if episodes:
            self.sql("INSERT INTO episodes VALUES (1)")
        self.forget_runs()
        return self

    def before_the_ledger(self, stamp: str) -> Stand:
        """A database the old script built: a schema, the stamp, and no ledger."""
        self.provisioned()
        self.sql("DROP TABLE schema_migrations")
        self.set_stamp(stamp)
        return self


@pytest.fixture(scope="module")
def scratch_url(conn, db_url):
    if shutil.which("psql") is None:  # pragma: no cover - CI installs the client
        pytest.skip("psql not installed")
    name = f"apply_schema_{os.getpid()}_test"
    conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    conn.execute(f'CREATE DATABASE "{name}"')
    yield urlunsplit(urlsplit(db_url)._replace(path=f"/{name}"))
    conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture()
def stand(tmp_path, scratch_url):
    with psycopg.connect(scratch_url, autocommit=True) as c:
        c.execute("DROP SCHEMA public CASCADE")
        c.execute("CREATE SCHEMA public")
    return Stand(tmp_path, scratch_url)


# ---------------------------------------------------------------------------
# An empty database, and a run after it
# ---------------------------------------------------------------------------


def test_an_empty_database_gets_every_file_once_in_order_and_a_row_for_each(stand):
    """Manual files too: there is nothing in it to be careful with."""
    ran = stand.run()
    assert ran.returncode == 0, ran.stderr
    assert stand.ran() == NAMES
    assert stand.ledger() == {n[:3]: (n, True) for n in NAMES}
    assert stand.stamp() == "008"


def test_a_second_run_applies_nothing(stand):
    stand.provisioned(episodes=False)
    ran = stand.run()
    assert ran.returncode == 0, ran.stderr
    assert stand.ran() == []
    assert "0 applied now" in ran.stdout


def test_a_file_with_no_row_runs_and_no_other(stand):
    stand.provisioned().unrecord("003")
    assert stand.run().returncode == 0
    assert stand.ran() == ["003_plain.sql"]
    assert stand.ledger()["003"] == ("003_plain.sql", True)


def test_a_file_that_fails_gets_no_row_and_the_next_run_takes_up_there(stand):
    """What an interrupted first boot leaves, and how it is finished: the files before the
    failure are not applied again (001 could not be), and the manual one still runs in
    turn, since the database holds no episodes."""
    stand.write("005_plain.sql", body="SELECT 1 / 0;")
    ran = stand.run()
    assert ran.returncode not in (0, 2)
    assert stand.ran() == NAMES[:4]
    assert sorted(stand.ledger()) == ALL[:4]
    assert stand.stamp() == "004"

    stand.write("005_plain.sql")
    ran = stand.run()
    assert ran.returncode == 0, ran.stderr
    assert stand.ran() == NAMES
    assert sorted(stand.ledger()) == ALL


def test_a_psqlrc_that_turns_autocommit_off_does_not_roll_the_files_back(stand, tmp_path):
    rc = tmp_path / "psqlrc"
    rc.write_text("\\set AUTOCOMMIT off\n")
    ran = stand.run(env={"PSQLRC": str(rc)})
    assert ran.returncode == 0, ran.stderr
    assert stand.ran() == NAMES
    assert sorted(stand.ledger()) == ALL


def test_the_embedding_width_still_reaches_the_files(stand):
    """As a psql variable, and substituted for the literal 2048 in the file's text."""
    stand.write(
        "003_plain.sql",
        body="INSERT INTO ran (file) VALUES (:'embed_dims' || ' ' || 'halfvec(2048)');",
    )
    ran = stand.run(env={"SYNAPSE_EMBED_DIMS": "1024"})
    assert ran.returncode == 0, ran.stderr
    assert stand.ran()[2] == "1024 halfvec(1024)"
    assert sorted(stand.ledger()) == ALL, "the markers are still read after the substitution"


# ---------------------------------------------------------------------------
# A database from before the ledger
# ---------------------------------------------------------------------------


def test_a_database_from_before_the_ledger_is_read_by_its_stamp(stand):
    """The files up to the stamp are recorded and not run (001 cannot run twice); the ones
    above it are applied."""
    stand.before_the_ledger("006")
    ran = stand.run()
    assert ran.returncode == 0, ran.stderr
    assert stand.ran() == ["007_plain.sql"]
    ledger = stand.ledger()
    assert {v: run for v, (_, run) in ledger.items()} == {
        **dict.fromkeys(ALL[:6], False),
        "007": True,
    }
    assert "008" not in ledger, "optional and manual: left for --only"
    assert stand.stamp() == "007"


def test_status_says_what_a_run_would_do_and_changes_nothing(stand):
    stand.before_the_ledger("006")
    ran = stand.run("--status")
    assert ran.returncode == 0, ran.stderr
    assert "schema_version stamp: 006" in ran.stdout
    assert "applied: 6 of 8 files (6 recorded without a run)" in ran.stdout
    assert "  - 007_plain.sql\n" in ran.stdout
    assert "  - 008_cleanup.sql (optional, manual: --only 008)\n" in ran.stdout
    assert not stand.has_ledger()
    assert stand.ran() == []
    assert stand.stamp() == "006"


def test_a_schema_with_neither_ledger_nor_stamp_is_refused(stand):
    """Nothing says which files it has had, and 001 would only fail."""
    stand.before_the_ledger("006")
    stand.sql("DELETE FROM synapse_meta")
    ran = stand.run()
    assert ran.returncode == 1
    assert "no ledger and no schema_version stamp" in ran.stderr
    assert stand.ran() == [] and not stand.has_ledger()


@pytest.mark.parametrize("stamp", ["099", "garbage"])
def test_a_stamp_that_names_no_file_here_starts_no_ledger(stand, stamp):
    """Newer than the checkout, the ledger would start short of files the database has; not a
    number at all, there is nothing to read."""
    stand.before_the_ledger(stamp)
    ran = stand.run()
    assert ran.returncode == 1
    assert stand.ran() == [] and not stand.has_ledger()


# ---------------------------------------------------------------------------
# Manual files on a database that holds episodes
# ---------------------------------------------------------------------------


def test_an_upgrade_leaves_an_optional_manual_file_alone_until_it_is_named(stand):
    stand.provisioned().unrecord("007", "008")
    ran = stand.run()
    assert ran.returncode == 0, ran.stderr
    assert stand.ran() == ["007_plain.sql"]
    assert "--only 008" in ran.stdout
    assert "008" not in stand.ledger()

    ran = stand.run("--only", "008")
    assert ran.returncode == 0, ran.stderr
    assert stand.ran() == ["007_plain.sql", "008_cleanup.sql"]
    assert stand.ledger()["008"] == ("008_cleanup.sql", True)
    assert stand.stamp() == "008"


def test_an_upgrade_stops_before_a_required_manual_file(stand):
    stand.provisioned().unrecord("005", "006", "007")
    ran = stand.run()
    assert ran.returncode == 2
    assert "--only 006" in ran.stderr
    assert stand.ran() == ["005_plain.sql"], "the file before it ran, the one after did not"

    refused = stand.run("--only", "007")
    assert refused.returncode == 1 and "006_manual.sql comes before it" in refused.stderr
    assert stand.ran() == ["005_plain.sql"]

    assert stand.run("--only", "006").returncode == 0
    assert stand.run().returncode == 0
    assert stand.ran() == ["005_plain.sql", "006_manual.sql", "007_plain.sql"]
    assert sorted(stand.ledger()) == ALL


def test_only_steps_over_nothing_an_upgrade_would_apply(stand):
    """An optional file is still applied by an upgrade, so --only does not jump it; the one
    file it may pass is optional and manual, which an upgrade leaves alone."""
    stand.provisioned().unrecord("004", "005")
    refused = stand.run("--only", "005")
    assert refused.returncode == 1 and "004_optional.sql comes before it" in refused.stderr
    assert stand.ran() == []

    stand.write("004_optional.sql", (OPTIONAL, MANUAL))
    assert stand.run("--only", "005").returncode == 0
    assert stand.ran() == ["005_plain.sql"]
    assert "004" not in stand.ledger()


def test_only_on_a_recorded_file_does_nothing(stand):
    stand.provisioned()
    ran = stand.run("--only", "006")
    assert ran.returncode == 0 and "already recorded" in ran.stdout
    assert stand.ran() == []


# ---------------------------------------------------------------------------
# The stamp, and what the script refuses
# ---------------------------------------------------------------------------


def test_the_stamp_never_moves_down(stand):
    """A file that stamped a higher number itself (067 to 069 do) keeps it."""
    stand.provisioned().unrecord("007")
    stand.set_stamp("050")
    assert stand.run().returncode == 0
    assert stand.ran() == ["007_plain.sql"]
    assert stand.stamp() == "050"


def test_a_number_recorded_under_another_name_is_refused(stand):
    """Two branches took the same number, and the database has had the other one's file."""
    stand.provisioned().unrecord("007")
    stand.sql("UPDATE schema_migrations SET filename = '003_other.sql' WHERE version = '003'")
    ran = stand.run()
    assert ran.returncode == 1
    assert "003 is recorded as 003_other.sql" in ran.stderr
    assert stand.ran() == []
    assert "CLASH: 003 is recorded as 003_other.sql" in stand.run("--status").stdout


@pytest.mark.parametrize(
    ("names", "says"),
    [
        (["001_base.sql", "002_meta.sql", "002_again.sql"], "share the number 002"),
        (["002_meta.sql", "001_base.sql"], "FILES must rise"),
        (["001_base.sql", "0002_meta.sql"], "is not named NNN_name.sql"),
    ],
)
def test_a_list_the_ledger_cannot_key_is_refused_before_the_database_is_touched(
    tmp_path, scratch_url, names, says
):
    ran = Stand(tmp_path, scratch_url, names).run()
    assert ran.returncode == 1 and says in ran.stderr


def test_only_takes_a_number_the_list_has(stand):
    assert stand.run("--only", "123").returncode == 1
    assert stand.run("--only", "8").returncode == 1
    assert not stand.has_ledger()
