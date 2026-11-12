"""The test database at the schema states 066, 067 and 069 ran on, and back.

The suite's database is built from every file in schema/, so it is past 069: the four
one-memory tables have no ``group_id``. The tests of those files as migrations need the
database they were written for, and the runner gives each slot one database and no other,
so these rebuild that state in place, on emptied tables, and put the database back the way
apply_schema.sh left it afterwards. Each rebuild runs the older files themselves: the column
is added back, and 017, 035, 041 and 066 build what they build while it exists.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

SCHEMA = Path(__file__).resolve().parents[2] / "schema"
SQL_067 = SCHEMA / "067_kg_one_memory.sql"
SQL_069 = SCHEMA / "069_one_memory_cleanup.sql"
TABLES = ("kg_entities", "kg_relationships", "notes", "preferences")
#: 067's owner-only indexes; at 066 they did not exist yet.
ANY_INDEXES = ("kg_rel_pair_any", "kg_rel_tgt_any", "kg_entities_norm_any")
#: The names 069 builds again on owner_id alone; the older files build them with group_id.
REBUILT_BY_069 = ("kg_entities_one_self", "notes_live_idx", "preferences_live_idx")
#: The files that build what leads with group_id, in apply order.
OLDER_FILES = ("017_kg_postgres.sql", "035_preferences.sql", "041_notes.sql")


def psql(
    db_url: str,
    path: Path,
    *,
    check: bool = True,
    env: dict[str, str] | None = None,
    sql: str | None = None,
    stop: bool = True,
) -> subprocess.CompletedProcess:
    """Pipe a file to psql on stdin, the way a migration is applied to a real database:
    ``sql`` sends that text in its place, and ``stop=False`` leaves ON_ERROR_STOP unset, the
    way a client that goes on after an error would.

    The result and the error name the file, never the URL: a failure's message or repr must
    not print the database credential."""
    args = ["psql", db_url, "-X", "-q", *(["-v", "ON_ERROR_STOP=1"] if stop else [])]
    if sql is None:
        with path.open() as f:
            ran = subprocess.run(args, stdin=f, capture_output=True, text=True, env=env)
    else:
        ran = subprocess.run(args, input=sql, capture_output=True, text=True, env=env)
    code = ran.returncode
    if check and code:
        raise AssertionError(f"psql {path.name} exited {code}: {ran.stderr[-2000:]}")
    return subprocess.CompletedProcess(["psql", path.name], code, ran.stdout, ran.stderr)


def stamp(conn) -> str | None:
    row = conn.execute("SELECT value FROM synapse_meta WHERE key = 'schema_version'").fetchone()
    return None if row is None else row[0]


def set_stamp(conn, value: str) -> None:
    conn.execute("UPDATE synapse_meta SET value = %s WHERE key = 'schema_version'", (value,))


def wipe(conn) -> None:
    """Empty the four tables and drop the 067 ledger."""
    conn.execute(
        "TRUNCATE kg_relationships, kg_entities, notes, preferences RESTART IDENTITY CASCADE"
    )
    for (t,) in conn.execute(
        "SELECT tablename FROM pg_tables WHERE tablename LIKE 'kg_pile_merge_067_%'"
    ).fetchall():
        conn.execute(f'DROP TABLE IF EXISTS "{t}"')


def to_066(conn, db_url: str) -> None:
    """The database as 066 left it, empty: ``group_id`` on the four tables with no default,
    the nine indexes that lead with it, the insert trigger, no owner-only index, stamped 066."""
    wipe(conn)
    for t in TABLES:
        conn.execute(f"ALTER TABLE {t} ADD COLUMN IF NOT EXISTS group_id TEXT NOT NULL")
        conn.execute(f"ALTER TABLE {t} ALTER COLUMN group_id DROP DEFAULT")
    for ix in (*REBUILT_BY_069, *ANY_INDEXES):
        conn.execute(f"DROP INDEX IF EXISTS {ix}")
    for name in (*OLDER_FILES, "066_kg_fact_audience.sql"):
        psql(db_url, SCHEMA / name)
    set_stamp(conn, "066")


def to_068(conn, db_url: str) -> None:
    """The database 069 is applied to, empty: 066, then 067, stamped 068."""
    to_066(conn, db_url)
    psql(db_url, SQL_067)
    set_stamp(conn, "068")


def to_069(conn, db_url: str, before: str | None) -> None:
    """Back to what apply_schema.sh leaves: the four tables emptied, 067 finished, 069
    applied, and the stamp as it was ``before`` when that was higher."""
    wipe(conn)
    psql(db_url, SQL_067)
    psql(db_url, SQL_069)
    if before is not None and before > (stamp(conn) or ""):
        set_stamp(conn, before)
