"""Schema 068 against a real database: the rename, and the trigger that makes an older
build safe on a renamed database.

068 renames the span ids three parsers wrote under their transcript format (``jsonl:``,
``openai:``, ``cursor-sqlite:``) to the platform's name, and leaves a trigger that rewrites
those spellings on every write. The trigger is what the deploy order rests on: the file is
applied while the previous image is still serving, and that image goes on writing
``jsonl:<uuid>``. Without the rewrite, each turn it re-sends would be stored a second time.

Runs against the shared Postgres test DB (tagged into the ``db`` xdist group via
conftest). Every session id starts with a per-test ``span068-<hex>-`` prefix and each test
deletes exactly its own rows before and after.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import psycopg
import pytest

_MIGRATION = Path(__file__).resolve().parents[1] / "schema" / "068_span_id_prefix_is_platform.sql"

_RENAMES = [
    ("jsonl:{x}", "claude_code:{x}"),
    ("openai:{x}", "chatgpt:{x}"),
    ("cursor-sqlite:{x}:bubble", "cursor:{x}:bubble"),
]
_UNTOUCHED = [
    "claude_code:{x}",
    "codex:msg_{x}",
    "claude_ai:sess:{x}",
    "cursor:{x}:14",
    "hermes:{x}",
    "jsonlx:{x}",  # shares letters with a renamed prefix, is not one
    "{x}",  # a bare id, as the Logfire importer wrote
]


@pytest.fixture()
def tag(conn):
    hx = uuid.uuid4().hex[:12]

    def _wipe():
        conn.execute("DELETE FROM episodes WHERE session_id LIKE %s", (f"span068-{hx}-%",))

    _wipe()
    yield hx
    _wipe()


def _insert(conn, sid: str, seq: int, span: str | None) -> None:
    conn.execute(
        "INSERT INTO episodes (session_id, sequence, content, span_id) VALUES (%s, %s, %s, %s)",
        (sid, seq, f"turn {seq}", span),
    )


def _spans(conn, sid: str) -> list[str | None]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT span_id FROM episodes WHERE session_id = %s ORDER BY sequence", (sid,)
        ).fetchall()
    ]


@pytest.mark.parametrize(("old", "new"), _RENAMES)
def test_an_old_spelling_is_stored_under_the_platform_name(conn, tag, old, new):
    sid = f"span068-{tag}-w"
    _insert(conn, sid, 1, old.format(x=tag))
    assert _spans(conn, sid) == [new.format(x=tag)]


@pytest.mark.parametrize("span", _UNTOUCHED)
def test_every_other_span_id_is_stored_as_sent(conn, tag, span):
    sid = f"span068-{tag}-u"
    _insert(conn, sid, 1, span.format(x=tag))
    _insert(conn, sid, 2, None)
    assert _spans(conn, sid) == [span.format(x=tag), None]


def test_an_update_to_an_old_spelling_is_rewritten_too(conn, tag):
    sid = f"span068-{tag}-up"
    _insert(conn, sid, 1, f"codex:{tag}")
    conn.execute("UPDATE episodes SET span_id = %s WHERE session_id = %s", (f"openai:{tag}", sid))
    assert _spans(conn, sid) == [f"chatgpt:{tag}"]


def test_an_older_builds_duplicate_hits_the_span_index_under_its_own_name(conn, tag):
    # What the ingest route catches and counts as "already stored": the constraint name
    # is the contract (mcp_server/ingest_route.py re-raises any other violation).
    _insert(conn, f"span068-{tag}-a", 1, f"claude_code:{tag}")
    with pytest.raises(psycopg.errors.UniqueViolation) as exc:
        _insert(conn, f"span068-{tag}-b", 1, f"jsonl:{tag}")
    assert exc.value.diag.constraint_name == "episodes_span_id_idx"


def test_a_rerun_of_the_file_renames_what_the_trigger_missed_and_nothing_else(conn, tag):
    # Rows as a database from before 068 holds them: written with the trigger off.
    sid = f"span068-{tag}-m"
    with conn.transaction():
        conn.execute("ALTER TABLE episodes DISABLE TRIGGER episodes_canonical_span_id")
        for seq, (old, _) in enumerate(_RENAMES, start=1):
            _insert(conn, sid, seq, old.format(x=tag))
        # Their own ids: "claude_code:<tag>" beside "jsonl:<tag>" would be the same turn
        # stored twice, which the file refuses (the test after this one).
        for seq, span in enumerate(_UNTOUCHED, start=len(_RENAMES) + 1):
            _insert(conn, sid, seq, span.format(x=f"keep-{tag}"))
        conn.execute("ALTER TABLE episodes ENABLE TRIGGER episodes_canonical_span_id")
    assert _spans(conn, sid)[: len(_RENAMES)] == [old.format(x=tag) for old, _ in _RENAMES]

    conn.execute(_MIGRATION.read_text())

    expected = [new.format(x=tag) for _, new in _RENAMES] + [
        s.format(x=f"keep-{tag}") for s in _UNTOUCHED
    ]
    assert _spans(conn, sid) == expected
    conn.execute(_MIGRATION.read_text())  # and again: a no-op
    assert _spans(conn, sid) == expected


def test_a_turn_stored_under_both_spellings_stops_the_file_and_changes_nothing(conn, tag):
    # The state a build newer than its database leaves: it looked for claude_code:<id>,
    # missed the stored jsonl:<id>, and stored the turn again.
    sid = f"span068-{tag}-c"
    with conn.transaction():
        conn.execute("ALTER TABLE episodes DISABLE TRIGGER episodes_canonical_span_id")
        _insert(conn, sid, 1, f"jsonl:{tag}")
        _insert(conn, sid, 2, f"jsonl:other-{tag}")
        conn.execute("ALTER TABLE episodes ENABLE TRIGGER episodes_canonical_span_id")
    _insert(conn, sid, 3, f"claude_code:{tag}")

    with pytest.raises(psycopg.errors.RaiseException, match="both the old and the new span id"):
        with conn.transaction():
            conn.execute(_MIGRATION.read_text())

    assert _spans(conn, sid) == [f"jsonl:{tag}", f"jsonl:other-{tag}", f"claude_code:{tag}"]


def test_an_older_build_resending_a_stored_turn_does_not_store_it_twice(
    conn, db_url, tag, monkeypatch
):
    """The route, end to end, with the parser of a build from before 068.

    That build spells Claude Code span ids ``jsonl:<uuid>``. It re-sends the turn before
    the new one on every Stop hook; on a renamed database neither its session index nor
    its global probe finds ``jsonl:<uuid>``, so only the trigger stands between the
    re-sent turn and a second copy."""
    from fastmcp import FastMCP
    from starlette.testclient import TestClient

    from ingestion.jsonl_client import JSONLParser
    from mcp_server.ingest_route import register

    sid = f"span068-{tag}-r"

    def _turn(text: str, last_uuid: str) -> list[dict]:
        base = {"sessionId": sid, "cwd": f"/work/{tag}", "timestamp": "2026-11-12T09:00:00Z"}
        return [
            {**base, "type": "user", "uuid": f"u-{last_uuid}", "message": {"role": "user", "content": f"q {text}"}},
            {
                **base,
                "type": "assistant",
                "uuid": last_uuid,
                "message": {"role": "assistant", "content": [{"type": "text", "text": f"a {text}"}]},
            },
        ]  # fmt: skip

    m = FastMCP("test-span-068")
    register(m, lambda: db_url, lambda r: True)
    client = TestClient(m.http_app())

    first = client.post("/ingest", json={"records": _turn(f"first {tag}", f"t1-{tag}")})
    assert first.status_code == 200 and first.json()["ingested"] == 1, first.text
    assert _spans(conn, sid) == [f"claude_code:t1-{tag}"]

    real_parse = JSONLParser.parse_records

    def _older_build(self, *args, **kwargs):
        episodes = real_parse(self, *args, **kwargs)
        for ep in episodes:
            ep.span_id = ep.span_id.replace("claude_code:", "jsonl:", 1)
        return episodes

    monkeypatch.setattr(JSONLParser, "parse_records", _older_build)
    # Content dedup off: it would absorb the re-sent turn before the INSERT, and this test
    # is about the path where it does not (a turn re-rendered with a [context] line).
    monkeypatch.setenv("SYNAPSE_CONTENT_DEDUP", "0")

    resent = client.post(
        "/ingest",
        json={"records": _turn(f"first {tag}", f"t1-{tag}") + _turn(f"second {tag}", f"t2-{tag}")},
    )

    assert resent.status_code == 200, resent.text
    assert resent.json() == {"status": "ok", "ingested": 1, "span_collisions": 1}
    assert _spans(conn, sid) == [f"claude_code:t1-{tag}", f"claude_code:t2-{tag}"]
