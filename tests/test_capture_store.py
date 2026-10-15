"""ingestion.capture.store: caps, dedup, savepoints and the dry run, against the scratch DB."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from ingestion.capture import ToolCall, capture_calls, enabled
from ingestion.capture import store as store_mod
from ingestion.capture.detect import Skip
from ingestion.db import Database

PAGE = "# A Real Article\n\n" + "The measured latency was low and the design held up well. " * 12
RESULTS = json.dumps(
    [
        {"title": "One", "url": "https://one.com/a", "snippet": "first"},
        {"title": "Two", "url": "https://two.com/b", "snippet": "second"},
    ]
)


@pytest.fixture(scope="module")
def db(db_url):
    d = Database(db_url)
    yield d
    d.close()


@pytest.fixture(autouse=True)
def _clean(conn):
    conn.execute("TRUNCATE web_artifacts, episodes, private_sessions RESTART IDENTITY CASCADE")
    yield
    conn.execute("TRUNCATE web_artifacts, episodes, private_sessions RESTART IDENTITY CASCADE")


def fetch(tid: str, url: str = "https://example.com/p", text: str = PAGE, **kw) -> ToolCall:
    fields = dict(
        tool_use_id=tid,
        tool_name="mcp__acme__fetch",
        tool_input={"url": url},
        result_text=text,
        is_error=False,
        session_id="s1",
        project="proj",
        timestamp=datetime(2026, 9, 1, 12, tzinfo=UTC),
    )
    return ToolCall(**{**fields, **kw})


def never_private(_session: str | None) -> bool:
    return False


def stored(conn, cols: str = "tool_use_id") -> list[tuple]:
    return conn.execute(f"SELECT {cols} FROM web_artifacts ORDER BY id").fetchall()


def run(db, calls, **kw):
    return capture_calls(db, calls, is_private=kw.pop("is_private", never_private), **kw)


def reasons(result) -> list[str]:
    return [d.reason if isinstance(d, Skip) else "captured" for _, d in result.outcomes]


def test_a_page_and_a_result_list_are_stored_with_their_provenance(db, conn):
    search = ToolCall(
        "toolu_s", "mcp__acme__find", {"query": "voyage"}, RESULTS, False, "s1", "proj",
        datetime(2026, 9, 1, 12, 1, tzinfo=UTC), surface_id="surf-1",
    )  # fmt: skip
    result = run(db, [fetch("toolu_p", surface_id="surf-1"), search])
    assert result.captured == 2 and result.errors == 0
    page, results = conn.execute(
        "SELECT kind, url, url_canonical, session_id, tool_name, metadata, fetched_at, "
        "query, item_count FROM web_artifacts ORDER BY id"
    ).fetchall()
    assert page[:5] == (
        "web_scrape",
        "https://example.com/p",
        "https://example.com/p",
        "s1",
        "mcp__acme__fetch",
    )
    assert page[5] == {
        "capture": "generic",
        "project": "proj",
        "surface_id": "surf-1",
        "kg_extract": False,
    }
    assert page[6] == datetime(2026, 9, 1, 12, tzinfo=UTC)
    assert (results[0], results[7], results[8]) == ("search_result_set", "voyage", 2)


def test_skips_are_counted_by_reason(db, conn):
    calls = [
        fetch("a"),
        fetch("b", url="http://localhost:3000/"),
        fetch("c", text="tiny"),
        ToolCall("d", "Bash", {"command": "ls"}, PAGE, False, "s1", "proj", None),
    ]
    result = run(db, calls)
    assert reasons(result) == ["captured", "non_public_url", "too_short", "shell"]
    assert result.skipped == {"non_public_url": 1, "too_short": 1, "shell": 1}
    assert stored(conn) == [("a",)]


def test_a_dry_run_reports_what_a_live_run_does_and_stores_nothing(db, conn):
    calls = [fetch("a"), fetch("b", url="https://example.com/q"), fetch("c", url="http://nas.lan/")]
    dry = run(db, calls, dry_run=True)
    assert stored(conn) == []
    live = run(db, calls)
    assert reasons(dry) == reasons(live) == ["captured", "captured", "non_public_url"]
    assert stored(conn) == [("a",), ("b",)]
    # ...and it knows what is stored already.
    assert reasons(run(db, calls, dry_run=True)) == ["duplicate", "duplicate", "non_public_url"]


def test_shipping_the_same_calls_again_stores_nothing_new(db, conn):
    calls = [fetch("a"), fetch("b", url="https://example.com/q")]
    run(db, calls)
    assert reasons(run(db, calls)) == ["duplicate", "duplicate"]
    assert len(stored(conn)) == 2


def test_the_same_page_from_another_call_is_one_stored_copy_until_it_changes(db, conn):
    run(db, [fetch("a")])
    assert reasons(run(db, [fetch("b")])) == ["duplicate"]
    assert reasons(run(db, [fetch("c", text=PAGE + " An update to the page.")])) == ["captured"]
    assert reasons(run(db, [fetch("d", url="https://example.com/other")])) == ["captured"]
    assert [r[0] for r in stored(conn)] == ["a", "c", "d"]


def test_the_same_results_for_the_same_query_are_one_copy(db, conn):
    def search(tid, query="voyage"):
        return ToolCall(tid, "mcp__acme__find", {"query": query}, RESULTS, False, "s1", "p", None)

    assert reasons(run(db, [search("a"), search("b")])) == ["captured", "duplicate"]
    assert reasons(run(db, [search("c", "other query")])) == ["captured"]


def test_duplicates_inside_one_batch_are_caught(db, conn):
    assert reasons(run(db, [fetch("a"), fetch("a"), fetch("b")])) == [
        "captured",
        "duplicate",
        "duplicate",
    ]


def test_the_per_post_cap(db, conn, monkeypatch):
    monkeypatch.setattr(store_mod, "MAX_CAPTURES_PER_POST", 2)
    calls = [fetch(f"t{i}", url=f"https://example.com/{i}") for i in range(4)]
    assert reasons(run(db, calls)) == ["captured", "captured", "cap_post", "cap_post"]
    assert len(stored(conn)) == 2


def test_the_per_day_cap_counts_the_day_of_the_call(db, conn, monkeypatch):
    monkeypatch.setattr(store_mod, "MAX_CAPTURES_PER_DAY", 2)
    day = datetime(2026, 9, 1, 8, tzinfo=UTC)
    calls = [fetch(f"d{i}", url=f"https://example.com/{i}", timestamp=day) for i in range(3)]
    assert reasons(run(db, calls)) == ["captured", "captured", "cap_day"]
    # Another POST the same day is over the cap; another day is not.
    assert reasons(run(db, [fetch("late", url="https://example.com/9", timestamp=day)])) == [
        "cap_day"
    ]
    tomorrow = day + timedelta(days=1)
    assert reasons(run(db, [fetch("next", url="https://example.com/9", timestamp=tomorrow)])) == [
        "captured"
    ]


def test_a_private_session_is_never_captured(db, conn):
    calls = [
        fetch("a", session_id="secret"),
        fetch("b", url="https://example.com/q", session_id="open"),
    ]
    result = run(db, calls, is_private=lambda s: s == "secret")
    assert reasons(result) == ["private", "captured"]
    assert stored(conn) == [("b",)]


def test_a_failing_private_lookup_fails_closed(db, conn):
    def broken(_session):
        raise RuntimeError("lookup down")

    result = run(db, [fetch("a")], is_private=broken)
    assert result.errors == 1 and result.outcomes == [] and stored(conn) == []


def test_one_failing_row_costs_one_artifact(db, conn, monkeypatch):
    real = store_mod.insert_artifact

    def flaky(c, row):
        if row["tool_use_id"] == "bad":
            c.execute("SELECT 1/0")  # aborts the statement, as a bad row would
        return real(c, row)

    monkeypatch.setattr(store_mod, "insert_artifact", flaky)
    calls = [
        fetch("a"),
        fetch("bad", url="https://example.com/b"),
        fetch("c", url="https://example.com/c"),
    ]
    result = run(db, calls)
    assert result.errors == 1 and result.captured == 2
    assert stored(conn) == [("a",), ("c",)]


def test_the_episode_of_a_call_is_linked(db, conn):
    conn.execute(
        "INSERT INTO episodes (session_id, sequence, content, span_id) "
        "VALUES ('s1', 1, 'turn', 'jsonl:u1'), ('s1', 2, 'turn 2', 'jsonl:u2')"
    )
    ids = dict(conn.execute("SELECT span_id, id FROM episodes").fetchall())
    run(
        db,
        [
            fetch("a", span_id="jsonl:u1"),
            fetch("b", url="https://example.com/b", span_id="jsonl:u2", episode_id=ids["jsonl:u2"]),
            fetch("c", url="https://example.com/c", span_id=None),
            fetch("d", url="https://example.com/d", span_id="jsonl:gone"),
        ],
    )
    assert dict(stored(conn, "tool_use_id, parent_episode_id")) == {
        "a": ids["jsonl:u1"],
        "b": ids["jsonl:u2"],
        "c": None,
        "d": None,
    }


def test_enabled_reads_the_switch(monkeypatch):
    monkeypatch.delenv("SYNAPSE_WEB_CAPTURE", raising=False)
    assert enabled() is False
    for value, expected in (("on", True), ("ON", True), ("off", False), ("", False)):
        monkeypatch.setenv("SYNAPSE_WEB_CAPTURE", value)
        assert enabled() is expected
    monkeypatch.setenv("SYNAPSE_WEB_CAPTURE", "true")
    with pytest.raises(ValueError, match="SYNAPSE_WEB_CAPTURE"):
        enabled()
