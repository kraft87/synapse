"""DB-backed tests for the board (mcp_server/board.py) + the /context route.

Covers the render contract (banner, grouping/order, line format, project scoping,
superseded exclusion), the hard cap (line overflow drop order + token cap), the
timeline section, the machine-token route gate, and kind='board' telemetry. Skips
cleanly when no test DB is reachable (mirrors test_notes_store.py)."""

from __future__ import annotations

import os
import re

import psycopg
import pytest

_DB_URL = os.environ.get(
    "SYNAPSE_TEST_URL", "postgresql://synapse:synapse@127.0.0.1:5432/synapse_test"
)

# Skip the whole module if the shared Postgres test DB isn't up — these tests are DB-only.
try:
    _probe = psycopg.connect(_DB_URL, connect_timeout=2)
    _probe.close()
except Exception:  # pragma: no cover - environment dependent
    pytest.skip("no test DB reachable", allow_module_level=True)

from starlette.testclient import TestClient  # noqa: E402

from ingestion.db import Database  # noqa: E402
from ingestion.surfaces import FULL_TRUST  # noqa: E402
from mcp_server.board import _OWNER, build_board, record_board_metrics, register  # noqa: E402
from tests.helpers.surfaces import register_full, register_restricted  # noqa: E402


def _board(db_url, project):
    """build_board as a FULL-TRUST surface — the shape every render test below assumes.

    Serving is fail-closed since schema 053: an unnamed surface is restricted and sees
    nothing, which would turn every rendering assertion in this file into "board is
    empty". These tests are about layout, ordering, and caps, so they state the trust
    verdict once, here. The audience-scoping tests call ``build_board`` directly."""
    return build_board(db_url, project, trust=FULL_TRUST)


def _wipe(conn):
    conn.execute("TRUNCATE episodes RESTART IDENTITY CASCADE")
    conn.execute("DELETE FROM notes")
    conn.execute("DELETE FROM timeline_events")
    conn.execute("DELETE FROM recall_metrics")
    conn.execute("DELETE FROM recall_feedback")
    conn.execute("DELETE FROM surfaces")


def _note(db, *, type="user", hook="User prefers X", project=None):
    """Board tests never exercise the KNN, so notes land with NULL embedding."""
    return db.insert_note(
        owner_id=_OWNER,
        project=project,
        type=type,
        hook=hook,
        body=f"Body of: {hook}",
        embedding=None,
        embed_model=None,
        source_ref=None,
    )


def _episode(conn, session, seq, project, days_ago=0):
    conn.execute(
        "INSERT INTO episodes (session_id, sequence, project, content, created_at) "
        "VALUES (%s, %s, %s, %s, now() - make_interval(days => %s))",
        (session, seq, project, f"turn {seq}", days_ago),
    )


def _event(conn, fact, *, salience=2, days_ago=1, project="alpha", ref="tb:0"):
    conn.execute(
        "INSERT INTO timeline_events (t_valid, fact, source, source_ref, project, salience) "
        "VALUES (now() - make_interval(days => %s), %s, 'test', %s, %s, %s)",
        (days_ago, fact, ref, project, salience),
    )


# ---------------------------------------------------------------------------
# Render contract
# ---------------------------------------------------------------------------


def test_empty_db_renders_banner_only(conn, db_url):
    _wipe(conn)
    out = _board(db_url, None)
    assert out["status"] == "ok"
    assert out["n_notes"] == 0 and out["overflow"] == 0 and out["note_ids"] == []
    text = out["text"]
    assert text.startswith("[Synapse board — project: all]")
    assert "0 episodes across 0 projects." in text
    assert "Absence from this board means SEARCH (recall)" in text
    assert "##" not in text  # empty sections omitted


def test_banner_says_how_to_word_a_query_about_the_owner(conn, db_url, monkeypatch):
    """Facts and notes are third person under the owner's stored name. A client that reads
    only the board (no MCP instructions) is told that name here, in the server's words."""
    _wipe(conn)
    monkeypatch.setenv("SYNAPSE_OWNER_NAME", "Sam")
    text = _board(db_url, None)["text"]
    assert (
        'Memory is written in third person and calls the user "Sam": in a recall query '
        'about the user, write "Sam", not "I" or "my".'
    ) in text
    assert text.index("Absence from this board") < text.index("Memory is written in third person")


def test_banner_counts_and_recent_projects(conn, db_url):
    _wipe(conn)
    _episode(conn, "s1", 1, "alpha", days_ago=5)
    _episode(conn, "s1", 2, "alpha", days_ago=5)
    _episode(conn, "s2", 1, "beta", days_ago=1)
    text = _board(db_url, "alpha")["text"]
    assert text.startswith("[Synapse board — project: alpha]")
    # beta has the most recent activity, so it leads the most-recent list.
    assert "3 episodes across 2 projects (most recent: beta, alpha)." in text


def test_banner_null_project_episodes_do_not_consume_slots(conn, db_url):
    """A NULL-project episode group must not eat one of the 12 most-recent slots or
    skew the project count: 12 real projects + newer project-less episodes -> all 12
    listed, P counts 12."""
    _wipe(conn)
    for i in range(12):
        _episode(conn, f"s{i}", 1, f"proj{i:02d}", days_ago=i + 1)
    # Most recent activity of all — would win a LIMIT slot if not excluded in SQL.
    _episode(conn, "s-null", 1, None, days_ago=0)
    text = _board(db_url, None)["text"]
    assert "13 episodes across 12 projects" in text
    for i in range(12):
        assert f"proj{i:02d}" in text


def _many_projects(conn):
    """20 labeled projects plus unlabeled rows: newest activity runs proj00 (newest) to
    proj19, except that proj03 and proj04 share proj01's newest timestamp exactly (ties
    break by name), and every project has an older episode too, so "newest" means
    max(created_at)."""
    _wipe(conn)
    for i in range(20):
        days = 1 if i in (3, 4) else i
        conn.execute(
            "INSERT INTO episodes (session_id, sequence, project, content, created_at) "
            "VALUES (%s, 1, %s, 'old', '2026-01-01'), "
            "       (%s, 2, %s, 'new', timestamptz '2026-10-01 12:00' - make_interval(days => %s))",
            (f"s{i}", f"proj{i:02d}", f"s{i}", f"proj{i:02d}", days),
        )
    _episode(conn, "s-null", 1, None, days_ago=0)
    conn.execute(
        "INSERT INTO episodes (session_id, sequence, project, content) VALUES ('s-blank', 1, '', 'x')"
    )


# proj01, proj03 and proj04 share a newest timestamp, so they order by name.
_NEWEST_FIRST = ["proj00", "proj01", "proj03", "proj04", "proj02"] + [
    f"proj{i:02d}" for i in range(5, 12)
]


def test_banner_counts_every_project_and_lists_the_newest_twelve(conn, db_url):
    """The count is every labeled project, not the length of the capped list (it said
    "across 12 projects" with 157 in prod). Same-day projects order by their newest
    episode, then by name."""
    from mcp_server.board import _banner_stats

    _many_projects(conn)
    n_episodes, n_projects, names = _banner_stats(db_url)
    assert (n_episodes, n_projects) == (42, 20)
    assert names == _NEWEST_FIRST
    text = _board(db_url, None)["text"]
    assert f"42 episodes across 20 projects (most recent: {', '.join(_NEWEST_FIRST)})." in text


def test_skip_scan_and_group_by_agree(conn, db_url):
    """The index-backed skip scan returns exactly what the GROUP BY it replaces returns,
    count and order included, on labeled, unlabeled and tied rows."""
    from mcp_server.board import _RECENT_PROJECTS_SKIP_SCAN

    _many_projects(conn)
    group_by = conn.execute(
        "SELECT project, count(*) OVER () FROM episodes "
        "WHERE project IS NOT NULL AND project <> '' "
        "GROUP BY project ORDER BY max(created_at) DESC, project LIMIT %s",
        (12,),
    ).fetchall()
    skip_scan = conn.execute(_RECENT_PROJECTS_SKIP_SCAN, (12,)).fetchall()
    assert skip_scan == group_by
    assert [r[0] for r in skip_scan] == _NEWEST_FIRST and skip_scan[0][1] == 20
    _wipe(conn)  # and on an empty table
    assert conn.execute(_RECENT_PROJECTS_SKIP_SCAN, (12,)).fetchall() == []


def test_banner_keeps_the_group_by_until_the_index_exists(conn, db_url, monkeypatch):
    """Without schema 061's index the skip scan's per-project max is far slower than the
    seq scan, so the banner must not run it where the index is missing (the probe also
    requires indisvalid, for an interrupted CONCURRENTLY build)."""
    from ingestion.schema_check import ColumnProbe
    from mcp_server import board

    _many_projects(conn)
    assert ColumnProbe(board.PROJECT_RECENCY_INDEX._sql, "t").ready(conn, "with-index")

    monkeypatch.setattr(board, "_RECENT_PROJECTS_SKIP_SCAN", "SELECT no_such_column")
    monkeypatch.setattr(
        board, "PROJECT_RECENCY_INDEX", ColumnProbe(board.PROJECT_RECENCY_INDEX._sql, "t")
    )
    conn.execute("DROP INDEX episodes_project_created_idx")
    try:
        assert board._banner_stats(db_url)[1:] == (20, _NEWEST_FIRST)  # the GROUP BY served
    finally:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS episodes_project_created_idx "
            "ON episodes (project, created_at)"
        )


def test_grouping_order_and_line_format(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    nid = _note(db, type="feedback", hook="Never use tables in answers")
    _note(db, type="user", hook="User works in a homelab")
    _note(db, type="project", hook="Board PR in flight", project="alpha")
    _note(db, type="reference", hook="See the notes design doc")
    db.close()
    text = _board(db_url, "alpha")["text"]
    order = [
        text.index("## Rules & feedback"),
        text.index("## User"),
        text.index("## Project: alpha"),
        text.index("## References"),
    ]
    assert order == sorted(order)  # feedback -> user -> project -> reference
    line = next(ln for ln in text.splitlines() if "Never use tables" in ln)
    assert re.fullmatch(rf"- Never use tables in answers \(n:{nid}, upd \d{{2}}-\d{{2}}\)", line)


def test_project_scoping(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    _note(db, type="project", hook="Alpha project state", project="alpha")
    _note(db, type="project", hook="Beta project state", project="beta")
    _note(db, type="user", hook="Global user fact")
    db.close()
    text = _board(db_url, "alpha")["text"]
    assert "Alpha project state" in text
    assert "Beta project state" not in text  # other projects' notes excluded
    assert "Global user fact" in text  # globals always present
    # No project scope -> global set only, no project section at all.
    text_all = _board(db_url, None)["text"]
    assert "Alpha project state" not in text_all and "## Project" not in text_all


def test_superseded_notes_excluded(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    old = _note(db, type="user", hook="User prefers dark mode")
    new = _note(db, type="user", hook="User now prefers light mode")
    db.supersede_note(old, new)
    db.close()
    out = _board(db_url, None)
    assert "User now prefers light mode" in out["text"]
    assert "User prefers dark mode" not in out["text"]
    assert out["note_ids"] == [new]


# ---------------------------------------------------------------------------
# Hard cap / overflow
# ---------------------------------------------------------------------------


def test_line_overflow_drops_oldest_project_notes_first(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    _note(db, type="feedback", hook="Feedback rule one")
    _note(db, type="feedback", hook="Feedback rule two")
    ids = [
        _note(db, type="project", hook=f"Project note {i:03d}", project="alpha") for i in range(90)
    ]
    db.close()
    # Stagger updated_at: note 000 oldest ... 089 newest.
    for age, nid in enumerate(reversed(ids)):
        conn.execute(
            "UPDATE notes SET updated_at = now() - make_interval(days => %s) WHERE id = %s",
            (age, nid),
        )
    out = _board(db_url, "alpha")
    text = out["text"]
    assert out["overflow"] > 0
    assert text.count("\n") + 1 <= 80
    assert f"(+ {out['overflow']} notes not shown)" in text
    # Oldest-updated project notes dropped first; newest survive; feedback survives.
    assert "Project note 000" not in text
    assert "Project note 089" in text
    assert "Feedback rule one" in text and "Feedback rule two" in text
    assert out["n_notes"] == 92 - out["overflow"]


def test_token_cap_on_long_hooks(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    for i in range(5):
        _note(db, type="user", hook=f"Long note {i} " + "x" * 2000)
    db.close()
    out = _board(db_url, None)
    assert out["overflow"] >= 1  # far under 80 lines — the token cap did this
    assert len(out["text"]) // 4 <= 2000


def test_long_event_facts_clamped_and_feedback_survives(conn, db_url):
    """Verbose timeline facts (POST /timeline/events accepts unbounded text) must not
    bust the hard cap or evict curated notes: event lines are clamped at render, so
    the fixed section stays bounded and the cap loop never drains feedback."""
    _wipe(conn)
    for i in range(5):
        _event(conn, f"Event {i} " + "y" * 3000, salience=2, days_ago=1, ref=f"tb:long{i}")
    db = Database(db_url)
    _note(db, type="feedback", hook="Feedback must survive event floods")
    db.close()
    out = _board(db_url, None)
    text = out["text"]
    assert out["overflow"] == 0
    assert "Feedback must survive event floods" in text
    assert text.count("\n") + 1 <= 80 and len(text) // 4 <= 2000  # both caps hold
    assert "y" * 250 not in text  # each event fact clamped, not served whole
    assert all(len(ln) <= 250 for ln in text.splitlines())


def test_cap_loop_keeps_notes_when_floor_over_cap(conn, db_url):
    """If the fixed portion alone busts the token cap (pathological project names in
    the banner), dropping notes cannot reach the cap — the loop must keep them all
    instead of draining the board for zero benefit."""
    _wipe(conn)
    for i in range(12):
        _episode(conn, f"s{i}", 1, f"project-{i}-" + "z" * 800, days_ago=i)
    db = Database(db_url)
    _note(db, type="feedback", hook="Keep me")
    db.close()
    out = _board(db_url, None)
    assert out["overflow"] == 0 and out["n_notes"] == 1
    assert "Keep me" in out["text"]


def test_large_board_matches_old_cap_loop_in_few_renders(conn, db_url, monkeypatch):
    """240 live notes through the real DB path (prod had ~230 when the one-render-per-drop
    loop cost ~300 ms a request): same text, kept ids and overflow as the old loop on the
    same rows, from a handful of renders instead of one per dropped note."""
    import math

    from mcp_server import board
    from tests.test_board_cap import _old_cap_loop

    _wipe(conn)
    db = Database(db_url)
    types = ("feedback", "user", "project", "reference")
    ids = [
        _note(db, type=types[i % 4], hook=f"Note {i:03d} " + "w" * (i % 70), project="alpha")
        for i in range(240)
    ]
    db.close()
    for age, nid in enumerate(ids):
        conn.execute(
            "UPDATE notes SET updated_at = now() - make_interval(hours => %s) WHERE id = %s",
            (age, nid),
        )
    _event(conn, "Shipped the board", salience=2, days_ago=1, ref="tb:big")

    seen: dict[str, object] = {}
    real_fit, real_render = board._fit_notes, board._render
    renders = 0

    def capture(*args):
        seen["args"] = args
        return real_fit(*args)

    def counting_render(*args, **kwargs):
        nonlocal renders
        renders += 1
        return real_render(*args, **kwargs)

    monkeypatch.setattr(board, "_fit_notes", capture)
    monkeypatch.setattr(board, "_render", counting_render)
    out = _board(db_url, "alpha")
    monkeypatch.setattr(board, "_render", real_render)

    text, kept, dropped = _old_cap_loop(*seen["args"])
    assert out["text"] == text
    assert out["note_ids"] == [n["id"] for n in kept]
    assert out["overflow"] == dropped and dropped > 150
    assert renders <= 2 + math.ceil(math.log2(240))


def test_overflow_line_separated_when_all_notes_dropped(conn, db_url):
    """A board whose every note was evicted still renders the overflow line with a
    blank line after the banner, not jammed directly under it."""
    _wipe(conn)
    db = Database(db_url)
    _note(db, type="user", hook="Giant " + "x" * 9000)  # one note, over the cap alone
    db.close()
    out = _board(db_url, None)
    assert out["n_notes"] == 0 and out["overflow"] == 1
    assert out["text"].endswith("\n\n(+ 1 notes not shown)")


# ---------------------------------------------------------------------------
# Timeline section
# ---------------------------------------------------------------------------


def test_timeline_section_present_and_absent(conn, db_url):
    _wipe(conn)
    _event(conn, "Shipped the board", salience=2, days_ago=1, ref="tb:1")
    text = _board(db_url, None)["text"]
    assert "## Last 7 days" in text
    assert re.search(r"- \d{2}-\d{2} \(alpha\): Shipped the board.* \(t:\d+\)", text)

    _wipe(conn)
    _event(conn, "Routine tweak", salience=1, days_ago=1, ref="tb:2")  # below min_salience
    _event(conn, "Old milestone", salience=2, days_ago=30, ref="tb:3")  # outside window
    assert "## Last 7 days" not in _board(db_url, None)["text"]


# ---------------------------------------------------------------------------
# /context route + telemetry
# ---------------------------------------------------------------------------

_TOKEN = "test-board-token"


def _client(db_url, get_recall=None, resolve_trust=None):
    """``resolve_trust`` stands in for the server's bearer resolution. Omitted, every
    request is UNKNOWN, which is what the route does when it is given no resolver."""
    from fastmcp import FastMCP

    def authorized(request):
        return request.headers.get("authorization", "") == f"Bearer {_TOKEN}"

    test_mcp = FastMCP("test-board")
    register(test_mcp, db_url, authorized, get_recall=get_recall, resolve_trust=resolve_trust)
    return TestClient(test_mcp.http_app())


def _full_trust(_request):
    return FULL_TRUST


def test_route_auth_and_project_param(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    _note(db, type="project", hook="Alpha project state", project="alpha")
    db.close()
    with _client(db_url, resolve_trust=_full_trust) as client:
        assert client.get("/context").status_code == 401  # no token
        r = client.get("/context", headers={"Authorization": f"Bearer {_TOKEN}"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok" and "note_ids" not in body
        assert "Alpha project state" not in body["text"]  # unscoped -> global set only
        r2 = client.get("/context?project=alpha", headers={"Authorization": f"Bearer {_TOKEN}"})
        assert "Alpha project state" in r2.json()["text"]


def test_route_serves_the_preferences_block_with_the_board_when_asked(conn, db_url):
    """One request for a session start: the board and the preferences, both rendered
    here. A client from before ``?parts=`` asks for neither by name and gets the board."""
    _wipe(conn)
    conn.execute("DELETE FROM preferences WHERE owner_id = %s", (_OWNER,))
    db = Database(db_url)
    _note(db, type="feedback", hook="User prefers the proper fix")
    db.insert_preference(
        owner_id=_OWNER,
        project=None,
        pref="User wants short answers",
        polarity="rule",
        embedding=None,
        embed_model=None,
        source_ref="ep:1",
    )
    db.close()
    h = {"Authorization": f"Bearer {_TOKEN}"}
    with _client(db_url, resolve_trust=_full_trust) as client:
        old = client.get("/context", headers=h).json()
        both = client.get("/context?parts=board,preferences", headers=h).json()
        prefs = client.get("/context?parts=preferences", headers=h).json()
    assert "preferences" not in old and "User prefers the proper fix" in old["text"]
    assert both["text"] == old["text"] and "note_ids" not in both
    assert both["preferences"] == "[Synapse preferences]\n  - (rule) User wants short answers"
    assert prefs == {"status": "ok", "trust": "full", "preferences": both["preferences"]}

    # A caller the server cannot place is served no preference, as on /preferences/top.
    with _client(db_url) as client:
        unknown = client.get("/context?parts=board,preferences", headers=h).json()
    assert unknown["trust"] == "restricted" and unknown["preferences"] == ""
    conn.execute("DELETE FROM preferences WHERE owner_id = %s", (_OWNER,))
    _wipe(conn)


def test_route_records_board_telemetry(conn, db_url):
    from mcp_server.recall import Recall

    _wipe(conn)
    db = Database(db_url)
    kept = _note(db, type="user", hook="Telemetry fixture note")
    db.close()
    engine = Recall(db_url=db_url, voyage_api_key="")
    with _client(db_url, get_recall=lambda: engine, resolve_trust=_full_trust) as client:
        r = client.get("/context", headers={"Authorization": f"Bearer {_TOKEN}"})
        assert r.status_code == 200
    # The metrics write is fire-and-forget on a single-worker FIFO pool: a barrier
    # task completing proves the row insert before ours has finished.
    engine._async_executor.submit(lambda: None).result(timeout=10)
    row = conn.execute(
        "SELECT source, ms_total, chars, est_tokens, served_ids FROM recall_metrics "
        "WHERE kind = 'board'"
    ).fetchone()
    assert row is not None
    source, ms_total, chars, est_tokens, served_ids = row
    assert source == "http"
    assert ms_total is not None and chars > 0 and est_tokens == chars // 4
    # Served form ("n:N") — joinable against recall_feedback's helpful/noise ids.
    assert served_ids["notes"] == [f"n:{kept}"]
    assert served_ids["n_notes"] == 1 and served_ids["overflow"] == 0


def test_record_board_metrics_is_fail_soft():
    class Boom:
        def record_event(self, m):
            raise RuntimeError("nope")

    # Must swallow, never raise back into the serve path.
    record_board_metrics(Boom(), "http", 1.0, {"text": "t", "note_ids": [1]})


# ---------------------------------------------------------------------------
# Audience scoping (schema 053): what a restricted surface's board contains
# ---------------------------------------------------------------------------


def _audience_fixture(conn, db_url):
    """Two projects, two audiences, a private note and a work note, plus timeline
    events and episodes on both sides. Returns the ids the assertions key on."""
    _wipe(conn)
    db = Database(db_url)
    personal_note = _note(db, type="user", hook="User keeps a personal journal")
    work_note = _note(db, type="user", hook="User prefers tabs over spaces")
    alpha_note = _note(db, type="project", hook="Alpha runs on Postgres", project="alpha")
    beta_note = _note(db, type="project", hook="Beta is a side project", project="beta")
    db.close()
    conn.execute(
        "UPDATE notes SET audience = 'work-safe' WHERE id = ANY(%s)", ([work_note, alpha_note],)
    )
    _episode(conn, "s-alpha", 1, "alpha")
    _episode(conn, "s-beta", 1, "beta")
    # One day apiece: the board keeps a single event per day, so a shared day would hide two.
    _event(conn, "Alpha shipped a release", project="alpha", days_ago=1, ref="aud:1")
    _event(conn, "Beta got a new logo", project="beta", days_ago=2, ref="aud:2")
    _event(conn, "Unlabeled milestone", project=None, days_ago=3, ref="aud:3")
    return {
        "personal": personal_note,
        "work": work_note,
        "alpha": alpha_note,
        "beta": beta_note,
    }


def test_restricted_board_filters_notes_digest_and_banner(conn, db_url):
    """The whole enforcement surface of the board, in one restricted serve."""
    ids = _audience_fixture(conn, db_url)
    sid = register_restricted(conn, ["alpha"])
    out = build_board(db_url, "alpha", sid)
    text = out["text"]

    assert out["trust"] == "restricted"
    # Notes: work-safe only, whatever their type or project.
    assert "User prefers tabs over spaces" in text
    assert "Alpha runs on Postgres" in text
    assert "User keeps a personal journal" not in text
    assert ids["personal"] not in out["note_ids"]
    # Digest: allowlisted projects only, and NULL-project events are excluded — an
    # unlabeled event has no provenance that could clear it.
    assert "Alpha shipped a release" in text
    assert "Beta got a new logo" not in text
    assert "Unlabeled milestone" not in text
    # Banner: project names are themselves disclosure, so they follow the allowlist.
    assert "alpha" in text and "beta" not in text
    assert text.splitlines()[1].startswith("1 episodes across 1 projects")


def test_full_trust_board_is_unfiltered(conn, db_url):
    """The control: the same corpus on a trusted host serves everything."""
    _audience_fixture(conn, db_url)
    sid = register_full(conn)
    text = build_board(db_url, "alpha", sid)["text"]
    assert "User keeps a personal journal" in text
    assert "User prefers tabs over spaces" in text
    assert "Beta got a new logo" in text
    assert "2 episodes across 2 projects" in text


def test_unknown_surface_is_restricted_with_empty_allowlist(conn, db_url):
    """No surfaces row means restricted with an EMPTY allowlist.

    Concretely: work-safe notes still serve (that tier is defined as content safe to
    leave a trusted host), but nothing project-scoped does — no episodes in the banner,
    no timeline digest, no project notes. A request naming no surface at all is treated
    identically; there is no "unauthenticated means trusted" path."""
    ids = _audience_fixture(conn, db_url)
    for surface in ("never-registered-host", None):
        out = build_board(db_url, "alpha", surface)
        assert out["trust"] == "restricted", surface
        assert ids["personal"] not in out["note_ids"], surface
        assert "User prefers tabs over spaces" in out["text"], surface
        # Empty allowlist: project = ANY('{}') matches nothing, NULL project included.
        assert "Alpha shipped a release" not in out["text"], surface
        assert "0 episodes across 0 projects" in out["text"], surface


def test_surface_lookup_error_degrades_to_restricted_not_unfiltered(conn, db_url, monkeypatch):
    """A failing surfaces lookup must not become "serve everything". This is the
    degradation the spec singles out: the board already degrades sections to empty on a
    missing table, and that posture must not invert into unfiltered here."""
    ids = _audience_fixture(conn, db_url)
    sid = register_full(conn)  # the row EXISTS and says full — only the lookup is broken

    import mcp_server.board as board_mod
    from ingestion.surfaces import lookup_surface as real_lookup

    # Drive the REAL fail-closed path with a DSN nothing is listening on, so the board's
    # other queries keep working and only the trust lookup fails — the exact shape of a
    # surfaces outage.
    monkeypatch.setattr(
        board_mod,
        "lookup_surface",
        lambda _u, s: real_lookup("postgresql://synapse:synapse@127.0.0.1:1/nope", s),
    )
    out = build_board(db_url, "alpha", sid)
    assert out["trust"] == "restricted"
    assert ids["personal"] not in out["note_ids"]
    assert "User keeps a personal journal" not in out["text"]
    assert "0 episodes across 0 projects" in out["text"]


def test_route_enforces_the_resolved_verdict_not_the_surface_param(conn, db_url):
    """The /context route serves what the CREDENTIAL resolves to. ``?surface=`` used to
    select the row for a root-token caller; naming a full-trust row must change nothing."""
    from ingestion.surfaces import SurfaceTrust

    _audience_fixture(conn, db_url)
    full = register_full(conn, "legacy-full-host")
    work = SurfaceTrust(surface_id="dev-w", trust="restricted", allowed_projects=("alpha",))
    h = {"Authorization": f"Bearer {_TOKEN}"}
    with _client(db_url, resolve_trust=lambda _r: work) as client:
        body = client.get(f"/context?project=alpha&surface={full}", headers=h).json()
        assert body["trust"] == "restricted"
        assert "User keeps a personal journal" not in body["text"]
        assert "User prefers tabs over spaces" in body["text"]
        assert "Alpha shipped a release" in body["text"]  # its OWN allowlist still serves


def test_route_without_a_resolver_serves_unknown_whatever_surface_is_named(conn, db_url):
    """No resolver means no credential evidence: UNKNOWN, restricted with an empty
    allowlist. A ``?surface=`` naming a full-trust row is not a fallback."""
    _audience_fixture(conn, db_url)
    full = register_full(conn, "legacy-full-host")
    h = {"Authorization": f"Bearer {_TOKEN}"}
    with _client(db_url) as client:
        named = client.get(f"/context?project=alpha&surface={full}", headers=h).json()
        bare = client.get("/context?project=alpha", headers=h).json()
    for body in (named, bare):
        assert body["trust"] == "restricted"
        assert "User keeps a personal journal" not in body["text"]
        assert "Alpha shipped a release" not in body["text"]
    assert named["text"] == bare["text"]


# ---------------------------------------------------------------------------
# Per-type quotas, helpful-first ranking, one timeline line per day
# ---------------------------------------------------------------------------


def _rate_helpful(conn, *note_ids, reports=1):
    for _ in range(reports):
        conn.execute(
            "INSERT INTO recall_feedback (query, helpful) VALUES ('q', %s::jsonb)",
            ("[" + ",".join(f'"n:{i}"' for i in note_ids) + "]",),
        )


def _flood(conn, db_url, *, feedback=60, user=40, project=20, reference=20):
    """A prod-shaped board: far more notes than fit, the feedback alone enough to fill the
    cap. Hooks are ~100 chars; updated_at descends with creation order inside each type."""
    db = Database(db_url)
    ids: dict[str, list[int]] = {"feedback": [], "user": [], "project": [], "reference": []}
    for type_, n in (
        ("feedback", feedback),
        ("user", user),
        ("project", project),
        ("reference", reference),
    ):
        for i in range(n):
            hook = f"{type_} note {i:03d} " + "w" * 90
            ids[type_].append(_note(db, type=type_, hook=hook, project="alpha"))
    db.close()
    for group in ids.values():
        for age, nid in enumerate(group):  # first created = newest
            conn.execute(
                "UPDATE notes SET updated_at = now() - make_interval(hours => %s) WHERE id = %s",
                (age, nid),
            )
    return ids


def _types_shown(conn, out):
    rows = conn.execute("SELECT type FROM notes WHERE id = ANY(%s)", (out["note_ids"],)).fetchall()
    shown = {"feedback": 0, "user": 0, "project": 0, "reference": 0}
    for (t,) in rows:
        shown[t] += 1
    return shown


def test_feedback_flood_no_longer_crowds_out_the_other_types(conn, db_url):
    """The bug: 60 feedback hooks alone fill the cap, so user, project and reference notes
    were all dropped, board after board. Every type must now be on the board, in
    proportion to its quota."""
    _wipe(conn)
    _flood(conn, db_url)
    out = _board(db_url, "alpha")
    shown = _types_shown(conn, out)
    assert out["overflow"] > 0 and len(out["text"]) // 4 <= 2000
    assert all(n > 0 for n in shown.values()), shown
    assert shown["feedback"] > shown["user"] > shown["project"] > shown["reference"], shown
    assert shown["feedback"] < 60  # not the whole budget


def test_unused_quota_flows_to_the_other_types(conn, db_url):
    """No project notes on the board (project=None) and three user notes: the project
    share goes to feedback and references, and every user note is shown."""
    _wipe(conn)
    ids = _flood(conn, db_url, feedback=60, user=3, project=0, reference=60)
    out = _board(db_url, None)
    shown = _types_shown(conn, out)
    assert shown["user"] == 3 and shown["project"] == 0
    assert set(ids["user"]) <= set(out["note_ids"])
    assert shown["feedback"] > shown["reference"] > 0
    assert shown["feedback"] + shown["reference"] + 3 == out["n_notes"]


def test_helpful_ratings_outrank_recency_within_a_type(conn, db_url):
    """A user note nobody has updated in a long time but that recall's raters keep marking
    helpful is shown ahead of newer, unrated ones; two ratings beat one."""
    _wipe(conn)
    ids = _flood(conn, db_url)
    oldest_user, second_oldest = ids["user"][-1], ids["user"][-2]
    _rate_helpful(conn, oldest_user, second_oldest)
    _rate_helpful(conn, oldest_user)  # two reports for the oldest, one for the second
    out = _board(db_url, "alpha")
    shown_user = [i for i in out["note_ids"] if i in set(ids["user"])]
    assert oldest_user in shown_user and second_oldest in shown_user
    # The rest of the user quota goes to the newest unrated notes.
    assert set(ids["user"][: len(shown_user) - 2]) <= set(shown_user)


def test_unrated_boards_rank_by_recency_and_keep_section_order(conn, db_url):
    """With no ratings the kept notes of a type are its newest, and the sections still
    render feedback -> user -> project -> reference with each one newest-first."""
    _wipe(conn)
    ids = _flood(conn, db_url)
    out = _board(db_url, "alpha")
    kept = set(out["note_ids"])
    for group in ids.values():
        flags = [i in kept for i in group]  # group[0] is the newest
        assert flags == sorted(flags, reverse=True)  # a newest-first prefix, no gaps
    assert out["note_ids"] == sorted(
        out["note_ids"],
        key=lambda i: (
            next(k for k, g in enumerate(ids.values()) if i in g),
            ids[next(t for t, g in ids.items() if i in g)].index(i),
        ),
    )


def test_a_board_that_fits_is_untouched_by_ratings(conn, db_url):
    """Nothing is dropped, so ranking cannot change what is shown or the order it is
    shown in."""
    _wipe(conn)
    db = Database(db_url)
    a = _note(db, type="user", hook="User fact A")
    b = _note(db, type="user", hook="User fact B")
    db.close()
    before = _board(db_url, None)
    _rate_helpful(conn, a, a, b)
    after = _board(db_url, None)
    assert before["text"] == after["text"] and after["overflow"] == 0


def test_restricted_board_shares_the_quotas(conn, db_url):
    """A work laptop's board is cut the same way: its work-safe feedback flood no longer
    hides its work-safe user notes."""
    _wipe(conn)
    ids = _flood(conn, db_url, feedback=60, user=12, project=0, reference=30)
    conn.execute("UPDATE notes SET audience = 'work-safe'")
    sid = register_restricted(conn, ["alpha"])
    out = build_board(db_url, None, sid)
    shown = _types_shown(conn, out)
    assert out["trust"] == "restricted" and out["overflow"] > 0
    assert shown["user"] > 0 and shown["reference"] > 0 and shown["feedback"] > shown["user"]
    assert len(out["text"]) // 4 <= 2000
    assert set(out["note_ids"]) <= {i for g in ids.values() for i in g}


def test_timeline_is_one_line_per_day_and_at_most_five_days(conn, db_url):
    """Seven days of milestones, several on some days: the board carries each day's
    highest-salience event, for the five newest days."""
    _wipe(conn)
    for days_ago in range(1, 8):
        _event(
            conn,
            f"Plain event day {days_ago}",
            salience=2,
            days_ago=days_ago,
            ref=f"tb:p{days_ago}",
        )
    _event(conn, "Big event on day two", salience=4, days_ago=2, ref="tb:big2")
    _event(conn, "Other project same day", salience=3, days_ago=3, project="beta", ref="tb:b3")
    text = _board(db_url, None)["text"]
    lines = [ln for ln in text.splitlines() if re.search(r"\(t:\d+\)$", ln)]
    assert len(lines) == 5
    dates = [ln[2:7] for ln in lines]
    assert dates == sorted(dates, reverse=True) and len(set(dates)) == 5
    assert "Big event on day two" in text and "Plain event day 2" not in text
    assert "Other project same day" in text and "Plain event day 3" not in text
    assert "Plain event day 1" in text and "Plain event day 6" not in text


def test_timeline_share_of_the_board_is_small(conn, db_url):
    """Verbose milestones every day used to take about a quarter of the token budget."""
    _wipe(conn)
    for days_ago in range(1, 8):
        for k in range(3):
            _event(
                conn,
                f"Milestone {days_ago}-{k} " + "m" * 400,
                salience=2,
                days_ago=days_ago,
                ref=f"tb:{days_ago}{k}",
            )
    _flood(conn, db_url)
    text = _board(db_url, "alpha")["text"]
    section = text.split("## Last 7 days", 1)[1]
    assert len(section) < 0.15 * len(text)


def test_a_busy_day_cannot_crowd_older_days_off_the_timeline(conn, db_url):
    """Sixty qualifying events on the newest day (20 projects x the per-project-day
    limit of three) must not push the oldest day's single event out of the pool the
    one-per-day cut draws from."""
    _wipe(conn)
    for p in range(20):
        for k in range(3):
            _event(conn, f"Busy day {p}-{k}", days_ago=1, project=f"busy{p}", ref=f"tb:{p}-{k}")
    _event(conn, "Quiet day five", days_ago=5, ref="tb:quiet")
    text = _board(db_url, None)["text"]
    assert "Quiet day five" in text
    assert sum("Busy day" in ln for ln in text.splitlines()) == 1
