"""DB-backed tests for the operator dashboard routes (mcp_server/dashboard_routes.py).

Covers the /dash/api/* contract (catalog, feed merge + keyset cursor + type-filter
behavior, episode detail, derived facts/events, session ordering, entity dossier,
search per-type + total_by_type, flag toggle + audit + feed reflection) and the
unauthenticated static bundle routes (503 without a build, 200 + traversal rejection
with a monkeypatched dist dir). ALL fixture data is synthetic — this repo is public.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import psycopg
import pytest
from psycopg.types.json import Json
from starlette.testclient import TestClient

import mcp_server.dashboard_routes as dr
from mcp_server.dashboard_routes import register

pytestmark = pytest.mark.db

_TOKEN = "test-dash-token"
_H = {"Authorization": f"Bearer {_TOKEN}"}
_BASE = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)


def _t(minutes: int) -> datetime:
    return _BASE + timedelta(minutes=minutes)


def _wipe(conn):
    conn.execute(
        "TRUNCATE episodes, kg_entities, kg_relationships, timeline_events, "
        "notes, preferences, chunks, extraction_queue, dream_runs, "
        "dashboard_flags, dashboard_audit, recall_metrics "
        "RESTART IDENTITY CASCADE"
    )


@pytest.fixture()
def clean(conn):
    """Wipe the tables these tests touch before AND after — deterministic counts for a
    serial DB-group run, and no synthetic rows leak past the test."""
    _wipe(conn)
    yield
    _wipe(conn)


# ---------------------------------------------------------------------------
# Synthetic-row insert helpers (all return the new id)
# ---------------------------------------------------------------------------


def _episode(
    conn,
    *,
    session="s-dash",
    seq=1,
    project="synapse",
    source="claude-code",
    content="turn body",
    created_at=None,
    platform="claude_code",
    model="opus",
    human="hi",
    assistant="hello",
):
    return conn.execute(
        "INSERT INTO episodes "
        "(session_id, sequence, project, source, platform, model, human_turn, assistant_turn, "
        " content, created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (
            session,
            seq,
            project,
            source,
            platform,
            model,
            human,
            assistant,
            content,
            created_at or _t(0),
        ),
    ).fetchone()[0]


# The KG helpers write only the bitemporal t_* columns and entity_subtype, never the
# FalkorDB-era created_at / valid_at / invalid_at / entity_type mirrors: the dashboard
# reads the former, so a read that slips back onto a legacy column fails here.
def _entity(conn, uuid, *, name, entity_type="Technology", summary="a thing", group_id="technical"):
    return conn.execute(
        "INSERT INTO kg_entities (uuid, group_id, name, entity_subtype, summary, t_created) "
        "VALUES (%s,%s,%s,%s,%s,%s) RETURNING uuid",
        (uuid, group_id, name, entity_type, summary, _t(0)),
    ).fetchone()[0]


def _spokes(conn, hub, n, *, prefix=None):
    """n live edges from `hub` to throwaway endpoints: gives it a live degree of n."""
    for i in range(n):
        _rel(conn, f"{prefix or hub}-sp{i}", src=hub, tgt=f"{prefix or hub}-leaf{i}")


def _rel(
    conn,
    uuid,
    *,
    src,
    tgt,
    name="uses",
    fact="x uses y",
    group_id="technical",
    episodes=None,
    t_valid=None,
    t_invalid=None,
    created_at=None,
    retrieval_count=0,
):
    return conn.execute(
        "INSERT INTO kg_relationships "
        "(uuid, group_id, src_uuid, tgt_uuid, name, fact, episodes, retrieval_count, "
        " t_created, t_valid, t_invalid) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING uuid",
        (
            uuid,
            group_id,
            src,
            tgt,
            name,
            fact,
            Json(episodes) if episodes is not None else None,
            retrieval_count,
            created_at or _t(0),
            t_valid,
            t_invalid,
        ),
    ).fetchone()[0]


def _event(
    conn,
    *,
    fact="shipped a thing",
    source="chat",
    source_ref="tb:1",
    project="synapse",
    salience=2,
    domain="technical",
    event_type=None,
    t_valid=None,
    ingested_at=None,
    reported_refs=(),
):
    return conn.execute(
        "INSERT INTO timeline_events "
        "(t_valid, fact, source, source_ref, project, salience, domain, event_type, ingested_at, "
        " reported_refs) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (
            t_valid or _t(0),
            fact,
            source,
            source_ref,
            project,
            salience,
            domain,
            event_type,
            ingested_at or _t(0),
            list(reported_refs),
        ),
    ).fetchone()[0]


def _note(conn, *, hook="User prefers X", group_id="technical", source_ref=None):
    return conn.execute(
        "INSERT INTO notes (owner_id, group_id, type, hook, body, source_ref) "
        "VALUES ('default',%s,'user',%s,'body',%s) RETURNING id",
        (group_id, hook, source_ref),
    ).fetchone()[0]


def _pref(conn, *, pref="User prefers bullet lists", group_id="technical", source_ref=None):
    return conn.execute(
        "INSERT INTO preferences (owner_id, group_id, pref, polarity, source_ref) "
        "VALUES ('default',%s,%s,'like',%s) RETURNING id",
        (group_id, pref, source_ref),
    ).fetchone()[0]


def _chunk(conn, *, session="s-dash", start=1, end=3, episode_ids, content="chunk body"):
    return conn.execute(
        "INSERT INTO chunks (session_id, start_sequence, end_sequence, episode_ids, "
        " project, content) VALUES (%s,%s,%s,%s,'synapse',%s) RETURNING id",
        (session, start, end, Json(episode_ids), content),
    ).fetchone()[0]


def _metric(
    conn,
    *,
    kind="recall",
    source="dashboard",
    query="q",
    ms_total=100.0,  # float32-exact values only (recall_metrics REAL cols)
    est_tokens=100,
    rerank_top_score=0.5,
    created_at=None,
    legs=None,  # optional {embed,bm25,vector,kg,web,rerank,notes} → ms_* columns
):
    legs = legs or {}
    return conn.execute(
        "INSERT INTO recall_metrics (kind, source, query, ms_total, est_tokens, "
        " rerank_top_score, created_at, ms_embed, ms_bm25, ms_vector, ms_kg, ms_web, "
        " ms_rerank, ms_notes) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (
            kind,
            source,
            query,
            ms_total,
            est_tokens,
            rerank_top_score,
            created_at or _t(0),
            legs.get("embed"),
            legs.get("bm25"),
            legs.get("vector"),
            legs.get("kg"),
            legs.get("web"),
            legs.get("rerank"),
            legs.get("notes"),
        ),
    ).fetchone()[0]


def _queue(
    conn,
    *,
    content="turn body",
    status="pending",
    episode_id=None,
    error=None,
    attempts=0,
    enqueued_at=None,
    processed_at=None,
    content_type="episode",
    session_id=None,
):
    return conn.execute(
        "INSERT INTO extraction_queue "
        "(episode_id, content, status, error, attempts, enqueued_at, processed_at, "
        " content_type, session_id) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (
            episode_id,
            content,
            status,
            error,
            attempts,
            enqueued_at,
            processed_at,
            content_type,
            session_id,
        ),
    ).fetchone()[0]


def _dream_run(conn, *, started_at, finished_at, counts, samples=None, stages=None, ok=True):
    return conn.execute(
        "INSERT INTO dream_runs (started_at, finished_at, stages, counts, samples, ok) "
        "VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
        (
            started_at,
            finished_at,
            Json(stages or {}),
            Json(counts),
            Json(samples or {}),
            ok,
        ),
    ).fetchone()[0]


def _client(db_url):
    from fastmcp import FastMCP

    def authorized(request):
        return request.headers.get("authorization", "") == f"Bearer {_TOKEN}"

    test_mcp = FastMCP("test-dash")
    register(test_mcp, db_url, authorized)
    return TestClient(test_mcp.http_app())


# ---------------------------------------------------------------------------
# Auth gate
# ---------------------------------------------------------------------------


def test_api_routes_require_machine_token(clean, conn, db_url):
    with _client(db_url) as client:
        for path in (
            "/dash/api/catalog",
            "/dash/api/feed",
            "/dash/api/session/s1",
            "/dash/api/search?q=x",
            "/dash/api/flags",
        ):
            r = client.get(path)
            assert r.status_code == 401
            assert r.json() == {"status": "error", "detail": "unauthorized"}
        assert client.post("/dash/api/flag", json={"kind": "note", "id": "1"}).status_code == 401
        # A valid token clears the gate.
        assert client.get("/dash/api/catalog", headers=_H).status_code == 200


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_catalog_counts(clean, conn, db_url):
    _episode(conn, session="a", seq=1, project="synapse", source="claude-code")
    _episode(conn, session="a", seq=2, project="synapse", source="claude-code")
    _episode(conn, session="b", seq=1, project=None, source="cursor")
    _entity(conn, "e1", name="Widget", group_id="personal")
    _rel(conn, "r1", src="e1", tgt="e1", group_id="technical")
    _event(conn, source_ref="tb:c1", domain="technical")
    with _client(db_url) as client:
        body = client.get("/dash/api/catalog", headers=_H).json()
    assert {"name": "synapse", "count": 2} in body["projects"]
    assert {"name": "untagged", "count": 1} in body["projects"]
    assert {"name": "claude-code", "count": 2} in body["sources"]
    assert {"name": "cursor", "count": 1} in body["sources"]
    # group_ids: union of fact/entity group_id + timeline domain, plain string list.
    assert body["group_ids"] == ["personal", "technical"]


# ---------------------------------------------------------------------------
# Feed — merge order + keyset cursor + type-filter behavior
# ---------------------------------------------------------------------------


def _feed_page(client, cursor=None, **params):
    q = dict(params)
    if cursor:
        q["cursor"] = cursor
    r = client.get("/dash/api/feed", params=q, headers=_H)
    assert r.status_code == 200
    return r.json()


def test_feed_merge_order_and_cursor_pagination(clean, conn, db_url):
    # Six items, distinct write times; reverse-chron order is A F1 E1 B F2 E2.
    a = _episode(conn, session="s", seq=1, content="ep A", created_at=_t(6))
    b = _episode(conn, session="s", seq=2, content="ep B", created_at=_t(3))
    _rel(conn, "f1", src="x", tgt="y", fact="fact one", created_at=_t(5))
    _rel(conn, "f2", src="x", tgt="y", fact="fact two", created_at=_t(2))
    e1 = _event(conn, fact="event one", source_ref="tb:e1", ingested_at=_t(4))
    e2 = _event(conn, fact="event two", source_ref="tb:e2", ingested_at=_t(1))

    with _client(db_url) as client:
        p1 = _feed_page(client, limit=3)
        assert [(i["type"], i["id"]) for i in p1["items"]] == [
            ("episode", str(a)),
            ("fact", "f1"),
            ("timeline_event", str(e1)),
        ]
        assert p1["next_cursor"]
        # timeline item carries top-level sal; fact carries inline data.
        assert p1["items"][2]["sal"] == 0.9
        assert p1["items"][1]["data"]["fact"] == "fact one"

        p2 = _feed_page(client, cursor=p1["next_cursor"], limit=3)
        assert [(i["type"], i["id"]) for i in p2["items"]] == [
            ("episode", str(b)),
            ("fact", "f2"),
            ("timeline_event", str(e2)),
        ]
        # A full second page means one more (empty) fetch drains the streams.
        p3 = _feed_page(client, cursor=p2["next_cursor"], limit=3)
        assert p3["items"] == [] and p3["next_cursor"] is None

    # No id appeared twice across pages.
    ids = [(i["type"], i["id"]) for i in p1["items"] + p2["items"]]
    assert len(ids) == len(set(ids)) == 6


def test_feed_filters_apply_only_where_column_exists(clean, conn, db_url):
    ep_syn = _episode(
        conn,
        session="s",
        seq=1,
        project="synapse",
        source="claude-code",
        content="syn ep",
        created_at=_t(10),
    )
    ep_cur = _episode(
        conn,
        session="s",
        seq=2,
        project="synapse",
        source="cursor",
        content="cursor ep",
        created_at=_t(9),
    )
    _rel(conn, "ft", src="x", tgt="y", fact="tech fact", group_id="technical", created_at=_t(8))
    _rel(conn, "fp", src="x", tgt="y", fact="personal fact", group_id="personal", created_at=_t(7))
    et = _event(
        conn,
        fact="tech event",
        source_ref="tb:t",
        project="synapse",
        domain="technical",
        ingested_at=_t(6),
    )
    ep_other = _event(
        conn,
        fact="personal event",
        source_ref="tb:p",
        project="other",
        domain="personal",
        ingested_at=_t(5),
    )

    with _client(db_url) as client:
        # group_id filter: hits facts (group_id) + timeline (domain); episodes NOT excluded.
        g = _feed_page(client, group_id="technical", limit=50)
        got = {(i["type"], i["id"]) for i in g["items"]}
        assert ("fact", "ft") in got and ("timeline_event", str(et)) in got
        assert ("episode", str(ep_syn)) in got  # episodes lack group_id -> not excluded
        assert ("fact", "fp") not in got and ("timeline_event", str(ep_other)) not in got

        # project filter: hits episodes + timeline; facts lack project -> NOT excluded.
        pj = _feed_page(client, project="synapse", limit=50)
        got = {(i["type"], i["id"]) for i in pj["items"]}
        assert ("episode", str(ep_syn)) in got and ("episode", str(ep_cur)) in got
        assert ("timeline_event", str(et)) in got
        assert ("fact", "ft") in got and ("fact", "fp") in got  # facts not excluded
        assert ("timeline_event", str(ep_other)) not in got  # project 'other' excluded

        # source filter: narrows episodes only.
        sc = _feed_page(client, source="claude-code", limit=50)
        got = {(i["type"], i["id"]) for i in sc["items"]}
        assert ("episode", str(ep_syn)) in got
        assert ("episode", str(ep_cur)) not in got  # source 'cursor' excluded
        assert ("fact", "ft") in got and ("timeline_event", str(et)) in got  # not excluded


def test_feed_untagged_filter_matches_null_empty_and_literal(clean, conn, db_url):
    """'untagged' is the catalog's bucket for NULL and '' (and a literal 'untagged'
    label lands in the same bucket); a real label is a plain equality the
    (project, created_at) index can serve."""
    null_ep = _episode(conn, session="u", seq=1, project=None, source=None, created_at=_t(4))
    blank_ep = _episode(conn, session="u", seq=2, project="", source="", created_at=_t(3))
    lit_ep = _episode(conn, session="u", seq=3, project="untagged", created_at=_t(2))
    tagged = _episode(conn, session="u", seq=4, project="synapse", created_at=_t(1))
    null_ev = _event(conn, fact="unlabelled event", source_ref="tb:u", project=None)
    tagged_ev = _event(conn, fact="labelled event", source_ref="tb:t", project="synapse")

    with _client(db_url) as client:
        got = {(i["type"], i["id"]) for i in _feed_page(client, project="untagged")["items"]}
        assert {("episode", str(e)) for e in (null_ep, blank_ep, lit_ep)} <= got
        assert ("timeline_event", str(null_ev)) in got
        assert ("episode", str(tagged)) not in got
        assert ("timeline_event", str(tagged_ev)) not in got

        got = {(i["type"], i["id"]) for i in _feed_page(client, source="untagged")["items"]}
        assert {("episode", str(null_ep)), ("episode", str(blank_ep))} <= got
        assert ("episode", str(tagged)) not in got

        got = {(i["type"], i["id"]) for i in _feed_page(client, project="synapse")["items"]}
        assert ("episode", str(tagged)) in got and ("timeline_event", str(tagged_ev)) in got
        assert ("episode", str(null_ep)) not in got

        hits = client.get(
            "/dash/api/search", params={"q": "turn", "project": "untagged"}, headers=_H
        ).json()["hits"]
        assert {h["id"] for h in hits} == {str(null_ep), str(blank_ep), str(lit_ep)}


def test_timeline_item_resolves_the_kth_event_ref(clean, conn, db_url):
    eid = _episode(conn, session="k", seq=1, content="turn with two events", created_at=_t(0))
    ev = _event(conn, fact="second event of the turn", source_ref=f"ep:{eid}#2", ingested_at=_t(5))
    with _client(db_url) as client:
        items = _feed_page(client)["items"]
        tl = client.get("/dash/api/timeline", headers=_H).json()["events"]
    by_key = {(i["type"], i["id"]): i for i in items}
    assert by_key[("timeline_event", str(ev))]["data"]["episode_id"] == eid
    assert tl[0]["episode_id"] == eid


def test_kg_reads_migration_backfills_legacy_only_rows(clean, conn, db_url):
    """The optional migration behind the t_created / entity_subtype reads copies the
    FalkorDB-era mirror into rows that only ever had it, so they stay on the feed and
    keep their type; a re-run changes nothing. The image runs without it (marker)."""
    from ingestion.schema_check import is_optional_migration

    schema = Path(__file__).resolve().parent.parent / "schema"
    (path,) = schema.glob("*_kg_dashboard_reads.sql")
    assert is_optional_migration(path)

    conn.execute(
        "INSERT INTO kg_entities (uuid, group_id, name, entity_type, created_at) "
        "VALUES ('legacy-e', 'technical', 'Old Tool', 'Tool', %s)",
        (_t(0),),
    )
    conn.execute(
        "INSERT INTO kg_relationships (uuid, group_id, src_uuid, tgt_uuid, fact, created_at) "
        "VALUES ('legacy-r', 'technical', 'legacy-e', 'legacy-e', 'an old fact', %s)",
        (_t(1),),
    )
    with _client(db_url) as client:
        before = _feed_page(client)["items"]
        assert ("fact", "legacy-r") not in {(i["type"], i["id"]) for i in before}

        conn.execute(path.read_text())
        conn.execute(path.read_text())  # idempotent

        items = {(i["type"], i["id"]): i for i in _feed_page(client)["items"]}
        assert datetime.fromisoformat(items[("fact", "legacy-r")]["ts"]) == _t(1)
        ent = client.get("/dash/api/entity/legacy-e", headers=_H).json()["entity"]
        assert ent["entity_type"] == "Tool"
        assert datetime.fromisoformat(ent["created_at"]) == _t(0)


# ---------------------------------------------------------------------------
# Episode detail + derived
# ---------------------------------------------------------------------------


def test_episode_detail(clean, conn, db_url):
    eid = _episode(
        conn,
        session="sx",
        seq=7,
        project="synapse",
        source="claude-code",
        content="full turn text",
        human="do X",
        assistant="did X",
    )
    with _client(db_url) as client:
        body = client.get(f"/dash/api/episode/{eid}", headers=_H).json()
        assert body["id"] == eid and body["session_id"] == "sx" and body["sequence"] == 7
        assert body["project"] == "synapse" and body["source"] == "claude-code"
        assert body["platform"] == "claude_code" and body["model"] == "opus"
        assert body["human_turn"] == "do X" and body["assistant_turn"] == "did X"
        assert body["content"] == "full turn text" and body["flagged"] is False
        assert body["created_at"] is not None
        # Missing episode -> 404, contract error shape.
        r = client.get("/dash/api/episode/999999", headers=_H)
        assert r.status_code == 404 and r.json()["status"] == "error"
        # Non-numeric id -> 400.
        assert client.get("/dash/api/episode/abc", headers=_H).status_code == 400


def test_episode_derived_containment_and_ep_ref(clean, conn, db_url):
    eid = _episode(conn, session="d", seq=1, content="derived source")
    other = _episode(conn, session="d", seq=2, content="unrelated")
    # Fact via jsonb array containment — numeric AND string element forms both resolve.
    _rel(conn, "dn", src="a", tgt="b", fact="numeric-linked", episodes=[eid])
    _rel(conn, "ds", src="a", tgt="b", fact="string-linked", episodes=[str(eid)])
    _rel(conn, "dx", src="a", tgt="b", fact="other-linked", episodes=[other])
    # Timeline events via source_ref: 'ep:<id>' for a turn's first event, 'ep:<id>#<k>'
    # for the k-th (ingestion/timeline_gate.py). The id-prefix near-miss must not match.
    ev = _event(conn, fact="linked event", source_ref=f"ep:{eid}", t_valid=_t(2))
    ev2 = _event(conn, fact="second event", source_ref=f"ep:{eid}#2", t_valid=_t(1))
    _event(conn, fact="other event", source_ref=f"ep:{other}")
    _event(conn, fact="prefix near-miss", source_ref=f"ep:{eid}9")
    with _client(db_url) as client:
        body = client.get(f"/dash/api/episode/{eid}/derived", headers=_H).json()
    fact_uuids = {f["uuid"] for f in body["facts"]}
    assert fact_uuids == {"dn", "ds"}  # both containment forms, other excluded
    assert [e["id"] for e in body["timeline_events"]] == [ev, ev2]


def test_episode_id_from_ref_reads_the_kth_event_form():
    assert dr._episode_id_from_ref("ep:12") == 12
    assert dr._episode_id_from_ref("ep:12#3") == 12
    assert dr._episode_id_from_ref("ep:12#") is None
    assert dr._episode_id_from_ref("ep:x#2") is None
    assert dr._episode_id_from_ref("deadbeef") is None
    assert dr._episode_id_from_ref(None) is None


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def test_session_ordering_and_highlight_echo(clean, conn, db_url):
    # Insert out of sequence order to prove ORDER BY sequence.
    _episode(conn, session="sess", seq=2, content="second", created_at=_t(2))
    first = _episode(conn, session="sess", seq=1, content="first", created_at=_t(1))
    _episode(conn, session="sess", seq=3, content="third", created_at=_t(3))
    with _client(db_url) as client:
        body = client.get(f"/dash/api/session/sess?highlight={first}", headers=_H).json()
    assert body["session_id"] == "sess"
    assert body["project"] == "synapse" and body["source"] == "claude-code"
    assert body["highlight"] == first  # echoed as int
    assert body["total"] == 3
    assert [e["sequence"] for e in body["episodes"]] == [1, 2, 3]
    # The drawer renders human_turn/assistant_turn only; content is not shipped.
    assert set(body["episodes"][0]) == {
        "id",
        "sequence",
        "created_at",
        "human_turn",
        "assistant_turn",
    }


def test_session_cap_windows_around_the_highlight(clean, conn, db_url, monkeypatch):
    """A session longer than the cap is cut to a window around the highlighted turn,
    so the drawer can always show the turn it was opened for."""
    monkeypatch.setattr(dr, "_SESSION_CAP", 4)
    ids = {seq: _episode(conn, session="long", seq=seq, created_at=_t(seq)) for seq in range(1, 11)}
    with _client(db_url) as client:
        hl = client.get(f"/dash/api/session/long?highlight={ids[8]}", headers=_H).json()
        assert [e["sequence"] for e in hl["episodes"]] == [6, 7, 8, 9]
        # The window slides back from the session's end to stay full.
        end = client.get(f"/dash/api/session/long?highlight={ids[10]}", headers=_H).json()
        assert [e["sequence"] for e in end["episodes"]] == [7, 8, 9, 10]
        # No highlight (or one outside the session): the first turns, as before.
        plain = client.get("/dash/api/session/long", headers=_H).json()
        assert [e["sequence"] for e in plain["episodes"]] == [1, 2, 3, 4]
        stray = client.get("/dash/api/session/long?highlight=999999", headers=_H).json()
        assert [e["sequence"] for e in stray["episodes"]] == [1, 2, 3, 4]
        # The window is not the session: the drawer is told how many turns there are.
        assert hl["total"] == end["total"] == plain["total"] == stray["total"] == 10


def test_session_unknown_id_is_a_404(clean, conn, db_url):
    """A session with no stored turns is not an empty session: the drawer says so."""
    with _client(db_url) as client:
        r = client.get("/dash/api/session/nope", headers=_H)
    assert r.status_code == 404
    assert r.json() == {"status": "error", "detail": "session not found"}


# ---------------------------------------------------------------------------
# Entity dossier
# ---------------------------------------------------------------------------


def test_entity_dossier(clean, conn, db_url):
    _entity(
        conn,
        "ent-main",
        name="MainThing",
        entity_type="Technology",
        summary="the subject",
    )
    _entity(conn, "ent-o1", name="OtherOne")
    _entity(conn, "ent-o2", name="OtherTwo")
    ep_a = _episode(conn, session="m", seq=1, content="mention A", created_at=_t(5))
    ep_b = _episode(conn, session="m", seq=2, content="mention B", created_at=_t(2))
    # Live edge (main is src) + superseded edge (main is tgt).
    _rel(
        conn,
        "rl",
        src="ent-main",
        tgt="ent-o1",
        name="uses",
        fact="main uses o1",
        episodes=[ep_a],
        retrieval_count=5,
        t_valid=_t(1),
        t_invalid=None,
    )
    _rel(
        conn,
        "rs",
        src="ent-o2",
        tgt="ent-main",
        name="owned",
        fact="o2 owned main",
        episodes=[ep_b],
        retrieval_count=3,
        t_valid=_t(0),
        t_invalid=_t(4),
    )

    with _client(db_url) as client:
        body = client.get("/dash/api/entity/ent-main", headers=_H).json()
        assert body["entity"]["name"] == "MainThing"
        assert body["entity"]["entity_type"] == "Technology" and body["entity"]["degree"] == 1
        # edges = live count, facts = live+superseded, served = sum(retrieval_count).
        assert body["stats"] == {"edges": 1, "facts": 2, "served": 8}

        by_uuid = {f["uuid"]: f for f in body["facts"]}
        assert by_uuid["rl"]["other"]["name"] == "OtherOne"  # main is src -> other is tgt
        assert by_uuid["rl"]["t_invalid"] is None and by_uuid["rl"]["provenance_episode_id"] == ep_a
        assert by_uuid["rs"]["other"]["name"] == "OtherTwo"  # main is tgt -> other is src
        assert by_uuid["rs"]["t_invalid"] is not None  # superseded carries t_invalid
        assert by_uuid["rs"]["provenance_episode_id"] == ep_b

        # Mentions: distinct provenance episodes, newest first, with paging + total.
        assert body["mentions"]["total"] == 2
        assert [m["episode_id"] for m in body["mentions"]["items"]] == [ep_a, ep_b]
        paged = client.get("/dash/api/entity/ent-main?mentions_offset=1", headers=_H).json()
        assert paged["mentions"]["offset"] == 1
        assert [m["episode_id"] for m in paged["mentions"]["items"]] == [ep_b]

        # Flagging a fact reflects in the dossier's fact row.
        client.post("/dash/api/flag", json={"kind": "fact", "id": "rl"}, headers=_H)
        after = client.get("/dash/api/entity/ent-main", headers=_H).json()
        assert {f["uuid"]: f["flagged"] for f in after["facts"]}["rl"] is True

        assert client.get("/dash/api/entity/nope", headers=_H).status_code == 404


def test_entity_dossier_pages_facts_live_first(clean, conn, db_url, monkeypatch):
    """The header counts every fact, so the list pages to match: live facts first (newest
    first), superseded ones after, `facts_offset` walks the rest."""
    monkeypatch.setattr(dr, "_ENTITY_FACTS_PAGE", 3)
    _entity(conn, "hub", name="Hub")
    _entity(conn, "o", name="Other")
    # Five facts. The two superseded ones are the NEWEST by t_valid, and one live fact has no
    # t_valid at all: a plain t_valid-DESC cut would fill page 1 with the dead ones.
    for name, t_valid, t_invalid in (
        ("dead-new", _t(50), _t(60)),
        ("dead-newer", _t(40), _t(45)),
        ("live-old", _t(1), None),
        ("live-mid", _t(10), None),
        ("live-undated", None, None),
    ):
        _rel(conn, name, src="hub", tgt="o", t_valid=t_valid, t_invalid=t_invalid)
    with _client(db_url) as client:
        p1 = client.get("/dash/api/entity/hub", headers=_H).json()
        p2 = client.get("/dash/api/entity/hub?facts_offset=3", headers=_H).json()
        p3 = client.get("/dash/api/entity/hub?facts_offset=99", headers=_H).json()
    assert [f["uuid"] for f in p1["facts"]] == ["live-mid", "live-old", "live-undated"]
    assert [f["uuid"] for f in p2["facts"]] == ["dead-new", "dead-newer"]
    assert p3["facts"] == []
    assert p1["stats"]["facts"] == p2["stats"]["facts"] == 5  # the count the UI pages against


def test_entity_degree_counts_live_edges_not_the_stored_column(clean, conn, db_url):
    """kg_entities.degree is a one-off backfill that nothing maintains (0 for every entity
    extraction wrote since). Every dashboard read counts live edges instead."""
    _entity(conn, "fresh", name="Fresh entity")  # stored degree 0: what the extractor writes
    _entity(conn, "stale", name="Stale entity")
    conn.execute("UPDATE kg_entities SET degree = 57 WHERE uuid = 'stale'")  # old backfill
    _spokes(conn, "fresh", 5)
    _spokes(conn, "stale", 2)
    _rel(conn, "gone", src="stale", tgt="x", t_invalid=_t(3))  # superseded: not an edge
    with _client(db_url) as client:
        dossier = client.get("/dash/api/entity/fresh", headers=_H).json()
        search = client.get(
            "/dash/api/search", params={"q": "entity", "type": "entities"}, headers=_H
        ).json()
        ta = client.get("/dash/api/graph/entities", params={"q": "entity"}, headers=_H).json()
        nb = client.get("/dash/api/graph/neighborhood?entity=fresh&depth=1", headers=_H).json()
    assert dossier["entity"]["degree"] == dossier["stats"]["edges"] == 5
    # Ranked by live degree: fresh (5) outranks stale (2) although its stored column says 0.
    assert [(h["id"], h["meta"]["degree"]) for h in search["hits"]] == [("fresh", 5), ("stale", 2)]
    assert [(e["uuid"], e["degree"]) for e in ta] == [("fresh", 5), ("stale", 2)]
    assert {n["uuid"]: n["degree"] for n in nb["nodes"]}["fresh"] == 5


def test_graph_seed_and_truncation_use_live_degree(clean, conn, db_url):
    # Two same-name nodes: the stored column favours the dead one, live edges the real one.
    _gent(conn, "dup-dead", name="Twin")
    _gent(conn, "dup-live", name="Twin")
    conn.execute("UPDATE kg_entities SET degree = 99 WHERE uuid = 'dup-dead'")
    _spokes(conn, "dup-live", 3)
    # Truncation keeps the highest LIVE degree: 'busy' has stored 0 but three edges.
    _gent(conn, "t-seed", name="Seed")
    _gent(conn, "t-busy", name="Busy")
    _gent(conn, "t-idle", name="Idle")
    conn.execute("UPDATE kg_entities SET degree = 50 WHERE uuid = 't-idle'")
    _rel(conn, "t1", src="t-seed", tgt="t-busy")
    _rel(conn, "t2", src="t-seed", tgt="t-idle")
    _spokes(conn, "t-busy", 3, prefix="tb")
    with _client(db_url) as client:
        by_norm = client.get("/dash/api/graph/neighborhood?entity=Twin", headers=_H).json()
        by_like = client.get("/dash/api/graph/neighborhood?entity=twi", headers=_H).json()
        cut = client.get(
            "/dash/api/graph/neighborhood?entity=t-seed&depth=1&limit=2", headers=_H
        ).json()
    assert by_norm["seed"] == by_like["seed"] == "dup-live"
    assert cut["truncated"] is True
    assert {n["uuid"] for n in cut["nodes"]} == {"t-seed", "t-busy"}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_per_type_and_total_by_type(clean, conn, db_url):
    tok = "zorptastic"
    epid = _episode(conn, session="q", seq=1, content=f"the {tok} subsystem failed")
    _rel(conn, "sr", src="a", tgt="b", fact=f"{tok} relates to widget", episodes=[epid])
    evid = _event(conn, fact=f"shipped {tok}", source_ref="tb:z")
    _entity(conn, "se", name=f"{tok} engine")
    _spokes(conn, "se", 4)

    with _client(db_url) as client:
        ep = client.get(f"/dash/api/search?q={tok}&type=episodes", headers=_H).json()
        # total_by_type is always all four, regardless of requested type.
        assert ep["total_by_type"] == {"episodes": 1, "facts": 1, "entities": 1, "events": 1}
        assert ep["hits"][0]["type"] == "episodes" and ep["hits"][0]["id"] == str(epid)
        assert tok in ep["hits"][0]["snippet"]
        assert ep["hits"][0]["meta"]["session_id"] == "q"

        fa = client.get(f"/dash/api/search?q={tok}&type=facts", headers=_H).json()
        assert fa["hits"][0]["type"] == "facts" and fa["hits"][0]["id"] == "sr"
        assert fa["hits"][0]["meta"]["episode_id"] == epid  # additive deep-link field

        ev = client.get(f"/dash/api/search?q={tok}&type=events", headers=_H).json()
        assert ev["hits"][0]["type"] == "events" and ev["hits"][0]["id"] == str(evid)
        # a git-sourced event has no episode to open: the SHA is all the hit can show
        assert ev["hits"][0]["meta"]["episode_id"] is None
        assert ev["hits"][0]["meta"]["source_ref"] == "tb:z"

        en = client.get(f"/dash/api/search?q={tok}&type=entities", headers=_H).json()
        assert en["hits"][0]["type"] == "entities" and en["hits"][0]["id"] == "se"
        assert en["hits"][0]["meta"] == {
            "name": f"{tok} engine",
            "entity_type": "Technology",
            "degree": 4,
        }


def test_entity_name_matching_is_a_literal_substring(clean, conn, db_url):
    """'%' and '_' typed into the omnibox, the graph typeahead or the seed box are text, not
    LIKE wildcards: 'my_skill' must not pull in 'myXskill' ahead of the real entity."""
    _entity(conn, "w1", name="myXskill")
    _spokes(conn, "w1", 3)
    _entity(conn, "w2", name="my_skill")
    _entity(conn, "w3", name="100% done")
    with _client(db_url) as client:
        for q, want in (("my_skill", ["w2"]), ("_", ["w2"]), ("%", ["w3"]), ("y_s", ["w2"])):
            se = client.get(
                "/dash/api/search", params={"q": q, "type": "entities"}, headers=_H
            ).json()
            assert [h["id"] for h in se["hits"]] == want, q
            assert se["total_by_type"]["entities"] == len(want), q
            ta = client.get("/dash/api/graph/entities", params={"q": q}, headers=_H).json()
            assert [e["uuid"] for e in ta] == want, q
        # The seed fallback (no uuid, no normalized_name hit) resolves to the literal match.
        for q, want in (("my_skill", "w2"), ("%", "w3")):
            nb = client.get("/dash/api/graph/neighborhood", params={"entity": q}, headers=_H).json()
            assert nb["seed"] == want, q
        # A backslash is text too, not an escape that swallows the next character.
        assert client.get("/dash/api/graph/entities", params={"q": "\\"}, headers=_H).json() == []


def test_search_survives_query_language_metacharacters(clean, conn, db_url):
    from ingestion.db_connection import bm25_query_text

    assert bm25_query_text('  err:"E42" (retry)* ') == "err  E42   retry"
    tok = "metachar"
    _episode(conn, session="mc", seq=1, content=f"the {tok} E42 failure")
    with _client(db_url) as client:
        r = client.get(f"/dash/api/search?q={tok}%3A%22E42%22*&type=episodes", headers=_H)
        assert r.status_code == 200 and len(r.json()["hits"]) == 1


def test_search_load_more_skips_the_tab_counts(clean, conn, db_url):
    """The four counts belong to the first page of a query; a later page carries only hits
    (the client keeps the totals it already holds)."""
    tok = "quibblesome"
    for i in range(3):
        _episode(conn, session="lm", seq=i + 1, content=f"the {tok} thing number {i}")

    with _client(db_url) as client:
        first = client.get(f"/dash/api/search?q={tok}&type=episodes&limit=2", headers=_H).json()
        assert first["total_by_type"]["episodes"] == 3 and len(first["hits"]) == 2
        more = client.get(
            f"/dash/api/search?q={tok}&type=episodes&limit=2&offset=2", headers=_H
        ).json()
        assert len(more["hits"]) == 1 and "total_by_type" not in more


# ---------------------------------------------------------------------------
# Flags — toggle + audit + feed reflection + gist resolution
# ---------------------------------------------------------------------------


def test_flag_toggle_audit_and_feed_reflection(clean, conn, db_url):
    eid = _episode(conn, session="f", seq=1, content="flag me", created_at=_t(1))
    with _client(db_url) as client:
        # insert
        r = client.post(
            "/dash/api/flag",
            json={"kind": "episode", "id": str(eid), "note": "suspect"},
            headers=_H,
        )
        assert r.json() == {"status": "ok", "flagged": True}
        flags = client.get("/dash/api/flags", headers=_H).json()["flags"]
        assert len(flags) == 1
        assert flags[0]["kind"] == "episode" and flags[0]["item_id"] == str(eid)
        assert flags[0]["note"] == "suspect" and flags[0]["gist"] == "flag me"
        # feed reflects flagged: true
        feed = client.get("/dash/api/feed", headers=_H).json()
        assert feed["items"][0]["flagged"] is True

        # unflag (toggle off)
        assert (
            client.post(
                "/dash/api/flag", json={"kind": "episode", "id": str(eid)}, headers=_H
            ).json()["flagged"]
            is False
        )
        assert client.get("/dash/api/flags", headers=_H).json()["flags"] == []

        # re-flag (partial-unique index must allow a fresh active row)
        assert (
            client.post(
                "/dash/api/flag", json={"kind": "episode", "id": str(eid)}, headers=_H
            ).json()["flagged"]
            is True
        )

    # Two flag rows for this item (one removed, one active); three audit rows in order.
    rows = conn.execute(
        "SELECT removed_at IS NULL AS active FROM dashboard_flags "
        "WHERE kind='episode' AND item_id=%s ORDER BY id",
        (str(eid),),
    ).fetchall()
    assert [r[0] for r in rows] == [False, True]
    audit = conn.execute(
        "SELECT action FROM dashboard_audit WHERE item_id=%s ORDER BY id", (str(eid),)
    ).fetchall()
    assert [a[0] for a in audit] == ["flag", "unflag", "flag"]


def test_flag_with_explicit_state_is_idempotent(clean, conn, db_url):
    """`flagged` says which state the operator wants, so a stale button or the Flags tab's
    'unflag' can't invert the intent; a repeat changes nothing and writes no audit row.
    Without `flagged` the route still toggles (older clients)."""
    eid = _episode(conn, session="f2", seq=1, content="flag me", created_at=_t(1))

    def post(**body):
        with _client(db_url) as client:
            r = client.post(
                "/dash/api/flag", json={"kind": "episode", "id": str(eid), **body}, headers=_H
            )
        assert r.status_code == 200
        return r.json()["flagged"]

    assert [post(flagged=True), post(flagged=True)] == [True, True]
    assert [post(flagged=False), post(flagged=False)] == [False, False]
    assert post() is True  # no state given: toggle
    assert post() is False
    assert post(flagged=False) is False

    active = conn.execute(
        "SELECT count(*) FROM dashboard_flags WHERE item_id=%s AND removed_at IS NULL",
        (str(eid),),
    ).fetchone()[0]
    assert active == 0
    audit = conn.execute(
        "SELECT action FROM dashboard_audit WHERE item_id=%s ORDER BY id", (str(eid),)
    ).fetchall()
    assert [a[0] for a in audit] == ["flag", "unflag", "flag", "unflag"]


def test_flag_concurrent_requests_settle_without_error(clean, conn, db_url):
    """Two simultaneous requests used to collide on the partial-unique index (500) while
    the server ended up flagged. Whatever the interleaving, every request now answers 200
    with a state, and exactly one active row exists."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    eid = _episode(conn, session="f3", seq=1, content="flag me", created_at=_t(1))
    n = 8
    gate = Barrier(n)

    def hit(_):
        gate.wait()
        return dr._set_flag(db_url, "episode", str(eid), None, True)

    with ThreadPoolExecutor(n) as pool:
        assert list(pool.map(hit, range(n))) == [True] * n
    assert (
        conn.execute(
            "SELECT count(*) FROM dashboard_flags WHERE item_id=%s AND removed_at IS NULL",
            (str(eid),),
        ).fetchone()[0]
        == 1
    )
    assert (
        conn.execute(
            "SELECT count(*) FROM dashboard_audit WHERE item_id=%s", (str(eid),)
        ).fetchone()[0]
        == 1
    )


def test_flag_rejects_non_boolean_state(clean, conn, db_url):
    with _client(db_url) as client:
        r = client.post(
            "/dash/api/flag", json={"kind": "note", "id": "1", "flagged": "yes"}, headers=_H
        )
    assert r.status_code == 400


def test_flags_list_resolves_gist_per_kind(clean, conn, db_url):
    eid = _episode(conn, session="g", seq=1, content="episode gist line")
    _rel(conn, "gf", src="a", tgt="b", fact="fact gist line")
    evid = _event(conn, fact="event gist line", source_ref="tb:g")
    nid = _note(conn, hook="note gist line")
    pid = _pref(conn, pref="pref gist line")
    with _client(db_url) as client:
        for kind, item in (
            ("episode", str(eid)),
            ("fact", "gf"),
            ("timeline_event", str(evid)),
            ("note", str(nid)),
            ("preference", str(pid)),
        ):
            client.post("/dash/api/flag", json={"kind": kind, "id": item}, headers=_H)
        flags = {
            (f["kind"], f["item_id"]): f["gist"]
            for f in client.get("/dash/api/flags", headers=_H).json()["flags"]
        }
    assert flags[("episode", str(eid))] == "episode gist line"
    assert flags[("fact", "gf")] == "fact gist line"
    assert flags[("timeline_event", str(evid))] == "event gist line"
    assert flags[("note", str(nid))] == "note gist line"
    assert flags[("preference", str(pid))] == "pref gist line"


def test_flags_list_carries_a_notes_full_body_for_inline_display(clean, conn, db_url):
    """A note has no turn to jump to, so the Flags tab shows its hook and body in place;
    every other kind reports no body."""
    nid = _note(conn, hook="note hook")
    conn.execute("UPDATE notes SET body = %s WHERE id = %s", ("line one\nline two", nid))
    evid = _event(conn, fact="event text", source_ref="tb:b")
    with _client(db_url) as client:
        for kind, item in (("note", str(nid)), ("timeline_event", str(evid))):
            client.post("/dash/api/flag", json={"kind": kind, "id": item}, headers=_H)
        flags = client.get("/dash/api/flags", headers=_H).json()["flags"]
    bodies = {f["kind"]: f["body"] for f in flags}
    assert bodies == {"note": "line one\nline two", "timeline_event": None}


def test_flags_list_names_the_episode_each_flag_leads_to(clean, conn, db_url):
    """Jump needs a target: a fact's first provenance episode, the ep:<id> turn an event,
    preference or note was born from, and the episode itself. A ref that is not a turn
    (a git SHA) or whose episode is gone resolves to null."""
    ep = _episode(conn, session="g2", seq=1, content="source turn")
    gone = ep + 1000
    _entity(conn, "ge1", name="Widget")
    _rel(conn, "gf-ok", src="ge1", tgt="ge1", episodes=[ep, gone])
    _rel(conn, "gf-none", src="ge1", tgt="ge1", episodes=None)
    ev_ok = _event(conn, source_ref=f"ep:{ep}#2")
    ev_sha = _event(conn, source="git:synapse", source_ref="deadbeef")
    ev_gone = _event(conn, source_ref=f"ep:{gone}")
    pid = _pref(conn, pref="p", source_ref=f"ep:{ep}")
    nid = _note(conn, hook="n", source_ref=f"ep:{ep}")
    with _client(db_url) as client:
        for kind, item in (
            ("episode", str(ep)),
            ("fact", "gf-ok"),
            ("fact", "gf-none"),
            ("timeline_event", str(ev_ok)),
            ("timeline_event", str(ev_sha)),
            ("timeline_event", str(ev_gone)),
            ("preference", str(pid)),
            ("note", str(nid)),
        ):
            client.post("/dash/api/flag", json={"kind": kind, "id": item}, headers=_H)
        flags = client.get("/dash/api/flags", headers=_H).json()["flags"]
    got = {(f["kind"], f["item_id"]): f["episode_id"] for f in flags}
    assert got == {
        ("episode", str(ep)): ep,
        ("fact", "gf-ok"): ep,
        ("fact", "gf-none"): None,
        ("timeline_event", str(ev_ok)): ep,
        ("timeline_event", str(ev_sha)): None,
        ("timeline_event", str(ev_gone)): None,
        ("preference", str(pid)): ep,
        ("note", str(nid)): ep,
    }


def test_flag_rejects_bad_input(clean, conn, db_url):
    with _client(db_url) as client:
        assert (
            client.post("/dash/api/flag", json={"kind": "bogus", "id": "1"}, headers=_H).status_code
            == 400
        )
        assert (
            client.post("/dash/api/flag", json={"kind": "note"}, headers=_H).status_code == 400
        )  # missing id


# ---------------------------------------------------------------------------
# Recall history (phase 2) — kind filter, newest-first order, limit cap, token gate
# ---------------------------------------------------------------------------


def test_recall_history_shape_order_and_kind_filter(clean, conn, db_url):
    # Three recall rows at distinct times + a non-recall row that must be excluded.
    _metric(
        conn, query="oldest", ms_total=120.0, est_tokens=400, rerank_top_score=0.5, created_at=_t(1)
    )
    _metric(
        conn,
        query="newest",
        ms_total=350.0,
        est_tokens=1900,
        rerank_top_score=0.875,
        created_at=_t(3),
    )
    _metric(
        conn,
        query="middle",
        ms_total=210.0,
        est_tokens=900,
        rerank_top_score=0.75,
        created_at=_t(2),
    )
    _metric(conn, kind="episodes", query="drilldown", created_at=_t(9))  # excluded by kind filter

    with _client(db_url) as client:
        body = client.get("/dash/api/recall/history", headers=_H).json()
    items = body["items"]
    # Newest first; the kind='episodes' row is filtered out despite being most recent.
    assert [i["query"] for i in items] == ["newest", "middle", "oldest"]
    top = items[0]
    assert top["source"] == "dashboard" and top["ms_total"] == 350.0
    assert top["est_tokens"] == 1900 and top["rerank_top_score"] == 0.875
    assert isinstance(top["id"], int) and top["created_at"] is not None
    # Exactly the contract's column set — no leaked telemetry columns.
    assert set(top) == {
        "id",
        "created_at",
        "query",
        "source",
        "ms_total",
        "est_tokens",
        "rerank_top_score",
    }


def test_recall_history_limit_cap_and_default(clean, conn, db_url):
    for i in range(5):
        _metric(conn, query=f"q{i}", created_at=_t(i))
    with _client(db_url) as client:
        # explicit small limit is honored
        assert len(client.get("/dash/api/recall/history?limit=2", headers=_H).json()["items"]) == 2
        # over-cap clamps (all 5 returned, no error) — the 200 cap just bounds the read
        assert (
            len(client.get("/dash/api/recall/history?limit=9999", headers=_H).json()["items"]) == 5
        )
        # unparseable limit falls back to the default (≥ 5, so all 5 returned)
        assert (
            len(client.get("/dash/api/recall/history?limit=abc", headers=_H).json()["items"]) == 5
        )


def test_recall_history_requires_token(clean, conn, db_url):
    with _client(db_url) as client:
        r = client.get("/dash/api/recall/history")
        assert r.status_code == 401
        assert r.json() == {"status": "error", "detail": "unauthorized"}


# Proposals (phase 2b) — unified skills + config review
# ---------------------------------------------------------------------------


@pytest.fixture()
def clean_proposals(conn):
    """Wipe the public tables these tests touch (episodes for provenance) + both lane
    proposal tables + dashboard_audit, before AND after — deterministic ids for a serial
    DB-group run, and no synthetic row (proposal OR episode) leaks past the test."""

    def _w():
        _wipe(conn)  # episodes, kg, timeline, notes, prefs, dashboard_flags/audit
        conn.execute("TRUNCATE skills_lane.skill_gap_candidates RESTART IDENTITY CASCADE")
        conn.execute("TRUNCATE config_lane.config_proposals RESTART IDENTITY CASCADE")
        # a skills approve applies the proposal to the registry (the lane's accept)
        conn.execute("DELETE FROM skills_lane.skill_registry WHERE name = 'latency-triage'")
        conn.execute("DELETE FROM skills_lane.skill_history WHERE name = 'latency-triage'")

    _w()
    yield
    _w()


def _skill_proposal(
    conn,
    *,
    kind="derive",
    name="latency-triage",
    status="proposed",
    summary="Recurring recall latency debugging with no reusable playbook.",
    proposal_body="# latency-triage\n\nWhen recall p95 regresses, read the waterfall.\n\n- rerank\n- kg",
    evidence=None,
    created_at=None,
):
    ev = (
        evidence
        if evidence is not None
        else [
            {
                "session_id": "sess-sk",
                "class": "grounded",
                "signal": "explicit_request",
                "why": "operator asked for a reusable playbook",
            }
        ]
    )
    # schema/023 constraint: an accepted/promoted candidate must carry grounded_weight > 0.
    gw = 3.0 if status in ("accepted", "promoted") else 0.0
    gs = 1 if status in ("accepted", "promoted") else 0
    return conn.execute(
        "INSERT INTO skills_lane.skill_gap_candidates "
        "(kind, name, status, summary, proposal_body, evidence, grounded_weight, "
        " grounded_sessions, created_at) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (kind, name, status, summary, proposal_body, Json(ev), gw, gs, created_at or _t(0)),
    ).fetchone()[0]


def _config_proposal(
    conn,
    *,
    kind="edit",
    file_key="CLAUDE.md",
    scope="general",
    status="proposed",
    summary="Add the raw-SQL / no-ORM rule the operator restated 5x.",
    diff="--- a/CLAUDE.md\n+++ b/CLAUDE.md\n@@ -1 +1,2 @@\n context\n+Prefer raw SQL over the ORM.",
    evidence=None,
    created_at=None,
):
    ev = (
        evidence
        if evidence is not None
        else [{"session_id": "sess-cf", "signal": "correction", "why": "restated the rule 5 times"}]
    )
    return conn.execute(
        "INSERT INTO config_lane.config_proposals "
        "(kind, file_key, scope, status, summary, diff, evidence, created_at) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (kind, file_key, scope, status, summary, diff, Json(ev), created_at or _t(0)),
    ).fetchone()[0]


def test_proposals_unified_list_and_pending_count(clean_proposals, conn, db_url):
    sk_prop = _skill_proposal(conn, name="latency-triage", status="proposed", created_at=_t(5))
    cf_prop = _config_proposal(conn, file_key="CLAUDE.md", status="proposed", created_at=_t(4))
    _skill_proposal(conn, name="graph-inspect", status="accepted", created_at=_t(3))
    _config_proposal(conn, file_key="rules/testing.md", status="rejected", created_at=_t(2))
    _skill_proposal(conn, name="not-yet", status="observe", created_at=_t(1))  # excluded

    with _client(db_url) as client:
        body = client.get("/dash/api/proposals", headers=_H).json()
        # pending_count = both lanes' 'proposed' rows; nav badge reads this.
        assert body["pending_count"] == 2
        rows = body["proposals"]
        ids = [r["id"] for r in rows]
        assert f"skill:{sk_prop}" in ids and f"config:{cf_prop}" in ids
        assert "observe" not in {r["status"] for r in rows}  # observe not a proposal
        # namespaced ids + normalized kinds + names (config uses file_key).
        by_id = {r["id"]: r for r in rows}
        assert by_id[f"skill:{sk_prop}"]["kind"] == "skill"
        assert by_id[f"skill:{sk_prop}"]["gist"].startswith("Recurring recall latency")
        assert by_id[f"config:{cf_prop}"]["kind"] == "config-edit"
        assert by_id[f"config:{cf_prop}"]["name"] == "CLAUDE.md"
        assert all("age_days" in r and "created_at" in r for r in rows)
        # merged newest-first across lanes (created_at DESC).
        assert ids == sorted(ids, key=lambda i: by_id[i]["created_at"], reverse=True)

        # status filter narrows the view (pending_count unchanged).
        only_prop = client.get("/dash/api/proposals?status=proposed", headers=_H).json()
        assert {r["status"] for r in only_prop["proposals"]} == {"proposed"}
        assert only_prop["pending_count"] == 2

        # kind filter narrows to one lane.
        only_skill = client.get("/dash/api/proposals?kind=skill", headers=_H).json()
        assert {r["kind"] for r in only_skill["proposals"]} == {"skill"}


def test_proposals_list_keeps_old_proposed_rows_behind_a_decided_backlog(
    clean_proposals, conn, db_url, monkeypatch
):
    """The per-lane cap must not hide a still-'proposed' row behind newer decided ones:
    proposed rows come first inside each lane's window, and ?status= is applied in SQL."""
    monkeypatch.setattr(dr, "_PROPOSALS_LANE_CAP", 3)
    old_skill = _skill_proposal(conn, name="old-skill", status="proposed", created_at=_t(1))
    old_cfg = _config_proposal(conn, file_key="old.md", status="proposed", created_at=_t(1))
    for i in range(4):  # newer decided rows that would fill each lane's window
        _skill_proposal(conn, name=f"done-{i}", status="rejected", created_at=_t(10 + i))
        _config_proposal(conn, file_key=f"done-{i}.md", status="rejected", created_at=_t(10 + i))

    with _client(db_url) as client:
        ids = [r["id"] for r in client.get("/dash/api/proposals", headers=_H).json()["proposals"]]
        assert f"skill:{old_skill}" in ids and f"config:{old_cfg}" in ids
        only = client.get("/dash/api/proposals?status=rejected", headers=_H).json()
        assert {r["status"] for r in only["proposals"]} == {"rejected"}
        assert len(only["proposals"]) == 6  # the cap applies per lane, after the filter
        # a status no row has is an empty view, not an error
        assert (
            client.get("/dash/api/proposals?status=applied", headers=_H).json()["proposals"] == []
        )


def test_proposal_detail_payload_types_and_provenance(clean_proposals, conn, db_url):
    # Episode sharing the skill evidence's session_id: the session is named, the turn is not.
    ep = _episode(conn, session="sess-sk", seq=1, content="the p95 regression session")
    sk = _skill_proposal(conn, name="latency-triage", status="proposed")
    cf = _config_proposal(conn, file_key="CLAUDE.md", status="proposed")

    with _client(db_url) as client:
        sd = client.get(f"/dash/api/proposals/skill:{sk}", headers=_H).json()
        assert sd["id"] == f"skill:{sk}" and sd["kind"] == "skill"
        assert sd["payload"]["type"] == "markdown"
        assert sd["payload"]["content"].startswith("# latency-triage")
        assert (
            isinstance(sd["evidence"], list) and sd["evidence"][0]["signal"] == "explicit_request"
        )
        assert sd["provenance_sessions"] == ["sess-sk"]
        assert ep not in sd["provenance_episodes"]  # nothing in the evidence names a turn
        assert sd["audit_log"] == []  # not decided yet

        cd = client.get(f"/dash/api/proposals/config:{cf}", headers=_H).json()
        assert cd["kind"] == "config-edit" and cd["name"] == "CLAUDE.md"
        assert cd["payload"]["type"] == "diff"
        assert "+++ b/CLAUDE.md" in cd["payload"]["content"]

        # malformed id -> 400, missing -> 404.
        assert client.get("/dash/api/proposals/bogus", headers=_H).status_code == 400
        assert client.get("/dash/api/proposals/skill:999999", headers=_H).status_code == 404


def test_proposal_provenance_names_cited_turns_and_every_evidence_session(
    clean_proposals, conn, db_url
):
    """Chips used to be the lowest-id turns of the evidence sessions under one global cap of
    40, so later sessions got none and the turns shown were not the ones cited. Now explicit
    refs are the episode chips, and every evidence session is listed once, in order."""
    cited = _episode(conn, session="sess-a", seq=5, content="the cited turn")
    for i in range(1, 60):  # more early turns than the old cap would have shown
        _episode(conn, session="sess-a", seq=100 + i, content=f"filler {i}")
    _episode(conn, session="sess-b", seq=1, content="late session turn")
    evidence = [
        {"session_id": "sess-a", "signal": "correction", "episode_id": cited},
        {"session_id": "sess-b", "signal": "correction", "source_ref": f"ep:{cited}#2"},
        {"session_id": "sess-a", "signal": "correction"},
        {"session_id": "sess-c", "signal": "correction", "episode_id": str(cited + 1)},
        {"signal": "correction", "why": "no session"},
    ]
    sk = _skill_proposal(conn, name="latency-triage", evidence=evidence)
    with _client(db_url) as client:
        d = client.get(f"/dash/api/proposals/skill:{sk}", headers=_H).json()
    assert d["provenance_episodes"] == [cited, cited + 1]
    assert d["provenance_sessions"] == ["sess-a", "sess-b", "sess-c"]


def test_proposal_decision_approve_writes_state_and_audit(clean_proposals, conn, db_url):
    sk = _skill_proposal(conn, name="latency-triage", status="proposed")
    with _client(db_url) as client:
        r = client.post(
            f"/dash/api/proposals/skill:{sk}/decision",
            json={"action": "approve", "note": "clear win"},
            headers=_H,
        )
        assert r.status_code == 200
        # dashboard approve maps to the skills lane's 'accept', which APPLIES the proposal:
        # the drafted body lands in the skill registry and the row goes to 'promoted'.
        assert r.json()["status"] == "promoted"

    # Lane row transitioned; dashboard_audit carries the namespaced id + note.
    row = conn.execute(
        "SELECT status FROM skills_lane.skill_gap_candidates WHERE id=%s", (sk,)
    ).fetchone()
    assert row[0] == "promoted"
    reg = conn.execute(
        "SELECT body, status FROM skills_lane.skill_registry WHERE name='latency-triage'"
    ).fetchone()
    assert reg[0].startswith("# latency-triage") and reg[1] == "active"
    audit = conn.execute(
        "SELECT action, kind, item_id, detail FROM dashboard_audit WHERE item_id=%s",
        (f"skill:{sk}",),
    ).fetchone()
    assert audit[0] == "proposal_approve" and audit[1] == "skill" and audit[2] == f"skill:{sk}"
    assert audit[3]["note"] == "clear win"

    # Detail's audit_log reflects the decision.
    with _client(db_url) as client:
        detail = client.get(f"/dash/api/proposals/skill:{sk}", headers=_H).json()
    assert [a["action"] for a in detail["audit_log"]] == ["proposal_approve"]
    assert detail["audit_log"][0]["note"] == "clear win"


def test_proposal_approve_without_draft_is_refused_and_not_audited(clean_proposals, conn, db_url):
    sk = _skill_proposal(conn, name="latency-triage", status="proposed", proposal_body=None)
    with _client(db_url) as client:
        r = client.post(
            f"/dash/api/proposals/skill:{sk}/decision",
            json={"action": "approve", "note": "looks right"},
            headers=_H,
        )
        assert r.status_code == 200
        assert r.json()["status"] == "refused" and "no draft yet" in r.json()["detail"]

    assert (
        conn.execute(
            "SELECT status FROM skills_lane.skill_gap_candidates WHERE id=%s", (sk,)
        ).fetchone()[0]
        == "proposed"
    )
    assert (
        conn.execute(
            "SELECT count(*) FROM dashboard_audit WHERE item_id=%s", (f"skill:{sk}",)
        ).fetchone()[0]
        == 0
    )


def test_proposal_decision_only_applies_to_a_proposed_row(clean_proposals, conn, db_url):
    """A pane loaded earlier (another tab, the CLI) can show 'proposed' for a row that
    has moved on. Approving an applied config proposal used to put it back to 'accepted'
    (the CLI would apply it again) and rejecting a promoted skill left it live in the
    registry with a 30-day cooldown. Anything but 'proposed' is refused and not audited."""
    applied = _config_proposal(conn, file_key="a.md", status="applied")
    rejected = _config_proposal(conn, file_key="b.md", status="rejected")
    promoted = _skill_proposal(conn, name="latency-triage", status="promoted")
    with _client(db_url) as client:
        for pid, action in (
            (f"config:{applied}", "approve"),
            (f"config:{applied}", "reject"),
            (f"config:{rejected}", "approve"),
            (f"skill:{promoted}", "reject"),
            (f"skill:{promoted}", "approve"),
        ):
            r = client.post(
                f"/dash/api/proposals/{pid}/decision",
                json={"action": action, "note": "second thoughts"},
                headers=_H,
            )
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "refused", (pid, action)
            assert "not 'proposed'" in body["detail"]
    assert conn.execute(
        "SELECT status FROM config_lane.config_proposals WHERE id = ANY(%s) ORDER BY id",
        ([applied, rejected],),
    ).fetchall() == [("applied",), ("rejected",)]
    assert conn.execute(
        "SELECT status, rejected_until FROM skills_lane.skill_gap_candidates WHERE id=%s",
        (promoted,),
    ).fetchone() == ("promoted", None)
    assert conn.execute("SELECT count(*) FROM dashboard_audit").fetchone()[0] == 0


def test_proposal_double_decision_is_refused_the_second_time(clean_proposals, conn, db_url):
    cf = _config_proposal(conn, file_key="rules/testing.md", status="proposed")
    with _client(db_url) as client:
        url = f"/dash/api/proposals/config:{cf}/decision"
        first = client.post(url, json={"action": "reject", "note": "too generic"}, headers=_H)
        again = client.post(url, json={"action": "reject", "note": "too generic"}, headers=_H)
    assert first.json()["status"] == "rejected"
    assert again.json()["status"] == "refused"
    assert (
        conn.execute(
            "SELECT count(*) FROM dashboard_audit WHERE item_id=%s", (f"config:{cf}",)
        ).fetchone()[0]
        == 1
    )


def test_proposal_reject_requires_note_and_records_reason(clean_proposals, conn, db_url):
    cf = _config_proposal(conn, file_key="rules/testing.md", status="proposed")
    with _client(db_url) as client:
        # reject with no note -> 400, row untouched.
        no_note = client.post(
            f"/dash/api/proposals/config:{cf}/decision", json={"action": "reject"}, headers=_H
        )
        assert no_note.status_code == 400
        blank = client.post(
            f"/dash/api/proposals/config:{cf}/decision",
            json={"action": "reject", "note": "   "},
            headers=_H,
        )
        assert blank.status_code == 400
        assert (
            conn.execute(
                "SELECT status FROM config_lane.config_proposals WHERE id=%s", (cf,)
            ).fetchone()[0]
            == "proposed"
        )
        # reject with a reason -> lane 'rejected', reason stored + audited.
        ok = client.post(
            f"/dash/api/proposals/config:{cf}/decision",
            json={"action": "reject", "note": "too generic"},
            headers=_H,
        )
        assert ok.status_code == 200 and ok.json()["status"] == "rejected"

    row = conn.execute(
        "SELECT status, reject_reason FROM config_lane.config_proposals WHERE id=%s", (cf,)
    ).fetchone()
    assert row[0] == "rejected" and row[1] == "too generic"
    audit = conn.execute(
        "SELECT action, detail FROM dashboard_audit WHERE item_id=%s", (f"config:{cf}",)
    ).fetchone()
    assert audit[0] == "proposal_reject" and audit[1]["note"] == "too generic"


def test_skill_reject_sets_30day_cooldown(clean_proposals, conn, db_url):
    """The skills lane implements a 30-day reject cooldown (rejected_until); the dashboard
    reject surfaces it via the lane's own act. The config lane has no such column (gap noted
    in docs/dashboard-contract.md)."""
    sk = _skill_proposal(conn, name="latency-triage", status="proposed")
    with _client(db_url) as client:
        client.post(
            f"/dash/api/proposals/skill:{sk}/decision",
            json={"action": "reject", "note": "one-off, not worth a skill"},
            headers=_H,
        )
    row = conn.execute(
        "SELECT status, reject_reason, rejected_until FROM skills_lane.skill_gap_candidates "
        "WHERE id=%s",
        (sk,),
    ).fetchone()
    assert row[0] == "rejected" and row[1] == "one-off, not worth a skill"
    assert row[2] is not None  # cooldown set ~30 days out
    days = (row[2] - datetime.now(UTC)).days
    assert 27 <= days <= 30


def test_proposals_require_machine_token(clean_proposals, conn, db_url):
    sk = _skill_proposal(conn, status="proposed")
    with _client(db_url) as client:
        assert client.get("/dash/api/proposals").status_code == 401
        assert client.get(f"/dash/api/proposals/skill:{sk}").status_code == 401
        assert (
            client.post(
                f"/dash/api/proposals/skill:{sk}/decision", json={"action": "approve"}
            ).status_code
            == 401
        )


# ---------------------------------------------------------------------------
# Phase 3 — SSE live stream (043 NOTIFY trigger + ring buffer + /dash/api/stream)
# ---------------------------------------------------------------------------


def test_dash_notify_trigger_fires_on_insert(clean, conn, db_url):
    """schema/043 arms AFTER INSERT triggers that pg_notify('dash_feed', {type,id}). LISTEN
    on a side connection, insert one of each feed type, assert the tiny payloads arrive with
    the right id kind (episode/timeline -> numeric id, fact -> uuid string)."""
    import json as _json

    listener = psycopg.connect(db_url, autocommit=True)
    try:
        listener.execute("LISTEN dash_feed")
        eid = _episode(conn, session="n", seq=1, content="notify me")
        _rel(conn, "nf", src="a", tgt="b", fact="notify fact")
        evid = _event(conn, fact="notify event", source_ref="tb:n")
        got: dict[str, object] = {}
        for n in listener.notifies(timeout=10, stop_after=3):
            p = _json.loads(n.payload)
            got[p["type"]] = p["id"]
        assert got["episode"] == eid  # bigint id -> JSON number -> int
        assert got["fact"] == "nf"  # uuid -> JSON string
        assert got["timeline_event"] == evid
    finally:
        listener.close()


def test_feed_event_buffer_replay_and_reset():
    """Pure-unit: the ring buffer's resume() replay/reset logic — no DB, no HTTP."""
    from mcp_server.dashboard_routes import _FeedEventBuffer

    buf = _FeedEventBuffer(capacity=3)
    ids = [buf.append({"type": "episode", "id": str(i)}) for i in (1, 2, 3)]
    assert ids == [1, 2, 3]
    assert buf.head == 3 and buf.oldest() == 1

    mode, ev = buf.resume(1)  # still present -> replay the missed tail
    assert mode == "replay" and [e[0] for e in ev] == [2, 3]
    assert buf.resume(3) == ("replay", [])  # exactly current -> nothing to replay
    assert buf.resume(99)[0] == "reset"  # client ahead of us (buffer reset) -> resync

    buf.append({"type": "episode", "id": "4"})
    buf.append({"type": "episode", "id": "5"})  # capacity 3 -> evicts ids 1,2; keeps 3,4,5
    assert buf.oldest() == 3 and buf.head == 5
    assert buf.resume(1)[0] == "reset"  # gap: id 2 was evicted -> reset
    mode, ev = buf.resume(2)  # contiguous (3 is the next kept id) -> replay 3,4,5
    assert mode == "replay" and [e[0] for e in ev] == [3, 4, 5]


def test_stream_requires_token(clean, conn, db_url):
    with _client(db_url) as client:
        r = client.get("/dash/api/stream")
        assert r.status_code == 401
        assert r.json() == {"status": "error", "detail": "unauthorized"}


async def test_stream_manager_hydrates_notify_into_buffer(clean, conn, db_url):
    """The LISTEN worker + hydration seam (the flake-proof core of the feature): subscribe
    starts the worker, an insert fires NOTIFY, the worker hydrates the row into the ring
    buffer as a full FeedItem. Deterministic — waits for the worker's `listening` event
    before inserting, so the NOTIFY is never raced against worker startup."""
    from mcp_server.dashboard_routes import _StreamManager

    mgr = _StreamManager(db_url)
    mgr.subscribe()
    try:
        await asyncio.wait_for(mgr.listening.wait(), timeout=10)
        eid = _episode(conn, session="strm", seq=1, content="stream me", project="synapse")
        for _ in range(100):  # up to ~10s for the hydrated item to land in the buffer
            if mgr.buffer.head >= 1:
                break
            await asyncio.sleep(0.1)
        assert mgr.buffer.head >= 1, "worker did not hydrate the NOTIFY into the buffer"
        _, item = mgr.buffer._events[-1]
        assert item["type"] == "episode" and item["id"] == str(eid)
        assert item["gist"] == "stream me" and item["flagged"] is False
        assert item["project"] == "synapse"
        # The status refresher populated the shared snapshot the header badge reads.
        await asyncio.wait_for(mgr.status_ready.wait(), timeout=10)
        assert set(mgr.status) == {"queue_depth", "processing", "failed", "active"}
        assert isinstance(mgr.status["queue_depth"], int)
    finally:
        mgr.unsubscribe()


async def test_stream_manager_relistens_after_the_connection_drops(
    clean, conn, db_url, monkeypatch
):
    """A dropped LISTEN connection (Postgres restart, network blip) must not leave the
    feed silently stale while clients stay subscribed: the worker reconnects, marks the
    gap with a `reset` event in the ring (clients refetch page 1), and delivers again."""
    from mcp_server.dashboard_routes import _StreamManager

    monkeypatch.setattr(dr, "_STREAM_RECONNECT_MIN_S", 0.05)
    mgr = _StreamManager(db_url)
    mgr.subscribe()
    try:
        await asyncio.wait_for(mgr.listening.wait(), timeout=10)
        conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname = current_database() AND pid <> pg_backend_pid() "
            "AND query LIKE 'LISTEN dash_feed%'"
        )
        for _ in range(100):  # the drop is seen, then the worker is listening again
            await asyncio.sleep(0.1)
            if mgr.buffer.head >= 1 and mgr.listening.is_set():
                break
        assert mgr.buffer.head == 1 and mgr.listening.is_set(), "worker did not re-LISTEN"
        assert mgr.buffer._events[-1][1] == {"type": "reset"}

        eid = _episode(conn, session="strm2", seq=1, content="after the drop")
        for _ in range(100):
            if mgr.buffer.head >= 2:
                break
            await asyncio.sleep(0.1)
        assert mgr.buffer.head == 2, "delivery did not resume after the reconnect"
        assert mgr.buffer._events[-1][1]["id"] == str(eid)
    finally:
        mgr.unsubscribe()


async def _never_disconnected() -> bool:
    return False


async def _frames_until(stream, event: str, limit: int = 30) -> list[str]:
    """Pull frames off an _event_stream until one is `event:`-named (inclusive)."""
    out: list[str] = []
    for _ in range(limit):
        frame = await asyncio.wait_for(stream.__anext__(), timeout=10)
        out.append(frame)
        if frame.startswith(f"event: {event}\n"):
            return out
    raise AssertionError(f"no {event} frame in {limit} frames: {out}")


def _frame_payload(frame: str) -> dict:
    import json as _json

    return _json.loads(frame.split("data: ", 1)[1])


def test_processing_status_counts_pending_processing_and_failed(clean, conn, db_url):
    """The header badge needs more than the pending count: claimed rows (processing) used to
    vanish from it, and failed rows never showed. queue_depth/active stay for compatibility."""
    for status in ("pending", "pending", "processing", "failed", "done"):
        _queue(conn, status=status, enqueued_at=_t(0))
    assert dr._processing_status(db_url) == {
        "queue_depth": 2,
        "processing": 1,
        "failed": 1,
        "active": True,
    }


async def test_stream_first_frames_are_ready_then_a_fresh_status(clean, conn, db_url):
    """A connection opens with `ready` (the stream is positioned and LISTENing: the client
    refetches page 1 now) and then a status snapshot computed for THIS connection, not the
    zeros the manager starts with or the previous run's leftovers."""
    from mcp_server.dashboard_routes import _event_stream, _StreamManager

    _queue(conn, status="pending", enqueued_at=_t(0))
    _queue(conn, status="processing", enqueued_at=_t(0))
    mgr = _StreamManager(db_url)
    stream = _event_stream(mgr, _never_disconnected, None)
    try:
        frames = await _frames_until(stream, "processing_status")
        assert frames[0] == "event: ready\ndata: {}\n\n"
        assert _frame_payload(frames[-1]) == {
            "queue_depth": 1,
            "processing": 1,
            "failed": 0,
            "active": True,
        }
    finally:
        await stream.aclose()
    assert mgr.status is None  # the last client left: the snapshot is dropped, not kept stale

    # Next connection, after the queue changed: its first status reflects the change.
    _queue(conn, status="pending", enqueued_at=_t(0))
    stream = _event_stream(mgr, _never_disconnected, None)
    try:
        frames = await _frames_until(stream, "processing_status")
        assert _frame_payload(frames[-1])["queue_depth"] == 2
    finally:
        await stream.aclose()


async def test_stream_delivers_a_row_committed_right_after_ready(clean, conn, db_url):
    """`ready` means the lazily-started worker is LISTENing: a row committed the moment the
    client sees it (the client's page-1 fetch follows it) is delivered, not lost to the
    worker's startup the way a first-subscriber NOTIFY used to be."""
    from mcp_server.dashboard_routes import _event_stream, _StreamManager

    mgr = _StreamManager(db_url)
    stream = _event_stream(mgr, _never_disconnected, None)
    try:
        await _frames_until(stream, "ready")
        assert mgr.listening.is_set()
        eid = _episode(conn, session="after-ready", seq=1, content="right after ready")
        frames = await _frames_until(stream, "new_episode")
        assert _frame_payload(frames[-1])["id"] == str(eid)
    finally:
        await stream.aclose()


async def test_stream_status_is_dropped_when_the_refresh_fails(clean, db_url, monkeypatch):
    """A count that can no longer be read is withdrawn (no status frames), not served stale."""
    from mcp_server.dashboard_routes import _StreamManager

    def boom(_url):
        raise RuntimeError("db down")

    monkeypatch.setattr(dr, "_processing_status", boom)
    mgr = _StreamManager(db_url)
    mgr.status = {"queue_depth": 9, "active": False}
    mgr.subscribe()
    try:
        await asyncio.wait_for(mgr.status_ready.wait(), timeout=10)
        assert mgr.status is None
    finally:
        mgr.unsubscribe()


async def test_stream_replays_buffered_events_after_ready(clean, db_url):
    from mcp_server.dashboard_routes import _event_stream, _StreamManager

    mgr = _StreamManager(db_url)
    for i in (1, 2, 3):
        mgr.buffer.append({"type": "episode", "id": str(i)})
    stream = _event_stream(mgr, _never_disconnected, 1)  # client has event 1, missed 2 and 3
    try:
        frames = await _frames_until(stream, "processing_status")
        assert [f.split("\n")[0] for f in frames] == [
            "event: ready",
            "event: new_episode",
            "event: new_episode",
            "event: processing_status",
        ]
        assert [f.split("\n")[1] for f in frames[1:3]] == ["id: 2", "id: 3"]
    finally:
        await stream.aclose()


async def test_stream_resets_a_connection_the_ring_buffer_outran(clean, db_url):
    """A reader blocked on a yield while the ring evicts events it has not sent must be told
    (`reset`: refetch page 1), the same as a resume across the gap, not skipped silently."""
    from mcp_server.dashboard_routes import _event_stream, _FeedEventBuffer, _StreamManager

    mgr = _StreamManager(db_url)
    mgr.buffer = _FeedEventBuffer(capacity=3)
    stream = _event_stream(mgr, _never_disconnected, None)
    try:
        await _frames_until(stream, "processing_status")  # suspended on that yield
        for i in range(1, 6):  # a burst of 5 into 3 slots: events 1 and 2 are evicted
            mgr.buffer.append({"type": "episode", "id": str(i)})
        frame = await asyncio.wait_for(stream.__anext__(), timeout=10)
        assert frame == "event: reset\ndata: {}\n\n"
        # It resumed from the head: later events stream normally, the evicted ones are not
        # replayed as if nothing happened.
        mgr.buffer.append({"type": "episode", "id": "6"})
        frames = await _frames_until(stream, "new_episode")
        assert frames[-1].splitlines()[1] == "id: 6"
    finally:
        await stream.aclose()


def test_sse_frame_format():
    """The wire framing helper: `event:` line, optional `id:` line (feed events only), a
    `data:` line, blank-line terminated. This + the manager-seam test above cover the SSE
    route's moving parts without an HTTP body read.

    NOTE: there is deliberately NO end-to-end TestClient streaming test. Starlette's
    TestClient (httpx ASGITransport) BUFFERS the response body, so it never yields frames
    from an unbounded text/event-stream and a `client.stream(...).iter_lines()` read hangs
    forever. Per the build brief, the route is instead proven at the seam: the trigger fires
    NOTIFY (test_dash_notify_trigger_fires_on_insert), the LISTEN worker hydrates it into the
    ring buffer (test_stream_manager_hydrates_notify_into_buffer), the buffer replay/reset
    logic (test_feed_event_buffer_replay_and_reset), the auth gate (test_stream_requires_token),
    and the frame format (here)."""
    from mcp_server.dashboard_routes import _sse_frame

    ep = _sse_frame("new_episode", '{"type":"episode","id":"42"}', 7)
    assert ep == 'event: new_episode\nid: 7\ndata: {"type":"episode","id":"42"}\n\n'
    # processing_status / reset carry no id (only feed events advance Last-Event-ID).
    st = _sse_frame("processing_status", '{"queue_depth":3,"active":true}')
    assert st == 'event: processing_status\ndata: {"queue_depth":3,"active":true}\n\n'
    assert "id:" not in st
    assert _sse_frame("reset", "{}").endswith("\n\n")


# Metrics (phase 4) — recall percentiles / ingestion / corpus
# ---------------------------------------------------------------------------


def test_parse_window_caps_and_defaults():
    """Pure-function window parsing: unit suffixes, 1h floor, 30d cap, safe fallback."""
    from mcp_server.dashboard_routes import _parse_window

    assert _parse_window(None, 7 * 86400) == 7 * 86400
    assert _parse_window("48h", 7 * 86400) == 48 * 3600
    assert _parse_window("2d", 7 * 86400) == 2 * 86400
    assert _parse_window("90d", 7 * 86400) == 30 * 86400  # capped at 30d
    assert _parse_window("30m", 7 * 86400) == 3600  # floored at 1h
    assert _parse_window("garbage", 111) == 111


def test_metrics_recall_series_percentiles_slowest_histogram(clean, conn, db_url):
    now = datetime.now(UTC)
    h = now.replace(minute=0, second=0, microsecond=0)  # current hour-bucket start
    legs = {
        "embed": 10.0,
        "bm25": 20.0,
        "vector": 30.0,
        "kg": 40.0,
        "web": 5.0,
        "rerank": 50.0,
        "notes": 15.0,
    }
    # bucket A (current hour): three recall rows, ms 100/200/300 (float32-exact)
    _metric(
        conn,
        query="q-slow",
        ms_total=300.0,
        est_tokens=1200,
        rerank_top_score=0.95,
        created_at=h + timedelta(minutes=1),
        legs=legs,
    )
    _metric(
        conn,
        query="q-mid",
        ms_total=200.0,
        est_tokens=1000,
        rerank_top_score=0.55,
        created_at=h + timedelta(minutes=2),
        legs=legs,
    )
    _metric(
        conn,
        query="q-fast",
        ms_total=100.0,
        est_tokens=800,
        rerank_top_score=0.35,
        created_at=h + timedelta(minutes=3),
        legs=legs,
    )
    # bucket B (previous hour): one recall row
    _metric(
        conn,
        query="q-prev",
        ms_total=150.0,
        est_tokens=500,
        rerank_top_score=0.72,
        created_at=h - timedelta(minutes=55),
    )
    # a non-recall row (kind='episodes') must be EXCLUDED from every recall aggregate
    _metric(
        conn,
        kind="episodes",
        query="q-episodes",
        ms_total=9999.0,
        created_at=h + timedelta(minutes=4),
    )

    with _client(db_url) as client:
        r = client.get("/dash/api/metrics/recall?window=7d", headers=_H)
    assert r.status_code == 200
    body = r.json()

    assert len(body["series"]) == 2  # two distinct hour buckets
    big = next(b for b in body["series"] if b["calls"] == 3)
    assert big["p50"] == 200.0  # percentile_cont(0.5) over [100,200,300]
    assert big["p95"] == 290.0  # percentile_cont(0.95) interpolates 200→300
    assert big["tokens_p50"] == 1000
    # only timed legs appear (notes included; the retired timeline/prefs legs never do)
    assert big["legs_p50"] == {
        "embed": 10.0,
        "bm25": 20.0,
        "vector": 30.0,
        "kg": 40.0,
        "web": 5.0,
        "rerank": 50.0,
        "notes": 15.0,
    }

    slow = body["slowest"]
    assert slow[0]["query"] == "q-slow" and slow[0]["ms_total"] == 300.0
    assert all(s["ms_total"] != 9999.0 for s in slow)  # episodes row excluded
    assert len(slow) <= 10

    hist = body["score_hist"]
    assert len(hist) == 10
    by_lo = {round(b["lo"], 1): b["n"] for b in hist}
    # 0.35→[0.3,0.4), 0.55→[0.5,0.6), 0.72→[0.7,0.8), 0.95→[0.9,1.0)
    assert by_lo[0.3] == 1 and by_lo[0.5] == 1 and by_lo[0.7] == 1 and by_lo[0.9] == 1
    assert sum(b["n"] for b in hist) == 4


def test_metrics_recall_retired_legs_are_not_reported(clean, conn, db_url):
    """Rows written while the timeline/prefs legs ran still carry their timings; the engine
    no longer produces those legs, so the per-leg stack must not draw them."""
    mid = _metric(conn, created_at=datetime.now(UTC) - timedelta(minutes=5))
    conn.execute("UPDATE recall_metrics SET ms_timeline = 7, ms_prefs = 9 WHERE id = %s", (mid,))
    with _client(db_url) as client:
        series = client.get("/dash/api/metrics/recall?window=7d", headers=_H).json()["series"]
    assert series[0]["legs_p50"] == {}


def test_metrics_recall_summary_is_window_percentiles(clean, conn, db_url):
    """The tiles are percentiles of the window's calls, not an average of per-hour
    percentiles: nine calls at 100 ms in one hour and one at 2000 ms in another average out
    to ~290 ms, but the window p50 is 100 and p95 is 1145."""
    h = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    for i in range(9):
        _metric(
            conn,
            ms_total=100.0,
            est_tokens=500,
            created_at=h - timedelta(hours=3) + timedelta(minutes=i),
        )
    _metric(conn, ms_total=2000.0, est_tokens=1500, created_at=h - timedelta(hours=1))
    _metric(conn, kind="episodes", ms_total=9999.0, created_at=h - timedelta(hours=1))
    with _client(db_url) as client:
        body = client.get("/dash/api/metrics/recall?window=7d", headers=_H).json()
    assert body["summary"] == {"calls": 10, "p50": 100.0, "p95": 1145.0, "tokens_p50": 500}
    assert [p["calls"] for p in body["series"]] == [9, 1]


def test_metrics_recall_summary_empty_window(clean, conn, db_url):
    with _client(db_url) as client:
        body = client.get("/dash/api/metrics/recall?window=7d", headers=_H).json()
    assert body["summary"] == {"calls": 0, "p50": None, "p95": None, "tokens_p50": None}


def test_metrics_ingestion_shape(clean, conn, db_url):
    now = datetime.now(UTC)
    # live snapshot: 2 recent pending + 1 stale pending (5d old) = 3 pending; 1 processing; 1 failed
    _queue(conn, status="pending", enqueued_at=now - timedelta(minutes=10))
    _queue(conn, status="pending", enqueued_at=now - timedelta(minutes=20))
    _queue(conn, status="pending", enqueued_at=now - timedelta(days=5))  # outside the 48h series
    _queue(conn, status="processing", enqueued_at=now - timedelta(minutes=30))
    _queue(
        conn,
        status="failed",
        error="boom timeout",
        attempts=3,
        enqueued_at=now - timedelta(hours=2),
        processed_at=now - timedelta(hours=1),
    )
    _queue(
        conn,
        status="done",
        enqueued_at=now - timedelta(hours=3),
        processed_at=now - timedelta(hours=2, minutes=30),
    )
    run_id = _dream_run(
        conn,
        started_at=now - timedelta(hours=9),
        finished_at=now - timedelta(hours=9) + timedelta(seconds=41),
        counts={"proposals_raised": 2, "config_proposals": 1},
        samples={
            "proposals": [{"id": "config:4", "kind": "config-edit", "name": "rules/learned.md"}]
        },
        stages={"config": {"ran": True, "ok": True}},
    )

    with _client(db_url) as client:
        r = client.get("/dash/api/metrics/ingestion?window=48h", headers=_H)
    assert r.status_code == 200
    b = r.json()

    assert b["queue_depth"] == 3  # live pending count (incl the stale one)
    assert b["queue"] == {"pending": 3, "processing": 1, "failed": 1}
    # enqueued/hour: only rows enqueued within 48h (5 of 6; the 5-day-old one is out)
    assert sum(p["n"] for p in b["throughput"]["enqueued_per_hour"]) == 5
    # completed/hour: rows that finished (status done) within 48h; the failed row is no completion
    assert sum(p["n"] for p in b["throughput"]["completed_per_hour"]) == 1
    assert len(b["failures"]) == 1
    assert b["failures"][0]["error"] == "boom timeout" and b["failures"][0]["attempts"] == 3
    assert b["last_dream"]["id"] == run_id
    assert b["last_dream"]["counts"]["proposals_raised"] == 2
    assert round(b["last_dream"]["duration_s"]) == 41
    assert b["last_dream"]["samples"]["proposals"][0]["id"] == "config:4"


def test_metrics_ingestion_failures_stay_inside_the_window(clean, conn, db_url):
    """A failed row stamps processed_at too, so it must not count as a completion; and the
    failures panel is the window's, not every failure since the table began."""
    now = datetime.now(UTC)
    _queue(
        conn,
        status="failed",
        error="recent",
        enqueued_at=now - timedelta(hours=3),
        processed_at=now - timedelta(hours=2),
    )
    _queue(
        conn,
        status="failed",
        error="last month",
        enqueued_at=now - timedelta(days=30),
        processed_at=now - timedelta(days=29),
    )
    _queue(conn, status="done", enqueued_at=now - timedelta(hours=3), processed_at=now)
    with _client(db_url) as client:
        b = client.get("/dash/api/metrics/ingestion?window=48h", headers=_H).json()
    assert [f["error"] for f in b["failures"]] == ["recent"]
    assert sum(p["n"] for p in b["throughput"]["completed_per_hour"]) == 1
    assert b["queue"]["failed"] == 2  # the live snapshot still counts both


def test_metrics_ingestion_last_dream_prefers_a_finished_run(clean, conn, db_url):
    """A crashed run (finished_at NULL) must not pin the 'last run' tile until the next
    nightly run: the latest FINISHED run wins; an unfinished one is the fallback."""
    now = datetime.now(UTC)
    done = _dream_run(
        conn, started_at=now - timedelta(hours=30), finished_at=now - timedelta(hours=29), counts={}
    )
    crashed = _dream_run(
        conn, started_at=now - timedelta(hours=6), finished_at=None, counts={}, ok=None
    )
    with _client(db_url) as client:
        assert (
            client.get("/dash/api/metrics/ingestion", headers=_H).json()["last_dream"]["id"] == done
        )
        conn.execute("DELETE FROM dream_runs WHERE id = %s", (done,))
        only = client.get("/dash/api/metrics/ingestion", headers=_H).json()["last_dream"]
    assert only["id"] == crashed and only["finished_at"] is None


def test_metrics_ingestion_empty_last_dream(clean, conn, db_url):
    """Fresh install: no dream runs recorded, empty queue → null last_dream, empty series."""
    with _client(db_url) as client:
        b = client.get("/dash/api/metrics/ingestion?window=48h", headers=_H).json()
    assert b["last_dream"] is None
    assert b["queue_depth"] == 0
    assert b["throughput"]["enqueued_per_hour"] == []


def test_metrics_corpus_shape_and_cache(clean, conn, db_url):
    now = datetime.now(UTC)
    _episode(
        conn,
        session="s1",
        seq=1,
        project="alpha",
        source="claude-code",
        created_at=now - timedelta(days=1),
    )
    _episode(
        conn,
        session="s1",
        seq=2,
        project="alpha",
        source="cursor",
        created_at=now - timedelta(days=2),
    )
    _episode(
        conn,
        session="s2",
        seq=1,
        project="beta",
        source="claude-code",
        created_at=now - timedelta(days=40),
    )  # outside the 30d sparkline
    _note(conn)
    _pref(conn)
    _rel(conn, "c-r1", src="x", tgt="y", created_at=now - timedelta(days=1))

    with _client(db_url) as client:
        first = client.get("/dash/api/metrics/corpus", headers=_H).json()
        # Insert MORE after the first call; the 1h in-process cache must serve the SAME body.
        _episode(
            conn,
            session="s3",
            seq=1,
            project="alpha",
            source="claude-code",
            created_at=now - timedelta(hours=1),
        )
        second = client.get("/dash/api/metrics/corpus", headers=_H).json()
    assert second == first  # served from cache (else episodes would read 4 rows)

    tables = {t["name"]: t for t in first["tables"]}
    assert {"episodes", "notes", "chunks"} <= set(tables)
    ep = tables["episodes"]
    assert ep["rows"] == 3 and ep["rows_estimated"] is False  # exact count on a small table
    assert len(ep["spark_30d"]) == 30
    assert ep["delta_30d"] == 2  # only the two within 30d
    assert tables["kg_relationships"]["delta_30d"] == 1  # sparkline off t_created
    proj = {p["name"]: p["n"] for p in first["by_project"]}
    assert proj["alpha"] == 2 and proj["beta"] == 1
    src = {p["name"]: p["n"] for p in first["by_source"]}
    assert src["claude-code"] == 2 and src["cursor"] == 1


def test_metrics_require_machine_token(clean, conn, db_url):
    with _client(db_url) as client:
        assert client.get("/dash/api/metrics/recall").status_code == 401
        assert client.get("/dash/api/metrics/ingestion").status_code == 401
        assert client.get("/dash/api/metrics/corpus").status_code == 401


# ---------------------------------------------------------------------------
# Static bundle routes (UNAUTHENTICATED) — 503 without a build, 200 + traversal reject
# ---------------------------------------------------------------------------


def test_static_503_without_bundle(tmp_path, monkeypatch, db_url):
    monkeypatch.setattr(dr, "_DIST_DIR", tmp_path / "missing")
    with _client(db_url) as client:
        r = client.get("/dash")  # no auth header — static routes are open
        assert r.status_code == 503 and r.json()["detail"] == "bundle not built"
        assert client.get("/dash/app.js").status_code == 503


def test_static_serves_bundle_and_rejects_traversal(tmp_path, monkeypatch, db_url):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>dash</title>")
    (dist / "app.js").write_text("console.log('dash')")
    (dist / "secret.txt").write_text("not an asset")  # sits in dist root, not assets/
    (dist / "assets" / "font.woff2").write_bytes(b"woff2-bytes")
    monkeypatch.setattr(dr, "_DIST_DIR", dist)

    with _client(db_url) as client:
        idx = client.get("/dash")
        assert idx.status_code == 200
        assert idx.headers["content-type"].startswith("text/html")
        assert idx.headers["cache-control"] == "no-cache"
        assert "dash" in idx.text

        js = client.get("/dash/app.js")
        assert js.status_code == 200
        assert js.headers["content-type"].startswith("application/javascript")

        font = client.get("/dash/assets/font.woff2")
        assert font.status_code == 200
        assert font.headers["content-type"] == "font/woff2"
        assert "immutable" in font.headers["cache-control"]

        # Not in the assets whitelist -> 404.
        assert client.get("/dash/assets/missing.woff2").status_code == 404
        # A dist-root file requested through /assets/ is NOT in the assets listing -> 404
        # (whitelist-by-listing blocks escaping the assets dir).
        assert client.get("/dash/assets/secret.txt").status_code == 404
        assert client.get("/dash/assets/app.js").status_code == 404
        # Encoded traversal never resolves to a real asset.
        assert client.get("/dash/assets/%2e%2e%2fapp.js").status_code == 404


# ---------------------------------------------------------------------------
# Phase 5 — Timeline (Events + Preferences), Dream report, Behavior files
# ---------------------------------------------------------------------------


def _pref_row(
    conn,
    *,
    pref,
    polarity="like",
    group_id="technical",
    assert_count=1,
    first_seen=None,
    last_asserted=None,
    t_invalid=None,
    superseded_by=None,
):
    return conn.execute(
        "INSERT INTO preferences (owner_id, group_id, pref, polarity, assert_count, "
        " first_seen, last_asserted, t_invalid, superseded_by) "
        "VALUES ('default',%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (
            group_id,
            pref,
            polarity,
            assert_count,
            first_seen or _t(0),
            last_asserted or _t(0),
            t_invalid,
            superseded_by,
        ),
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# Graph explorer (phase 6): typeahead + BFS neighborhood, as-of, truncation,
# name-seed resolution, token gates.
# ---------------------------------------------------------------------------


def _gent(conn, uuid, *, name, entity_type="Technology", normalized_name=None):
    """Entity insert that ALSO sets normalized_name (the shared _entity helper doesn't),
    so the exact-normalized-name seed path is exercisable. entity_supertype mirrors
    the subtype — the graph payloads serve BOTH (client colors by supertype)."""
    return conn.execute(
        "INSERT INTO kg_entities "
        "(uuid, group_id, name, normalized_name, entity_subtype, entity_supertype, summary, "
        " t_created) "
        "VALUES (%s,'technical',%s,%s,%s,%s,%s,%s) RETURNING uuid",
        (
            uuid,
            name,
            normalized_name or name.lower(),
            entity_type,
            entity_type,
            f"{name} summary",
            _t(0),
        ),
    ).fetchone()[0]


def _config_reg(conn, *, file_key, content, surface_id="hostA", scope="global", updated_at=None):
    import hashlib

    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    conn.execute(
        "INSERT INTO config_lane.config_registry "
        "(surface_id, scope, file_key, abs_path, content, content_hash, updated_at) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s)",
        (surface_id, scope, file_key, f"/home/x/{file_key}", content, digest, updated_at or _t(0)),
    )


@pytest.fixture()
def clean_behavior(conn):
    """Wipe the config-lane mirror before AND after (the phase-5 behavior tests own it —
    _wipe doesn't touch config_lane, so this keeps them deterministic and leak-free)."""

    def _w():
        conn.execute("TRUNCATE config_lane.config_registry RESTART IDENTITY CASCADE")

    _w()
    yield
    _w()


# ---- Timeline: Events ----


def test_timeline_events_default_fields_and_salience(clean, conn, db_url):
    eid = _episode(conn, session="tl", seq=1, content="the shipping session")
    # Distinct t_valid; coarse salience 2/1/0 -> sal 0.9/0.6/0.3; ep: ref resolves, sha doesn't.
    e_hi = _event(
        conn,
        fact="rerank pool capped at 96",
        source="chat",
        source_ref=f"ep:{eid}",
        salience=2,
        domain="technical",
        event_type="decision",
        t_valid=_t(30),
    )
    e_mid = _event(
        conn,
        fact="took a rest day",
        source="chat",
        source_ref="deadbeef",  # git-style ref -> episode_id None
        salience=1,
        domain="personal",
        event_type="finding",
        t_valid=_t(20),
    )
    e_lo = _event(
        conn,
        fact="untyped note",
        source="chat",
        source_ref="ep:99999",
        salience=0,
        domain="technical",
        event_type=None,  # untyped: shows in the default (no-chip) view
        t_valid=_t(10),
    )

    with _client(db_url) as client:
        body = client.get("/dash/api/timeline", headers=_H).json()
    ids = [e["id"] for e in body["events"]]
    assert ids == [e_hi, e_mid, e_lo]  # t_valid DESC
    by_id = {e["id"]: e for e in body["events"]}
    assert by_id[e_hi]["sal"] == 0.9 and by_id[e_hi]["salience"] == 2
    assert by_id[e_hi]["event_type"] == "decision" and by_id[e_hi]["episode_id"] == eid
    assert by_id[e_mid]["sal"] == 0.6 and by_id[e_mid]["episode_id"] is None
    assert by_id[e_lo]["sal"] == 0.3 and by_id[e_lo]["event_type"] is None  # untyped shown
    assert by_id[e_hi]["flagged"] is False


def test_timeline_keyset_pagination_with_ties(clean, conn, db_url):
    # Two events share a t_valid (tie); the (t_valid, id) keyset must not drop either.
    a = _event(conn, fact="tie A", source_ref="tb:a", t_valid=_t(10))
    b = _event(conn, fact="tie B", source_ref="tb:b", t_valid=_t(10))  # higher id, same t_valid
    c = _event(conn, fact="older C", source_ref="tb:c", t_valid=_t(5))
    with _client(db_url) as client:
        p1 = client.get("/dash/api/timeline", params={"limit": 1}, headers=_H).json()
        assert [e["id"] for e in p1["events"]] == [b]  # id DESC breaks the tie
        assert p1["next_before"]
        # params= lets TestClient URL-encode the cursor's '+' offset (the real client uses
        # URLSearchParams, which does the same) — a raw f-string would decode '+' to a space.
        p2 = client.get(
            "/dash/api/timeline", params={"limit": 1, "before": p1["next_before"]}, headers=_H
        ).json()
        assert [e["id"] for e in p2["events"]] == [a]  # tie sibling NOT skipped
        p3 = client.get(
            "/dash/api/timeline", params={"limit": 1, "before": p2["next_before"]}, headers=_H
        ).json()
        assert [e["id"] for e in p3["events"]] == [c]
        # A full page (len == limit) always yields a cursor, like the feed; the next fetch
        # drains the stream (empty page, no further cursor).
        assert p3["next_before"]
        p4 = client.get(
            "/dash/api/timeline", params={"limit": 1, "before": p3["next_before"]}, headers=_H
        ).json()
        assert p4["events"] == [] and p4["next_before"] is None
    # No id appeared twice across the pages.
    seen = [e["id"] for e in p1["events"] + p2["events"] + p3["events"]]
    assert sorted(seen) == sorted({a, b, c})


def test_timeline_type_and_domain_filters(clean, conn, db_url):
    decision = _event(
        conn, fact="decided", source_ref="tb:w", event_type="decision", domain="technical"
    )
    finding = _event(
        conn, fact="found out", source_ref="tb:l", event_type="finding", domain="personal"
    )
    untyped = _event(conn, fact="untyped", source_ref="tb:u", event_type=None, domain="technical")
    with _client(db_url) as client:
        # type chip narrows to that event_type; untyped rows drop out when a chip is active.
        only = client.get("/dash/api/timeline?type=decision", headers=_H).json()
        assert {e["id"] for e in only["events"]} == {decision}
        # group_id maps to the domain column (schema 038).
        personal = client.get("/dash/api/timeline?group_id=personal", headers=_H).json()
        assert {e["id"] for e in personal["events"]} == {finding}
        # No filter -> everything, including the untyped row.
        allrows = client.get("/dash/api/timeline", headers=_H).json()
        assert {e["id"] for e in allrows["events"]} == {decision, finding, untyped}


def test_timeline_rejects_a_type_the_pipeline_never_writes(clean, conn, db_url):
    """Every writer emits decision|action|finding|milestone; any other chip value is a
    client bug, so it is a 400 rather than a silently empty list."""
    _event(conn, fact="decided", source_ref="tb:w", event_type="decision")
    with _client(db_url) as client:
        for t in ("decision", "action", "finding", "milestone"):
            assert client.get(f"/dash/api/timeline?type={t}", headers=_H).status_code == 200
        r = client.get("/dash/api/timeline?type=work", headers=_H)
        assert r.status_code == 400 and "type" in r.json()["detail"]


def test_group_filter_keeps_events_with_no_domain(clean, conn, db_url):
    """schema 038: a NULL domain FAILS OPEN at read. An unlabelled event (legacy row, gate
    parse miss, pushed non-git event) stays visible under either group filter, in the
    timeline, the feed and the search tab (count and hits alike)."""
    tok = "unlabelledquux"
    tech = _event(conn, fact=f"{tok} tech", source_ref="tb:t", domain="technical")
    pers = _event(conn, fact=f"{tok} personal", source_ref="tb:p", domain="personal")
    blank = _event(conn, fact=f"{tok} blank", source_ref="tb:n", domain=None)
    with _client(db_url) as client:
        for group, want in (
            ("technical", {tech, blank}),
            ("personal", {pers, blank}),
            (None, {tech, pers, blank}),
        ):
            params = {"group_id": group} if group else {}
            tl = client.get("/dash/api/timeline", params=params, headers=_H).json()
            assert {e["id"] for e in tl["events"]} == want, group
            feed = _feed_page(client, **params)
            got = {int(i["id"]) for i in feed["items"] if i["type"] == "timeline_event"}
            assert got == want, group
            se = client.get(
                "/dash/api/search",
                params={"q": tok, "type": "events", **params},
                headers=_H,
            ).json()
            assert {int(h["id"]) for h in se["hits"]} == want, group
            assert se["total_by_type"]["events"] == len(want), group


def test_timeline_months_counts_each_month_in_the_requested_zone(clean, conn, db_url):
    """The jump-to-date options come from the server: every month that has events under the
    active filters, newest first, bucketed in the viewer's zone. An evening event the UTC
    string files under the next month belongs to the month the viewer lived it in, and
    `before` is that month's end, the cursor that loads the month and everything older."""
    # 2026-10-01T00:30Z is 2026-09-30 20:30 in Toronto; 2026-09-15 and 2026-07-04 are plain.
    late = _event(
        conn, fact="evening", source_ref="m:1", t_valid=datetime(2026, 10, 1, 0, 30, tzinfo=UTC)
    )
    sep = _event(
        conn, fact="mid", source_ref="m:2", t_valid=datetime(2026, 9, 15, 16, 0, tzinfo=UTC)
    )
    jul = _event(
        conn, fact="old", source_ref="m:3", event_type="decision",
        t_valid=datetime(2026, 7, 4, 16, 0, tzinfo=UTC), domain="personal",
    )  # fmt: skip
    with _client(db_url) as client:
        got = client.get(
            "/dash/api/timeline/months", params={"tz": "America/Toronto"}, headers=_H
        ).json()
        assert [(m["month"], m["n"]) for m in got["months"]] == [("2026-09", 2), ("2026-07", 1)]
        # The jump boundary is the first instant of October in Toronto (EDT, UTC-4).
        assert got["months"][0]["before"].startswith("2026-10-01T04:00:00")
        utc = client.get("/dash/api/timeline/months", params={"tz": "UTC"}, headers=_H).json()
        assert [(m["month"], m["n"]) for m in utc["months"]] == [
            ("2026-10", 1), ("2026-09", 1), ("2026-07", 1),
        ]  # fmt: skip
        # Same filters as the events list: type, group.
        typed = client.get(
            "/dash/api/timeline/months", params={"tz": "UTC", "type": "decision"}, headers=_H
        ).json()
        assert [m["month"] for m in typed["months"]] == ["2026-07"]
        grp = client.get(
            "/dash/api/timeline/months", params={"tz": "UTC", "group_id": "personal"}, headers=_H
        ).json()
        assert [m["month"] for m in grp["months"]] == ["2026-07"]
        # Loading a month through its `before` returns that month and older, nothing newer.
        page = client.get(
            "/dash/api/timeline", params={"before": got["months"][0]["before"]}, headers=_H
        ).json()
        assert [e["id"] for e in page["events"]] == [late, sep, jul]
        july = client.get(
            "/dash/api/timeline", params={"before": got["months"][1]["before"]}, headers=_H
        ).json()
        assert [e["id"] for e in july["events"]] == [jul]
        # A zone Postgres cannot resolve is a 400, not a 500.
        assert (
            client.get(
                "/dash/api/timeline/months", params={"tz": "Mars/Olympus"}, headers=_H
            ).status_code
            == 400
        )
        assert client.get("/dash/api/timeline/months", headers=_H).status_code == 400


def test_timeline_flag_reflection(clean, conn, db_url):
    ev = _event(conn, fact="flag this event", source_ref="tb:f")
    with _client(db_url) as client:
        client.post("/dash/api/flag", json={"kind": "timeline_event", "id": str(ev)}, headers=_H)
        body = client.get("/dash/api/timeline", headers=_H).json()
    assert body["events"][0]["flagged"] is True


# ---- Timeline: Preferences ----


def test_preferences_sort_live_first_and_supersede_join(clean, conn, db_url):
    live1 = _pref_row(
        conn, pref="Prefers dark theme", polarity="like", assert_count=5, last_asserted=_t(50)
    )
    live2 = _pref_row(
        conn,
        pref="Dislikes ORMs — raw SQL only",
        polarity="dislike",
        assert_count=9,
        last_asserted=_t(30),
    )
    live_new = _pref_row(
        conn, pref="self-hosted pgvector", polarity="like", assert_count=2, last_asserted=_t(10)
    )
    old = _pref_row(
        conn,
        pref="Liked hosted vector DBs",
        polarity="like",
        assert_count=3,
        last_asserted=_t(5),
        t_invalid=_t(20),
        superseded_by=live_new,
    )

    with _client(db_url) as client:
        rec = client.get("/dash/api/preferences?sort=recency", headers=_H).json()["preferences"]
        # Live rows first (by last_asserted DESC), superseded last.
        assert [p["id"] for p in rec] == [live1, live2, live_new, old]
        ac = client.get("/dash/api/preferences?sort=assert_count", headers=_H).json()["preferences"]
        # Live rows first (by assert_count DESC) — a different order than recency.
        assert [p["id"] for p in ac] == [live2, live1, live_new, old]

        row = {p["id"]: p for p in rec}[old]
        assert row["t_invalid"] is not None
        assert row["superseded_by"] == live_new
        assert row["superseded_by_text"] == "self-hosted pgvector"
        assert row["polarity"] == "like" and row["assert_count"] == 3
        live_row = {p["id"]: p for p in rec}[live1]
        assert live_row["superseded_by"] is None and live_row["superseded_by_text"] is None


def test_preferences_flag_reflection(clean, conn, db_url):
    pid = _pref_row(conn, pref="Prefers bullet lists", assert_count=1)
    with _client(db_url) as client:
        client.post("/dash/api/flag", json={"kind": "preference", "id": str(pid)}, headers=_H)
        prefs = client.get("/dash/api/preferences", headers=_H).json()["preferences"]
    assert {p["id"]: p["flagged"] for p in prefs}[pid] is True


# ---- Dream report ----


def test_dream_report_off_dream_runs(clean, conn, db_url):
    r_old = _dream_run(
        conn,
        started_at=_t(0),
        finished_at=_t(1),
        counts={"proposals_raised": 1},
        samples={"proposals": [{"id": "skill:1", "kind": "skill", "name": "a"}]},
        stages={"skills": {"ran": True, "ok": True}},
        ok=True,
    )
    r_new = _dream_run(
        conn,
        started_at=_t(120),
        finished_at=_t(121),
        counts={"proposals_raised": 3, "config_proposals": 2},
        samples={"proposals": [{"id": "config:4", "kind": "config-edit", "name": "rules/x.md"}]},
        stages={"config": {"ran": True, "ok": True}},
        ok=True,
    )
    with _client(db_url) as client:
        runs = client.get("/dash/api/dream/report", headers=_H).json()["runs"]
    assert [r["id"] for r in runs] == [r_new, r_old]  # newest first
    top = runs[0]
    assert top["counts"]["proposals_raised"] == 3 and top["counts"]["config_proposals"] == 2
    assert top["samples"]["proposals"][0]["id"] == "config:4"
    assert top["stages"] == {"config": {"ran": True, "ok": True}}
    assert round(top["duration_s"]) == 60 and top["ok"] is True and top["errors"] == []


def test_dream_report_empty(clean, conn, db_url):
    with _client(db_url) as client:
        assert client.get("/dash/api/dream/report", headers=_H).json() == {"runs": []}


# ---- Behavior files ----


def test_behavior_files_grouping_and_content(clean_behavior, conn, db_url):
    _config_reg(
        conn,
        file_key="CLAUDE.md",
        content="# CLAUDE\nSee [[voice]] and [[job_crons]].\nAgain [[voice]].",
    )
    _config_reg(conn, file_key="rules/voice.md", content="Voice rules. Ref [[CLAUDE]].")
    _config_reg(conn, file_key="memory/note1.md", content="note body [[voice]]")
    _config_reg(conn, file_key="AGENTS.md", content="misc, no links")

    with _client(db_url) as client:
        files = client.get("/dash/api/behavior/files", headers=_H).json()
        names = [g["name"] for g in files["groups"]]
        assert names == ["CLAUDE.md", "rules", "memory notes", "other"]  # fixed display order
        by_group = {g["name"]: g["files"] for g in files["groups"]}
        assert [f["file_key"] for f in by_group["CLAUDE.md"]] == ["CLAUDE.md"]
        assert [f["file_key"] for f in by_group["rules"]] == ["rules/voice.md"]
        assert [f["file_key"] for f in by_group["memory notes"]] == ["memory/note1.md"]
        assert [f["file_key"] for f in by_group["other"]] == ["AGENTS.md"]
        cm = by_group["CLAUDE.md"][0]
        assert cm["surface_id"] == "hostA" and cm["scope"] == "global"
        assert cm["size"] > 0 and cm["updated_at"] is not None

        # file detail: content + meta + deduped, ordered wikilinks
        fd = client.get("/dash/api/behavior/file?key=CLAUDE.md&scope=global", headers=_H).json()
        assert fd["file_key"] == "CLAUDE.md" and "[[voice]]" in fd["content"]
        assert fd["links"] == ["voice", "job_crons"]  # first-seen order, deduped
        assert fd["meta"]["surface_id"] == "hostA" and fd["meta"]["scope"] == "global"
        assert fd["meta"]["content_hash"] and fd["meta"]["size"] == len(fd["content"])

        # missing key -> 400; unknown file -> 404
        assert client.get("/dash/api/behavior/file", headers=_H).status_code == 400
        assert (
            client.get("/dash/api/behavior/file?key=nope.md&scope=global", headers=_H).status_code
            == 404
        )


def test_behavior_file_surface_disambiguation(clean_behavior, conn, db_url):
    # Same file_key on two surfaces (PK includes surface_id) — explicit surface selects one.
    _config_reg(conn, file_key="CLAUDE.md", content="host A body", surface_id="hostA")
    _config_reg(conn, file_key="CLAUDE.md", content="host B body", surface_id="hostB")
    with _client(db_url) as client:
        a = client.get(
            "/dash/api/behavior/file?key=CLAUDE.md&scope=global&surface=hostB", headers=_H
        ).json()
    assert a["content"] == "host B body" and a["meta"]["surface_id"] == "hostB"


# ---- Token gates ----


def test_phase5_routes_require_machine_token(clean, conn, db_url):
    with _client(db_url) as client:
        for path in (
            "/dash/api/timeline",
            "/dash/api/timeline?type=work",
            "/dash/api/timeline?tz=Not/AZone",
            "/dash/api/timeline/months",
            "/dash/api/timeline/months?tz=UTC",
            "/dash/api/timeline/months?tz=Not/AZone",
            "/dash/api/preferences",
            "/dash/api/dream/report",
            "/dash/api/behavior/files",
            "/dash/api/behavior/file?key=CLAUDE.md",
        ):
            r = client.get(path)
            assert r.status_code == 401
            assert r.json() == {"status": "error", "detail": "unauthorized"}


def _mini_graph(conn):
    """9 entities, ~10 edges, one superseded + one not-yet-valid. Live degrees come out
    hub 5 > pg 3 > cc 2 > the rest 1, so truncation order is deterministic. Returns the
    seed episode id for provenance checks."""
    _gent(conn, "g-hub", name="Synapse", entity_type="Project")
    _gent(conn, "g-pg", name="Postgres")
    _gent(conn, "g-cc", name="Claude Code")
    _gent(conn, "g-kg", name="knowledge graph", entity_type="Concept")
    _gent(conn, "g-anthropic", name="Anthropic", entity_type="Organization")
    _gent(conn, "g-nuc", name="homelab NUC")  # depth-2 (via g-pg)
    _gent(conn, "g-sse", name="SSE stream")  # depth-2 (via g-cc)
    _gent(conn, "g-far", name="embedding cache")  # depth-2 (via g-pg)
    _gent(conn, "g-future", name="future thing")  # only via a not-yet-valid edge

    ep = _episode(conn, session="g", seq=1, content="synapse stores in postgres")
    # Depth-1 (from g-hub) live edges.
    _rel(
        conn,
        "e1",
        src="g-hub",
        tgt="g-pg",
        name="stores in",
        fact="Synapse stores in Postgres",
        episodes=[ep],
        t_valid=_t(0),
    )
    _rel(
        conn,
        "e2",
        src="g-hub",
        tgt="g-cc",
        name="ingests from",
        fact="Synapse ingests from Claude Code",
        t_valid=_t(0),
    )
    _rel(
        conn,
        "e3",
        src="g-hub",
        tgt="g-kg",
        name="extracts",
        fact="Synapse extracts a knowledge graph",
        t_valid=_t(0),
    )
    _rel(
        conn,
        "e4",
        src="g-hub",
        tgt="g-anthropic",
        name="built by",
        fact="Synapse built by Anthropic",
        t_valid=_t(0),
    )
    # Depth-2 edges.
    _rel(
        conn,
        "e5",
        src="g-pg",
        tgt="g-nuc",
        name="runs on",
        fact="Postgres runs on the homelab NUC",
        t_valid=_t(0),
    )
    _rel(
        conn,
        "e6",
        src="g-pg",
        tgt="g-far",
        name="caches in",
        fact="Postgres caches in the embedding cache",
        t_valid=_t(0),
    )
    _rel(
        conn,
        "e7",
        src="g-cc",
        tgt="g-sse",
        name="emits",
        fact="Claude Code emits an SSE stream",
        t_valid=_t(0),
    )
    # Superseded edge (t_valid before base, invalidated before base) — always included when
    # as_of >= its t_valid; the client dashes it.
    _rel(
        conn,
        "e-super",
        src="g-hub",
        tgt="g-pg",
        name="used",
        fact="Synapse used SQLite",
        t_valid=_t(-1000),
        t_invalid=_t(-10),
    )
    # Not-yet-valid edge — hidden when as_of < its t_valid.
    _rel(
        conn,
        "e-future",
        src="g-hub",
        tgt="g-future",
        name="will use",
        fact="Synapse will use a future thing",
        t_valid=_t(100),
    )
    return ep


def test_root_redirects_to_dash(clean, conn, db_url):
    with _client(db_url) as client:
        r = client.get("/", follow_redirects=False)
        assert r.status_code == 302 and r.headers["location"] == "/dash"


def test_graph_nodes_carry_supertype(clean, conn, db_url):
    _mini_graph(conn)
    with _client(db_url) as client:
        nb = client.get("/dash/api/graph/neighborhood?entity=g-hub&depth=1", headers=_H).json()
        assert all("supertype" in n for n in nb["nodes"])
        # entity_type on the wire is the taxonomy subtype (entity_subtype, schema 020).
        assert {n["uuid"]: n["entity_type"] for n in nb["nodes"]}["g-hub"] == "Project"
        ta = client.get("/dash/api/graph/entities?q=Synapse", headers=_H).json()
        assert all("supertype" in e for e in ta)
        assert [e["entity_type"] for e in ta] == ["Project"]


def test_graph_typeahead(clean, conn, db_url):
    _mini_graph(conn)
    with _client(db_url) as client:
        # ILIKE on name, degree DESC — matches carrying 'a' come back highest-degree first.
        r = client.get("/dash/api/graph/entities?q=a", headers=_H).json()
        assert isinstance(r, list) and len(r) >= 3
        degs = [row["degree"] for row in r]
        assert degs == sorted(degs, reverse=True)  # degree DESC ordering
        assert set(r[0].keys()) == {"uuid", "name", "entity_type", "supertype", "degree"}

        # A precise substring resolves to the one entity.
        pg = client.get("/dash/api/graph/entities?q=Postgres", headers=_H).json()
        assert [row["uuid"] for row in pg] == ["g-pg"]

        # limit caps the result count.
        capped = client.get("/dash/api/graph/entities?q=a&limit=2", headers=_H).json()
        assert len(capped) == 2

        # Empty q -> empty list.
        assert client.get("/dash/api/graph/entities?q=", headers=_H).json() == []


def test_graph_neighborhood_depth_1_vs_2(clean, conn, db_url):
    ep = _mini_graph(conn)
    with _client(db_url) as client:
        d1 = client.get("/dash/api/graph/neighborhood?entity=g-hub&depth=1", headers=_H).json()
        d1_nodes = {n["uuid"] for n in d1["nodes"]}
        # depth 1: seed + its direct neighbors (incl. the not-yet-valid future edge, since
        # as_of is unset) — but NOT the depth-2 nodes.
        assert d1_nodes == {"g-hub", "g-pg", "g-cc", "g-kg", "g-anthropic", "g-future"}
        assert d1["seed"] == "g-hub" and d1["truncated"] is False
        assert d1["nodes"][0]["uuid"] == "g-hub"  # seed sorted first
        # node payload shape
        assert set(d1["nodes"][0].keys()) == {
            "uuid",
            "name",
            "entity_type",
            "supertype",
            "degree",
            "summary",
        }
        # edge payload shape + provenance resolution
        e1 = next(e for e in d1["edges"] if e["uuid"] == "e1")
        assert e1["src"] == "g-hub" and e1["tgt"] == "g-pg" and e1["provenance_episode_id"] == ep
        assert set(e1.keys()) == {
            "uuid",
            "src",
            "tgt",
            "name",
            "fact",
            "t_valid",
            "t_invalid",
            "provenance_episode_id",
            "retrieval_count",
        }

        d2 = client.get("/dash/api/graph/neighborhood?entity=g-hub&depth=2", headers=_H).json()
        d2_nodes = {n["uuid"] for n in d2["nodes"]}
        # depth 2 pulls in the second ring via g-pg / g-cc.
        assert {"g-nuc", "g-far", "g-sse"} <= d2_nodes
        assert d2["truncated"] is False


def test_graph_neighborhood_as_of_filter(clean, conn, db_url):
    _mini_graph(conn)
    with _client(db_url) as client:
        # quote() so the '+' in the ISO offset survives as an offset, not a decoded space.
        as_of = quote(_t(50).isoformat())
        r = client.get(
            f"/dash/api/graph/neighborhood?entity=g-hub&depth=1&as_of={as_of}", headers=_H
        ).json()
        node_ids = {n["uuid"] for n in r["nodes"]}
        edge_ids = {e["uuid"] for e in r["edges"]}
        # not-yet-valid edge (t_valid=_t(100) > as_of) is hidden -> its exclusive node drops.
        assert "g-future" not in node_ids
        assert "e-future" not in edge_ids
        # superseded edge (t_valid=_t(-1000) <= as_of, t_invalid set) is INCLUDED, dashed client-side.
        sup = next(e for e in r["edges"] if e["uuid"] == "e-super")
        assert sup["t_invalid"] is not None
        # live edges still present.
        assert "e1" in edge_ids


def test_graph_neighborhood_truncation_keeps_highest_degree(clean, conn, db_url):
    _mini_graph(conn)
    with _client(db_url) as client:
        r = client.get(
            "/dash/api/graph/neighborhood?entity=g-hub&depth=2&limit=3", headers=_H
        ).json()
        assert r["truncated"] is True
        kept = {n["uuid"] for n in r["nodes"]}
        # budget 3 -> seed (always) + the two highest-degree reachable nodes (g-pg=3, g-cc=2 live edges).
        assert kept == {"g-hub", "g-pg", "g-cc"}
        # every returned edge connects two kept nodes.
        for e in r["edges"]:
            assert e["src"] in kept and e["tgt"] in kept


def test_graph_neighborhood_edge_cap_prunes_orphans(clean, conn, db_url):
    """A dense hub must not ship an unbounded hairball: over edge_cap, seed-adjacent +
    most-retrieved edges win, truncated flips, and nodes the cut orphaned are pruned
    (regression: the live 707-degree hub returned 150 nodes / ~2K edges and hung the
    browser in layout)."""
    _gent(conn, "ec-hub", name="Hub")
    for i in range(4):
        _gent(conn, f"ec-n{i}", name=f"Node {i}")
    # 4 seed-adjacent edges with distinct retrieval_counts + 2 leaf-leaf edges.
    for i in range(4):
        _rel(conn, f"ec-e{i}", src="ec-hub", tgt=f"ec-n{i}", retrieval_count=10 - i)
    _rel(conn, "ec-x1", src="ec-n0", tgt="ec-n1", retrieval_count=99)
    _rel(conn, "ec-x2", src="ec-n2", tgt="ec-n3", retrieval_count=98)

    out = dr._graph_neighborhood(db_url, "ec-hub", 1, None, 150, edge_cap=3)
    assert out is not None and out["truncated"] is True
    # Seed-adjacent edges beat the higher-retrieval leaf-leaf edges; within the
    # adjacent set, retrieval_count orders the cut.
    assert {e["uuid"] for e in out["edges"]} == {"ec-e0", "ec-e1", "ec-e2"}
    kept = {n["uuid"] for n in out["nodes"]}
    assert kept == {"ec-hub", "ec-n0", "ec-n1", "ec-n2"}  # ec-n3 orphaned -> pruned


def test_graph_neighborhood_name_seed_resolution(clean, conn, db_url):
    _mini_graph(conn)
    with _client(db_url) as client:
        # exact normalized_name (case-insensitive input).
        exact = client.get("/dash/api/graph/neighborhood?entity=Synapse", headers=_H).json()
        assert exact["seed"] == "g-hub"
        # no exact normalized_name -> best ILIKE match by degree.
        fuzzy = client.get("/dash/api/graph/neighborhood?entity=postgr", headers=_H).json()
        assert fuzzy["seed"] == "g-pg"
        # unknown seed -> 404.
        assert (
            client.get("/dash/api/graph/neighborhood?entity=nope-xyz", headers=_H).status_code
            == 404
        )
        # missing entity param -> 400.
        assert client.get("/dash/api/graph/neighborhood", headers=_H).status_code == 400


def test_graph_endpoints_require_machine_token(clean, conn, db_url):
    _mini_graph(conn)
    with _client(db_url) as client:
        for path in ("/dash/api/graph/entities?q=a", "/dash/api/graph/neighborhood?entity=g-hub"):
            r = client.get(path)
            assert r.status_code == 401
            assert r.json() == {"status": "error", "detail": "unauthorized"}


# ---------------------------------------------------------------------------
# Delete (phase 7) — cascade semantics, non-cascade, audit
# ---------------------------------------------------------------------------


def test_delete_episode_facts_sole_vs_shared_provenance(clean, conn, db_url):
    """THE rule (contract §Phase 7): a fact whose provenance names ONLY the deleted
    episode dies; a fact sharing provenance with a surviving episode loses the id from
    its array and lives. Regression guard for the 2026-06 cleanup, where deleting by
    blanket provenance match would have taken 1467 legitimate facts out with 330 bad ones."""
    keep = _episode(conn, session="d1", seq=1, content="survivor turn")
    doomed = _episode(conn, session="d1", seq=2, content="doomed turn")
    _entity(conn, "d-e1", name="Widget")
    _rel(conn, "d-sole", src="d-e1", tgt="d-e1", fact="sole fact", episodes=[doomed])
    _rel(conn, "d-shared", src="d-e1", tgt="d-e1", fact="shared fact", episodes=[keep, doomed])
    _rel(conn, "d-other", src="d-e1", tgt="d-e1", fact="unrelated fact", episodes=[keep])
    # Defensive: the extractor's array element type isn't guaranteed, so the numeric
    # STRING form must be matched by the same cascade.
    _rel(conn, "d-str", src="d-e1", tgt="d-e1", fact="string-provenance", episodes=[str(doomed)])

    with _client(db_url) as client:
        r = client.request("DELETE", f"/dash/api/episodes/{doomed}", headers=_H)
        assert r.status_code == 200
        body = r.json()
    assert body["episode_id"] == doomed
    assert body["facts_deleted"] == 2  # d-sole + d-str
    assert body["facts_unlinked"] == 1  # d-shared

    live = dict(
        conn.execute("SELECT uuid, episodes FROM kg_relationships ORDER BY uuid").fetchall()
    )
    assert set(live) == {"d-other", "d-shared"}
    assert live["d-shared"] == [keep]  # the deleted id is gone, the survivor's stays
    assert live["d-other"] == [keep]
    assert conn.execute("SELECT count(*) FROM episodes").fetchone()[0] == 1


def test_delete_episode_removes_chunks_including_mixed_windows(clean, conn, db_url):
    """A chunk's content is the verbatim concatenation of its episode window, so a chunk
    citing the deleted episode CANNOT be kept — mixed windows go too, and their count is
    reported separately so the UI can admit to the collateral."""
    keep = _episode(conn, session="d2", seq=1, content="survivor turn")
    doomed = _episode(conn, session="d2", seq=2, content="doomed turn")
    _chunk(conn, session="d2", start=1, end=1, episode_ids=[doomed], content="pure window")
    _chunk(conn, session="d2", start=1, end=2, episode_ids=[keep, doomed], content="mixed window")
    _chunk(conn, session="d2", start=2, end=3, episode_ids=[keep], content="untouched window")

    with _client(db_url) as client:
        body = client.request("DELETE", f"/dash/api/episodes/{doomed}", headers=_H).json()
    assert body["chunks_deleted"] == 2
    assert body["chunks_mixed_deleted"] == 1
    rows = conn.execute("SELECT content FROM chunks").fetchall()
    assert [r[0] for r in rows] == ["untouched window"]


def test_delete_episode_removes_timeline_events_and_queue_rows(clean, conn, db_url):
    """Chat timeline events (`ep:<id>` / `ep:<id>#<k>`) are sole-provenance by
    construction — they go. Events for other episodes, and the id-prefix near-miss
    `ep:<id>9`, must survive. extraction_queue rows follow the schema-001 FK."""
    doomed = _episode(conn, session="d3", seq=1, content="doomed turn")
    keep = _episode(conn, session="d3", seq=2, content="survivor turn")
    _event(conn, fact="event one", source="chat", source_ref=f"ep:{doomed}")
    _event(conn, fact="event two", source="chat", source_ref=f"ep:{doomed}#2")
    _event(conn, fact="other episode", source="chat", source_ref=f"ep:{keep}")
    _event(conn, fact="prefix near-miss", source="chat", source_ref=f"ep:{doomed}9")
    _event(conn, fact="a commit", source="git:synapse", source_ref="deadbeef")
    _queue(conn, content="doomed turn", episode_id=doomed, enqueued_at=_t(0))
    _queue(conn, content="survivor turn", episode_id=keep, enqueued_at=_t(0))
    _queue(conn, content="session summary", episode_id=None, enqueued_at=_t(0))

    with _client(db_url) as client:
        body = client.request("DELETE", f"/dash/api/episodes/{doomed}", headers=_H).json()
    assert body["events_deleted"] == 2
    facts = {r[0] for r in conn.execute("SELECT fact FROM timeline_events").fetchall()}
    assert facts == {"other episode", "prefix near-miss", "a commit"}
    # The FK cascade took only the doomed episode's queue row.
    left = {r[0] for r in conn.execute("SELECT content FROM extraction_queue").fetchall()}
    assert left == {"survivor turn", "session summary"}


def test_delete_episode_leaves_standing_rows_and_analytics_alone(clean, conn, db_url):
    """Deliberate non-cascade (contract §Phase 7): preferences/notes are reconciled
    standing rows whose source_ref is a birth certificate, not ownership; recall_metrics
    is a historical log. Deleting them on one episode delete IS the blanket-provenance
    mistake the sole-vs-shared rule exists to prevent."""
    doomed = _episode(conn, session="d4", seq=1, content="doomed turn")
    pid = _pref(conn, pref="User prefers bullet lists", source_ref=f"ep:{doomed}")
    nid = _note(conn, hook="User prefers bullet lists", source_ref=f"ep:{doomed}")
    mid = _metric(conn, query="a past recall")

    with _client(db_url) as client:
        assert (
            client.request("DELETE", f"/dash/api/episodes/{doomed}", headers=_H).status_code == 200
        )
    assert conn.execute("SELECT count(*) FROM preferences WHERE id=%s", (pid,)).fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM notes WHERE id=%s", (nid,)).fetchone()[0] == 1
    assert (
        conn.execute("SELECT count(*) FROM recall_metrics WHERE id=%s", (mid,)).fetchone()[0] == 1
    )


def test_delete_episode_retires_flag_and_audits(clean, conn, db_url):
    doomed = _episode(conn, session="d5", seq=1, content="doomed turn")
    with _client(db_url) as client:
        client.post("/dash/api/flag", json={"kind": "episode", "id": str(doomed)}, headers=_H)
        assert client.get("/dash/api/flags", headers=_H).json()["flags"] != []
        body = client.request("DELETE", f"/dash/api/episodes/{doomed}", headers=_H).json()
        # No dangling flag pointing at a row that no longer exists.
        assert client.get("/dash/api/flags", headers=_H).json()["flags"] == []
    audit = conn.execute(
        "SELECT action, kind, detail FROM dashboard_audit WHERE item_id=%s ORDER BY id",
        (str(doomed),),
    ).fetchall()
    assert [a[0] for a in audit] == ["flag", "delete_episode"]
    assert audit[1][1] == "episode"
    assert audit[1][2] == body  # the cascade summary IS the audit detail


def test_delete_episode_removes_chunk_queue_rows_in_every_status(clean, conn, db_url):
    """Chunk queue rows are keyed by (session_id, content) with no episode_id, so the 001
    FK never reaches them. A pending one would re-extract the deleted text into facts with
    no provenance; a done one still stores it verbatim. Both must go with the chunk."""
    keep = _episode(conn, session="d6", seq=1, content="survivor turn")
    doomed = _episode(conn, session="d6", seq=2, content="doomed turn with sk-SECRET")
    window = "survivor turn\ndoomed turn with sk-SECRET"
    _chunk(conn, session="d6", start=1, end=2, episode_ids=[keep, doomed], content=window)
    _chunk(conn, session="d6", start=3, end=4, episode_ids=[keep], content="untouched window")
    for status in ("pending", "processing", "done", "failed"):
        _queue(
            conn,
            content=window,
            status=status,
            content_type="chunk",
            session_id="d6",
            enqueued_at=_t(0),
        )
    _queue(
        conn, content=window, content_type="chunk", session_id="other-session", enqueued_at=_t(0)
    )
    _queue(
        conn, content="untouched window", content_type="chunk", session_id="d6", enqueued_at=_t(0)
    )

    with _client(db_url) as client:
        r = client.request("DELETE", f"/dash/api/episodes/{doomed}", headers=_H)
    assert r.status_code == 200
    left = conn.execute("SELECT content, session_id FROM extraction_queue ORDER BY id").fetchall()
    assert left == [(window, "other-session"), ("untouched window", "d6")]


def test_delete_episode_reports_cascaded_fact_and_event_ids(clean, conn, db_url):
    """The client drops the cascade's rows from its lists: the response names the facts
    and events that went, and only those (an unlinked fact is still there)."""
    keep = _episode(conn, session="d7", seq=1, content="survivor turn")
    doomed = _episode(conn, session="d7", seq=2, content="doomed turn")
    _entity(conn, "i-e1", name="Widget")
    _rel(conn, "i-sole", src="i-e1", tgt="i-e1", episodes=[doomed])
    _rel(conn, "i-shared", src="i-e1", tgt="i-e1", episodes=[keep, doomed])
    ev = _event(conn, fact="event one", source_ref=f"ep:{doomed}")
    _event(conn, fact="other episode", source_ref=f"ep:{keep}")

    with _client(db_url) as client:
        body = client.request("DELETE", f"/dash/api/episodes/{doomed}", headers=_H).json()
    assert body["deleted_fact_ids"] == ["i-sole"]
    assert body["deleted_event_ids"] == [ev]


def test_delete_episode_retires_flags_on_cascaded_facts_and_events(clean, conn, db_url):
    """A flag on a row the cascade removed must not outlive it as a ghost in the Flags
    list; flags on rows that survive (an unlinked fact) stay."""
    keep = _episode(conn, session="d8", seq=1, content="survivor turn")
    doomed = _episode(conn, session="d8", seq=2, content="doomed turn")
    _entity(conn, "j-e1", name="Widget")
    _rel(conn, "j-sole", src="j-e1", tgt="j-e1", episodes=[doomed])
    _rel(conn, "j-shared", src="j-e1", tgt="j-e1", episodes=[keep, doomed])
    ev = _event(conn, fact="event one", source_ref=f"ep:{doomed}")
    with _client(db_url) as client:
        for kind, item in (
            ("episode", str(doomed)),
            ("fact", "j-sole"),
            ("fact", "j-shared"),
            ("timeline_event", str(ev)),
        ):
            client.post("/dash/api/flag", json={"kind": kind, "id": item}, headers=_H)
        client.request("DELETE", f"/dash/api/episodes/{doomed}", headers=_H)
        flags = client.get("/dash/api/flags", headers=_H).json()["flags"]
    assert [(f["kind"], f["item_id"]) for f in flags] == [("fact", "j-shared")]


def test_delete_episode_deletes_retold_events_nothing_backs(clean, conn, db_url):
    """A re-told event whose re-telling turns are all gone (or were never recorded: rows
    merged before schema 104 have an empty list) goes with its owner turn: keeping it
    would leave text from the deleted turn that no stored turn backs."""
    doomed = _episode(conn, session="d9", seq=1, content="doomed turn")
    sole = _event(conn, fact="told once", source_ref=f"ep:{doomed}")
    unrecorded = _event(conn, fact="told twice, who unknown", source_ref=f"ep:{doomed}#2")
    only_doomed = _event(
        conn, fact="told by itself", source_ref=f"ep:{doomed}#3", reported_refs=[f"ep:{doomed}"]
    )

    with _client(db_url) as client:
        body = client.request("DELETE", f"/dash/api/episodes/{doomed}", headers=_H).json()
    assert body["events_deleted"] == 3
    assert sorted(body["deleted_event_ids"]) == sorted([sole, unrecorded, only_doomed])
    assert conn.execute("SELECT count(*) FROM timeline_events").fetchone()[0] == 0


def test_delete_episode_repoints_a_retold_event_to_a_surviving_turn(clean, conn, db_url):
    """An event a later turn re-told is backed by that turn: it moves to the first
    re-telling turn that survives instead of being deleted with the turn that wrote it,
    and again when that one goes, until no stored turn backs it."""
    keep = _episode(conn, session="d10", seq=1, content="survivor turn")
    doomed = _episode(conn, session="d10", seq=2, content="doomed turn")
    second = _episode(conn, session="d10", seq=3, content="re-told it too")
    retold = _event(
        conn,
        fact="told thrice",
        source_ref=f"ep:{doomed}",
        reported_refs=[f"ep:{second}", f"ep:{keep}"],
    )
    # The survivor already owns ep:<keep>, so a row moved onto it takes the next free suffix.
    own = _event(conn, fact="survivor's own", source_ref=f"ep:{keep}")

    def stored(event_id):
        return tuple(
            conn.execute(
                "SELECT source_ref, reported_refs FROM timeline_events WHERE id = %s", (event_id,)
            ).fetchone()
        )

    with _client(db_url) as client:
        body = client.request("DELETE", f"/dash/api/episodes/{doomed}", headers=_H).json()
        assert body["events_deleted"] == 0 and body["deleted_event_ids"] == []
        assert stored(retold) == (f"ep:{second}", [f"ep:{second}", f"ep:{keep}"])

        client.request("DELETE", f"/dash/api/episodes/{second}", headers=_H)
        # The turn that is gone re-told nothing any more.
        assert stored(retold) == (f"ep:{keep}#2", [f"ep:{keep}"])

        body = client.request("DELETE", f"/dash/api/episodes/{keep}", headers=_H).json()
    assert sorted(body["deleted_event_ids"]) == sorted([retold, own])
    assert conn.execute("SELECT count(*) FROM timeline_events").fetchone()[0] == 0


def test_delete_episode_without_reported_refs_deletes_retold_events(
    clean, conn, db_url, monkeypatch
):
    """Before schema 104 is applied the cascade cannot tell who re-told an event."""
    from ingestion import db_timeline

    monkeypatch.setattr(db_timeline.REPORTED_REFS_PROBE, "ready", lambda *a, **k: False)
    keep = _episode(conn, session="d11", seq=1, content="survivor turn")
    doomed = _episode(conn, session="d11", seq=2, content="doomed turn")
    _event(conn, fact="told twice", source_ref=f"ep:{doomed}", reported_refs=[f"ep:{keep}"])

    with _client(db_url) as client:
        body = client.request("DELETE", f"/dash/api/episodes/{doomed}", headers=_H).json()
    assert body["events_deleted"] == 1


def test_delete_bad_id_missing_row_and_auth(clean, conn, db_url):
    with _client(db_url) as client:
        assert client.request("DELETE", "/dash/api/episodes/1").status_code == 401
        assert client.request("DELETE", "/dash/api/facts/x").status_code == 401
        assert client.request("DELETE", "/dash/api/episodes/abc", headers=_H).status_code == 400
        r = client.request("DELETE", "/dash/api/episodes/999999", headers=_H)
        assert r.status_code == 404
        assert r.json() == {"status": "error", "detail": "episode not found"}
        r = client.request("DELETE", "/dash/api/facts/no-such-uuid", headers=_H)
        assert r.status_code == 404
        assert r.json() == {"status": "error", "detail": "fact not found"}


def test_delete_fact_single_edge_and_supersession_pointer(clean, conn, db_url):
    """A fact is a leaf: no cascade, only the dangling schema-028 pointer cleanup on the
    retired edges this one had superseded."""
    _entity(conn, "f-e1", name="Widget")
    _rel(conn, "f-old", src="f-e1", tgt="f-e1", fact="old fact", t_invalid=_t(2))
    _rel(conn, "f-new", src="f-e1", tgt="f-e1", fact="new fact")
    _rel(conn, "f-keep", src="f-e1", tgt="f-e1", fact="untouched fact")
    conn.execute("UPDATE kg_relationships SET invalidated_by='f-new' WHERE uuid='f-old'")

    with _client(db_url) as client:
        body = client.request("DELETE", "/dash/api/facts/f-new", headers=_H).json()
    assert body == {"fact_id": "f-new", "deleted": True, "supersession_links_cleared": 1}
    rows = dict(
        conn.execute("SELECT uuid, invalidated_by FROM kg_relationships ORDER BY uuid").fetchall()
    )
    assert set(rows) == {"f-keep", "f-old"}
    assert rows["f-old"] is None  # no edge cites a uuid that no longer exists
    audit = conn.execute(
        "SELECT action, kind FROM dashboard_audit WHERE item_id='f-new'"
    ).fetchall()
    assert audit == [("delete_fact", "fact")]
