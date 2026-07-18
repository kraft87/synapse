"""DB-backed tests for the operator dashboard routes (mcp_server/dashboard_routes.py).

Covers the /dash/api/* contract (catalog, feed merge + keyset cursor + type-filter
behavior, episode detail, derived facts/events, session ordering, entity dossier,
search per-type + total_by_type, flag toggle + audit + feed reflection) and the
unauthenticated static bundle routes (503 without a build, 200 + traversal rejection
with a monkeypatched dist dir). Skips cleanly when no test DB is reachable (mirrors
test_board.py). ALL fixture data is synthetic — this repo is public.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from urllib.parse import quote

import psycopg
import pytest
from psycopg.types.json import Json

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

import mcp_server.dashboard_routes as dr  # noqa: E402
from mcp_server.dashboard_routes import register  # noqa: E402

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


def _entity(
    conn, uuid, *, name, entity_type="Technology", summary="a thing", degree=0, group_id="technical"
):
    return conn.execute(
        "INSERT INTO kg_entities (uuid, group_id, name, entity_type, summary, degree, created_at) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING uuid",
        (uuid, group_id, name, entity_type, summary, degree, _t(0)),
    ).fetchone()[0]


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
        " created_at, t_valid, t_invalid) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING uuid",
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
):
    return conn.execute(
        "INSERT INTO timeline_events "
        "(t_valid, fact, source, source_ref, project, salience, domain, event_type, ingested_at) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
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
        ),
    ).fetchone()[0]


def _note(conn, *, hook="User prefers X", group_id="technical"):
    return conn.execute(
        "INSERT INTO notes (owner_id, group_id, type, hook, body) "
        "VALUES ('default',%s,'user',%s,'body') RETURNING id",
        (group_id, hook),
    ).fetchone()[0]


def _pref(conn, *, pref="User prefers bullet lists", group_id="technical"):
    return conn.execute(
        "INSERT INTO preferences (owner_id, group_id, pref, polarity) "
        "VALUES ('default',%s,%s,'like') RETURNING id",
        (group_id, pref),
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
    legs=None,  # optional {embed,bm25,vector,kg,web,rerank,timeline,prefs} → ms_* columns
):
    legs = legs or {}
    return conn.execute(
        "INSERT INTO recall_metrics (kind, source, query, ms_total, est_tokens, "
        " rerank_top_score, created_at, ms_embed, ms_bm25, ms_vector, ms_kg, ms_web, "
        " ms_rerank, ms_timeline, ms_prefs) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
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
            legs.get("timeline"),
            legs.get("prefs"),
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
):
    return conn.execute(
        "INSERT INTO extraction_queue "
        "(episode_id, content, status, error, attempts, enqueued_at, processed_at) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (episode_id, content, status, error, attempts, enqueued_at, processed_at),
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
    # Timeline event via ep:<id> source_ref.
    ev = _event(conn, fact="linked event", source_ref=f"ep:{eid}")
    _event(conn, fact="other event", source_ref=f"ep:{other}")
    with _client(db_url) as client:
        body = client.get(f"/dash/api/episode/{eid}/derived", headers=_H).json()
    fact_uuids = {f["uuid"] for f in body["facts"]}
    assert fact_uuids == {"dn", "ds"}  # both containment forms, other excluded
    assert [e["id"] for e in body["timeline_events"]] == [ev]


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
    assert [e["sequence"] for e in body["episodes"]] == [1, 2, 3]
    assert body["episodes"][0]["content"] == "first"


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
        degree=1,
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


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_per_type_and_total_by_type(clean, conn, db_url):
    tok = "zorptastic"
    epid = _episode(conn, session="q", seq=1, content=f"the {tok} subsystem failed")
    _rel(conn, "sr", src="a", tgt="b", fact=f"{tok} relates to widget", episodes=[epid])
    evid = _event(conn, fact=f"shipped {tok}", source_ref="tb:z")
    _entity(conn, "se", name=f"{tok} engine", degree=4)

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

        en = client.get(f"/dash/api/search?q={tok}&type=entities", headers=_H).json()
        assert en["hits"][0]["type"] == "entities" and en["hits"][0]["id"] == "se"
        assert en["hits"][0]["meta"] == {
            "name": f"{tok} engine",
            "entity_type": "Technology",
            "degree": 4,
        }


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


def test_proposal_detail_payload_types_and_provenance(clean_proposals, conn, db_url):
    # Episode sharing the skill evidence's session_id → best-effort provenance resolves.
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
        assert ep in sd["provenance_episodes"]  # resolved from evidence session_id
        assert sd["audit_log"] == []  # not decided yet

        cd = client.get(f"/dash/api/proposals/config:{cf}", headers=_H).json()
        assert cd["kind"] == "config-edit" and cd["name"] == "CLAUDE.md"
        assert cd["payload"]["type"] == "diff"
        assert "+++ b/CLAUDE.md" in cd["payload"]["content"]

        # malformed id -> 400, missing -> 404.
        assert client.get("/dash/api/proposals/bogus", headers=_H).status_code == 400
        assert client.get("/dash/api/proposals/skill:999999", headers=_H).status_code == 404


def test_proposal_decision_approve_writes_state_and_audit(clean_proposals, conn, db_url):
    sk = _skill_proposal(conn, name="latency-triage", status="proposed")
    with _client(db_url) as client:
        r = client.post(
            f"/dash/api/proposals/skill:{sk}/decision",
            json={"action": "approve", "note": "clear win"},
            headers=_H,
        )
        assert r.status_code == 200
        # dashboard approve maps to the skills lane's 'accept' (NOT promote — materializing
        # stays with the lane); the lane returns status 'accepted'.
        assert r.json()["status"] == "accepted"

    # Lane row transitioned; dashboard_audit carries the namespaced id + note.
    row = conn.execute(
        "SELECT status FROM skills_lane.skill_gap_candidates WHERE id=%s", (sk,)
    ).fetchone()
    assert row[0] == "accepted"
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
        assert set(mgr.status) == {"queue_depth", "active"}
        assert isinstance(mgr.status["queue_depth"], int)
    finally:
        mgr.unsubscribe()


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
    legs = {"embed": 10.0, "bm25": 20.0, "vector": 30.0, "kg": 40.0, "web": 5.0, "rerank": 50.0}
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
    # only timed legs appear; timeline/prefs were NULL so they're absent
    assert big["legs_p50"] == {
        "embed": 10.0,
        "bm25": 20.0,
        "vector": 30.0,
        "kg": 40.0,
        "web": 5.0,
        "rerank": 50.0,
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
    # completed/hour: rows with processed_at within 48h (failed + done)
    assert sum(p["n"] for p in b["throughput"]["completed_per_hour"]) == 2
    assert len(b["failures"]) == 1
    assert b["failures"][0]["error"] == "boom timeout" and b["failures"][0]["attempts"] == 3
    assert b["last_dream"]["id"] == run_id
    assert b["last_dream"]["counts"]["proposals_raised"] == 2
    assert round(b["last_dream"]["duration_s"]) == 41
    assert b["last_dream"]["samples"]["proposals"][0]["id"] == "config:4"


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


def _gent(conn, uuid, *, name, degree, entity_type="Technology", normalized_name=None):
    """Entity insert that ALSO sets normalized_name (the shared _entity helper doesn't),
    so the exact-normalized-name seed path is exercisable. entity_supertype mirrors
    entity_type — the graph payloads serve BOTH (client colors by supertype)."""
    return conn.execute(
        "INSERT INTO kg_entities "
        "(uuid, group_id, name, normalized_name, entity_type, entity_supertype, summary, "
        " degree, created_at) "
        "VALUES (%s,'technical',%s,%s,%s,%s,%s,%s,%s) RETURNING uuid",
        (
            uuid,
            name,
            normalized_name or name.lower(),
            entity_type,
            entity_type,
            f"{name} summary",
            degree,
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
        event_type="work",
        t_valid=_t(30),
    )
    e_mid = _event(
        conn,
        fact="took a rest day",
        source="chat",
        source_ref="deadbeef",  # git-style ref -> episode_id None
        salience=1,
        domain="personal",
        event_type="health",
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
    assert by_id[e_hi]["event_type"] == "work" and by_id[e_hi]["episode_id"] == eid
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
    work = _event(conn, fact="work event", source_ref="tb:w", event_type="work", domain="technical")
    life = _event(conn, fact="life event", source_ref="tb:l", event_type="life", domain="personal")
    untyped = _event(conn, fact="untyped", source_ref="tb:u", event_type=None, domain="technical")
    with _client(db_url) as client:
        # type chip narrows to that event_type; untyped rows drop out when a chip is active.
        only_work = client.get("/dash/api/timeline?type=work", headers=_H).json()
        assert {e["id"] for e in only_work["events"]} == {work}
        # group_id maps to the domain column (schema 038).
        personal = client.get("/dash/api/timeline?group_id=personal", headers=_H).json()
        assert {e["id"] for e in personal["events"]} == {life}
        # No filter -> everything, including the untyped row.
        allrows = client.get("/dash/api/timeline", headers=_H).json()
        assert {e["id"] for e in allrows["events"]} == {work, life, untyped}


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


def test_behavior_files_grouping_content_and_linkgraph(clean_behavior, conn, db_url):
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

        # linkgraph: node per logical file_key + one edge per [[wikilink]] instance
        lg = client.get("/dash/api/behavior/linkgraph", headers=_H).json()
        node_keys = {n["file_key"] for n in lg["nodes"]}
        assert node_keys == {"CLAUDE.md", "rules/voice.md", "memory/note1.md", "AGENTS.md"}
        edge_pairs = {(e["source"], e["target"]) for e in lg["edges"]}
        assert ("CLAUDE.md", "voice") in edge_pairs and ("CLAUDE.md", "job_crons") in edge_pairs
        assert ("rules/voice.md", "CLAUDE") in edge_pairs
        assert ("memory/note1.md", "voice") in edge_pairs
        # CLAUDE.md's duplicate [[voice]] is deduped in links but the edge list holds one 'voice'.
        assert sum(1 for e in lg["edges"] if e["source"] == "CLAUDE.md") == 2


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
            "/dash/api/preferences",
            "/dash/api/dream/report",
            "/dash/api/behavior/files",
            "/dash/api/behavior/file?key=CLAUDE.md",
            "/dash/api/behavior/linkgraph",
        ):
            r = client.get(path)
            assert r.status_code == 401
            assert r.json() == {"status": "error", "detail": "unauthorized"}


def _mini_graph(conn):
    """6-8 entities, ~10 edges, one superseded + one not-yet-valid. Degrees are set so
    truncation order is deterministic. Returns the seed episode id for provenance checks."""
    _gent(conn, "g-hub", name="Synapse", degree=10, entity_type="Project")
    _gent(conn, "g-pg", name="Postgres", degree=8)
    _gent(conn, "g-cc", name="Claude Code", degree=6)
    _gent(conn, "g-kg", name="knowledge graph", degree=5, entity_type="Concept")
    _gent(conn, "g-anthropic", name="Anthropic", degree=4, entity_type="Organization")
    _gent(conn, "g-nuc", name="homelab NUC", degree=3)  # depth-2 (via g-pg)
    _gent(conn, "g-sse", name="SSE stream", degree=2)  # depth-2 (via g-cc)
    _gent(conn, "g-far", name="embedding cache", degree=1)  # depth-2 (via g-pg)
    _gent(conn, "g-future", name="future thing", degree=0)  # only via a not-yet-valid edge

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
        ta = client.get("/dash/api/graph/entities?q=Synapse", headers=_H).json()
        assert all("supertype" in e for e in ta)


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
        # budget 3 -> seed (always) + the two highest-degree reachable nodes (g-pg=8, g-cc=6).
        assert kept == {"g-hub", "g-pg", "g-cc"}
        # every returned edge connects two kept nodes.
        for e in r["edges"]:
            assert e["src"] in kept and e["tgt"] in kept


def test_graph_neighborhood_edge_cap_prunes_orphans(clean, conn, db_url):
    """A dense hub must not ship an unbounded hairball: over edge_cap, seed-adjacent +
    most-retrieved edges win, truncated flips, and nodes the cut orphaned are pruned
    (regression: the live 707-degree hub returned 150 nodes / ~2K edges and hung the
    browser in layout)."""
    _gent(conn, "ec-hub", name="Hub", degree=9)
    for i in range(4):
        _gent(conn, f"ec-n{i}", name=f"Node {i}", degree=4 - i)
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
