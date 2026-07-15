"""DB-backed tests for the operator dashboard routes (mcp_server/dashboard_routes.py).

Covers the /dash/api/* contract (catalog, feed merge + keyset cursor + type-filter
behavior, episode detail, derived facts/events, session ordering, entity dossier,
search per-type + total_by_type, flag toggle + audit + feed reflection) and the
unauthenticated static bundle routes (503 without a build, 200 + traversal rejection
with a monkeypatched dist dir). Skips cleanly when no test DB is reachable (mirrors
test_board.py). ALL fixture data is synthetic — this repo is public.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

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
        "notes, preferences, dashboard_flags, dashboard_audit RESTART IDENTITY CASCADE"
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
    t_valid=None,
    ingested_at=None,
):
    return conn.execute(
        "INSERT INTO timeline_events "
        "(t_valid, fact, source, source_ref, project, salience, domain, ingested_at) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (
            t_valid or _t(0),
            fact,
            source,
            source_ref,
            project,
            salience,
            domain,
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
