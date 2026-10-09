"""DB-backed test for Recall._self_session_facts — the facts half of self-exclusion.

A fact is the calling session's own when EVERY source episode in its ``episodes``
provenance belongs to that session. Reinforced facts (an older source too), facts with
no recorded source, and malformed provenance all keep serving.
"""

from __future__ import annotations

import uuid

from mcp_server.recall import Recall
from tests.helpers.embed import GROUP


def _episode(conn, session_id: str, seq: int) -> int:
    return conn.execute(
        "INSERT INTO episodes (session_id, sequence, content) VALUES (%s, %s, 'x') RETURNING id",
        (session_id, seq),
    ).fetchone()[0]


def _fact(conn, fact_uuid: str, episodes: str | None) -> None:
    conn.execute(
        "INSERT INTO kg_relationships (uuid, owner_id, group_id, src_uuid, tgt_uuid, name, fact,"
        " episodes) VALUES (%s, 'default', %s, 'a', 'b', 'R', %s, %s::jsonb)",
        (fact_uuid, GROUP, f"fact {fact_uuid}", episodes),
    )


def test_only_facts_sourced_entirely_from_the_session_are_its_own(conn, db_url):
    me, other = f"me-{uuid.uuid4().hex[:8]}", f"other-{uuid.uuid4().hex[:8]}"
    m1, m2 = _episode(conn, me, 1), _episode(conn, me, 2)
    o1 = _episode(conn, other, 1)
    tag = uuid.uuid4().hex[:8]
    own, own_text, mixed, none_, empty, junk, missing = (
        f"{tag}-{k}" for k in ("own", "own-text", "mixed", "none", "empty", "junk", "missing")
    )
    _fact(conn, own, f"[{m1}, {m2}]")
    _fact(conn, own_text, f'["{m1}"]')  # ids stored as JSON strings resolve the same way
    _fact(conn, mixed, f"[{m1}, {o1}]")  # reinforced by this session, sourced earlier: serve
    _fact(conn, none_, None)  # no provenance (web artifact): never excluded
    _fact(conn, empty, "[]")
    _fact(conn, junk, '["not-an-id"]')
    _fact(conn, missing, "[999999999]")  # a deleted source episode is not this session's

    r = Recall(db_url, "")
    got = r._self_session_facts([own, own_text, mixed, none_, empty, junk, missing], me)
    assert got == {own, own_text}
    assert r._self_session_facts([own, own_text], other) == set()
    assert r._self_session_facts([], me) == set()


def test_lookup_failure_fails_open(db_url):
    r = Recall(db_url, "")

    def _boom():
        raise RuntimeError("db down")

    r._ensure_pg = _boom
    assert r._self_session_facts(["f-1"], "me") == set()


def _superseded(conn, old_uuid: str, new_uuid: str, old_eps: str, new_eps: str) -> None:
    conn.execute(
        "INSERT INTO kg_relationships (uuid, owner_id, group_id, src_uuid, tgt_uuid, name, fact,"
        " episodes, t_invalid, invalidated_by) VALUES"
        " (%s, 'default', %s, 'a', 'b', 'R', %s, %s::jsonb, now(), %s),"
        " (%s, 'default', %s, 'a', 'b', 'R', %s, %s::jsonb, NULL, NULL)",
        (
            old_uuid,
            GROUP,
            f"old {old_uuid}",
            old_eps,
            new_uuid,
            new_uuid,
            GROUP,
            f"now {new_uuid}",
            new_eps,
        ),
    )


def test_episode_overlay_skips_successors_from_the_calling_session(conn, db_url):
    me, other = f"me-{uuid.uuid4().hex[:8]}", f"other-{uuid.uuid4().hex[:8]}"
    old_ep = _episode(conn, other, 1)  # the served (older) episode
    mine, theirs = _episode(conn, me, 1), _episode(conn, other, 2)
    tag = uuid.uuid4().hex[:8]
    _superseded(conn, f"{tag}-p1", f"{tag}-n-mine", f"[{old_ep}]", f"[{mine}]")
    _superseded(conn, f"{tag}-p2", f"{tag}-n-theirs", f"[{old_ep}]", f"[{theirs}]")

    r = Recall(db_url, "")
    everyone = r._episode_supersessions([old_ep], GROUP)
    assert sorted(everyone[old_ep]) == [f"now {tag}-n-mine", f"now {tag}-n-theirs"]
    assert r._episode_supersessions([old_ep], GROUP, self_session=me) == {
        old_ep: [f"now {tag}-n-theirs"]
    }
