"""DB-backed tests for the config-lane ingest routes (mcp_server/config_sync_routes).

Covers the mirror contract the plugin push depends on: publish upserts a (surface, file_key) row,
a re-push of unchanged content is a no-op (changed=False), a real edit updates content + hash,
list/fetch return what was published, and two surfaces keep separate rows for the same file_key.
"""

from __future__ import annotations

from mcp_server.config_sync_routes import _fetch_config, _list_configs, _publish_config


def _pub(db_url, surface, key, content, path="/home/u/.claude/x"):
    return _publish_config(
        db_url,
        {"surface": surface, "file_key": key, "abs_path": path, "content": content},
    )


def test_publish_then_fetch(conn, db_url):
    conn.execute("TRUNCATE config_lane.config_registry")
    r = _pub(db_url, "boxA", "rules/voice.md", "be terse", path="/home/u/.claude/rules/voice.md")
    assert r["changed"] is True
    got = _fetch_config(db_url, "boxA", "rules/voice.md")
    assert got["found"] and got["content"] == "be terse"
    assert got["abs_path"] == "/home/u/.claude/rules/voice.md"
    assert got["content_hash"] == r["content_hash"]


def test_unchanged_republish_is_noop(conn, db_url):
    conn.execute("TRUNCATE config_lane.config_registry")
    _pub(db_url, "boxA", "CLAUDE.md", "same body")
    again = _pub(db_url, "boxA", "CLAUDE.md", "same body")
    assert again["changed"] is False  # identical content -> nothing written
    edit = _pub(db_url, "boxA", "CLAUDE.md", "new body")
    assert edit["changed"] is True
    assert _fetch_config(db_url, "boxA", "CLAUDE.md")["content"] == "new body"


def test_list_scoped_to_surface(conn, db_url):
    conn.execute("TRUNCATE config_lane.config_registry")
    _pub(db_url, "boxA", "CLAUDE.md", "A body")
    _pub(db_url, "boxA", "rules/voice.md", "A voice")
    _pub(db_url, "boxB", "CLAUDE.md", "B body")  # same file_key, different surface = separate row
    a = {f["file_key"] for f in _list_configs(db_url, "boxA")["files"]}
    assert a == {"CLAUDE.md", "rules/voice.md"}
    b = _list_configs(db_url, "boxB")["files"]
    assert [f["file_key"] for f in b] == ["CLAUDE.md"]
    assert (
        _fetch_config(db_url, "boxB", "CLAUDE.md")["content"] == "B body"
    )  # not clobbered by boxA


def test_fetch_missing(conn, db_url):
    conn.execute("TRUNCATE config_lane.config_registry")
    assert _fetch_config(db_url, "boxA", "nope.md") == {"found": False}


def test_scope_separates_same_file_key(conn, db_url):
    # The same file_key (CLAUDE.md) under global vs project scope are distinct rows on one surface.
    conn.execute("TRUNCATE config_lane.config_registry")
    _publish_config(
        db_url,
        {"surface": "boxA", "scope": "global", "file_key": "CLAUDE.md", "content": "global body"},
    )
    _publish_config(
        db_url,
        {
            "surface": "boxA",
            "scope": "project:repo",
            "file_key": "CLAUDE.md",
            "content": "project body",
        },
    )
    assert _fetch_config(db_url, "boxA", "CLAUDE.md", "global")["content"] == "global body"
    assert _fetch_config(db_url, "boxA", "CLAUDE.md", "project:repo")["content"] == "project body"
    scopes = {(f["scope"], f["file_key"]) for f in _list_configs(db_url, "boxA")["files"]}
    assert scopes == {("global", "CLAUDE.md"), ("project:repo", "CLAUDE.md")}
