"""DB-backed tests for the config-lane mirror route (mcp_server/config_sync_routes._publish_config).

Covers the mirror contract the plugin push depends on: publish upserts a (surface, scope,
file_key) row, a re-push of unchanged content is a no-op, a real edit updates content + hash, a
batch lands in one call, and two surfaces keep separate rows for the same file_key.
"""

from __future__ import annotations

import pytest

from mcp_server.config_sync_routes import _publish_config


@pytest.fixture()
def mirror(conn, db_url):
    conn.execute("TRUNCATE config_lane.config_registry")

    def pub(surface, key, content, scope="global", path="/home/u/.claude/x"):
        file = {"scope": scope, "file_key": key, "abs_path": path, "content": content}
        return _publish_config(db_url, surface, [file])

    def read(surface, key, scope="global"):
        return conn.execute(
            "SELECT content, abs_path, content_hash FROM config_lane.config_registry "
            "WHERE surface_id = %s AND scope = %s AND file_key = %s",
            (surface, scope, key),
        ).fetchone()

    return pub, read


def test_publish_stores_content_path_and_hash(mirror):
    pub, read = mirror
    r = pub("boxA", "rules/voice.md", "be terse", path="/home/u/.claude/rules/voice.md")
    assert r == {"status": "ok", "changed": 1}
    content, path, digest = read("boxA", "rules/voice.md")
    assert (content, path) == ("be terse", "/home/u/.claude/rules/voice.md")
    assert len(digest) == 64


def test_unchanged_republish_is_noop(mirror):
    pub, read = mirror
    pub("boxA", "CLAUDE.md", "same body")
    assert pub("boxA", "CLAUDE.md", "same body")["changed"] == 0  # identical -> nothing written
    assert pub("boxA", "CLAUDE.md", "new body")["changed"] == 1
    assert read("boxA", "CLAUDE.md")[0] == "new body"


def test_one_batch_publishes_every_file_and_counts_the_changed(mirror, db_url):
    pub, read = mirror
    pub("boxA", "CLAUDE.md", "global body")
    files = [
        {"scope": "global", "file_key": "CLAUDE.md", "content": "global body"},  # unchanged
        {"scope": "global", "file_key": "rules/voice.md", "content": "voice"},
        {"scope": "project:repo", "file_key": "CLAUDE.md", "content": "project body"},
    ]
    assert _publish_config(db_url, "boxA", files) == {"status": "ok", "changed": 2}
    assert read("boxA", "rules/voice.md")[0] == "voice"
    # the same file_key under global vs project scope are distinct rows on one surface
    assert read("boxA", "CLAUDE.md")[0] == "global body"
    assert read("boxA", "CLAUDE.md", "project:repo")[0] == "project body"


def test_surfaces_keep_separate_rows_for_the_same_file(mirror):
    pub, read = mirror
    pub("boxA", "CLAUDE.md", "A body")
    pub("boxB", "CLAUDE.md", "B body")
    assert read("boxA", "CLAUDE.md")[0] == "A body"  # not clobbered by boxB
    assert read("boxB", "CLAUDE.md")[0] == "B body"
