"""PgSkillsProvider — serves skills_lane skills as skill:// MCP resources from PG.

DB-backed: tagged into the `db` xdist group via conftest's _DB_FILES. Seeds one throwaway
skill (+ a bundled file), exercises the provider methods directly (no HTTP/MCP client), and
asserts the exact wire shape fastmcp's client `sync_skills`/`download_skill` depends on:
the SKILL.md resource, the _manifest JSON (incl. the SKILL.md entry, sha256:-prefixed
hashes), and a bundled file served through the template.
"""

from __future__ import annotations

import json

import pytest

from mcp_server.skills_provider import PgSkillsProvider

_SKILL = "_pgprov_test_skill"
_BODY = "---\nname: t\ndescription: test\n---\n# Test\nbody here"


@pytest.fixture()
def seeded(conn, db_url):
    conn.execute("DELETE FROM skills_lane.skill_registry WHERE name = %s", (_SKILL,))
    conn.execute(
        "INSERT INTO skills_lane.skill_registry (name, description, body, scope, status) "
        "VALUES (%s, %s, %s, 'global', 'active')",
        (_SKILL, "test skill desc", _BODY),
    )
    conn.execute(
        "INSERT INTO skills_lane.skill_files "
        "(skill_name, path, content, sha256, size, is_executable) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (_SKILL, "scripts/run.py", b"print('hi')", "deadbeef", 11, True),
    )
    yield db_url
    conn.execute("DELETE FROM skills_lane.skill_registry WHERE name = %s", (_SKILL,))


async def test_lists_skill_and_manifest(seeded):
    p = PgSkillsProvider(seeded)
    uris = {str(r.uri) for r in await p._list_resources()}
    assert f"skill://{_SKILL}/SKILL.md" in uris
    assert f"skill://{_SKILL}/_manifest" in uris


async def test_skill_md_serves_body(seeded):
    p = PgSkillsProvider(seeded)
    res = await p._get_resource(f"skill://{_SKILL}/SKILL.md")
    assert res is not None
    assert "body here" in await res.read()


async def test_manifest_shape(seeded):
    p = PgSkillsProvider(seeded)
    res = await p._get_resource(f"skill://{_SKILL}/_manifest")
    assert res is not None
    data = json.loads(await res.read())
    assert data["skill"] == _SKILL
    by_path = {f["path"]: f for f in data["files"]}
    # SKILL.md is itself a manifest entry (download_skill iterates every file)
    assert "SKILL.md" in by_path and "scripts/run.py" in by_path
    assert all(f["hash"].startswith("sha256:") for f in data["files"])
    assert by_path["SKILL.md"]["size"] == len(_BODY.encode("utf-8"))


async def test_bundled_file_via_template(seeded):
    p = PgSkillsProvider(seeded)
    # Bundled files are NOT served by _get_resource (return None) ...
    assert await p._get_resource(f"skill://{_SKILL}/scripts/run.py") is None
    # ... they resolve through the template, mirroring the stock provider.
    tmpl = await p._get_resource_template(f"skill://{_SKILL}/scripts/run.py")
    assert tmpl is not None
    assert await tmpl.read({"path": "scripts/run.py"}) == "print('hi')"


async def test_inactive_skill_not_served(seeded, conn):
    conn.execute(
        "UPDATE skills_lane.skill_registry SET status = 'proposed' WHERE name = %s", (_SKILL,)
    )
    p = PgSkillsProvider(seeded)
    uris = {str(r.uri) for r in await p._list_resources()}
    assert f"skill://{_SKILL}/SKILL.md" not in uris
    assert await p._get_resource(f"skill://{_SKILL}/SKILL.md") is None
