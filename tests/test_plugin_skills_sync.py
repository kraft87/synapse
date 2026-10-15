"""plugin/scripts/skills_sync.py main(): scope routing must never sync one folder twice.

A session started in the home directory has CLAUDE_PROJECT_DIR=$HOME, so the project
skills folder ($HOME/.claude/skills) is the global folder. Syncing it under both scopes
flipped every skill's scope on each session start (skill_registry is unique on name),
leaving the global listing empty for every other host and for the Codex plugin.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "plugin" / "scripts"


def _load_sync(monkeypatch, tmp_path, project_dir):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CLAUDE_SKILLS_DIR", str(tmp_path / ".claude" / "skills"))
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(project_dir))
    monkeypatch.setenv("SYNAPSE_SKILLS_SYNC", "1")
    monkeypatch.delenv("SYNAPSE_INGEST_TOKEN", raising=False)
    monkeypatch.delitem(sys.modules, "config", raising=False)
    monkeypatch.syspath_prepend(str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "synapse_plugin_skills_sync_test", _SCRIPTS / "skills_sync.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_post(published):
    def post_json(path, payload, timeout=30.0):
        if path == "/skills/list":
            return {"skills": []}
        if path == "/skills/publish":
            published.append(payload["scope"])
            return {"status": "ok"}
        raise AssertionError(f"unexpected route {path}")

    return post_json


def _write_skill(folder: Path, name: str) -> None:
    (folder / name).mkdir(parents=True)
    (folder / name / "SKILL.md").write_text(f"---\ndescription: {name}\n---\n{name}\n")


def test_home_project_syncs_global_scope_once(monkeypatch, tmp_path, capsys):
    module = _load_sync(monkeypatch, tmp_path, tmp_path)
    _write_skill(tmp_path / ".claude" / "skills", "alpha")
    published: list[str] = []
    monkeypatch.setattr(module.config, "post_json", _fake_post(published))

    module.main()

    assert published == ["global"]
    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["reloadSkills"] is True


def test_real_project_dir_still_syncs_project_scope(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    _write_skill(repo / ".claude" / "skills", "beta")
    module = _load_sync(monkeypatch, tmp_path, repo)
    published: list[str] = []
    monkeypatch.setattr(module.config, "post_json", _fake_post(published))

    module.main()

    assert published == ["project:repo"]


class _Server:
    """A /skills/* stub: the listing it serves, a body for /skills/fetch, and every call."""

    def __init__(self, skills: list[dict], fetched: dict | None = None) -> None:
        self.skills = skills
        self.fetched = fetched or {}
        self.calls: list[str] = []

    def post_json(self, path, payload, timeout=30.0):
        self.calls.append(path)
        if path == "/skills/list":
            return {"skills": self.skills, "publish": True}
        if path == "/skills/fetch":
            return self.fetched
        return {"status": "ok"}


def test_listing_content_hash_is_compared_without_the_body(monkeypatch, tmp_path):
    """The server names each skill's whole-skill hash, so the listing needs no body: an
    identical copy settles on the hash alone."""
    module = _load_sync(monkeypatch, tmp_path, tmp_path)
    skills_dir = tmp_path / ".claude" / "skills"
    _write_skill(skills_dir, "alpha")
    body = (skills_dir / "alpha" / "SKILL.md").read_text()
    server = _Server([{"name": "alpha", "content_hash": module._whole_hash(body, []), "files": []}])

    assert module._sync("global", skills_dir, client=server) == (0, 0, 0)
    assert server.calls == ["/skills/list"]


def test_listing_without_content_hash_still_compares_the_body(monkeypatch, tmp_path):
    """A server from before the field: the hash is computed from the body it sends."""
    module = _load_sync(monkeypatch, tmp_path, tmp_path)
    skills_dir = tmp_path / ".claude" / "skills"
    _write_skill(skills_dir, "alpha")
    body = (skills_dir / "alpha" / "SKILL.md").read_text()
    server = _Server([{"name": "alpha", "body": body, "files": []}])

    assert module._sync("global", skills_dir, client=server) == (0, 0, 0)
    assert server.calls == ["/skills/list"]


def test_a_different_content_hash_pulls_and_remembers_that_hash(monkeypatch, tmp_path):
    module = _load_sync(monkeypatch, tmp_path, tmp_path)
    skills_dir = tmp_path / ".claude" / "skills"
    _write_skill(skills_dir, "alpha")
    newer = "2999-01-01T00:00:00+00:00"
    theirs = module._whole_hash("---\ndescription: alpha\n---\nedited elsewhere\n", [])
    server = _Server(
        [{"name": "alpha", "content_hash": theirs, "files": [], "content_modified_at": newer}],
        fetched={
            "found": True,
            "body": "---\ndescription: alpha\n---\nedited elsewhere\n",
            "files": [],
        },
    )

    assert module._sync("global", skills_dir, client=server) == (1, 0, 0)
    assert "/skills/fetch" in server.calls
    assert theirs in module._load_ledger(skills_dir)["alpha"]


def test_a_folder_being_synced_is_skipped_by_a_second_sync(monkeypatch, tmp_path):
    """Two session starts (or a compaction beside one) used to run two two-way syncs over
    one folder, sharing the pull temp file and the ledgers. Now the second skips."""
    module = _load_sync(monkeypatch, tmp_path, tmp_path)
    skills_dir = tmp_path / ".claude" / "skills"
    _write_skill(skills_dir, "alpha")
    server = _Server([])
    lock_path = module._lock_path(skills_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "w") as held:
        module.lock_exclusive(held, blocking=False)
        assert module._sync("global", skills_dir, client=server) == (0, 0, 0)
        assert server.calls == []  # not even the listing

        # the lock is per folder: another folder syncs while this one is held
        other = tmp_path / "other" / "skills"
        _write_skill(other, "beta")
        assert module._sync("global", other, client=server) == (0, 1, 0)

    assert module._sync("global", skills_dir, client=server) == (0, 1, 0)  # released
