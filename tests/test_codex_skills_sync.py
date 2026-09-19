"""Codex skills sync: scans only the user-skills folder, shares the Claude engine, locks."""

from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path

import pytest

_PLUGIN = Path(__file__).resolve().parents[1] / "plugin-codex"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeServer:
    """In-memory stand-in for the /skills/* routes, keyed by (scope, name)."""

    def __init__(self):
        self.skills: dict[tuple[str, str], dict] = {}
        self.calls: list[str] = []

    def post_json(self, path, payload, timeout=30.0):
        self.calls.append(path)
        if path == "/skills/list":
            return {
                "skills": [
                    {
                        "name": name,
                        "body": s["body"],
                        "files": [{"path": f["path"], "sha256": f["sha256"]} for f in s["files"]],
                        "content_modified_at": s["content_modified_at"],
                    }
                    for (scope, name), s in self.skills.items()
                    if scope == payload["scope"]
                ]
            }
        if path == "/skills/fetch":
            for (_scope, name), s in self.skills.items():
                if name == payload["name"]:
                    return {"found": True, **s}
            return {"found": False}
        if path == "/skills/publish":
            self.skills[(payload["scope"], payload["name"])] = {
                "body": payload["body"],
                "files": payload["files"],
                "content_modified_at": payload["content_modified_at"],
            }
            return {"status": "ok"}
        if path == "/skills/overwrite":
            return {"status": "ok"}
        raise AssertionError(f"unexpected route {path}")


@pytest.fixture
def sync(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return _load("synapse_codex_skills_sync_test", _PLUGIN / "scripts" / "skills_sync.py")


def _seed_server(server, name, body, files=()):
    encoded = []
    for rel, content in files:
        import hashlib

        encoded.append(
            {
                "path": rel,
                "content_b64": base64.b64encode(content).decode(),
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
                "is_executable": False,
            }
        )
    server.skills[("global", name)] = {
        "body": body,
        "files": encoded,
        "content_modified_at": "2026-09-18T12:00:00+00:00",
    }


def test_pull_materializes_only_into_user_skills_dir(sync, tmp_path):
    server = FakeServer()
    _seed_server(server, "alpha", "---\ndescription: a\n---\nalpha body\n", [("ref.md", b"ref")])
    target = tmp_path / ".agents" / "skills"
    system = tmp_path / ".codex" / "skills" / ".system" / "bundled"
    system.mkdir(parents=True)
    (system / "SKILL.md").write_text("bundled\n")

    assert sync.run(server, target, tmp_path / ".synapse" / "lock") == (1, 0)

    assert (target / "alpha" / "SKILL.md").read_text() == "---\ndescription: a\n---\nalpha body\n"
    assert (target / "alpha" / "ref.md").read_bytes() == b"ref"
    assert (system / "SKILL.md").read_text() == "bundled\n"
    assert ("global", "bundled") not in server.skills
    assert "/skills/publish" not in server.calls


def test_push_local_only_skill_to_global_scope(sync, tmp_path):
    server = FakeServer()
    target = tmp_path / ".agents" / "skills"
    (target / "local").mkdir(parents=True)
    (target / "local" / "SKILL.md").write_text("---\ndescription: mine\n---\nlocal\n")

    assert sync.run(server, target, tmp_path / ".synapse" / "lock") == (0, 1)

    published = server.skills[("global", "local")]
    assert published["body"].endswith("local\n")
    assert sync.run(server, target, tmp_path / ".synapse" / "lock") == (0, 0)


def test_lock_held_skips_without_touching_server(sync, tmp_path):
    server = FakeServer()
    _seed_server(server, "alpha", "alpha\n")
    lock_path = tmp_path / ".synapse" / "lock"
    lock_path.parent.mkdir(parents=True)
    filelock = _load(
        "synapse_filelock_test", _PLUGIN.parent / "plugin" / "scripts" / "synapse_filelock.py"
    )
    with open(lock_path, "w") as held:
        filelock.lock_exclusive(held, blocking=False)
        assert sync.run(server, tmp_path / ".agents" / "skills", lock_path) is None
    assert server.calls == []
    assert not (tmp_path / ".agents" / "skills" / "alpha").exists()


def test_main_is_noop_unless_opted_in(sync, monkeypatch, capsys):
    monkeypatch.setattr(sync, "SKILLS_SYNC", False)
    calls = []
    monkeypatch.setattr(sync, "run", lambda *a: calls.append(a))
    assert sync.main() == 0
    assert calls == []
    assert capsys.readouterr().out == ""


def test_main_reports_counts_and_fails_open(sync, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sync, "SKILLS_SYNC", True)
    monkeypatch.setattr(sync, "SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(sync, "run", lambda *a: (2, 1))
    assert sync.main() == 0
    assert "pulled 2, pushed 1" in capsys.readouterr().out

    def boom(*a):
        raise ConnectionError("server down")

    monkeypatch.setattr(sync, "run", boom)
    assert sync.main() == 0
    assert capsys.readouterr().out == ""


def test_session_start_runs_sync_under_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    hook = _load("synapse_codex_session_start_test", _PLUGIN / "hooks" / "session_start.py")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["timeout"] = cmd, kw["timeout"]

    monkeypatch.setattr(hook.subprocess, "run", fake_run)
    monkeypatch.setenv("SYNAPSE_SKILLS_SYNC", "0")
    hook._sync_skills()
    assert seen == {}
    monkeypatch.setenv("SYNAPSE_SKILLS_SYNC", "1")
    monkeypatch.setenv("SYNAPSE_CODEX_SKILLS_SYNC_TIMEOUT", "3")
    hook._sync_skills()
    assert seen["cmd"][-1].endswith("skills_sync.py")
    assert seen["timeout"] == 3.0
    assert json.dumps(seen["cmd"])  # plain argv, no shell
