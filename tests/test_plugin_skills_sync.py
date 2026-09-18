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
