"""Tests for project ID resolution logic."""

import subprocess

from ingestion.project_id import _first_commit_hash, _read_dot_file, resolve_project_id


class TestReadDotFile:
    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / ".memory-project"
        f.write_text("my-project-id\n")
        assert _read_dot_file(tmp_path) == "my-project-id"

    def test_strips_whitespace(self, tmp_path):
        f = tmp_path / ".memory-project"
        f.write_text("  abc123  \n")
        assert _read_dot_file(tmp_path) == "abc123"

    def test_returns_none_when_missing(self, tmp_path):
        assert _read_dot_file(tmp_path) is None

    def test_returns_none_on_empty_file(self, tmp_path):
        (tmp_path / ".memory-project").write_text("   \n")
        assert _read_dot_file(tmp_path) is None


class TestFirstCommitHash:
    def test_returns_12char_hash_in_git_repo(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "t@t.com"], cwd=tmp_path, capture_output=True
        )
        subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, capture_output=True)
        (tmp_path / "f.txt").write_text("x")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

        result = _first_commit_hash(tmp_path)
        assert result is not None
        assert len(result) == 12
        assert all(c in "0123456789abcdef" for c in result)

    def test_returns_none_outside_git_repo(self, tmp_path):
        assert _first_commit_hash(tmp_path) is None

    def test_returns_none_on_empty_repo(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        assert _first_commit_hash(tmp_path) is None


class TestResolveProjectId:
    def test_dot_file_takes_priority(self, tmp_path, monkeypatch):
        (tmp_path / ".memory-project").write_text("dot-file-id")
        monkeypatch.setenv("MEMORY_PROJECT", "env-id")
        assert resolve_project_id(cwd=tmp_path) == "dot-file-id"

    def test_git_hash_second_priority(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MEMORY_PROJECT", raising=False)
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "t@t.com"], cwd=tmp_path, capture_output=True
        )
        subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, capture_output=True)
        (tmp_path / "f.txt").write_text("x")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

        result = resolve_project_id(cwd=tmp_path)
        assert len(result) == 12

    def test_env_var_third_priority(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEMORY_PROJECT", "env-project")
        result = resolve_project_id(cwd=tmp_path)
        assert result == "env-project"

    def test_fallback_is_deterministic(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MEMORY_PROJECT", raising=False)
        r1 = resolve_project_id(cwd=tmp_path)
        r2 = resolve_project_id(cwd=tmp_path)
        assert r1 == r2
        assert len(r1) > 0

    def test_fallback_differs_for_different_paths(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MEMORY_PROJECT", raising=False)
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        assert resolve_project_id(cwd=dir_a) != resolve_project_id(cwd=dir_b)
