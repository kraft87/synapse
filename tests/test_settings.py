"""Process configuration: the environment first, then one .env file, the same everywhere."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion import settings


@pytest.fixture()
def env_file(monkeypatch, tmp_path):
    """Point the fallback file at a temp path; returns a writer for its contents."""
    path = tmp_path / "synapse.env"
    monkeypatch.setenv("SYNAPSE_ENV_FILE", str(path))

    def write(text: str) -> None:
        path.write_text(text)

    return write


def test_the_file_is_the_repo_root_env_not_the_working_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("SYNAPSE_ENV_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SYNAPSE_DB_URL=postgresql://cwd/db\n")
    assert settings.env_file_path() == Path(settings.__file__).resolve().parent.parent / ".env"
    assert settings.env_file_path() != tmp_path / ".env"


def test_a_missing_file_is_empty_config(monkeypatch):
    monkeypatch.setenv("SYNAPSE_ENV_FILE", "/nonexistent/synapse-test.env")
    monkeypatch.delenv("SOME_KEY", raising=False)
    assert settings.env_file_values() == {}
    assert settings.setting("SOME_KEY") == ""
    assert settings.setting("SOME_KEY", "fallback") == "fallback"


def test_the_environment_wins_and_an_empty_variable_falls_through(monkeypatch, env_file):
    env_file("SOME_KEY=from-file\nOTHER_KEY=other-file\n")
    monkeypatch.setenv("SOME_KEY", "from-env")
    monkeypatch.setenv("OTHER_KEY", "")
    assert settings.setting("SOME_KEY") == "from-env"
    assert settings.setting("OTHER_KEY") == "other-file"


def test_file_syntax_matches_what_compose_reads(env_file):
    env_file(
        "# a comment\n"
        "\n"
        "PLAIN=value\n"
        "  SPACED = padded  \n"
        'DOUBLE="quoted # kept"\n'
        "SINGLE='single'\n"
        "TRAILING=value  # dropped\n"
        "URL=postgresql://u:p@host:5432/db?x=a=b\n"
        "EMPTY=\n"
        "REPEATED=first\n"
        "REPEATED=second\n"
        "no equals sign\n"
    )
    assert settings.env_file_values() == {
        "PLAIN": "value",
        "SPACED": "padded",
        "DOUBLE": "quoted # kept",
        "SINGLE": "single",
        "TRAILING": "value",
        "URL": "postgresql://u:p@host:5432/db?x=a=b",
        "EMPTY": "",
        "REPEATED": "second",
    }


def test_require_setting_names_the_key_and_the_file(monkeypatch, env_file):
    env_file("")
    monkeypatch.delenv("NEEDED_KEY", raising=False)
    with pytest.raises(RuntimeError, match=r"NEEDED_KEY is not set .*synapse\.env"):
        settings.require_setting("NEEDED_KEY")
    env_file("NEEDED_KEY=present\n")
    assert settings.require_setting("NEEDED_KEY") == "present"


def test_poller_settings_run_from_the_env_file_alone(monkeypatch, env_file):
    """A bare ``python -m ingestion`` with only a .env: every key it needs comes from it."""
    for key in ("SYNAPSE_DB_URL", "POLL_INTERVAL_SECONDS", "VOYAGE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    env_file("SYNAPSE_DB_URL=postgresql://file/db\nVOYAGE_API_KEY=pa-file\n")
    cfg = settings.get_settings()
    assert (cfg.db_url, cfg.poll_interval_seconds, cfg.voyage_api_key) == (
        "postgresql://file/db",
        300,
        "pa-file",
    )


def test_poller_settings_need_a_database(monkeypatch, env_file):
    env_file("")
    monkeypatch.delenv("SYNAPSE_DB_URL", raising=False)
    with pytest.raises(RuntimeError, match="SYNAPSE_DB_URL is not set"):
        settings.get_settings()


def test_poller_interval_must_be_a_number(monkeypatch, env_file):
    env_file("")
    monkeypatch.setenv("SYNAPSE_DB_URL", "postgresql://x/db")
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "soon")
    with pytest.raises(ValueError, match="POLL_INTERVAL_SECONDS must be a whole number"):
        settings.get_settings()
