"""The client bootstrap helper is reversible and never touches what it did not create."""

from __future__ import annotations

import json
import re
import shlex
import tomllib

from mcp_gateway import client_setup as cs


def test_install_then_remove_round_trip(tmp_path):
    assert cs.main(["install-bootstrap", "--client", "claude", "--skills-dir", str(tmp_path)]) == 0
    md = tmp_path / cs.STUB_NAME / "SKILL.md"
    assert md.read_text() == cs.STUB
    # Idempotent.
    assert cs.main(["install-bootstrap", "--client", "claude", "--skills-dir", str(tmp_path)]) == 0
    assert cs.main(["remove-bootstrap", "--client", "claude", "--skills-dir", str(tmp_path)]) == 0
    assert not (tmp_path / cs.STUB_NAME).exists()
    assert tmp_path.exists()  # the skills folder itself is never removed


def test_dry_run_writes_nothing(tmp_path):
    assert (
        cs.main(
            ["install-bootstrap", "--client", "codex", "--skills-dir", str(tmp_path), "--dry-run"]
        )
        == 0
    )
    assert list(tmp_path.iterdir()) == []


def test_foreign_or_edited_folders_are_left_alone(tmp_path):
    foreign = tmp_path / cs.STUB_NAME
    foreign.mkdir()
    (foreign / "SKILL.md").write_text("---\nname: mine\n---\nhand written\n")
    assert cs.main(["install-bootstrap", "--client", "claude", "--skills-dir", str(tmp_path)]) == 1
    assert cs.main(["remove-bootstrap", "--client", "claude", "--skills-dir", str(tmp_path)]) == 1
    assert (foreign / "SKILL.md").read_text().endswith("hand written\n")

    (foreign / "SKILL.md").write_text(cs.STUB)
    (foreign / "notes.md").write_text("user addition")
    assert cs.main(["remove-bootstrap", "--client", "claude", "--skills-dir", str(tmp_path)]) == 1
    assert (foreign / "notes.md").exists()


def test_codex_default_dir_follows_plugin_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNAPSE_CODEX_SKILLS_DIR", str(tmp_path))
    assert cs._default_dir("codex") == tmp_path


def test_stub_points_at_the_gateway_served_workflow_under_a_distinct_name():
    assert "skill://gateway-research/SKILL.md" in cs.STUB
    # A different name from the gateway-served skill, so skills sync can never make two
    # different documents answer to the same skill:// URI.
    assert cs.STUB_NAME != "gateway-research"
    assert re.search(r"^name: synapse-gateway-research$", cs.STUB, re.M)


def test_snippets_are_valid_client_config(capsys):
    assert cs.main(["snippets", "--gateway-url", "https://gw.example:8766/mcp"]) == 0
    out = capsys.readouterr().out
    add_json = next(line for line in out.splitlines() if line.startswith("claude mcp add-json"))
    cfg = json.loads(shlex.split(add_json)[-1])
    assert cfg["url"] == "https://gw.example:8766/mcp"
    assert "--gateway" in cfg["headersHelper"]
    toml_block = out.split("## Codex — add to ~/.codex/config.toml\n", 1)[1].split(
        "## Codex — remove"
    )[0]
    server = tomllib.loads(toml_block)["mcp_servers"]["synapse-gateway"]
    assert server["url"] == "https://gw.example:8766/mcp"
    assert shlex.split(server["http_headers_helper"])[-3:] == ["--url", server["url"], "--gateway"]
    assert "Bearer ${SYNAPSE_INGEST_TOKEN}" in out  # a reference, never a token value


def test_user_edits_that_keep_the_marker_are_preserved(tmp_path):
    """The marker line is a hint for humans, not proof of ownership: an edited copy is the
    user's now, and neither install (overwrite) nor remove (delete) may touch it."""
    args = ["--client", "claude", "--skills-dir", str(tmp_path)]
    assert cs.main(["install-bootstrap", *args]) == 0
    md = tmp_path / cs.STUB_NAME / "SKILL.md"
    edited = cs.STUB.replace("say so and stop", "say so, then try the direct connection")
    assert cs.MARKER in edited and edited != cs.STUB
    md.write_text(edited)

    assert cs.main(["install-bootstrap", *args]) == 1
    assert md.read_text() == edited
    assert cs.main(["remove-bootstrap", *args]) == 1
    assert md.read_text() == edited
    assert cs.main(["remove-bootstrap", *args, "--dry-run"]) == 1


def test_whitespace_only_edit_is_still_an_edit(tmp_path):
    args = ["--client", "codex", "--skills-dir", str(tmp_path)]
    assert cs.main(["install-bootstrap", *args]) == 0
    md = tmp_path / cs.STUB_NAME / "SKILL.md"
    md.write_text(cs.STUB + "\n")
    assert cs.main(["remove-bootstrap", *args]) == 1
    assert md.exists()


def test_saved_credential_snippets_drop_env_token_overrides(capsys):
    """Live pilot finding: a shell exporting the root token silently overrode the saved
    device credential. --saved-credential wraps the helper so the overrides are removed."""
    assert (
        cs.main(["snippets", "--gateway-url", "http://gw.example:8766", "--saved-credential"]) == 0
    )
    out = capsys.readouterr().out
    add_json = next(line for line in out.splitlines() if line.startswith("claude mcp add-json"))
    claude_helper = shlex.split(json.loads(shlex.split(add_json)[-1])["headersHelper"])
    toml_block = out.split("## Codex — add to ~/.codex/config.toml\n", 1)[1].split("## Optional")[0]
    codex_helper = shlex.split(
        tomllib.loads(toml_block)["mcp_servers"]["synapse-gateway"]["http_headers_helper"]
    )
    for argv in (claude_helper, codex_helper):
        assert argv[:5] == [
            "env",
            "-u",
            "SYNAPSE_INGEST_TOKEN",
            "-u",
            "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN",
        ]
        assert argv[-1] == "--gateway"
    assert cs.main(["snippets"]) == 0
    assert "env -u" not in capsys.readouterr().out.split("## Claude Code — add")[1]
