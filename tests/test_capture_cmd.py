"""`synapse capture` (plugin/scripts/capture_cmd.py): the dry run and the backfill.

Stdlib-only script loaded by path, no live server: urlopen is a recorder that answers like
the server routes do. The report itself is the server's (tests/test_capture_dry_run.py); here
it is the walk of the local history, what is posted where, and what is printed.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "plugin" / "scripts"
_NAMES = ("config", "ingest_hook", "import_history", "capture_cmd", "synapse_project")
_ENV = (
    "SYNAPSE_URL",
    "SYNAPSE_INGEST_URL",
    "SYNAPSE_INGEST_TOKEN",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN",
)

REPORT = {
    "status": "ok",
    "enabled": False,
    "calls": 3,
    "ignored": 1,
    "errors": 0,
    "tools": {
        "mcp__acme__fetch": {
            "captured": 1,
            "skipped": {"too_short": 1},
            "samples": [
                {
                    "url": "https://example.com/a",
                    "title": "A page",
                    "kind": "web_scrape",
                    "destination": "searchable",
                    "source": "generic",
                }
            ],
        }
    },
}


@pytest.fixture()
def cmd(monkeypatch, tmp_path) -> ModuleType:
    cfg_dir = tmp_path / "claude"
    cfg_dir.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cfg_dir))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SYNAPSE_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    for var in _ENV:
        monkeypatch.delenv(var, raising=False)
    for name in _NAMES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location("capture_cmd", _SCRIPTS / "capture_cmd.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in _NAMES:
        sys.modules.pop(name, None)
    sys.path.remove(str(_SCRIPTS))
    return mod


class Server:
    """urlopen, replaced: records every POST, answers like the two routes."""

    def __init__(self, monkeypatch, *, enabled: bool = False, probe_status: int | None = None):
        self.posts: list[tuple[str, dict]] = []
        self.enabled = enabled
        self.probe_status = probe_status
        monkeypatch.setattr(urllib.request, "urlopen", self._urlopen)

    def _urlopen(self, req, timeout=None):
        path = req.full_url.removeprefix("http://localhost:8765")
        body = json.loads(req.data)
        self.posts.append((path, body))
        if path == "/capture/dry-run" and not body["records"]:
            if self.probe_status:
                raise urllib.error.HTTPError(req.full_url, self.probe_status, "x", {}, None)
            reply = {"status": "ok", "enabled": self.enabled, "tools": {}}
        elif path == "/capture/dry-run":
            reply = REPORT
        else:
            reply = {"status": "ok", "ingested": 1, "captured": 2}
        return io.BytesIO(json.dumps(reply).encode())

    def paths(self) -> list[str]:
        return [p for p, _ in self.posts]


def rec(kind: str, uid: str, content, **extra) -> dict:
    return {
        "type": kind,
        "uuid": uid,
        "sessionId": "s-1",
        "cwd": "/work/app",
        "message": {"role": kind, "content": content},
        **extra,
    }


def turns(n: int, **extra) -> list[dict]:
    out = []
    for i in range(n):
        out += [
            rec("user", f"u{i}", f"question {i}", **extra),
            rec("assistant", f"a{i}", [{"type": "text", "text": f"answer {i}"}], **extra),
        ]
    return out


def write(path: Path, records: list[dict], age_days: float = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    t = time.time() - age_days * 86400
    os.utime(path, (t, t))
    return path


def rollout(path: Path) -> Path:
    items = [
        {"type": "session_meta", "payload": {"id": "c", "cwd": "/work/codex-app"}},
        {
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q"}]},
        },
    ]  # fmt: skip
    return write(path, items)


UUID = "0199aaaa-bbbb-cccc-dddd-eeeeffff0000"


@pytest.fixture()
def history(tmp_path):
    projects, codex = tmp_path / "projects", tmp_path / "codex"
    write(projects / "-work-app" / "sess-1.jsonl", turns(2))
    write(
        projects / "-work-app" / "sess-1" / "subagents" / "agent-9.jsonl",
        turns(1, isSidechain=True),
    )
    rollout(codex / "2026" / "09" / "01" / f"rollout-2026-09-01T10-00-00-{UUID}.jsonl")
    return ["--projects-dir", str(projects), "--codex-dir", str(codex)]


# ------------------------------------------------------------------- pieces


def test_since_is_a_span_or_a_date(cmd):
    now = 1_000_000_000.0
    assert cmd.parse_since("30d", now) == now - 30 * 86400
    assert cmd.parse_since("12h", now) == now - 12 * 3600
    assert cmd.parse_since("2w", now) == now - 14 * 86400
    assert cmd.parse_since("2026-09-01") == time.mktime((2026, 9, 1, 0, 0, 0, 0, 0, -1))
    with pytest.raises(Exception, match="not a span"):
        cmd.parse_since("last month")


def test_the_report_of_each_batch_adds_up(cmd):
    total = cmd.new_total()
    for _ in range(3):
        cmd.merge(total, REPORT)
    tool = total["tools"]["mcp__acme__fetch"]
    assert (total["calls"], total["ignored"]) == (9, 3)
    assert (tool["captured"], tool["skipped"], len(tool["samples"])) == (3, {"too_short": 3}, 3)
    cmd.merge(total, REPORT)
    assert len(tool["samples"]) == 3  # a few per tool, not one per page


def test_the_report_names_the_tools_and_says_whether_capture_is_on(cmd):
    total = cmd.new_total()
    cmd.merge(total, REPORT)
    text = cmd.render(total, enabled=False)
    assert "1 would be captured, 1 skipped, 1 had nothing to capture" in text
    assert "OFF on this server" in text
    assert "mcp__acme__fetch: 1 captured, skipped: too_short 1" in text
    assert "web_scrape -> searchable: https://example.com/a - A page" in text
    assert "ON on this server" in cmd.render(total, enabled=True)


# ------------------------------------------------------------------- the dry run


def test_the_dry_run_posts_every_transcript_to_the_dry_run_route_and_prints_the_report(
    cmd, history, monkeypatch, capsys
):
    server = Server(monkeypatch)
    assert cmd.main(["--dry-run", *history]) == 0
    out = capsys.readouterr().out

    assert set(server.paths()) == {"/capture/dry-run"}  # nothing is stored
    probe, *posted = server.posts
    assert probe[1] == {"records": []}
    claude = [b for _, b in posted if "format" not in b]
    codex = [b for _, b in posted if b.get("format") == "codex"]
    assert len(claude) == 2  # the session and its sub-agent transcript
    assert any(r.get("isSidechain") for b in claude for r in b["records"])
    assert all(b["project"] == "app" for b in claude)
    assert [(b["session_id"], b["project"]) for b in codex] == [(UUID, "codex-app")]
    assert "mcp__acme__fetch: 3 captured" in out and "OFF on this server" in out


def test_since_leaves_out_older_transcripts(cmd, history, tmp_path, monkeypatch, capsys):
    write(tmp_path / "projects" / "-work-old" / "sess-old.jsonl", turns(1), age_days=90)
    server = Server(monkeypatch)
    assert cmd.main(["--dry-run", "--since", "30d", *history]) == 0
    assert len(server.posts) == 1 + 3  # the question, then the three recent files
    capsys.readouterr()


def test_the_dry_run_ships_the_large_outputs_the_transcript_only_points_to(
    cmd, tmp_path, monkeypatch
):
    projects = tmp_path / "projects"
    saved = projects / "-work-app" / "sess" / "tool-results" / "toolu_a.txt"
    saved.parent.mkdir(parents=True)
    saved.write_text("the whole page " * 100)
    redirect = f"<persisted-output>\nOutput too large (2KB). Full output saved to: {saved}\n</persisted-output>"
    use = {"type": "tool_use", "id": "toolu_a", "name": "mcp__acme__fetch", "input": {"url": "https://example.com"}}  # fmt: skip
    res = {"type": "tool_result", "tool_use_id": "toolu_a", "content": redirect}
    write(
        projects / "-work-app" / "sess.jsonl",
        [rec("user", "u", "q"), rec("assistant", "a", [use]), rec("user", "r", [res])],
    )
    monkeypatch.setattr(cmd.config, "PROJECTS_DIR", projects)
    server = Server(monkeypatch)
    cmd.main(["--dry-run", "--projects-dir", str(projects), "--codex-dir", str(tmp_path / "none")])
    (body,) = [b for _, b in server.posts if b["records"]]
    assert body["records"][2]["message"]["content"][0]["content"] == "the whole page " * 100


def test_a_server_without_the_route_is_told_to_update(cmd, history, monkeypatch, capsys):
    server = Server(monkeypatch, probe_status=404)
    assert cmd.main(["--dry-run", *history]) == 1
    assert "no /capture/dry-run" in capsys.readouterr().err
    assert len(server.posts) == 1


# ------------------------------------------------------------------- the backfill


def test_backfill_refuses_while_the_server_has_capture_off(cmd, history, monkeypatch, capsys):
    server = Server(monkeypatch, enabled=False)
    assert cmd.main(["--backfill", "--yes", *history]) == 1
    assert "capture is off on this server" in capsys.readouterr().err
    assert "/ingest" not in server.paths()


def test_backfill_posts_the_history_to_ingest_once_capture_is_on(cmd, history, monkeypatch, capsys):
    server = Server(monkeypatch, enabled=True)
    assert cmd.main(["--backfill", "--yes", *history]) == 0
    assert set(server.paths()) == {"/capture/dry-run", "/ingest"}
    bodies = [b for p, b in server.posts if p == "/ingest"]
    assert len(bodies) == 3 and {b["source"] for b in bodies} == {"import", "codex-import"}
    assert "Done: 6 page(s) captured from 3/3 transcript(s)" in capsys.readouterr().out


def test_backfill_asks_before_it_posts(cmd, history, monkeypatch, capsys):
    server = Server(monkeypatch, enabled=True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    assert cmd.main(["--backfill", *history]) == 1
    assert "Aborted" in capsys.readouterr().out and "/ingest" not in server.paths()


def test_one_failed_file_costs_that_file(cmd, history, monkeypatch, capsys):
    server = Server(monkeypatch, enabled=True)
    real = server._urlopen

    def flaky(req, timeout=None):
        if (
            req.full_url.endswith("/ingest")
            and b"sess-1" not in req.data
            and b"answer 1" in req.data
        ):
            raise OSError("connection refused")
        return real(req, timeout)

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    assert cmd.main(["--backfill", "--yes", *history]) == 0
    out = capsys.readouterr().out
    assert "FAILED" in out and "1 transcript(s) failed" in out


def test_the_synapse_command_routes_capture_here(monkeypatch):
    sys.modules.pop("synapse_cli", None)
    spec = importlib.util.spec_from_file_location("synapse_cli", _SCRIPTS / "synapse_cli.py")
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(_SCRIPTS))
    try:
        spec.loader.exec_module(cli)
    finally:
        sys.path.remove(str(_SCRIPTS))
    assert cli._PASSTHROUGH["capture"] == "capture_cmd"
