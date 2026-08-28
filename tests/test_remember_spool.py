"""The remember() spool: a memory write must survive an MCP outage.

The defect this covers (2026-08-25): MCP OAuth was down mid-session, remember() could not be
called, and a user correction was silently lost for two days. Everything here is the local
half of the fix — the plugin script that queues an intent to disk and replays it over the
plain-HTTP lane that stayed up. No database and no live server: the replay target is a stub
HTTP server, so these are fast and cover the failure shapes a live server can't produce on
demand (dies mid-flush, confirms then the client crashes, 4xx vs 5xx).

Loaded by path like test_private_mode.py — the hooks live outside the package and run under
the CLI's bare Python.

The route half is covered DB-backed in test_remember_routes.py.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SPOOL = _REPO / "plugin" / "scripts" / "remember_spool.py"

_PLUGIN_ENV_VARS = (
    "SYNAPSE_URL",
    "SYNAPSE_INGEST_URL",
    "SYNAPSE_INGEST_TOKEN",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN",
)


class _StubServer:
    """A stand-in /remember/spool. Idempotent on intent_id like the real route, and able to
    fail on command so the durability paths are testable."""

    def __init__(self) -> None:
        self.seen: dict[str, int] = {}  # intent_id -> note id
        self.calls: list[dict] = []
        self.fail_after: int | None = None  # start failing once this many writes have landed
        self.status_for: dict[str, int] = {}  # intent_id -> HTTP status override
        self.next_note = 100
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # BaseHTTPRequestHandler's verb-dispatch API
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                outer.calls.append(body)
                if body.get("probe"):
                    return self._json(200, {"status": "ok", "probe": True})
                iid = body.get("intent_id")
                override = outer.status_for.get(iid)
                if override:
                    return self._json(override, {"status": "error", "detail": "refused"})
                if outer.fail_after is not None and len(outer.seen) >= outer.fail_after:
                    return self._json(503, {"status": "error", "detail": "server down"})
                if iid in outer.seen:  # the real route's idempotency contract
                    return self._json(
                        200, {"status": "ok", "outcome": "duplicate", "note_id": outer.seen[iid]}
                    )
                outer.next_note += 1
                outer.seen[iid] = outer.next_note
                return self._json(
                    200, {"status": "ok", "outcome": "created", "note_id": outer.next_note}
                )

            def _json(self, code: int, payload: dict) -> None:
                raw = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args) -> None:  # keep pytest output clean
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture()
def stub_server():
    s = _StubServer()
    yield s
    s.close()


def _load(monkeypatch, tmp_path, url: str) -> ModuleType:
    """A fresh spool module whose config, spool file and log all live under tmp_path."""
    cfg_dir = tmp_path / "claude"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cfg_dir))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SYNAPSE_SPOOL_LOG", str(tmp_path / "spool.log"))
    monkeypatch.setenv("SYNAPSE_SPOOL_TIMEOUT", "5")
    monkeypatch.setenv("SYNAPSE_SPOOL_PROBE_TIMEOUT", "5")
    for var in _PLUGIN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SYNAPSE_URL", url)
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("config", None)
    spec = importlib.util.spec_from_file_location("remember_spool_test", _SPOOL)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules.pop("config", None)
    return mod


@pytest.fixture()
def spool(monkeypatch, tmp_path, stub_server) -> ModuleType:
    return _load(monkeypatch, tmp_path, stub_server.url)


@pytest.fixture()
def offline(monkeypatch, tmp_path) -> ModuleType:
    """Same module, pointed at a port nothing is listening on — the outage case."""
    return _load(monkeypatch, tmp_path, "http://127.0.0.1:9")


def _add(mod, hook="User prefers X", body="Because Y.", type="user", **kw) -> dict:
    rec = mod.make_record(hook, body, type=type, **kw)
    mod.append(rec)
    return rec


# ---------------------------------------------------------------------------
# add / list / flush
# ---------------------------------------------------------------------------


def test_add_appends_a_durable_record(offline, capsys):
    assert (
        offline.main(["add", "--hook", "User prefers X", "--body", "Because Y.", "--type", "user"])
        == 0
    )
    recs = offline.load()
    assert len(recs) == 1
    r = recs[0]
    assert r["hook"] == "User prefers X" and r["body"] == "Because Y." and r["type"] == "user"
    assert r["origin"] == "cli" and r["id"] and r["ts"]
    # It is on DISK, not just in memory — that is the entire guarantee.
    assert json.loads(offline.SPOOL_PATH.read_text().strip())["id"] == r["id"]
    out = capsys.readouterr().out
    assert "spooled" in out and "queued, flushes at next session start" in out


def test_add_reads_json_from_stdin(offline, monkeypatch):
    payload = {"hook": "Hook from stdin", "body": "Multi\nline\nbody.", "type": "feedback"}
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(payload)))
    assert offline.main(["add"]) == 0
    r = offline.load()[0]
    assert r["hook"] == "Hook from stdin" and r["body"] == "Multi\nline\nbody."
    assert r["type"] == "feedback"


def test_add_rejects_a_half_intent(offline, capsys):
    assert offline.main(["add", "--hook", "hook only"]) == 2
    assert offline.load() == []
    assert "need --hook AND --body" in capsys.readouterr().err


def test_add_rejects_an_invalid_type(offline, capsys):
    assert offline.main(["add", "--hook", "h", "--body", "b", "--type", "bogus"]) == 2
    assert offline.load() == []
    assert "invalid type" in capsys.readouterr().err


def test_add_writes_through_immediately_when_the_server_is_up(spool, stub_server, capsys):
    """MCP absent but HTTP healthy is a real case — the user gets a write NOW, not next session."""
    assert spool.main(["add", "--hook", "User prefers X", "--body", "Because Y."]) == 0
    assert spool.load() == []  # flushed straight through
    assert len(stub_server.seen) == 1
    assert "written to memory now" in capsys.readouterr().out


def test_list_reports_queue_contents(offline, capsys):
    r = _add(offline, hook="Queued hook")
    assert offline.main(["list"]) == 0
    out = capsys.readouterr().out
    assert r["id"][:8] in out and "Queued hook" in out and "1 queued" in out


def test_list_on_empty_spool(offline, capsys):
    assert offline.main(["list"]) == 0
    assert "spool empty" in capsys.readouterr().out


def test_flush_writes_and_dequeues(spool, stub_server):
    a, b = _add(spool, hook="First"), _add(spool, hook="Second")
    res = spool.flush()
    assert [f["id"] for f in res["flushed"]] == [a["id"], b["id"]]
    assert res["left"] == 0 and res["error"] is None
    assert spool.load() == []
    # The payload the route contracts on
    posted = {c["intent_id"]: c for c in stub_server.calls if "intent_id" in c}
    assert posted[a["id"]]["hook"] == "First" and posted[a["id"]]["type"] == "user"


def test_flush_on_an_empty_spool_is_a_no_op(spool, stub_server):
    assert spool.flush() == {"flushed": [], "left": 0, "error": None, "busy": False}
    assert stub_server.calls == []


# ---------------------------------------------------------------------------
# Durability: mid-flush death, retries, concurrent appends
# ---------------------------------------------------------------------------


def test_flush_that_dies_midway_keeps_every_unconfirmed_intent(spool, stub_server):
    """The core durability claim: partial progress loses nothing and duplicates nothing."""
    recs = [_add(spool, hook=f"Note {i}") for i in range(4)]
    stub_server.fail_after = 2  # first two land, then the server falls over

    res = spool.flush()
    assert len(res["flushed"]) == 2 and res["error"] is not None
    left = spool.load()
    assert [r["id"] for r in left] == [recs[2]["id"], recs[3]["id"]]

    stub_server.fail_after = None
    res2 = spool.flush()
    assert [f["id"] for f in res2["flushed"]] == [recs[2]["id"], recs[3]["id"]]
    assert spool.load() == []
    # Four intents in, four writes out — no intent written twice.
    assert len(stub_server.seen) == 4


def test_replay_of_a_confirmed_intent_is_idempotent(spool, stub_server):
    """A confirm that never reaches the client (crash in the window) re-posts the SAME id;
    the server recognizes it and no second note is minted."""
    rec = _add(spool, hook="Written but not dequeued")
    assert spool.post_intent(rec)["outcome"] == "created"
    # The client died before remove_ids — the line is still on disk, exactly as designed.
    assert [r["id"] for r in spool.load()] == [rec["id"]]

    res = spool.flush()
    assert res["flushed"][0]["outcome"] == "duplicate"
    assert res["flushed"][0]["note_id"] == stub_server.seen[rec["id"]]
    assert spool.load() == []
    assert len(stub_server.seen) == 1  # one note, two posts


def test_flush_preserves_intents_appended_during_the_flush(spool, monkeypatch):
    """Removal is by id, never by truncation: a write spooled mid-flush must survive."""
    first = _add(spool, hook="Before flush")
    later: dict = {}
    real_post = spool.post_intent

    def _post(rec):
        out = real_post(rec)
        if not later:
            later.update(_add(spool, hook="Arrived mid-flush"))
        return out

    monkeypatch.setattr(spool, "post_intent", _post)
    res = spool.flush()
    assert [f["id"] for f in res["flushed"]] == [first["id"]]
    assert [r["id"] for r in spool.load()] == [later["id"]]


def test_flush_drops_a_4xx_refusal_but_keeps_a_5xx(spool, stub_server):
    """An unreplayable payload must not wedge the queue forever; a server fault must."""
    bad = _add(spool, hook="Rejected")
    good = _add(spool, hook="Fine")
    stub_server.status_for[bad["id"]] = 400
    res = spool.flush()
    assert [f["id"] for f in res["flushed"]] == [good["id"]]
    assert spool.load() == []  # the 400 was dropped, not retried

    other = _add(spool, hook="Transient")
    stub_server.status_for[other["id"]] = 503
    res = spool.flush()
    assert res["flushed"] == [] and res["error"]
    assert [r["id"] for r in spool.load()] == [other["id"]]


def test_flush_respects_its_budget(spool):
    recs = [_add(spool, hook=f"N{i}") for i in range(3)]
    res = spool.flush(budget=-1)  # already over budget on the first iteration
    assert res["flushed"] == [] and "budget" in (res["error"] or "")
    assert [r["id"] for r in spool.load()] == [r["id"] for r in recs]


def test_flush_is_capped_per_run(spool):
    recs = [_add(spool, hook=f"N{i}") for i in range(5)]
    res = spool.flush(max_items=2)
    assert len(res["flushed"]) == 2 and res["left"] == 3
    assert [r["id"] for r in spool.load()] == [r["id"] for r in recs[2:]]


def test_unwritable_records_are_dropped_not_retried(spool):
    spool.append({"id": "junk-id", "ts": "2026-08-25T00:00:00+00:00", "origin": "cli"})
    good = _add(spool, hook="Real")
    res = spool.flush()
    assert [f["id"] for f in res["flushed"]] == [good["id"]]
    assert spool.load() == []


# ---------------------------------------------------------------------------
# Capture path 1: PostToolUse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resp",
    [
        None,
        "",
        {"status": "error", "detail": "OAuth token expired"},
        {"isError": True, "content": [{"type": "text", "text": "connection refused"}]},
        [{"type": "text", "text": "MCP error -32000: Connection closed"}],
        {"content": [{"type": "text", "text": '{"status": "error", "detail": "boom"}'}]},
    ],
)
def test_failed_remember_is_spooled(offline, monkeypatch, capsys, resp):
    payload = {
        "hook_event_name": "PostToolUse",
        "session_id": "s-outage",
        "tool_name": "mcp__plugin_synapse_synapse__remember",
        "tool_input": {"hook": "User's correction", "body": "The real fact.", "type": "user"},
        "tool_response": resp,
    }
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(payload)))
    assert offline.main(["post-tool-use"]) == 0

    recs = offline.load()
    assert len(recs) == 1
    assert recs[0]["hook"] == "User's correction" and recs[0]["body"] == "The real fact."
    assert recs[0]["type"] == "user" and recs[0]["origin"] == "posttooluse"
    assert recs[0]["session_id"] == "s-outage"

    out = json.loads(capsys.readouterr().out)
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert out["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert "SPOOLED" in ctx and "QUEUED, not lost" in ctx


@pytest.mark.parametrize(
    "resp",
    [
        {"status": "ok", "note_id": 42, "outcome": "created"},
        {"content": [{"type": "text", "text": '{"status": "ok", "note_id": 42}'}]},
        [{"type": "text", "text": '{"note_id": 7, "outcome": "updated"}'}],
    ],
)
def test_successful_remember_spools_nothing(offline, monkeypatch, capsys, resp):
    payload = {
        "tool_input": {"hook": "h", "body": "b"},
        "tool_response": resp,
    }
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(payload)))
    assert offline.main(["post-tool-use"]) == 0
    assert offline.load() == []
    assert capsys.readouterr().out == ""


def test_post_tool_use_ignores_an_unrecoverable_call(offline, monkeypatch, capsys):
    """No hook/body/content in tool_input — nothing to queue, and no noise about it."""
    monkeypatch.setattr(
        "sys.stdin",
        __import__("io").StringIO(json.dumps({"tool_input": {}, "tool_response": None})),
    )
    assert offline.main(["post-tool-use"]) == 0
    assert offline.load() == [] and capsys.readouterr().out == ""


def test_post_tool_use_survives_garbage_stdin(offline, monkeypatch):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("not json at all"))
    assert offline.main(["post-tool-use"]) == 0


def test_legacy_content_form_is_spooled(offline, monkeypatch):
    payload = {
        "tool_input": {"content": "Chose A over B because latency."},
        "tool_response": {"status": "error"},
    }
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(payload)))
    assert offline.main(["post-tool-use"]) == 0
    rec = offline.load()[0]
    assert rec["content"] == "Chose A over B because latency." and offline.is_writable(rec)


# ---------------------------------------------------------------------------
# SessionStart
# ---------------------------------------------------------------------------


def _stdin(monkeypatch, payload="{}"):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(payload))


def test_session_start_is_silent_when_healthy_and_empty(spool, monkeypatch, capsys):
    _stdin(monkeypatch)
    assert spool.main(["session-start"]) == 0
    assert capsys.readouterr().out == ""


def test_session_start_flushes_and_reports(spool, monkeypatch, capsys, tmp_path):
    _add(spool, hook="First correction")
    _add(spool, hook="Second correction")
    _stdin(monkeypatch)
    assert spool.main(["session-start"]) == 0

    line = capsys.readouterr().out.strip()
    assert line.startswith("[Synapse] flushed 2 spooled memory write(s):")
    assert "First correction" in line and "Second correction" in line
    assert spool.load() == []
    log = (tmp_path / "spool.log").read_text()
    assert "flushed 2 spooled memory write(s)" in log and "OK flush" in log


def test_session_start_routes_writes_to_the_cli_while_the_server_is_down(
    offline, monkeypatch, capsys
):
    _stdin(monkeypatch)
    assert offline.main(["session-start"]) == 0
    out = capsys.readouterr().out
    assert "memory server is NOT reachable" in out
    assert "remember_spool.py add" in out and "--hook" in out
    assert "QUEUED" in out


def test_session_start_names_the_backlog_while_down(offline, monkeypatch, capsys):
    _add(offline, hook="Queued while down")
    _stdin(monkeypatch)
    assert offline.main(["session-start"]) == 0
    out = capsys.readouterr().out
    assert "1 memory write(s) are already queued" in out
    assert offline.load()  # still queued — an unreachable server dequeues nothing


def test_session_start_reports_a_partial_flush(spool, stub_server, monkeypatch, capsys):
    for i in range(3):
        _add(spool, hook=f"N{i}")
    stub_server.fail_after = 1
    _stdin(monkeypatch)
    assert spool.main(["session-start"]) == 0
    out = capsys.readouterr().out
    assert "flushed 1 spooled memory write(s)" in out and "2 still queued" in out
    assert len(spool.load()) == 2


def test_hook_paths_never_fail_the_turn(tmp_path):
    """Fail-soft contract: the hook entrypoints exit 0 on garbage input and a dead server.

    Run as real subprocesses — the ``__main__`` wrapper is what swallows the failure, and
    exit code is the only thing Claude Code looks at."""
    import os
    import subprocess

    env = {
        **os.environ,
        "SYNAPSE_URL": "http://127.0.0.1:9",
        "SYNAPSE_DATA_DIR": str(tmp_path / "data"),
        "SYNAPSE_SPOOL_LOG": str(tmp_path / "spool.log"),
        "SYNAPSE_SPOOL_PROBE_TIMEOUT": "2",
        "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
    }
    for var in _PLUGIN_ENV_VARS:
        if var != "SYNAPSE_URL":
            env.pop(var, None)
    for cmd in ("session-start", "post-tool-use"):
        p = subprocess.run(
            [sys.executable, str(_SPOOL), cmd],
            input="{ not json",
            capture_output=True,
            text=True,
            env=env,
        )
        assert p.returncode == 0, p.stderr
