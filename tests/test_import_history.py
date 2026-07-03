"""synapse-import (plugin/scripts/import_history.py) — bulk history import.

The importer must (a) discover session transcripts oldest-first and skip obvious
non-session files, (b) cut POST batches only at turn boundaries so a turn's
span_id never splits across two POSTs, (c) refuse to send anything until the
user confirms (the import costs LLM extraction), (d) resolve URL/token through
the same plugin config layer as the Stop hook, (e) fail soft per file and exit
non-zero only when every file failed, and (f) tell the user re-running/Ctrl-C is
safe (server dedups by span_id).

Stdlib-only script loaded by path (it lives outside the package, next to the
hook it shares its POST helper with). No live server: urlopen is monkeypatched.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from types import ModuleType

_REPO = Path(__file__).resolve().parents[1]
_IMPORT_SCRIPT = _REPO / "plugin" / "scripts" / "import_history.py"

# Every env var the plugin config layer consults for the endpoint/token —
# cleared so the tests are hermetic on any machine (dev box or CI).
_PLUGIN_ENV_VARS = (
    "SYNAPSE_URL",
    "SYNAPSE_INGEST_URL",
    "SYNAPSE_INGEST_TOKEN",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN",
)


def _isolated_plugin_env(monkeypatch, tmp_path, options: dict | None = None) -> None:
    """Point the plugin config layer at a scratch config dir: a controlled
    settings.json, no real env vars, no credentials file, no project .claude."""
    cfg_dir = tmp_path / "claude"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    if options is not None:
        (cfg_dir / "settings.json").write_text(
            json.dumps({"pluginConfigs": {"synapse@synapse": {"options": options}}}),
            encoding="utf-8",
        )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cfg_dir))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)  # keep Path.cwd()/.claude from finding real project settings
    for var in _PLUGIN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _load_importer() -> ModuleType:
    """Load the importer fresh. `config` and `ingest_hook` are popped around the
    load so URL/token re-resolve from the current env/settings rather than a
    module cached by an earlier test."""
    for name in ("config", "ingest_hook", "import_history"):
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location("import_history", _IMPORT_SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("config", "ingest_hook", "import_history"):
        sys.modules.pop(name, None)
    return mod


def _u(uid: str, text: str) -> dict:
    return {"type": "user", "uuid": uid, "sessionId": "s-1", "message": {"content": text}}


def _a(uid: str, text: str) -> dict:
    return {
        "type": "assistant",
        "uuid": uid,
        "sessionId": "s-1",
        "message": {"content": [{"type": "text", "text": text}]},
    }


def _turns(n: int) -> list[dict]:
    recs: list[dict] = []
    for i in range(1, n + 1):
        recs += [_u(f"u{i}", f"q{i}"), _a(f"a{i}", f"r{i}")]
    return recs


def _write_transcript(path: Path, records: list[dict], mtime: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _capture_posts(monkeypatch, ingested_per_call: int = 1, fail_urls: set[str] | None = None):
    """Replace urllib.request.urlopen with a recorder. Returns the call list:
    (url, parsed body, headers) per POST."""
    calls: list[tuple[str, dict, dict]] = []

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        if fail_urls and url in fail_urls:
            raise OSError("connection refused")
        calls.append((url, json.loads(req.data), dict(req.header_items())))
        return _FakeResponse(json.dumps({"status": "ok", "ingested": ingested_per_call}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discovery_oldest_first(monkeypatch, tmp_path):
    """Files come back sorted by mtime ascending, so the import lands in rough
    chronological order regardless of directory iteration order."""
    _isolated_plugin_env(monkeypatch, tmp_path)
    mod = _load_importer()
    root = tmp_path / "projects"
    now = time.time()
    _write_transcript(root / "proj-a" / "new.jsonl", _turns(1), mtime=now)
    _write_transcript(root / "proj-b" / "oldest.jsonl", _turns(1), mtime=now - 200)
    _write_transcript(root / "proj-a" / "middle.jsonl", _turns(1), mtime=now - 100)
    assert [p.name for p in mod.discover(root)] == ["oldest.jsonl", "middle.jsonl", "new.jsonl"]


def test_discovery_skips_non_session_files(monkeypatch, tmp_path):
    """Empty files, hidden files, and JSONL that isn't a transcript (plain text,
    records without a "type" key) are skipped; real transcripts survive."""
    _isolated_plugin_env(monkeypatch, tmp_path)
    mod = _load_importer()
    root = tmp_path / "projects"
    _write_transcript(root / "p" / "real.jsonl", _turns(1))
    (root / "p" / "empty.jsonl").write_text("", encoding="utf-8")
    (root / "p" / ".hidden.jsonl").write_text(json.dumps(_u("u", "q")), encoding="utf-8")
    (root / "p" / "notes.jsonl").write_text("just some text\nmore text\n", encoding="utf-8")
    (root / "p" / "other-tool.jsonl").write_text('{"foo": "bar"}\n', encoding="utf-8")
    (root / "p" / "readme.md").write_text("# not jsonl", encoding="utf-8")
    assert [p.name for p in mod.discover(root)] == ["real.jsonl"]


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def test_batches_cut_only_at_turn_boundaries(monkeypatch, tmp_path):
    """A batch may run past batch_size until the next user-turn start; every
    batch after the first begins at a turn boundary, and no turn is split
    (a partial turn would carry the wrong span_id)."""
    _isolated_plugin_env(monkeypatch, tmp_path)
    mod = _load_importer()
    recs = _turns(3)  # u1 a1 u2 a2 u3 a3
    got = list(mod.batches(recs, batch_size=3))
    assert [[r["uuid"] for r in b] for b in got] == [["u1", "a1", "u2", "a2"], ["u3", "a3"]]
    # every non-first batch starts at a turn boundary
    for b in got[1:]:
        assert b[0]["type"] == "user"
    # nothing dropped, order preserved
    assert [r["uuid"] for b in got for r in b] == [r["uuid"] for r in recs]


def test_batches_single_batch_when_under_size(monkeypatch, tmp_path):
    _isolated_plugin_env(monkeypatch, tmp_path)
    mod = _load_importer()
    recs = _turns(2)
    assert list(mod.batches(recs, batch_size=500)) == [recs]


# ---------------------------------------------------------------------------
# Confirmation gate
# ---------------------------------------------------------------------------


def test_no_confirmation_means_nothing_sent(monkeypatch, tmp_path):
    """Answering anything but y/yes aborts with exit 1 and ZERO POSTs — the
    summary/warning must print before any network traffic."""
    _isolated_plugin_env(monkeypatch, tmp_path)
    mod = _load_importer()
    calls = _capture_posts(monkeypatch)
    root = tmp_path / "projects"
    _write_transcript(root / "p" / "s.jsonl", _turns(2))
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    rc = mod.main(["--projects-dir", str(root)])
    assert rc == 1
    assert calls == []


def test_yes_flag_skips_prompt_and_ships(monkeypatch, tmp_path, capsys):
    _isolated_plugin_env(monkeypatch, tmp_path)
    mod = _load_importer()
    calls = _capture_posts(monkeypatch)

    def _no_prompt(*_):  # pragma: no cover - would hang a test run
        raise AssertionError("--yes must not prompt")

    monkeypatch.setattr("builtins.input", _no_prompt)
    root = tmp_path / "projects"
    _write_transcript(root / "p" / "s.jsonl", _turns(2))
    rc = mod.main(["--projects-dir", str(root), "--yes"])
    assert rc == 0
    assert len(calls) == 1
    assert [r["uuid"] for r in calls[0][1]["records"]] == ["u1", "a1", "u2", "a2"]
    assert calls[0][1]["source"] == "import"
    out = capsys.readouterr().out
    assert "[1/1]" in out and "ingested 1" in out


def test_interactive_y_ships(monkeypatch, tmp_path):
    _isolated_plugin_env(monkeypatch, tmp_path)
    mod = _load_importer()
    calls = _capture_posts(monkeypatch)
    root = tmp_path / "projects"
    _write_transcript(root / "p" / "s.jsonl", _turns(1))
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    assert mod.main(["--projects-dir", str(root)]) == 0
    assert len(calls) == 1


def test_summary_warns_costs_and_resume_safety(monkeypatch, tmp_path, capsys):
    """The pre-flight summary states scale (files/size/turns), the LLM-cost
    warning, and that Ctrl-C / re-running is safe (span_id dedup) — the messaging
    that makes the import feel non-scary."""
    _isolated_plugin_env(monkeypatch, tmp_path)
    mod = _load_importer()
    _capture_posts(monkeypatch)
    root = tmp_path / "projects"
    _write_transcript(root / "p" / "s.jsonl", _turns(3))
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    mod.main(["--projects-dir", str(root)])
    out = capsys.readouterr().out
    assert "1 transcript file(s)" in out
    assert "estimated turns  ~3" in out
    assert "KG extraction" in out
    assert "subscription usage or API credits" in out
    assert "Ctrl-C" in out and "dedup by" in out and "span_id" in out
    assert "Aborted" in out


def test_non_tty_requires_yes(monkeypatch, tmp_path):
    """No interactive stdin and no --yes → abort (exit 1), never hang or ship."""
    _isolated_plugin_env(monkeypatch, tmp_path)
    mod = _load_importer()
    calls = _capture_posts(monkeypatch)
    root = tmp_path / "projects"
    _write_transcript(root / "p" / "s.jsonl", _turns(1))
    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # EOF immediately
    assert mod.main(["--projects-dir", str(root)]) == 1
    assert calls == []


# ---------------------------------------------------------------------------
# Config resolution — same layer as the Stop hook
# ---------------------------------------------------------------------------


def test_posts_to_settings_url_with_token(monkeypatch, tmp_path):
    """URL + bearer come from the /plugin install answers in settings.json, via
    the exact ingest_hook/config path the Stop hook uses — never localhost when
    the user configured a server, and the token rides as Authorization."""
    _isolated_plugin_env(
        monkeypatch,
        tmp_path,
        options={
            "SYNAPSE_URL": "https://synapse.example.net",
            "SYNAPSE_INGEST_TOKEN": "tok-test-123",
        },
    )
    mod = _load_importer()
    calls = _capture_posts(monkeypatch)
    root = tmp_path / "projects"
    _write_transcript(root / "p" / "s.jsonl", _turns(1))
    assert mod.main(["--projects-dir", str(root), "--yes"]) == 0
    url, _body, headers = calls[0]
    assert url == "https://synapse.example.net/ingest"
    assert headers.get("Authorization") == "Bearer tok-test-123"


def test_env_override_wins_and_default_is_localhost(monkeypatch, tmp_path):
    _isolated_plugin_env(monkeypatch, tmp_path, options=None)
    mod = _load_importer()
    assert mod.ingest_hook.INGEST_URL == "http://localhost:8765/ingest"
    monkeypatch.setenv("SYNAPSE_URL", "http://10.0.0.5:8765")
    mod2 = _load_importer()
    assert mod2.ingest_hook.INGEST_URL == "http://10.0.0.5:8765/ingest"


# ---------------------------------------------------------------------------
# Fail-soft
# ---------------------------------------------------------------------------


def test_one_bad_file_does_not_stop_the_rest(monkeypatch, tmp_path, capsys):
    """A file whose POST fails is reported and skipped; the run continues and
    exits 0 because at least one file shipped."""
    _isolated_plugin_env(monkeypatch, tmp_path)
    mod = _load_importer()
    calls = _capture_posts(monkeypatch)
    root = tmp_path / "projects"
    now = time.time()
    _write_transcript(root / "p" / "bad.jsonl", _turns(1), mtime=now - 100)
    _write_transcript(root / "p" / "good.jsonl", _turns(1), mtime=now)

    real_ship = mod.ship_file

    def flaky_ship(path, batch_size):
        if path.name == "bad.jsonl":
            raise OSError("connection refused")
        return real_ship(path, batch_size)

    monkeypatch.setattr(mod, "ship_file", flaky_ship)
    rc = mod.main(["--projects-dir", str(root), "--yes"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "bad.jsonl → FAILED (OSError" in out
    assert "good.jsonl → ingested" in out
    assert "1 file(s) failed — re-run to retry" in out
    assert len(calls) == 1  # only good.jsonl reached the wire


def test_everything_failed_exits_nonzero(monkeypatch, tmp_path):
    _isolated_plugin_env(monkeypatch, tmp_path)
    mod = _load_importer()

    def fake_urlopen(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    root = tmp_path / "projects"
    _write_transcript(root / "p" / "s.jsonl", _turns(1))
    assert mod.main(["--projects-dir", str(root), "--yes"]) == 1


def test_empty_projects_dir_is_a_clean_noop(monkeypatch, tmp_path):
    _isolated_plugin_env(monkeypatch, tmp_path)
    mod = _load_importer()
    root = tmp_path / "projects"
    root.mkdir()
    assert mod.main(["--projects-dir", str(root)]) == 0
    assert mod.main(["--projects-dir", str(tmp_path / "missing")]) == 1


# ---------------------------------------------------------------------------
# Dedup-driven resume: re-running the same import is a no-op server-side
# ---------------------------------------------------------------------------


def test_rerun_ships_same_spans_for_server_dedup(monkeypatch, tmp_path):
    """Resume = just run it again: the client re-POSTs, identity is span_id, and
    the server skips stored turns. Both runs must therefore ship byte-identical
    record sets (no client-side state that could drift)."""
    _isolated_plugin_env(monkeypatch, tmp_path)
    root = tmp_path / "projects"
    _write_transcript(root / "p" / "s.jsonl", _turns(2))

    mod = _load_importer()
    calls1 = _capture_posts(monkeypatch, ingested_per_call=2)
    assert mod.main(["--projects-dir", str(root), "--yes"]) == 0
    mod2 = _load_importer()
    calls2 = _capture_posts(monkeypatch, ingested_per_call=0)  # server dedups everything
    assert mod2.main(["--projects-dir", str(root), "--yes"]) == 0
    assert [c[1] for c in calls1] == [c[1] for c in calls2]
