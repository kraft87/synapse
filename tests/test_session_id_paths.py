"""A session id is a plain file name before any hook joins it into a path.

Private mode keeps one marker per session at PRIVATE_DIR/<session_id>. The id comes
from a transcript record, a hook payload or the command line, so ``..``, an absolute
path or a backslash form would otherwise stat, write or delete a file outside
PRIVATE_DIR. Every hook applies the client core's rule (synapse_client/private.py
``plain_name``): the Claude ingest hook and the Codex stop hook read such an id as not
private and touch nothing; both private_mode.py toggles refuse it, their SessionEnd
handler leaves it alone, and the id is URL-quoted in /private-sessions/{id}. An ordinary
uuid works as before throughout. The config review CLI's file key, which also comes from
the server, is held inside CONFIG_DIR the same way.

Real stdlib servers on 127.0.0.1 only; every file a hook reads at import is under a
temp dir (scratch_user).
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import threading
import time
import urllib.parse
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.helpers.client_core import REPO, drop_scripts, load_script, scratch_user, use_core

_CODEX = REPO / "plugin-codex"
_UUID = "0b6c2f1e-4d3a-4c55-9a7e-2f1d8b9c0a11"
#: The hostile forms; "victim" is a file beside PRIVATE_DIR that no hook may touch.
_BAD = ["..", "../victim", "ABSOLUTE", "..\\victim"]


def _bad(sid: str, tmp_path: Path) -> str:
    return str(tmp_path / "victim") if sid == "ABSOLUTE" else sid


def _victims(tmp_path: Path, private_dir: Path, *, stale: bool = True) -> list[Path]:
    """Files a hostile id points at: tmp/victim (through ``..`` or the absolute form)
    and, where a backslash is a file-name character, PRIVATE_DIR/``..\\victim`` (on
    Windows that name climbs out). Stale (13h old), so a hook that took one for an
    expired marker would delete it."""
    private_dir.mkdir(parents=True, exist_ok=True)
    paths = [tmp_path / "victim"]
    if os.sep == "/":
        paths.append(private_dir / "..\\victim")
    old = time.time() - 13 * 3600
    for path in paths:
        path.write_text("keep", encoding="utf-8")
        if stale:
            os.utime(path, (old, old))
    return paths


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def scratch(monkeypatch, tmp_path) -> Iterator[Path]:
    """A scratch user; returns the private dir. Puts back sys.path and the modules the
    hooks import under shared names."""
    scratch_user(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    saved = {name: sys.modules.get(name) for name in ("common", "private_mode", "config_review")}
    yield Path(os.environ["SYNAPSE_PRIVATE_DIR"])
    drop_scripts()
    for name, mod in saved.items():
        sys.modules.pop(name, None)
        if mod is not None:
            sys.modules[name] = mod


def _codex_stop_hook() -> ModuleType:
    sys.modules.pop("common", None)
    return _load("codex_stop_hook_paths_test", _CODEX / "hooks" / "synapse_stop_hook.py")


def _private_mode(harness: str) -> ModuleType:
    if harness == "claude":
        sys.modules.pop("private_mode", None)
        return load_script("private_mode")
    sys.modules.pop("common", None)
    return _load("codex_private_mode_test", _CODEX / "scripts" / "private_mode.py")


# ---------------------------------------------------------------------------
# One rule: the core's, and nobody keeps a copy
# ---------------------------------------------------------------------------


def test_the_rule_is_the_cores_and_no_toggle_keeps_a_copy(scratch):
    use_core()
    from synapse_client.private import plain_name as core

    ids = ["s-1", _UUID, "...", ".hidden", "a.b", "C:x", ".", "..", "../x", "/etc/x"]
    ids += ["a/b", "..\\x", "C:\\x", "a\\b", "a\0b"]
    assert [core(i) for i in ids] == [True] * 6 + [False] * 9
    # The hooks' _is_private is the core's is_private, and both private_mode.py scripts
    # run the core's toggle (the tests below go through all four).
    for harness in ("claude", "codex"):
        assert not hasattr(_private_mode(harness), "_plain_name")


# ---------------------------------------------------------------------------
# Claude ingest hook: the record loop and _is_private
# ---------------------------------------------------------------------------


def _u(uid: str, sid: str) -> dict:
    return {"type": "user", "uuid": uid, "sessionId": sid, "message": {"content": "q"}}


def _a(uid: str, sid: str) -> dict:
    content = [{"type": "text", "text": "r"}]
    return {"type": "assistant", "uuid": uid, "sessionId": sid, "message": {"content": content}}


def _transcript(path: Path, record_sid: str) -> Path:
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in (_u("u1", record_sid), _a("a1", record_sid))),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("sid", _BAD)
def test_the_ingest_hook_reads_a_record_id_that_is_not_a_file_name_as_not_private(
    scratch, tmp_path, monkeypatch, sid
):
    """A resumed transcript's records carry their own sessionId, which the hook checks
    for a marker too. A hostile one used to stat and, as a stale marker, unlink a file
    outside PRIVATE_DIR (or read PRIVATE_DIR/.. as a live marker)."""
    hook = load_script("ingest_hook")
    victims = _victims(tmp_path, scratch)
    path = _transcript(tmp_path / "s-public.jsonl", _bad(sid, tmp_path))
    posted: list[Any] = []
    monkeypatch.setattr(hook, "_post_records", lambda recs, source="hook": posted.append(recs))

    assert hook._is_private(_bad(sid, tmp_path)) is False
    assert hook._ship(str(path), mode="catchup") == (1, 2)
    assert len(posted) == 1
    assert all(v.read_text(encoding="utf-8") == "keep" for v in victims)


def test_the_ingest_hook_still_reads_a_uuid_marker(scratch, tmp_path, monkeypatch):
    hook = load_script("ingest_hook")
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / _UUID).write_text("", encoding="utf-8")
    path = _transcript(tmp_path / "other-name.jsonl", _UUID)
    monkeypatch.setattr(hook, "_post_records", lambda *a, **k: pytest.fail("posted"))
    assert hook._ship(str(path), mode="catchup") == (0, 0)
    old = time.time() - 13 * 3600
    os.utime(scratch / _UUID, (old, old))
    assert hook._is_private(_UUID) is False and not (scratch / _UUID).exists()


# ---------------------------------------------------------------------------
# Codex stop hook
# ---------------------------------------------------------------------------


def _rollout(path: Path) -> Path:
    def item(role: str, kind: str, text: str) -> dict:
        content = [{"type": kind, "text": text}]
        return {"type": "response_item", "payload": {"type": "message", "role": role, "content": content}}  # fmt: skip

    recs = [item("user", "input_text", "q"), item("assistant", "output_text", "r")]
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    return path


@pytest.mark.parametrize("sid", _BAD)
def test_the_codex_stop_hook_reads_an_id_that_is_not_a_file_name_as_not_private(
    scratch, tmp_path, monkeypatch, sid
):
    """The hook tested PRIVATE_DIR/<id>.exists(): ``../victim`` or an absolute path
    named some other file, and the session was skipped as private on its say-so."""
    monkeypatch.setenv("SYNAPSE_CODEX_CURSORS", str(tmp_path / "state" / "cursors.json"))
    monkeypatch.setenv("SYNAPSE_CODEX_HOOK_LOG", str(tmp_path / "codex.log"))
    hook = _codex_stop_hook()
    victims = _victims(tmp_path, scratch, stale=False)
    path = _rollout(tmp_path / f"rollout-2026-01-31T09-12-00-{_UUID}.jsonl")
    posted: list[Any] = []
    monkeypatch.setattr(
        hook, "_post", lambda recs, s, project=None: posted.append(recs) or {"ingested": 1}
    )

    hook._ship({"session_id": _bad(sid, tmp_path), "transcript_path": str(path)})
    assert len(posted) == 1
    assert all(v.read_text(encoding="utf-8") == "keep" for v in victims)


def test_the_codex_stop_hook_still_reads_a_uuid_marker(scratch, tmp_path, monkeypatch):
    monkeypatch.setenv("SYNAPSE_CODEX_CURSORS", str(tmp_path / "state" / "cursors.json"))
    monkeypatch.setenv("SYNAPSE_CODEX_HOOK_LOG", str(tmp_path / "codex.log"))
    hook = _codex_stop_hook()
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / _UUID).write_text("", encoding="utf-8")
    path = _rollout(tmp_path / f"rollout-2026-01-31T09-12-00-{_UUID}.jsonl")
    monkeypatch.setattr(hook, "_post", lambda *a, **k: pytest.fail("posted"))
    assert hook._ship({"session_id": _UUID, "transcript_path": str(path)}) == (0, 0)


# ---------------------------------------------------------------------------
# Both private_mode.py toggles
# ---------------------------------------------------------------------------


class _PrivateSessions(ThreadingHTTPServer):
    """/private-sessions/{id}: records the raw path of every call, keeps the flag."""

    daemon_threads = True

    def __init__(self) -> None:
        rows: set[str] = set()
        self.paths: list[str] = []
        self.rows = rows
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self) -> None:
                outer.paths.append(self.path)
                sid = urllib.parse.unquote(self.path.rsplit("/", 1)[-1])
                if self.command == "PUT":
                    rows.add(sid)
                elif self.command == "DELETE":
                    rows.discard(sid)
                body = json.dumps({"status": "ok", "private": sid in rows}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_PUT = do_GET = do_DELETE = _reply

            def log_message(self, *args: Any) -> None:
                pass

        super().__init__(("127.0.0.1", 0), Handler)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


@pytest.fixture(params=["claude", "codex"])
def toggle(request, scratch, monkeypatch) -> Iterator[tuple[ModuleType, _PrivateSessions]]:
    srv = _PrivateSessions()
    threading.Thread(target=srv.serve_forever, args=(0.02,), daemon=True).start()
    monkeypatch.setenv("SYNAPSE_URL", srv.url)
    yield _private_mode(request.param), srv
    srv.shutdown()
    srv.server_close()


@pytest.mark.parametrize("sid", [*_BAD, "a\0b"])
def test_the_toggles_refuse_an_id_that_is_not_a_file_name(toggle, tmp_path, scratch, sid):
    """They used to refuse only ``/``, ``.`` and ``..``: a backslash form wrote (and
    ``off`` deleted) a marker by that name, which on Windows is outside PRIVATE_DIR."""
    mod, srv = toggle
    victims = _victims(tmp_path, scratch)
    for action in ("on", "off", "status"):
        assert mod.main([action, _bad(sid, tmp_path)]) == 1
    assert srv.paths == []
    assert all(v.read_text(encoding="utf-8") == "keep" for v in victims)
    assert sorted(p.name for p in scratch.iterdir()) == sorted(
        v.name for v in victims if v.parent == scratch
    )


@pytest.mark.parametrize("sid", _BAD)
def test_session_end_leaves_an_id_that_is_not_a_file_name_alone(
    toggle, tmp_path, scratch, monkeypatch, sid
):
    """The SessionEnd payload's id went straight into unlink()."""
    mod, srv = toggle
    victims = _victims(tmp_path, scratch)
    monkeypatch.setattr(mod.sys, "stdin", io.StringIO(json.dumps({"session_id": _bad(sid, tmp_path)})))  # fmt: skip
    assert mod.main(["--session-end"]) == 0
    assert all(v.read_text(encoding="utf-8") == "keep" for v in victims)
    assert srv.paths == []


def test_session_end_is_fail_soft_on_an_id_that_is_not_text(toggle, monkeypatch):
    mod, _ = toggle
    monkeypatch.setattr(mod.sys, "stdin", io.StringIO(json.dumps({"session_id": 42})))
    assert mod.main(["--session-end"]) == 0


def test_the_session_id_is_quoted_in_the_route(toggle):
    """A plain file name can still hold ``?``, ``#``, ``%`` or a space: unquoted, the
    server read a different id (or urllib refused the URL) and the read-back lied."""
    mod, srv = toggle
    sid = "s 1?x#y%z"
    assert mod.main(["on", sid]) == 0
    quoted = urllib.parse.quote(sid, safe="")
    assert srv.paths == [f"/private-sessions/{quoted}"] * 2
    assert srv.rows == {sid}


def test_an_ordinary_uuid_round_trips(toggle, scratch, monkeypatch):
    mod, srv = toggle
    assert mod.main(["on", _UUID]) == 0
    assert (scratch / _UUID).exists() and srv.rows == {_UUID}
    assert srv.paths == [f"/private-sessions/{_UUID}"] * 2
    monkeypatch.setattr(mod.sys, "stdin", io.StringIO(json.dumps({"session_id": _UUID})))
    assert mod.main(["--session-end"]) == 0
    assert not (scratch / _UUID).exists() and srv.rows == {_UUID}


# ---------------------------------------------------------------------------
# config_review: the server's file key stays inside CONFIG_DIR
# ---------------------------------------------------------------------------


@pytest.fixture()
def review(scratch, tmp_path, monkeypatch) -> tuple[ModuleType, Path]:
    """plugin/scripts/config_review.py with CONFIG_DIR at tmp/claude; returns it and
    that dir."""
    mod = load_script("config_review")
    cfg = tmp_path / "claude"
    cfg.mkdir()
    monkeypatch.setattr(mod.config, "CONFIG_DIR", cfg)
    return mod, cfg


def _outside(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in tmp_path.iterdir() if p.name not in ("claude", "home"))


@pytest.mark.parametrize(
    "key",
    ["../outside.md", "rules/../../outside.md", "ABSOLUTE", "..\\outside.md", "rules\\..\\..\\outside.md", "", ".", "rules/.."],
)  # fmt: skip
def test_config_review_refuses_a_key_outside_the_config_dir(review, tmp_path, key):
    mod, cfg = review
    before = _outside(tmp_path)
    key = str(tmp_path / "outside.md") if key == "ABSOLUTE" else key
    with pytest.raises(ValueError, match="outside"):
        mod._apply_rule(key, "Do not use em-dashes")
    assert list(cfg.iterdir()) == [] and _outside(tmp_path) == before


@pytest.mark.parametrize(
    ("key", "lands"),
    [
        ("CLAUDE.md", "CLAUDE.md"),
        ("rules/learned.md", "rules/learned.md"),
        (f"rules/../{_UUID}.md", f"{_UUID}.md"),
    ],
)
def test_config_review_still_writes_a_key_inside_the_config_dir(review, key, lands):
    mod, cfg = review
    path, wrote = mod._apply_rule(key, "Do not use em-dashes")
    assert wrote and Path(path).resolve() == (cfg / lands).resolve()
    assert "- Do not use em-dashes" in (cfg / lands).read_text(encoding="utf-8")


def test_config_review_follows_a_symlink_kept_inside_the_config_dir(review, tmp_path):
    """The check reads the key, not the resolved path: a rules dir symlinked to a
    dotfiles checkout still takes the write."""
    mod, cfg = review
    dotfiles = tmp_path / "dotfiles" / "rules"
    dotfiles.mkdir(parents=True)
    (cfg / "rules").symlink_to(dotfiles, target_is_directory=True)
    _, wrote = mod._apply_rule("rules/learned.md", "Filter out contract jobs")
    assert wrote
    assert "- Filter out contract jobs" in (dotfiles / "learned.md").read_text(encoding="utf-8")


def test_accept_with_a_key_outside_the_config_dir_writes_and_applies_nothing(
    review, tmp_path, monkeypatch, capsys
):
    mod, cfg = review
    before = _outside(tmp_path)
    accepted = {"file_key": "../outside.md", "rule": "r", "scope": "general"}
    monkeypatch.setattr(mod.config, "act", lambda *a, **k: accepted)
    posted: list[Any] = []
    monkeypatch.setattr(mod.config, "post_json", lambda *a, **k: posted.append(a))
    mod.cmd_accept(7, local=False)
    assert posted == []
    assert "nothing written and not applied" in capsys.readouterr().out
    assert list(cfg.iterdir()) == [] and _outside(tmp_path) == before
