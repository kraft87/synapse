"""Private mode: a session marked "off the record" must never reach memory.

Three legs, tested here without a database or a live server:

  * the Stop hook's LOCAL half (``plugin/scripts/ingest_hook.py``) — a live marker file
    stops every POST for that session, a stale marker is ignored AND deleted, and an
    unreadable marker directory counts as private (fail-safe direction: silently
    capturing a believed-private session is the unacceptable failure);
  * the SERVER chokepoint (``ingestion.private_sessions`` + ``ingestion.backfill``) —
    the predicate that drops private turns wherever a parsed turn becomes an episode,
    which is what covers the paths that bypass the hook entirely;
  * the TOGGLE CLI (``plugin/scripts/private_mode.py``) round-tripped against a stubbed
    HTTP endpoint — it must verify both writes and exit nonzero when either fails.

The hooks live outside the package and run under the CLI's bare Python, so they're
loaded by path (same as test_ingest_hook_cursor.py). The /ingest route's own drop is
covered DB-backed in test_private_session_routes.py.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[1]
_PLUGIN_HOOK = _REPO / "plugin" / "scripts" / "ingest_hook.py"
_ROOT_HOOK = _REPO / "scripts" / "synapse_ingest_hook.py"
_TOGGLE = _REPO / "plugin" / "scripts" / "private_mode.py"

_PLUGIN_ENV_VARS = (
    "SYNAPSE_URL",
    "SYNAPSE_INGEST_URL",
    "SYNAPSE_INGEST_TOKEN",
    "SYNAPSE_PRIVATE_DIR",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN",
)


def _isolate(monkeypatch, tmp_path) -> Path:
    """Point the plugin config layer entirely at tmp_path; returns the private dir."""
    cfg_dir = tmp_path / "claude"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    private_dir = tmp_path / "private"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cfg_dir))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SYNAPSE_INGEST_LOG", str(tmp_path / "hook.log"))
    monkeypatch.setenv("SYNAPSE_PRIVATE_DIR", str(private_dir))
    monkeypatch.chdir(tmp_path)
    for var in _PLUGIN_ENV_VARS:
        if var != "SYNAPSE_PRIVATE_DIR":
            monkeypatch.delenv(var, raising=False)
    return private_dir


def _load(path: Path, name: str) -> ModuleType:
    sys.modules.pop("config", None)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules.pop("config", None)
    return mod


@pytest.fixture()
def hook(monkeypatch, tmp_path) -> ModuleType:
    """A fresh plugin-hook module whose config/state/markers all live under tmp_path."""
    _isolate(monkeypatch, tmp_path)
    return _load(_PLUGIN_HOOK, "ingest_hook_private_test")


def _u(uid: str, sid: str = "s-private", text: str = "q") -> dict:
    return {"type": "user", "uuid": uid, "sessionId": sid, "message": {"content": text}}


def _a(uid: str, sid: str = "s-private", text: str = "r") -> dict:
    return {
        "type": "assistant",
        "uuid": uid,
        "sessionId": sid,
        "message": {"content": [{"type": "text", "text": text}]},
    }


def _write_jsonl(path: Path, recs: list[dict]) -> None:
    with open(path, "wb") as f:
        for r in recs:
            f.write(json.dumps(r).encode() + b"\n")


def _transcript(tmp_path: Path, sid: str) -> Path:
    """A 2-turn transcript named <session_id>.jsonl, like Claude Code writes."""
    path = tmp_path / f"{sid}.jsonl"
    _write_jsonl(path, [_u("u1", sid), _a("a1", sid), _u("u2", sid), _a("a2", sid)])
    return path


def _mark(private_dir: Path, sid: str, age_hours: float = 0.0) -> Path:
    private_dir.mkdir(parents=True, exist_ok=True)
    marker = private_dir / sid
    marker.write_text("", encoding="utf-8")
    if age_hours:
        old = time.time() - age_hours * 3600
        os.utime(marker, (old, old))
    return marker


# ---------------------------------------------------------------------------
# Hook: the local marker stops shipping
# ---------------------------------------------------------------------------


def test_marker_present_ships_nothing(hook, tmp_path, monkeypatch):
    """The whole point: with a live marker, _ship POSTs nothing and records no cursor."""
    sid = "s-private"
    path = _transcript(tmp_path, sid)
    _mark(Path(os.environ["SYNAPSE_PRIVATE_DIR"]), sid)
    posted: list[list[dict]] = []
    monkeypatch.setattr(hook, "_post_records", lambda recs, source="hook": posted.append(recs))

    assert hook._ship(str(path), mode="stop") == (0, 0)
    assert posted == []
    assert str(path) not in hook._load_state()  # cursor untouched — nothing was consumed


def test_marker_blocks_the_catchup_sweep_too(hook, tmp_path, monkeypatch):
    """The SessionStart sweep re-scans disk, so it must apply the same check per session:
    the private transcript is skipped while its neighbour still ships."""
    projects = tmp_path / "projects" / "proj"
    projects.mkdir(parents=True)
    private = _transcript(projects, "s-private")
    public = _transcript(projects, "s-public")
    _mark(Path(os.environ["SYNAPSE_PRIVATE_DIR"]), "s-private")
    old = time.time() - 3600  # outside ACTIVE_GRACE so the sweep considers both files
    for p in (private, public):
        os.utime(p, (old, old))
    posted: list[list[dict]] = []
    monkeypatch.setattr(hook, "_post_records", lambda recs, source="hook": posted.append(recs))

    hook._catchup(str(tmp_path / "projects"), skip_path="")

    shipped_sids = {r["sessionId"] for batch in posted for r in batch}
    assert shipped_sids == {"s-public"}


def test_marker_matches_record_session_id_not_just_filename(hook, tmp_path, monkeypatch):
    """A resumed/compacted transcript can carry a sessionId that differs from its
    filename, and it's the record's id that gets stored — so that id must be honoured."""
    path = tmp_path / "some-other-name.jsonl"
    _write_jsonl(path, [_u("u1", "s-resumed"), _a("a1", "s-resumed")])
    _mark(Path(os.environ["SYNAPSE_PRIVATE_DIR"]), "s-resumed")
    posted: list[list[dict]] = []
    monkeypatch.setattr(hook, "_post_records", lambda recs, source="hook": posted.append(recs))

    assert hook._ship(str(path), mode="stop") == (0, 0)
    assert posted == []


def test_stale_marker_is_ignored_and_removed(hook, tmp_path, monkeypatch):
    """Markers expire (12h): a session id is never reused, so an orphan left by a
    crashed SessionEnd would otherwise suppress nothing forever while looking live."""
    sid = "s-private"
    path = _transcript(tmp_path, sid)
    marker = _mark(Path(os.environ["SYNAPSE_PRIVATE_DIR"]), sid, age_hours=13)
    posted: list[list[dict]] = []
    monkeypatch.setattr(
        hook, "_post_records", lambda recs, source="hook": posted.append(recs) or "ok"
    )

    posts, shipped = hook._ship(str(path), mode="catchup")
    assert (posts, shipped) == (1, 4)
    assert not marker.exists()  # ignored AND cleaned up


def test_fresh_marker_survives_the_ttl_check(hook, tmp_path):
    sid = "s-private"
    _mark(Path(os.environ["SYNAPSE_PRIVATE_DIR"]), sid, age_hours=11)
    assert hook._is_private(sid) is True


def test_unreadable_marker_dir_skips_ingestion(hook, tmp_path, monkeypatch):
    """Fail-safe direction: if the marker can't be read we cannot prove the session is
    NOT private, so we skip. A deferred turn is recoverable; a silently captured one is
    not. (os.stat is patched rather than chmod'ed so the test holds when run as root.)"""
    sid = "s-private"
    path = _transcript(tmp_path, sid)
    _mark(Path(os.environ["SYNAPSE_PRIVATE_DIR"]), sid)
    posted: list[list[dict]] = []
    monkeypatch.setattr(hook, "_post_records", lambda recs, source="hook": posted.append(recs))
    monkeypatch.setattr(
        hook.os, "stat", lambda *a, **k: (_ for _ in ()).throw(PermissionError("EACCES"))
    )

    assert hook._is_private(sid) is True
    assert hook._ship(str(path), mode="stop") == (0, 0)
    assert posted == []


def test_no_marker_ships_normally(hook, tmp_path, monkeypatch):
    """Guard against the check turning into a blanket block."""
    path = _transcript(tmp_path, "s-public")
    posted: list[list[dict]] = []
    monkeypatch.setattr(
        hook, "_post_records", lambda recs, source="hook": posted.append(recs) or "ok"
    )
    assert hook._ship(str(path), mode="stop")[1] == 4
    assert len(posted) == 1


def test_root_hook_copy_honours_the_marker(monkeypatch, tmp_path):
    """The repo-root Stop hook (the pre-plugin deployment path) ships the same
    transcripts, so it carries the same check — otherwise private mode would be
    silently local-only on those installs."""
    _isolate(monkeypatch, tmp_path)
    mod = _load(_ROOT_HOOK, "root_hook_private_test")
    sid = "s-private"
    path = _transcript(tmp_path, sid)
    marker = _mark(Path(os.environ["SYNAPSE_PRIVATE_DIR"]), sid)

    def _boom(*a, **k):
        raise AssertionError("posted a private session")

    monkeypatch.setattr(mod.urllib.request, "urlopen", _boom)
    assert mod._is_private(sid) is True
    mod._ship(str(path))  # would raise via _boom if it POSTed

    os.utime(marker, (time.time() - 13 * 3600,) * 2)
    assert mod._is_private(sid) is False and not marker.exists()


# ---------------------------------------------------------------------------
# Server chokepoint: the predicate + its backfill wiring
# ---------------------------------------------------------------------------


class _StubDB:
    """Minimal Database stand-in for the backfill write path."""

    def __init__(self, private: set[str] | None = None) -> None:
        self.private = private or set()
        self.lookups: list[str] = []
        self.written: list[object] = []
        self.enqueued: list[object] = []

    # private-mode leg
    def is_private_session(self, session_id: str) -> bool:
        self.lookups.append(session_id)
        return session_id in self.private

    # backfill leg
    def get_session_episodes(self, session_id: str) -> list[dict]:
        return []

    def span_id_exists(self, span_id: str) -> bool:
        return False

    def upsert_episode(self, ep: object) -> int:
        self.written.append(ep)
        return len(self.written)

    def enqueue_extraction(self, item: object) -> None:
        self.enqueued.append(item)


def _episode(session_id: str, seq: int = 1):
    from ingestion.models import Episode

    return Episode(
        session_id=session_id,
        sequence=seq,
        span_id=f"jsonl:{session_id}:{seq}",
        content=f"[user] hello {seq}",
        project="demo",
    )


def test_predicate_drops_private_and_memoizes():
    from ingestion.private_sessions import PrivateSessions

    db = _StubDB(private={"s-private"})
    p = PrivateSessions(db)
    assert p.is_private("s-private") is True
    assert p.is_private("s-private") is True  # cached — one SELECT per session, not per turn
    assert p.is_private("s-public") is False
    assert p.is_private(None) is False and p.is_private("") is False
    assert db.lookups == ["s-private", "s-public"]


def test_backfill_drops_a_private_session():
    """The disk sweep is the path the marker file can't cover (the transcript outlives
    the marker), so the row has to stop it here."""
    from ingestion.backfill import _write_session_episodes, write_backfill_session

    db = _StubDB(private={"s-private"})
    eps = [_episode("s-private", 1), _episode("s-private", 2)]
    assert _write_session_episodes(db, "s-private", eps, "demo") == 0
    assert write_backfill_session(db, "s-private", eps) == 0
    assert db.written == [] and db.enqueued == []


def test_backfill_still_writes_a_normal_session():
    from ingestion.backfill import _write_session_episodes

    db = _StubDB(private={"s-private"})
    assert _write_session_episodes(db, "s-public", [_episode("s-public", 1)], "demo") == 1
    assert len(db.written) == 1


def test_predicate_fails_closed_on_lookup_error():
    """A lookup that raises must propagate out of the chokepoint (failing the POST /
    backfill run), never be swallowed into "not private"."""
    from ingestion.private_sessions import PrivateSessions

    class Boom(_StubDB):
        def is_private_session(self, session_id: str) -> bool:
            raise RuntimeError("connection reset")

    with pytest.raises(RuntimeError):
        PrivateSessions(Boom()).is_private("s-private")


# ---------------------------------------------------------------------------
# Toggle CLI round-trip against a stubbed endpoint
# ---------------------------------------------------------------------------


class _StubServer:
    """A stand-in for /private-sessions/{id}: records calls, serves the flag back."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.rows: set[str] = set()
        self.lie_on_put = False  # simulate a write that silently doesn't land
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self) -> None:
                sid = self.path.rsplit("/", 1)[-1]
                outer.calls.append((self.command, sid))
                if self.command == "PUT" and not outer.lie_on_put:
                    outer.rows.add(sid)
                elif self.command == "DELETE":
                    outer.rows.discard(sid)
                body = json.dumps({"status": "ok", "private": sid in outer.rows}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_PUT = do_GET = do_DELETE = _reply

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


@pytest.fixture()
def toggle(monkeypatch, tmp_path, stub_server) -> ModuleType:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("SYNAPSE_URL", stub_server.url)
    return _load(_TOGGLE, "private_mode_test")


def test_on_writes_marker_and_verifies_server(toggle, stub_server):
    sid = "s-abc"
    assert toggle.main(["on", sid]) == 0
    assert (toggle.PRIVATE_DIR / sid).exists()
    # wrote it, then READ IT BACK — "off the record" is never claimed on an unverified write
    assert stub_server.calls == [("PUT", sid), ("GET", sid)]
    assert sid in stub_server.rows


def test_on_exits_nonzero_when_the_server_write_does_not_land(toggle, stub_server, capsys):
    stub_server.lie_on_put = True
    assert toggle.main(["on", "s-abc"]) == 1
    assert "NOT off the record" in capsys.readouterr().err


def test_on_exits_nonzero_when_the_server_is_unreachable(monkeypatch, tmp_path, capsys):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("SYNAPSE_URL", "http://127.0.0.1:1")  # nothing listening
    mod = _load(_TOGGLE, "private_mode_unreachable_test")
    assert mod.main(["on", "s-abc"]) == 1
    err = capsys.readouterr().err
    assert "NOT off the record" in err
    # the local half still stands, and the failure says so rather than pretending
    assert (mod.PRIVATE_DIR / "s-abc").exists()
    assert "backfill" in err


def test_off_removes_marker_and_keeps_the_row(toggle, stub_server):
    """Turning private mode off must not retroactively expose what was already said:
    the transcript is still on disk, and the row is the only thing keeping it out."""
    sid = "s-abc"
    assert toggle.main(["on", sid]) == 0
    stub_server.calls.clear()
    assert toggle.main(["off", sid]) == 0
    assert not (toggle.PRIVATE_DIR / sid).exists()
    assert stub_server.calls == []  # no DELETE
    assert sid in stub_server.rows


def test_off_forget_deletes_the_row(toggle, stub_server):
    sid = "s-abc"
    assert toggle.main(["on", sid]) == 0
    stub_server.calls.clear()
    assert toggle.main(["off", sid, "--forget"]) == 0
    assert [c[0] for c in stub_server.calls] == ["DELETE", "GET"]
    assert sid not in stub_server.rows


def test_session_end_removes_marker_but_not_the_row(toggle, stub_server, monkeypatch):
    sid = "s-abc"
    assert toggle.main(["on", sid]) == 0
    stub_server.calls.clear()

    class _Stdin:
        @staticmethod
        def read() -> str:
            return json.dumps({"session_id": sid, "reason": "clear"})

    monkeypatch.setattr(toggle.sys, "stdin", _Stdin())
    assert toggle.main(["--session-end"]) == 0
    assert not (toggle.PRIVATE_DIR / sid).exists()
    assert stub_server.calls == []
    assert sid in stub_server.rows


def test_session_end_is_fail_soft_on_garbage_payload(toggle, monkeypatch):
    class _Stdin:
        @staticmethod
        def read() -> str:
            return "not json"

    monkeypatch.setattr(toggle.sys, "stdin", _Stdin())
    assert toggle.main(["--session-end"]) == 0  # a hook must never break session teardown


def test_bad_usage_exits_nonzero(toggle):
    assert toggle.main([]) == 2
    assert toggle.main(["on"]) == 2
    assert toggle.main(["on", "../escape"]) == 1  # path traversal is not a session id


def test_status_reports_both_halves(toggle, stub_server, capsys):
    sid = "s-abc"
    toggle.main(["on", sid])
    capsys.readouterr()
    assert toggle.main(["status", sid]) == 0
    assert "marker=yes server=yes" in capsys.readouterr().out
