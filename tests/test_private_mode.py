"""Private mode: a session marked "off the record" must never reach memory.

Three legs, tested here without a database or a live server:

  * the Stop hook's LOCAL half (``plugin/scripts/ingest_hook.py``) — a live marker file
    stops every POST for that session, a stale marker is ignored AND deleted, and an
    unreadable marker directory counts as private (fail-safe direction: silently
    capturing a believed-private session is the unacceptable failure);
  * the SERVER chokepoint (``ingestion.private_sessions`` + ``ingestion.episode_writer``) —
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
import subprocess
import sys
import time
import urllib.error
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import ModuleType
from urllib.parse import parse_qs, urlsplit

import pytest

from tests.helpers.http import LoopbackServer

_REPO = Path(__file__).resolve().parents[1]
_PLUGIN_HOOK = _REPO / "plugin" / "scripts" / "ingest_hook.py"
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

    # episode-writer leg: nothing is stored yet, nothing is a replay or tombstoned
    @contextmanager
    def atomic(self):
        yield

    def get_session_span_index(self, session_id: str) -> tuple[set[str], int]:
        return set(), 0

    def span_id_exists(self, span_id: str) -> bool:
        return False

    def content_dup_exists(self, project: str | None, content: str, cwd: str | None = None) -> bool:
        return False

    def is_tombstoned(self, span_id, project, content, cwd) -> bool:
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
    the marker), so the row has to stop it here: every backfill stores through the
    episode writer."""
    from ingestion.episode_writer import store_episodes

    db = _StubDB(private={"s-private"})
    stats = store_episodes(db, [_episode("s-private", 1), _episode("s-private", 2)], "demo")
    assert stats.written == 0 and stats.private == 2
    assert db.written == [] and db.enqueued == []


def test_backfill_still_writes_a_normal_session():
    from ingestion.episode_writer import store_episodes

    db = _StubDB(private={"s-private"})
    assert store_episodes(db, [_episode("s-public", 1)], "demo").written == 1
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
    """A stand-in for /private-sessions/{id}: records calls, serves the flag back.

    By default it behaves like a server from before purge existed: PUT answers only
    ``private``. ``stored_turns`` makes it report that count; ``purge`` ("ok", "refuse" or
    "error") makes it act on ``?purge=1`` the way the current route does."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.queries: list[dict[str, list[str]]] = []
        self.agents: list[str] = []
        self.rows: set[str] = set()
        self.lie_on_put = False  # simulate a write that silently doesn't land
        self.stored_turns: int | None = None
        self.purge: str | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _reply(self) -> None:
                url = urlsplit(self.path)
                sid = url.path.rsplit("/", 1)[-1]
                query = parse_qs(url.query)
                outer.calls.append((self.command, sid))
                outer.queries.append(query)
                outer.agents.append(self.headers.get("User-Agent", ""))
                if self.command == "PUT" and not outer.lie_on_put:
                    outer.rows.add(sid)
                elif self.command == "DELETE":
                    outer.rows.discard(sid)
                payload: dict = {"status": "ok", "private": sid in outer.rows}
                if self.command == "PUT":
                    if outer.stored_turns is not None:
                        payload["stored_turns"] = outer.stored_turns
                    purging = query.get("purge") == ["1"]
                    if purging and outer.purge == "refuse":
                        # the flag is committed first, so it stays set
                        detail = "purge refused: needs full trust"
                        return self._send(403, {"status": "error", "detail": detail})
                    if purging and outer.purge == "error":
                        # the purge transaction failed after the flag was committed
                        return self._send(500, {"status": "error", "detail": "purge failed"})
                    if purging and outer.purge == "ok":
                        payload["purged"] = {
                            "episodes_deleted": outer.stored_turns or 0,
                            "facts_deleted": 4,
                            "events_deleted": 1,
                            "preferences_deleted": 2,
                        }
                        payload["stored_turns"] = 0
                self._send(200, payload)

            do_PUT = do_GET = do_DELETE = _reply

            def log_message(self, *args) -> None:  # keep pytest output clean
                pass

        self._server = LoopbackServer(Handler)
        self.url = self._server.url

    def close(self) -> None:
        self._server.close()


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


def test_on_says_how_many_earlier_turns_are_still_stored(toggle, stub_server, capsys):
    """Private mode covers the session from now on; what was ingested before stays
    recallable, so the count and the way to remove it are printed, and nothing is deleted."""
    stub_server.stored_turns = 3
    assert toggle.main(["on", "s-abc"]) == 0
    out = capsys.readouterr().out
    assert "3 turn(s) of this session were already stored" in out and "--purge" in out
    assert "deleted" not in out and "purged" not in out
    assert stub_server.queries[0] == {}  # no purge was asked for


def test_on_is_quiet_about_earlier_turns_when_there_are_none_or_the_server_is_old(
    toggle, stub_server, capsys
):
    stub_server.stored_turns = 0
    assert toggle.main(["on", "s-abc"]) == 0
    assert "already stored" not in capsys.readouterr().out
    stub_server.stored_turns = None  # a server from before the count: nothing to claim
    assert toggle.main(["on", "s-def"]) == 0
    assert "already stored" not in capsys.readouterr().out


def test_on_purge_asks_the_server_to_delete_and_reports_what_it_deleted(
    toggle, stub_server, capsys
):
    stub_server.stored_turns = 3
    stub_server.purge = "ok"
    assert toggle.main(["on", "s-abc", "--purge"]) == 0
    assert stub_server.queries[0] == {"purge": ["1"]}
    out = capsys.readouterr().out
    assert "purged 3 stored turn(s)" in out and "4 fact(s)" in out and "2 preference(s)" in out
    assert "already stored" not in out
    assert "s-abc" in stub_server.rows


def test_on_purge_against_a_server_that_ignores_it_never_claims_a_deletion(
    toggle, stub_server, capsys
):
    """An older server drops the unknown query parameter and answers a plain 200."""
    assert toggle.main(["on", "s-abc", "--purge"]) == 1
    captured = capsys.readouterr()
    assert "purged" not in captured.out and "deleted" not in captured.out
    assert "--purge did not run" in captured.err
    assert "s-abc" in stub_server.rows  # private mode itself is on, and the output says so
    assert "private mode is ON" in captured.err


def test_on_purge_refused_keeps_private_mode_on_and_claims_no_deletion(toggle, stub_server, capsys):
    stub_server.purge = "refuse"
    assert toggle.main(["on", "s-abc", "--purge"]) == 1
    captured = capsys.readouterr()
    assert "purge refused: needs full trust" in captured.err
    assert "private mode is ON" in captured.err and "nothing is claimed deleted" in captured.err
    assert "NOT off the record" not in captured.err  # the flag did land
    assert "s-abc" in stub_server.rows
    assert [c[0] for c in stub_server.calls] == ["PUT", "GET"]  # the flag was read back


def test_on_purge_server_error_still_reports_private_mode_on(toggle, stub_server, capsys):
    """The route commits the flag before it purges, so a 5xx from the purge must not read as
    "server flag not set / NOT off the record": the read-back says the flag landed."""
    stub_server.purge = "error"
    assert toggle.main(["on", "s-abc", "--purge"]) == 1
    captured = capsys.readouterr()
    assert "private mode is ON" in captured.err and "nothing is claimed deleted" in captured.err
    assert "purge failed" in captured.err
    assert "NOT off the record" not in captured.err and "purged" not in captured.out
    assert [c[0] for c in stub_server.calls] == ["PUT", "GET"]


@pytest.mark.parametrize(
    "timeout",
    [
        # urlopen raises the bare form when the request went out and the reply never came
        # (http.client.getresponse() runs outside do_open's URLError wrapper)...
        TimeoutError("timed out"),
        # ...and the wrapped form only when connect or send timed out.
        urllib.error.URLError(TimeoutError("timed out")),
    ],
    ids=["bare", "wrapped"],
)
def test_on_purge_timeout_says_the_outcome_is_unknown(
    toggle, stub_server, monkeypatch, capsys, timeout
):
    """The purge runs in one long transaction; when the answer does not come in time it may
    still finish. The flag is read back, and the output claims neither state of the turns."""
    real = toggle.config.request_json

    def slow_purge(method, path, *args, **kwargs):
        if method == "PUT" and "purge=1" in path:
            real(method, path.split("?")[0], *args, **kwargs)  # the flag lands server-side
            raise timeout
        return real(method, path, *args, **kwargs)

    monkeypatch.setattr(toggle.config, "request_json", slow_purge)
    assert toggle.main(["on", "s-abc", "--purge"]) == 1
    captured = capsys.readouterr()
    assert "private mode is ON" in captured.err and "nothing is claimed deleted" in captured.err
    assert "outcome is unknown" in captured.err and "on --purge" in captured.err
    assert "NOT off the record" not in captured.err and "purged" not in captured.out
    assert [c[0] for c in stub_server.calls] == ["PUT", "GET"]


def test_purge_only_belongs_to_on(toggle, stub_server, capsys):
    assert toggle.main(["off", "s-abc", "--purge"]) == 2
    assert toggle.main(["status", "s-abc", "--purge"]) == 2
    assert "only applies to `on`" in capsys.readouterr().err
    assert stub_server.calls == []


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


def test_flag_shaped_session_id_is_rejected(toggle, stub_server, capsys):
    # `off --forget s-abc` (flag before id) must not no-op against a session
    # literally named "--forget" while s-abc silently stays private.
    assert toggle.main(["off", "--forget", "s-abc"]) == 1
    assert "looks like a flag" in capsys.readouterr().err
    assert stub_server.calls == []


# ---------------------------------------------------------------------------
# The Codex entry point is the shared script, not a copy of it
# ---------------------------------------------------------------------------

_CODEX_TOGGLE = _REPO / "plugin-codex" / "scripts" / "private_mode.py"


def _codex(tmp_path, stub_server, *args: str) -> subprocess.CompletedProcess:
    """plugin-codex/scripts/private_mode.py as Codex runs it: a subprocess in the session's
    directory, with the server URL only in the user-level Claude settings."""
    cfg_dir = tmp_path / "claude"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "settings.json").write_text(
        json.dumps(
            {"pluginConfigs": {"synapse@synapse": {"options": {"SYNAPSE_URL": stub_server.url}}}}
        )
    )
    project = tmp_path / "project"
    (project / ".claude").mkdir(parents=True, exist_ok=True)
    # A repository's own settings are not Codex's: they must not redirect the toggle.
    (project / ".claude" / "settings.json").write_text(
        json.dumps(
            {
                "pluginConfigs": {
                    "synapse@synapse": {"options": {"SYNAPSE_URL": "http://127.0.0.1:9"}}
                }
            }
        )
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("SYNAPSE_", "CLAUDE_PLUGIN_OPTION_"))
    }
    env |= {
        "HOME": str(tmp_path),
        "CLAUDE_CONFIG_DIR": str(cfg_dir),
        "SYNAPSE_HOME": str(tmp_path / "synapse-home"),
        "SYNAPSE_DATA_DIR": str(tmp_path / "data"),
        "SYNAPSE_PRIVATE_DIR": str(tmp_path / "private"),
    }
    return subprocess.run(
        [sys.executable, str(_CODEX_TOGGLE), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=project,
        check=False,
    )


def test_codex_toggle_runs_the_shared_script_as_codex(stub_server, tmp_path):
    stub_server.stored_turns = 2
    done = _codex(tmp_path, stub_server, "on", "s-abc")
    assert done.returncode == 0, done.stderr
    assert "2 turn(s) of this session were already stored" in done.stdout
    assert (tmp_path / "private" / "s-abc").exists()
    assert "s-abc" in stub_server.rows
    assert stub_server.agents and all(
        a.startswith("synapse-codex-plugin/") for a in stub_server.agents
    )


def test_codex_toggle_purges_by_name_and_refuses_a_flag_for_an_id(stub_server, tmp_path):
    stub_server.stored_turns = 2
    stub_server.purge = "ok"
    done = _codex(tmp_path, stub_server, "on", "s-abc", "--purge")
    assert done.returncode == 0, done.stderr
    assert "purged 2 stored turn(s)" in done.stdout
    assert stub_server.queries[-2] == {"purge": ["1"]}  # the PUT; the read-back GET follows
    assert _codex(tmp_path, stub_server, "status", "s-abc", "--purge").returncode == 2


def test_codex_toggle_against_an_older_server_does_not_claim_a_purge(stub_server, tmp_path):
    done = _codex(tmp_path, stub_server, "on", "s-abc", "--purge")
    assert done.returncode == 1
    assert "--purge did not run" in done.stderr


def test_codex_session_end_drops_the_marker(stub_server, tmp_path):
    assert _codex(tmp_path, stub_server, "on", "s-end").returncode == 0
    done = subprocess.run(
        [sys.executable, str(_CODEX_TOGGLE), "--session-end"],
        input=json.dumps({"session_id": "s-end"}),
        capture_output=True,
        text=True,
        env={**os.environ, "SYNAPSE_PRIVATE_DIR": str(tmp_path / "private")},
        check=False,
    )
    assert done.returncode == 0
    assert not (tmp_path / "private" / "s-end").exists()
