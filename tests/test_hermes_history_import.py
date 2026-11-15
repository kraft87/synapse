"""``synapse import hermes``: past Hermes sessions from ``state.db`` (history.py and
plugin/scripts/import_hermes.py).

The store is built here with the tables and columns Hermes v0.23.1 writes (schema 33,
hermes_state_common.py), the creation receipt of a local chat
(hermes_state_local.py ``commit_local_session``) and the row shapes of a branch
(hermes_state_mutation_branch.py) and a subagent (tools/delegate_tool.py). The text is
made up. Nothing here reaches a server: the client is a fake that dedups by span id, as
the ingest route does.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.helpers.client_core import PLUGIN, use_core

use_core()
from synapse_client import messages  # noqa: E402
from synapse_client.harnesses.hermes import history  # noqa: E402
from synapse_client.harnesses.hermes import messages as hm  # noqa: E402
from synapse_client.harnesses.hermes import sessions as hs  # noqa: E402
from synapse_client.settings import Settings  # noqa: E402
from synapse_client.turn_upload import TurnUploader  # noqa: E402
from synapse_client.upload import Uploader  # noqa: E402

_SCHEMA = """
CREATE TABLE schema_version (version INTEGER NOT NULL);
CREATE TABLE sessions (
    id TEXT PRIMARY KEY, source TEXT NOT NULL, created_source TEXT, user_id TEXT,
    session_key TEXT, chat_id TEXT, chat_type TEXT, model TEXT, model_config TEXT,
    parent_session_id TEXT, started_at REAL NOT NULL, cwd TEXT, title TEXT);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL,
    content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT, timestamp REAL NOT NULL,
    _compressed_summary INTEGER NOT NULL DEFAULT 0, active INTEGER NOT NULL DEFAULT 1,
    compacted INTEGER NOT NULL DEFAULT 0, api_content TEXT, display_kind TEXT,
    message_uid TEXT, absorbed_message_uids TEXT);
CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE session_admissions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL,
    principal_id TEXT NOT NULL, target_session_id TEXT NOT NULL);
"""
_T0 = 1_790_000_000.0


def _owner() -> str:
    """This machine's owner as Hermes names it: ``uid:`` on POSIX, ``sid:`` on Windows."""
    subject = hs.owner_subject()
    assert subject, "no owner subject on this machine"
    return subject


def _local(name: str) -> str:
    return "local-" + hashlib.sha256(name.encode()).hexdigest()


def _route(chat: str) -> str:
    return f"agent:main:local:dm:{chat}"


class Store:
    """A Hermes home with a ``state.db`` built the way Hermes writes one."""

    def __init__(self, home: Path, version: int = 33) -> None:
        self.home = home
        home.mkdir(parents=True, exist_ok=True)
        self.path = home / "state.db"
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(_SCHEMA)
        self.conn.execute("INSERT INTO schema_version VALUES (?)", (version,))
        self.conn.commit()
        self.clock = _T0
        self.uids = 0

    def close(self) -> None:
        self.conn.close()

    def chat(
        self,
        name: str,
        *,
        source: str = "local",
        created: str = "cli",
        title: str | None = None,
        principal: str | None = None,
        request: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
        parent: str | None = None,
        cwd: str = "/home/u/work/alpha",
    ) -> str:
        """A local chat the gateway created, with its creation receipt."""
        sid = _local(name)
        self.conn.execute(
            "INSERT INTO sessions (id, source, created_source, user_id, session_key, chat_id, "
            "chat_type, model, model_config, parent_session_id, started_at, title) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                sid,
                source,
                created,
                principal or _owner(),
                _route(sid),
                sid,
                "dm",
                "model-x",
                json.dumps(model_config) if model_config else None,
                parent,
                self.tick(),
                title,
            ),
        )
        receipt = {
            "principal_id": principal or _owner(),
            "request_id": "req-" + name,
            "session_id": sid,
            "route": _route(sid),
            "policy": {
                "source": created,
                "cwd": cwd,
                "request_json": json.dumps({"cwd": cwd, **(request or {})}),
            },
        }
        self.conn.execute(
            "INSERT INTO state_meta VALUES (?,?)",
            ("gateway.local_policy.v1:" + sid, json.dumps(receipt)),
        )
        self.conn.commit()
        return sid

    def row(self, sid: str, source: str, **columns: Any) -> str:
        """A session row with no chat identity (an in-process surface, a subagent)."""
        values = {"id": sid, "source": source, "started_at": self.tick(), "model": "model-x"}
        values.update(columns)
        keys = ", ".join(values)
        self.conn.execute(
            f"INSERT INTO sessions ({keys}) VALUES ({','.join('?' * len(values))})",
            tuple(values.values()),
        )
        self.conn.commit()
        return sid

    def tick(self) -> float:
        self.clock += 7.25
        return self.clock

    def uid(self) -> str:
        self.uids += 1
        return f"{self.uids:032x}"

    def add(self, sid: str, role: str, content: str = "", **columns: Any) -> dict[str, Any]:
        """One stored row; returns the dict Hermes would hold for it in memory."""
        values: dict[str, Any] = {
            "session_id": sid,
            "role": role,
            "content": content,
            "timestamp": self.tick(),
            "message_uid": self.uid(),
        }
        values.update(columns)
        keys = ", ".join(values)
        self.conn.execute(
            f"INSERT INTO messages ({keys}) VALUES ({','.join('?' * len(values))})",
            tuple(values.values()),
        )
        self.conn.commit()
        live = {
            "role": role,
            "content": content,
            "timestamp": values["timestamp"],
            "message_uid": values["message_uid"],
        }
        if values.get("tool_calls"):
            live["tool_calls"] = json.loads(values["tool_calls"])
        for key in ("tool_name", "display_kind"):
            if values.get(key):
                live[key] = values[key]
        return live

    def turn(self, sid: str, text: str, reply: str, *, tool: bool = False) -> list[dict]:
        rows = [self.add(sid, "user", text)]
        if tool:
            call = [{"id": "c1", "function": {"name": "recall", "arguments": '{"q": "x"}'}}]
            rows.append(self.add(sid, "assistant", "", tool_calls=json.dumps(call)))
            rows.append(self.add(sid, "tool", '{"facts": []}', tool_name="recall"))
        rows.append(self.add(sid, "assistant", reply))
        return rows

    def admit(self, sid: str, principal: str) -> None:
        self.conn.execute(
            "INSERT INTO session_admissions (request_id, principal_id, target_session_id) "
            "VALUES (?,?,?)",
            ("r", principal, sid),
        )
        self.conn.commit()

    def branch(self, parent: str, name: str) -> str:
        """``/branch``: a new chat whose rows are clones of the parent's, uids kept."""
        child = self.chat(name, model_config={"_branched_from": parent}, parent=parent)
        self.conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_call_id, tool_calls, "
            "tool_name, timestamp, _compressed_summary, display_kind, message_uid, "
            "absorbed_message_uids) SELECT ?, role, content, tool_call_id, tool_calls, "
            "tool_name, timestamp, _compressed_summary, display_kind, message_uid, "
            "absorbed_message_uids FROM messages WHERE session_id=? AND active=1 ORDER BY id",
            (child, parent),
        )
        self.conn.commit()
        return child


@pytest.fixture()
def store(tmp_path: Path):
    s = Store(tmp_path / "hermes")
    yield s
    s.close()


def _by_id(home: Path) -> dict[str, history.Session]:
    return {s.id: s for s in history.read(home)}


def _keys(session: history.Session) -> list[str]:
    return [messages.turn_key(turn) for _, turn in session.pairs]


# -- eligibility ---------------------------------------------------------------------


def test_owner_local_chats_are_imported(store: Store) -> None:
    cli = store.chat("cli")
    store.turn(cli, "first question", "first answer", tool=True)
    store.turn(cli, "second question", "second answer")
    store.admit(cli, _owner())
    tui = store.chat("tui", created="tui")
    store.turn(tui, "tui question", "tui answer")
    oneshot = store.chat("oneshot", source="oneshot", created="oneshot")
    store.turn(oneshot, "one shot", "pong")
    stdio = store.row("20261114_141210_9d4c39", "tui", cwd="/home/u/work/beta")
    store.turn(stdio, "in-process", "answer")
    found = _by_id(store.home)
    for sid in (cli, tui, oneshot, stdio):
        assert found[sid].reason == "", (sid, found[sid].reason)
    assert len(found[cli].pairs) == 2
    assert found[cli].cwd == "/home/u/work/alpha"
    assert found[cli].model == "model-x"
    assert found[stdio].cwd == "/home/u/work/beta"


@pytest.mark.parametrize(
    ("kind", "expect"),
    [
        ("subagent", "subagent"),
        ("api", "api_server"),
        ("tool", "tool"),
        ("telegram", "telegram"),
        ("cron", "cron"),
        ("bot_chat", "Bot Chat"),
        ("group", "Group Chat"),
        ("follows", "follows its profile"),
        ("not_owner", "owner"),
        ("foreign_turn", "another person or agent"),
        ("side_agent", "side agent"),
        ("acp_alone", "ACP"),
        ("bot_parent", "Bot Chat"),
    ],
)
def test_sessions_the_provider_refuses_are_skipped(store: Store, kind: str, expect: str) -> None:
    if kind == "subagent":
        delegator = store.chat("delegator")
        sid = store.row(
            "20261114_141210_aaaaaa",
            "subagent",
            parent_session_id=delegator,
            model_config=json.dumps({"_delegate_from": delegator}),
        )
    elif kind == "api":
        sid = store.row(
            "api-f0bacbd2005e6787",
            "api_server",
            user_id="api",
            chat_id="api-f0bacbd2005e6787",
            chat_type="dm",
            session_key="agent:main:api_server:dm:api-f0bacbd2005e6787",
        )
    elif kind == "tool":
        sid = store.chat("tool", source="tool", created="tool")
    elif kind == "telegram":
        sid = store.row(
            "tg-1", "telegram", user_id="42", chat_id="42", chat_type="dm", session_key="k"
        )
    elif kind == "cron":
        sid = store.row("cron_job_1", "cron")
    elif kind == "bot_chat":
        sid = store.chat("bot", title="Bot Chat")
    elif kind == "group":
        sid = store.chat("group", title="Group: lab · main")
    elif kind == "follows":
        sid = store.chat("follows", request={"follow_profile_config": True})
    elif kind == "not_owner":
        sid = store.chat("other", principal="uid:4242")
    elif kind == "foreign_turn":
        sid = store.chat("relay")
        store.admit(sid, _owner())
        store.admit(sid, "bot:peer-agent")
    elif kind == "side_agent":
        sid = store.row("bg_1a2b3c", "cli")
    elif kind == "acp_alone":
        sid = store.row("20261114_000000_acp001", "acp")
    else:
        bot = store.chat("bot-parent", title="Bot Chat")
        sid = store.branch(bot, "bot-branch")
    store.turn(sid, "hello", "hi")
    found = _by_id(store.home)[sid]
    assert found.reason, kind
    assert expect.lower() in found.reason.lower(), found.reason
    assert found.pairs == ()


def test_the_session_s_own_automation_is_the_owner_s(store: Store) -> None:
    """A background task reporting back is admitted as the session's own automation."""
    sid = store.chat("delegating")
    store.admit(sid, _owner())
    store.admit(sid, "automation:" + _route(sid))
    store.turn(sid, "q", "a")
    assert _by_id(store.home)[sid].reason == ""


def test_a_marked_chat_is_skipped_and_the_marks_are_only_read(store: Store) -> None:
    sid = store.chat("marked")
    store.turn(sid, "q", "a")
    clean = store.chat("clean")
    store.turn(clean, "q", "a")
    folder = store.home / "synapse" / "inert"
    folder.mkdir(parents=True)
    (folder / hashlib.sha256(sid.encode()).hexdigest()).write_text(f"{sid}\nbot\n")
    before = sorted(p.name for p in folder.iterdir())
    found = _by_id(store.home)
    assert "written into" in found[sid].reason
    assert found[clean].reason == ""
    assert sorted(p.name for p in folder.iterdir()) == before


def test_nothing_under_the_hermes_home_is_written(store: Store) -> None:
    bot = store.chat("bot", title="Bot Chat")  # the provider would mark this one
    store.turn(bot, "q", "a")
    sid = store.chat("plain")
    store.turn(sid, "q", "a")
    store.close()

    def snapshot() -> dict[str, tuple[int, str]]:
        out = {}
        for path in sorted(store.home.rglob("*")):
            if path.is_file():
                data = path.read_bytes()
                out[str(path)] = (path.stat().st_mtime_ns, hashlib.sha256(data).hexdigest())
        return out

    before = snapshot()
    history.read(store.home)
    assert snapshot() == before
    assert not (store.home / "synapse").exists()


# -- private mode --------------------------------------------------------------------


def test_private_sessions_and_their_continuations_are_skipped(store: Store) -> None:
    secret = store.chat("secret")
    store.turn(secret, "my codename", "ok")
    branch = store.branch(secret, "secret-branch")
    store.turn(branch, "later", "answer")
    plain = store.chat("plain")
    store.turn(plain, "q", "a")
    found = _by_id(store.home)
    assert found[branch].continues == (secret,)

    def is_private(sid: str) -> bool | None:
        return sid == secret

    assert history.private_reason(found[secret], is_private) == "private"
    assert "continues a private" in history.private_reason(found[branch], is_private)
    assert history.private_reason(found[plain], is_private) == ""
    unsure = history.private_reason(found[plain], lambda _sid: None)
    assert "could not be checked" in unsure


# -- copies and ids ------------------------------------------------------------------


def test_a_branch_imports_only_its_own_turns(store: Store) -> None:
    parent = store.chat("parent")
    store.turn(parent, "p1", "a1")
    store.turn(parent, "p2", "a2", tool=True)
    child = store.branch(parent, "child")
    store.turn(child, "c1", "b1")
    found = _by_id(store.home)
    assert len(found[parent].pairs) == 2
    assert found[child].copies == 2
    assert len(found[child].pairs) == 1
    context, turn = found[child].pairs[0]
    assert context is None  # the turn before it is the parent's
    assert turn[0]["content"] == "c1"
    every = _keys(found[parent]) + _keys(found[child])
    assert len(every) == len(set(every))


def test_compaction_clones_and_rewound_rows(store: Store) -> None:
    sid = store.chat("compacted")
    store.turn(sid, "old question", "old answer")
    kept = store.turn(sid, "kept question", "kept answer")
    # In-place compaction: everything archived, a summary row, the tail cloned.
    store.conn.execute("UPDATE messages SET active=0, compacted=1 WHERE session_id=?", (sid,))
    store.add(sid, "user", "summary of earlier turns", _compressed_summary=1)
    for row in kept:
        store.add(sid, row["role"], row["content"], message_uid=row["message_uid"])
    store.turn(sid, "new question", "new answer")
    gone = store.turn(sid, "rewound question", "rewound answer")
    store.conn.execute(
        "UPDATE messages SET active=0 WHERE message_uid IN (?, ?)",
        (gone[0]["message_uid"], gone[1]["message_uid"]),
    )
    store.conn.commit()
    pairs = _by_id(store.home)[sid].pairs
    assert [turn[0]["content"] for _, turn in pairs] == [
        "old question",
        "kept question",
        "new question",
    ]
    assert "summary" not in json.dumps([turn for _, turn in pairs])


class FakeServer:
    """The ingest route's dedup: one episode per span id, ``<client>:<id>`` of the
    turn's last record that carries one."""

    def __init__(self, settings: Settings, fail_after: int | None = None) -> None:
        self.settings = settings
        self.spans: dict[str, float | None] = {}
        self.bodies: list[dict[str, Any]] = []
        self.fail_after = fail_after
        self.private: set[str] = set()

    def ingest(self, payload: Any, timeout: float = 30, **_: Any) -> dict[str, Any]:
        from synapse_client import private

        private.check_payload(self.settings, payload)
        if self.fail_after is not None and len(self.bodies) >= self.fail_after:
            raise KeyboardInterrupt
        self.bodies.append(dict(payload))
        new = 0
        for turn in messages.turns(payload["records"]):
            if not messages.complete(turn):
                continue
            span = "hermes:" + messages.turn_key(turn)
            if span not in self.spans and payload["session_id"] not in self.private:
                self.spans[span] = turn[0].get("ts")
                new += 1
        # The context turn leads a batch and is counted as a duplicate, as on the server.
        return {"ingested": new}

    def private_get(self, session_id: str, timeout: float = 15) -> dict[str, Any]:
        return {"private": session_id in self.private}


def _settings(tmp_path: Path) -> Settings:
    return Settings.load(
        env={"SYNAPSE_URL": "http://synapse.test", "SYNAPSE_PRIVATE_DIR": str(tmp_path / "pd")},
        home=tmp_path / "synapse-home",
    )


def test_ids_match_the_live_provider(store: Store, tmp_path: Path) -> None:
    """The same conversation, uploaded live by the provider's TurnUploader and imported
    from the store, gives the same records and the same span ids."""
    sid = store.chat("live")
    live: list[dict[str, Any]] = []
    live += store.turn(sid, "first", "one", tool=True)
    live += store.turn(sid, "delegate this", "dispatched, I will report back")
    # A background delegate_task reports back: Hermes appends a row in the user's place.
    live.append(
        store.add(
            sid,
            "user",
            "[ASYNC DELEGATION BATCH COMPLETE — deleg_1]\nresult",
            display_kind="async_delegation_complete",
        )
    )
    live.append(store.add(sid, "assistant", "the subagent says 391"))
    live += store.turn(sid, "third", "three")

    settings = _settings(tmp_path)
    server = FakeServer(settings)
    uploader = TurnUploader(
        Uploader(server, None),  # type: ignore[arg-type]
        client="hermes",
        source="hermes-sync",
        lookback=10,
    )
    uploader.upload(hm.to_records(live), sid, timeout=5, finished=True)
    live_spans = dict(server.spans)

    session = _by_id(store.home)[sid]
    assert history.records(store.conn, sid) == hm.to_records(live)
    imported = ["hermes:" + key for key in _keys(session)]
    assert imported == list(live_spans)
    # The reply to the delegation's report is a turn of its own: the turn that
    # dispatched it is not sent again.
    assert len(imported) == 4
    # Each turn keeps its own time: the first record's ts is the stored timestamp.
    assert [turn[0]["ts"] for _, turn in session.pairs] == list(live_spans.values())
    imported_records = [p["records"] for p in history.payloads(session)]
    assert imported_records == [b["records"] for b in server.bodies]
    body = history.payloads(session)[0]
    assert body["client"] == "hermes" and body["format"] == "messages"
    assert body["cwd"] == "/home/u/work/alpha" and body["model"] == "model-x"


# -- schema --------------------------------------------------------------------------


def test_an_unknown_schema_version_stops_before_reading(tmp_path: Path) -> None:
    s = Store(tmp_path / "newer", version=34)
    s.close()
    with pytest.raises(history.SchemaError, match="schema version 34"):
        history.read(s.home)


def test_a_missing_column_stops_before_reading(tmp_path: Path) -> None:
    home = tmp_path / "odd"
    home.mkdir()
    conn = sqlite3.connect(home / "state.db")
    conn.executescript(
        _SCHEMA.replace("message_uid TEXT, ", "") + "INSERT INTO schema_version VALUES (33);"
    )
    conn.close()
    with pytest.raises(history.SchemaError, match=r"messages\.message_uid"):
        history.read(home)


def test_not_a_hermes_store(tmp_path: Path) -> None:
    home = tmp_path / "other"
    home.mkdir()
    sqlite3.connect(home / "state.db").close()
    with pytest.raises(history.SchemaError):
        history.read(home)


# -- the command ---------------------------------------------------------------------


def _script() -> ModuleType:
    path = PLUGIN / "scripts" / "import_hermes.py"
    spec = importlib.util.spec_from_file_location("import_hermes_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def command(store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _script()
    settings = _settings(tmp_path)
    server = FakeServer(settings)
    monkeypatch.setattr(module.Settings, "load", classmethod(lambda cls: settings))
    monkeypatch.setattr(module, "Client", lambda _settings, _agent: server)
    return module, server


def test_the_command_imports_and_a_rerun_adds_nothing(store, command, capsys) -> None:
    module, server = command
    a = store.chat("a")
    store.turn(a, "q1", "a1")
    store.turn(a, "q2", "a2")
    b = store.chat("b")
    store.turn(b, "q3", "a3")
    sub = store.row("20261114_141210_bbbbbb", "subagent")
    store.turn(sub, "sub q", "sub a")
    secret = store.chat("secret")
    store.turn(secret, "secret q", "secret a")
    server.private.add(secret)
    home = str(store.home)
    assert module.main(["--hermes-home", home, "--yes"]) == 0
    out = capsys.readouterr().out
    assert "2 session(s), 3 finished turn(s)" in out
    assert "Done: 3 new turn(s)" in out
    assert {body["session_id"] for body in server.bodies} == {a, b}
    # The rerun finds every turn in the ledger the provider shares: nothing is sent.
    posted = len(server.bodies)
    assert module.main(["--hermes-home", home, "--yes"]) == 0
    assert "Nothing to import." in capsys.readouterr().out
    assert len(server.bodies) == posted


def test_a_private_marker_keeps_a_session_out(store, command, tmp_path, capsys) -> None:
    module, server = command
    sid = store.chat("marked-private")
    store.turn(sid, "secret", "ok")
    (tmp_path / "pd").mkdir()
    (tmp_path / "pd" / sid).write_text("")
    assert module.main(["--hermes-home", str(store.home), "--yes"]) == 0
    assert server.bodies == []
    assert "private" in capsys.readouterr().out


def test_dry_run_and_declining_send_nothing(store, command, monkeypatch, capsys) -> None:
    module, server = command
    sid = store.chat("a")
    store.turn(sid, "q", "a")
    assert module.main(["--hermes-home", str(store.home), "--dry-run"]) == 0
    assert "1 turn(s)" in capsys.readouterr().out
    answers = iter(["", "", "n"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert module.main(["--hermes-home", str(store.home)]) == 1
    assert "Nothing was sent" in capsys.readouterr().out
    assert server.bodies == []


def test_an_interrupted_import_resumes(store, command, capsys) -> None:
    module, server = command
    sids = [store.chat(f"s{i}") for i in range(3)]
    for sid in sids:
        store.turn(sid, f"q {sid[-4:]}", "a")
    server.fail_after = 1
    assert module.main(["--hermes-home", str(store.home), "--yes"]) == 130
    assert "re-run to resume" in capsys.readouterr().out
    assert len(server.spans) == 1
    server.fail_after = None
    assert module.main(["--hermes-home", str(store.home), "--yes"]) == 0
    assert "Done: 2 new turn(s)" in capsys.readouterr().out
    assert len(server.spans) == 3


def test_the_command_refuses_an_unknown_schema(tmp_path, command, capsys) -> None:
    module, server = command
    s = Store(tmp_path / "newer", version=99)
    s.close()
    assert module.main(["--hermes-home", str(s.home), "--yes"]) == 1
    assert "schema version 99" in capsys.readouterr().err
    assert server.bodies == []


def test_import_hermes_is_a_subcommand_of_import(monkeypatch, tmp_path) -> None:
    from tests.test_import_history import _isolated_plugin_env

    _isolated_plugin_env(monkeypatch, tmp_path)
    scripts = str(PLUGIN / "scripts")
    monkeypatch.syspath_prepend(scripts)
    called: list[list[str]] = []
    fake = ModuleType("import_hermes")
    fake.main = lambda argv: called.append(argv) or 0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "import_hermes", fake)
    spec = importlib.util.spec_from_file_location(
        "import_history_under_test", PLUGIN / "scripts" / "import_history.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["hermes", "--hermes-home", "/x", "--dry-run"]) == 0
    assert called == [["--hermes-home", "/x", "--dry-run"]]
