"""A Hermes session store built with Hermes's own tables and row shapes, for tests.

The tables come from ``tests/fixtures/hermes_state_v33.sql``, dumped from a ``state.db``
Hermes v0.23.1 wrote. The row shapes are the ones that store held after real sessions
(a probe provider logged the store at every hook): a user row with
``display_metadata``; an assistant row that calls a tool, with empty text and the calls
as JSON; a tool row with ``tool_call_id``; a background report's user row with
``display_kind`` ``async_delegation_complete`` and a reply after it; a ``/model``
marker as a row of its own (``model_switch``) and the next message as another; a
``/skill`` row holding the whole scaffold; an in-place compaction that archives every
row (``active=0, compacted=1``) and inserts copies with new row ids and the same
``message_uid``, its summary taking the uid of a row it replaces. Each ``add`` returns
the dict Hermes holds for the row in its live list, so a test can hand the provider
the list Hermes would hand it. The text is made up.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

DDL = Path(__file__).resolve().parent.parent / "fixtures" / "hermes_state_v33.sql"
T0 = 1_794_724_800.0


def call(name: str = "terminal", arguments: str = '{"command": "ls"}', cid: str = "") -> dict:
    """A tool call as Hermes stores it in ``tool_calls``."""
    cid = cid or "call-" + uuid.uuid4().hex[:12]
    return {
        "id": cid,
        "call_id": cid,
        "response_item_id": "fc_" + cid,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class HermesStore:
    """``<home>/state.db`` as Hermes writes it. Not safe across threads."""

    def __init__(self, home: Path, version: int = 33) -> None:
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.path = self.home / "state.db"
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(DDL.read_text(encoding="utf-8"))
        self.conn.execute("INSERT INTO schema_version VALUES (?)", (version,))
        self.conn.commit()
        self.clock = T0
        self.count = 0

    def close(self) -> None:
        self.conn.close()

    def tick(self) -> float:
        self.clock += 1.0
        return self.clock

    def uid(self) -> str:
        self.count += 1
        return f"{self.count:08x}" + uuid.uuid4().hex[:24]

    def session(
        self,
        sid: str,
        *,
        source: str = "tui",
        parent: str | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> str:
        """A session row as the stdio TUI or Desktop writes it (no chat identity)."""
        self.conn.execute(
            "INSERT INTO sessions (id, source, parent_session_id, model_config, started_at) "
            "VALUES (?,?,?,?,?)",
            (sid, source, parent, json.dumps(model_config) if model_config else None, self.tick()),
        )
        self.conn.commit()
        return sid

    def add(
        self,
        sid: str,
        role: str,
        content: Any = "",
        *,
        display_kind: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        summary: bool = False,
        display_metadata: dict[str, Any] | None = None,
        uid: str | None = None,
        active: int = 1,
        compacted: int = 0,
        live: Any = None,
    ) -> dict[str, Any]:
        """One row, committed, and the live dict Hermes holds for it (``live``: its
        content there, when Hermes stores another, as for a picture)."""
        uid = uid or self.uid()
        ts = self.tick()
        stored = content if isinstance(content, str) else json.dumps(content)
        self.conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_call_id, tool_calls, "
            "tool_name, timestamp, _compressed_summary, active, compacted, display_kind, "
            "display_metadata, message_uid) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                sid,
                role,
                stored,
                tool_call_id,
                json.dumps(tool_calls) if tool_calls else None,
                tool_name,
                ts,
                1 if summary else 0,
                active,
                compacted,
                display_kind,
                json.dumps(display_metadata) if display_metadata else None,
                uid,
            ),
        )
        self.conn.commit()
        shown = content if live is None else live
        held: dict[str, Any] = {"role": role, "content": shown, "timestamp": ts, "message_uid": uid}
        for key, value in (
            ("display_kind", display_kind),
            ("tool_calls", tool_calls),
            ("tool_name", tool_name),
            ("tool_call_id", tool_call_id),
            ("display_metadata", display_metadata),
        ):
            if value:
                held[key] = value
        if summary:
            held["_compressed_summary"] = True
        return held

    def turn(self, sid: str, user: str, reply: str, *, tools: bool = False) -> list[dict[str, Any]]:
        """A finished turn: the user's row, a tool call and its result when ``tools``,
        and the reply."""
        rows = [self.add(sid, "user", user, display_metadata={"submission_id": uuid.uuid4().hex})]
        if tools:
            one = call()
            rows.append(self.add(sid, "assistant", "", tool_calls=[one]))
            rows.append(
                self.add(
                    sid,
                    "tool",
                    json.dumps({"output": f"output for {user}", "exit_code": 0, "error": None}),
                    tool_name="terminal",
                    tool_call_id=one["id"],
                )
            )
        rows.append(self.add(sid, "assistant", reply))
        return rows

    def report(self, sid: str, text: str, reply: str) -> list[dict[str, Any]]:
        """A background task's report, a run Hermes starts itself, and its reply."""
        body = f"[ASYNC DELEGATION BATCH COMPLETE — deleg_x]\n{text}"
        return [
            self.add(sid, "user", body, display_kind="async_delegation_complete"),
            self.add(sid, "assistant", reply),
        ]

    def compact(self, sid: str, keep: int) -> list[dict[str, Any]]:
        """An in-place compaction (hermes_state_messages.py): every active row archived,
        a summary that takes the uid of the first row it replaces, and copies of the last
        ``keep`` rows with the same uids. Returns the live list Hermes holds after it."""
        rows = self.conn.execute(
            "SELECT role, content, tool_call_id, tool_calls, tool_name, display_kind, "
            "display_metadata, message_uid FROM messages WHERE session_id=? AND active=1 "
            "ORDER BY id",
            (sid,),
        ).fetchall()
        self.conn.execute(
            "UPDATE messages SET active=0, compacted=1 WHERE session_id=? AND active=1", (sid,)
        )
        self.conn.commit()
        replaced = rows[: len(rows) - keep]
        live: list[dict[str, Any]] = []
        if replaced:
            live.append(
                self.add(
                    sid,
                    "assistant",
                    "[PRIOR CONTEXT — for reference only; not a new message]\nsummary",
                    summary=True,
                    uid=replaced[-1][7],
                )
            )
        for role, content, cid, calls, name, kind, meta, uid in rows[len(rows) - keep :]:
            live.append(
                self.add(
                    sid,
                    role,
                    content,
                    tool_call_id=cid,
                    tool_calls=json.loads(calls) if calls else None,
                    tool_name=name,
                    display_kind=kind,
                    display_metadata=json.loads(meta) if meta else None,
                    uid=uid,
                )
            )
        return live

    def branch(self, parent: str, child: str) -> list[dict[str, Any]]:
        """``/branch``: a child session holding copies of the parent's rows, same uids."""
        self.session(child, parent=parent, model_config={"_branched_from": parent})
        rows = self.conn.execute(
            "SELECT role, content, tool_call_id, tool_calls, tool_name, display_kind, "
            "message_uid FROM messages WHERE session_id=? AND active=1 ORDER BY id",
            (parent,),
        ).fetchall()
        return [
            self.add(
                child,
                role,
                content,
                tool_call_id=cid,
                tool_calls=json.loads(calls) if calls else None,
                tool_name=name,
                display_kind=kind,
                uid=uid,
            )
            for role, content, cid, calls, name, kind, uid in rows
        ]

    def admit(self, sid: str, principal: str, request_id: str | None = None) -> None:
        """A turn the gateway admitted from ``principal``."""
        rid = request_id or uuid.uuid4().hex
        self.conn.execute(
            "INSERT INTO session_admissions (admission_id, request_id, principal_id, "
            "target_session_id, lineage_json, payload_json, payload_digest, intent, status) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, rid, principal, sid, "{}", "{}", "d", "queue", "terminal"),
        )
        self.conn.commit()
