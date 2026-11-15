"""Random Hermes conversations, uploaded as the provider uploads them: every user message
and every final reply is stored exactly once, and nothing Hermes masked leaves.

Each seed plays a conversation twice over: as the list Hermes hands a memory provider
(scrubbed of secrets, merged where Hermes merges, in the order Hermes ran it) and as
the rows it writes to its session store (tests/helpers/hermes_store.py: raw text, a
marker and the next message as two rows, a queued prompt's accept-time row, archived
rows). It has turns with and without tools, reports after finished turns (some
answered alike), nudges inside a run, ``/model`` markers and notes Hermes folds into
the next message, pictures, failed runs, ``/steer``, in-place compactions, a rotation
to a new session id inside a run, an agent rebuilt mid-session (its list loses every
``display_kind``), runs no upload ran for, flushes inside a run, uploads that fail and
wait in the outbox, a prompt queued during a run, rows Hermes could not write (no uid,
not in the store), a window where only the server's row keeps the session private
followed by ``off --forget`` and a restart, and secrets in the user's text. The events
review round 4 added (followups/hermes-turn-cut/review-round4/fuzz4.py): rows written
by a later flush, a store gone for a while, ``off --forget`` with no restart, a marker
and a note folded with the message, two prompts queued in one run, a compaction that
leaves a report first.

The upload is the provider's: the list converted with the store's kinds (source.py,
messages.py), the ledger (shipped.py), TurnUploader, Uploader and Outbox; the server is
the ingest route's own parser with its span and content dedup.

``SYNAPSE_FUZZ_SEEDS`` sets how many seeds run (default 40).
"""

from __future__ import annotations

import hashlib
import os
import random
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from ingestion.messages_client import MessagesParser
from tests.helpers.client_core import use_core
from tests.helpers.hermes_store import HermesStore, call

use_core()
from synapse_client import private  # noqa: E402
from synapse_client.harnesses.hermes import messages as hm  # noqa: E402
from synapse_client.harnesses.hermes import source  # noqa: E402
from synapse_client.outbox import Outbox  # noqa: E402
from synapse_client.settings import Settings  # noqa: E402
from synapse_client.shipped import Ledger  # noqa: E402
from synapse_client.turn_upload import TurnUploader  # noqa: E402
from synapse_client.upload import Uploader  # noqa: E402

SEEDS = int(os.environ.get("SYNAPSE_FUZZ_SEEDS", "40"))
MODEL = "[System: The active model for this chat has changed to gpt-x via provider y.]"
NUDGE = "[System: Continue now. Execute the required tool calls.]"
FAILED = (
    "This turn did not complete. Some actions may already have run; verify their effects "
    "before resending."
)
EVENTS = [
    "turn", "tools", "report", "alike", "nudged", "marker", "note", "picture", "failed",
    "steer", "compact", "rebuild", "skip", "down", "flush", "queued", "rotate", "unwritten",
    "private", "secret", "written_later", "store_gone", "forget_live", "fold2", "queued2",
    "report_first",
]  # fmt: skip


class Server:
    """The ingest route: the server's parser, one episode per span id and none whose
    content is already stored. ``down`` fails every request; a session in ``rows`` is
    dropped with a 200, as the server's private_sessions row drops it."""

    def __init__(self, tmp: Path) -> None:
        self.settings = Settings.load(
            env={"SYNAPSE_URL": "http://synapse.test", "SYNAPSE_PRIVATE_DIR": str(tmp / "pd")},
            home=tmp / "synapse-home",
        )
        self.down = False
        self.rows: set[str] = set()
        self.episodes: list[Any] = []
        self.spans: set[str] = set()
        self.contents: set[str] = set()
        self.wire: list[str] = []

    def ingest(self, payload: Any, timeout: float = 30, **_: Any) -> dict[str, Any]:
        private.check_payload(self.settings, payload)
        if self.down:
            raise urllib.error.URLError("down")
        self.wire.append(repr(payload["records"]))
        if payload["session_id"] in self.rows:
            return {"ingested": 0}
        for ep in MessagesParser().parse_records(
            payload["records"], "t", "proj", session_id=payload["session_id"], client="hermes"
        ):
            digest = hashlib.md5(ep.content.encode("utf-8"), usedforsecurity=False).hexdigest()
            if ep.span_id in self.spans or digest in self.contents:
                continue
            self.spans.add(ep.span_id)
            self.contents.add(digest)
            self.episodes.append(ep)
        return {"ingested": 1}

    def private_get(self, session_id: str, timeout: float = 15) -> dict[str, Any]:
        return {"private": session_id in self.rows}

    def private_put(self, session_id: str, timeout: float = 15) -> dict[str, Any]:
        self.rows.add(session_id)
        return {}


class Run:
    """One seed: the store, the list, the server, and the agent that uploads."""

    def __init__(self, seed: int, tmp: Path) -> None:
        self.rnd = random.Random(seed)
        self.seed = seed
        self.home = tmp / "hermes"
        self.store = HermesStore(self.home)
        self.sid = "sess-a"
        self.store.session(self.sid)
        self.server = Server(tmp)
        self.ledger = Ledger(self.server.settings.home / "shipped" / "hermes")
        self.outbox = Outbox(tmp / "outbox")
        self.live: list[dict[str, Any]] = []
        self.agent = self.new_agent()
        self.n = 0
        self.must: list[str] = []  # stored exactly once
        self.at_most: list[str] = []  # stored once or never (a failed run's ask)
        self.never: list[str] = []  # never on the wire
        self.alike = ""
        self.alike_uses = 0
        self.scrubbed: dict[str, str] = {}  # uid -> the text Hermes hands over for it

    def new_agent(self) -> TurnUploader:
        uploader = Uploader(self.server, self.outbox)  # type: ignore[arg-type]
        return TurnUploader(
            uploader, client="hermes", source="hermes-sync", lookback=10, ledger=self.ledger
        )

    def tok(self, kind: str) -> str:
        self.n += 1
        return f"{kind}-{self.seed}-{self.n}-text"

    def both(self, role: str, content: Any = "", **kw: Any) -> dict[str, Any]:
        """A row Hermes writes and keeps in its list."""
        row = self.store.add(self.sid, role, content, **kw)
        self.live.append(row)
        return row

    def sync(self, finished: bool = True) -> None:
        """A hook: the provider converts the list it was handed, with the store's kinds."""
        listed = [dict(m) for m in self.live]
        got = source.read(self.home, self.sid, listed)
        assert not got.mismatched, got.mismatched
        records = hm.to_records(listed, got.kinds)
        self.agent.upload(records, self.sid, timeout=5, finished=finished)
        if not self.server.down:
            self.agent.uploader.drain(5)

    # -- what can happen between two hooks -------------------------------------------

    def turn(self, tools: bool = False, user: str | None = None) -> None:
        user = user or self.tok("USER")
        final = self.tok("FINAL")
        self.must += [user, final]
        self.both("user", user)
        if tools:
            one = call()
            self.both("assistant", "", tool_calls=[one])
            self.both(
                "tool", self.tok("RESULT") + " output", tool_name="terminal", tool_call_id=one["id"]
            )
        self.both("assistant", final)

    def report(self, reply: str | None = None) -> None:
        text = self.tok("REPORT")
        self.never.append(text)
        self.both(
            "user", f"[ASYNC DELEGATION COMPLETE]\n{text}", display_kind="async_delegation_complete"
        )
        if reply is None:
            reply = self.tok("FINAL")
            self.must.append(reply)
        self.both("assistant", reply)

    def nudged(self) -> None:
        user, final = self.tok("USER"), self.tok("FINAL")
        self.must += [user, final]
        self.both("user", user)
        self.both("assistant", "INTERIM " + self.tok("INTERIM"))
        self.both("user", NUDGE)
        self.both("assistant", final)

    def folded(self, text: str, kind: str) -> None:
        """A row nobody answered, then a message: two rows in the store; one in the list,
        which Hermes folds (agent/agent_runtime_helpers.py ``_merge_consecutive_users``)."""
        user, final = self.tok("USER"), self.tok("FINAL")
        self.must += [user, final]
        marker = self.store.add(self.sid, "user", text, display_kind=kind)
        message = self.store.add(self.sid, "user", user)
        merged = {k: v for k, v in marker.items() if k != "display_kind"}
        merged["content"] = text + "\n\n" + user
        merged["_merged_turn_prefix"] = text
        merged["_absorbed_message_uids"] = [message["message_uid"]]
        self.live.append(merged)
        self.both("assistant", final)

    def marker(self) -> None:
        self.folded(MODEL, "model_switch")

    def note(self) -> None:
        text = self.tok("NOTE")
        self.never.append(text)
        self.folded(f"[IMPORTANT: Background process exited.]\n{text}", "process_complete")

    def picture(self) -> None:
        final = self.tok("FINAL")
        self.must.append(final)
        self.both("user", "[screenshot]", live=[{"type": "image_url", "image_url": {"url": "x"}}])
        self.both("assistant", final)

    def failed(self) -> None:
        user = self.tok("USER")
        self.at_most.append(user)
        self.both("user", user)
        self.both("assistant", FAILED, display_kind="failed_turn")

    def steer(self) -> None:
        user, final = self.tok("USER"), self.tok("FINAL")
        self.must += [user, final]
        self.both("user", user)
        self.both("assistant", "", tool_calls=[call()])
        self.both("tool", self.tok("RESULT") + " output", tool_name="terminal")
        self.both("user", "[OUT-OF-BAND USER MESSAGE]\nsteer", display_kind="steer")
        self.both("assistant", final)

    def compact(self) -> None:
        """Hermes flushes what it is about to compress (``on_pre_compress``), then
        compacts in place; the list it hands over after holds the copies."""
        self.sync(finished=False)
        self.live = [
            self.handed(m) for m in self.store.compact(self.sid, keep=self.rnd.randint(0, 6))
        ]

    def handed(self, row: dict[str, Any]) -> dict[str, Any]:
        uid = row.get("message_uid")
        return {**row, "content": self.scrubbed[uid]} if uid in self.scrubbed else row

    def rebuild(self) -> None:
        """Hermes rebuilt the agent (a model switch): its list lost every display_kind."""
        self.agent = self.new_agent()
        self.live = [{k: v for k, v in m.items() if k != "display_kind"} for m in self.live]

    def flush(self) -> None:
        """A flush inside a run, after an interim reply, then the rest of the run."""
        user, final = self.tok("USER"), self.tok("FINAL")
        self.must += [user, final]
        self.both("user", user)
        self.both("assistant", "INTERIM " + self.tok("INTERIM"))
        self.both("user", NUDGE)
        self.sync(finished=False)
        self.both("assistant", final)

    def queued(self) -> None:
        """A prompt queued while a run goes: written at accept time in the store, not in
        the run's list; written again when its own run starts (probe 20261115_050146)."""
        a_user, a_final, b_user, b_final = (self.tok(k) for k in ("USER", "FINAL", "USER", "FINAL"))
        self.must += [a_user, a_final, b_user, b_final]
        self.both("user", a_user)
        early = self.store.add(self.sid, "user", b_user, display_metadata={"_queued_prompt": True})
        one = call()
        self.both("assistant", "", tool_calls=[one])
        self.both(
            "tool", self.tok("RESULT") + " output", tool_name="terminal", tool_call_id=one["id"]
        )
        self.both("assistant", a_final)
        self.sync()
        self.store.conn.execute(
            "UPDATE messages SET active=0 WHERE message_uid=?", (early["message_uid"],)
        )
        self.store.conn.commit()
        self.both("user", b_user)
        self.both("assistant", b_final)

    def rotate(self) -> None:
        """A compaction that moves to a new session id inside a run: the run's rows are
        cloned into the child with their uids, the reply is written there."""
        user, final = self.tok("USER"), self.tok("FINAL")
        self.must += [user, final]
        one = call()
        run = [
            self.both("user", user),
            self.both("assistant", "", tool_calls=[one]),
            self.both(
                "tool", self.tok("RESULT") + " output", tool_name="terminal", tool_call_id=one["id"]
            ),
        ]
        self.sync(finished=False)  # on_pre_compress, before the compression moves on
        child = f"sess-{self.tok('child')}"
        self.store.session(child, parent=self.sid, model_config={"_compressed_from": self.sid})
        self.sid = child
        self.live = [self.store.add(child, "user", "[CONTEXT COMPACTION] summary", summary=True)]
        for r in run:
            self.both(
                r["role"],
                r["content"],
                uid=r["message_uid"],
                tool_calls=r.get("tool_calls"),
                tool_name=r.get("tool_name"),
                tool_call_id=r.get("tool_call_id"),
            )
        # (the run's own rows hold no secret: ``secret`` writes a whole run of its own)
        self.agent.start(child, 10)
        self.both("assistant", final)

    def unwritten(self) -> None:
        """Hermes could not write the run: its rows are in the list with no uid."""
        user, final = self.tok("USER"), self.tok("FINAL")
        self.must += [user, final]
        self.live += [{"role": "user", "content": user}, {"role": "assistant", "content": final}]

    def private(self) -> None:
        """``synapse private on``, a turn, the marker goes while the server's row stays
        (plain ``off``, or its twelve hours), a turn, then ``off --forget`` and a
        restart: both turns are stored once in the end."""
        marker = Path(self.server.settings.private_dir) / self.sid
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("", encoding="utf-8")
        self.server.rows.add(self.sid)
        self.turn()
        self.sync()
        marker.unlink()
        self.turn()
        self.sync()
        self.server.rows.discard(self.sid)
        self.ledger.forget(self.sid)  # what `synapse private off --forget` does here
        self.agent = self.new_agent()  # a restart

    def secret(self) -> None:
        """The user pasted a key: the store holds it, the list Hermes hands over does not."""
        key = "SECRET" + self.tok("KEY")
        self.never.append(key)
        user, final = self.tok("USER"), self.tok("FINAL")
        self.must += [user, final]
        row = self.store.add(self.sid, "user", f"{user} my key is {key}")
        self.scrubbed[row["message_uid"]] = f"{user} my key is [redacted]"
        self.live.append(self.handed(row))
        self.both("assistant", final)

    # -- review round 4 (followups/hermes-turn-cut/review-round4/fuzz4.py) -----------

    def written_later(self) -> None:
        """Rows Hermes could not write at one hook get their uids by the next."""
        user, final = self.tok("USER"), self.tok("FINAL")
        self.must += [user, final]
        rows = [
            {"role": "user", "content": user, "timestamp": 1e9 + self.n},
            {"role": "assistant", "content": final, "timestamp": 1e9 + self.n + 1},
        ]
        self.live += rows
        self.sync()
        for row in rows:
            stored = self.store.add(self.sid, row["role"], row["content"])
            row["message_uid"], row["timestamp"] = stored["message_uid"], stored["timestamp"]
        self.turn()

    def store_gone(self) -> None:
        """The store cannot be read for a while, and the agent is rebuilt meanwhile."""

        def version(n: int) -> None:
            self.store.conn.execute("UPDATE schema_version SET version=?", (n,))
            self.store.conn.commit()

        self.report()
        self.sync()
        version(99)
        self.rebuild()
        self.report()
        self.sync()
        self.turn()
        self.sync()
        version(33)
        self.turn()

    def forget_live(self) -> None:
        """``off --forget`` while the same provider goes on (no restart)."""
        marker = Path(self.server.settings.private_dir) / self.sid
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("", encoding="utf-8")
        self.server.rows.add(self.sid)
        self.turn()
        self.sync()
        marker.unlink()
        self.turn()
        self.sync()
        self.server.rows.discard(self.sid)
        self.ledger.forget(self.sid)
        self.turn()

    def fold2(self) -> None:
        """A marker, a note, then the message, folded into one list row."""
        user, final, note = self.tok("USER"), self.tok("FINAL"), self.tok("NOTE")
        self.must += [user, final]
        self.never.append(note)
        text = f"[IMPORTANT: Background process exited.]\n{note}"
        marker = self.store.add(self.sid, "user", MODEL, display_kind="model_switch")
        noted = self.store.add(self.sid, "user", text, display_kind="process_complete")
        message = self.store.add(self.sid, "user", user)
        merged = {k: v for k, v in marker.items() if k != "display_kind"}
        merged["content"] = MODEL + "\n\n" + text + "\n\n" + user
        merged["_merged_turn_prefix"] = MODEL + "\n\n" + text
        merged["_absorbed_message_uids"] = [noted["message_uid"], message["message_uid"]]
        self.live.append(merged)
        self.both("assistant", final)

    def queued2(self) -> None:
        """Two prompts queued during one run."""
        a, fa, b, fb, c, fc = (
            self.tok(k) for k in ("USER", "FINAL", "USER", "FINAL", "USER", "FINAL")
        )
        self.must += [a, fa, b, fb, c, fc]
        self.both("user", a)
        eb = self.store.add(self.sid, "user", b, display_metadata={"_queued_prompt": True})
        ec = self.store.add(self.sid, "user", c, display_metadata={"_queued_prompt": True})
        self.both("assistant", fa)
        self.sync()
        self.store.conn.execute(
            "UPDATE messages SET active=0 WHERE message_uid IN (?,?)",
            (eb["message_uid"], ec["message_uid"]),
        )
        self.store.conn.commit()
        self.both("user", b)
        ec2 = self.store.add(self.sid, "user", c, display_metadata={"_queued_prompt": True})
        self.both("assistant", fb)
        self.sync()
        self.store.conn.execute(
            "UPDATE messages SET active=0 WHERE message_uid=?", (ec2["message_uid"],)
        )
        self.store.conn.commit()
        self.both("user", c)
        self.both("assistant", fc)

    def report_first(self) -> None:
        """A compaction leaves a list that starts with a report."""
        self.report()
        self.sync()
        self.sync(finished=False)
        self.live = [self.handed(m) for m in self.store.compact(self.sid, keep=2)]
        self.turn()

    def play(self) -> None:
        self.turn()
        self.sync()  # the session's first upload: nothing before it to look back past
        self.alike = self.tok("ALIKE")
        for _ in range(self.rnd.randint(6, 26)):
            event = self.rnd.choice(EVENTS)
            if event == "compact":
                self.compact()
                continue
            if event == "rebuild":
                self.rebuild()
                continue
            if event == "skip":  # the run ends while no upload runs
                self.turn()
                continue
            if event == "private":
                self.private()
            elif event == "down":
                self.server.down = True
                self.turn()
                self.sync()
                self.server.down = False
                continue
            elif event == "alike":
                self.report(self.alike)
                self.alike_uses += 1
            elif event in ("turn", "tools"):
                self.turn(tools=event == "tools")
            else:
                getattr(self, event)()
            self.sync()
        self.turn()
        self.sync()

    def problems(self) -> list[str]:
        def times(token: str) -> int:
            n = 0
            for ep in self.server.episodes:
                lines = [ln for ln in ep.content.split("\n") if not ln.startswith("[context]")]
                n += any(token in ln for ln in lines)
            return n

        wire = "\n".join(self.server.wire)
        out = [f"{t} stored {times(t)} times" for t in self.must if times(t) != 1]
        if times(self.alike) != self.alike_uses:
            out.append(f"{self.alike}: {times(self.alike)} of {self.alike_uses} stored")
        out += [f"{t} stored {times(t)} times" for t in self.at_most if times(t) > 1]
        out += [f"{t} on the wire" for t in self.never if t in wire]
        return out


@pytest.mark.parametrize("seed", range(SEEDS))
def test_every_message_and_reply_is_stored_once(seed: int, tmp_path: Path) -> None:
    run = Run(seed, tmp_path)
    try:
        run.play()
    finally:
        run.store.close()
    assert run.problems() == []
