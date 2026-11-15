"""Random Hermes conversations, uploaded from the session store: every user message and
every final reply is stored exactly once.

Each seed writes a conversation to a store with Hermes's own tables and row shapes
(tests/helpers/hermes_store.py): turns with and without tools, reports after finished
turns (some answered alike), nudges inside a run after an interim reply, ``/model``
markers and notes before a message, pictures, failed runs, ``/steer``, in-place
compactions, a branch, an agent rebuilt mid-session, turns that end while no upload
runs (then a new agent), and uploads that fail and wait in the outbox. The uploads go
through the real reader (source.py, store.py), ledger (shipped.py), TurnUploader,
Uploader and Outbox; the server is the ingest route's own parser (``MessagesParser``)
with its span and content dedup.

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
from synapse_client.harnesses.hermes import source  # noqa: E402
from synapse_client.outbox import Outbox  # noqa: E402
from synapse_client.settings import Settings  # noqa: E402
from synapse_client.shipped import Ledger  # noqa: E402
from synapse_client.turn_upload import TurnUploader  # noqa: E402
from synapse_client.upload import Uploader  # noqa: E402

SEEDS = int(os.environ.get("SYNAPSE_FUZZ_SEEDS", "40"))
MODEL = "[System: The active model for this chat has changed to gpt-x via provider y.]"


class Server:
    """The ingest route: the server's parser, one episode per span id and none whose
    content is already stored in the project. ``down`` fails every request."""

    def __init__(self, tmp: Path) -> None:
        self.settings = Settings.load(
            env={"SYNAPSE_URL": "http://synapse.test", "SYNAPSE_PRIVATE_DIR": str(tmp / "pd")},
            home=tmp / "synapse-home",
        )
        self.down = False
        self.episodes: list[Any] = []
        self.spans: set[str] = set()
        self.contents: set[str] = set()

    def ingest(self, payload: Any, timeout: float = 30, **_: Any) -> dict[str, Any]:
        private.check_payload(self.settings, payload)
        if self.down:
            raise urllib.error.URLError("down")
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
        return {"private": False}

    def private_put(self, session_id: str, timeout: float = 15) -> dict[str, Any]:
        return {}


class Run:
    """One seed: the store, the server, and the agents that upload."""

    def __init__(self, seed: int, tmp: Path) -> None:
        self.rnd = random.Random(seed)
        self.seed = seed
        self.home = tmp / "hermes"
        self.store = HermesStore(self.home)
        self.sid = "sess-a"
        self.store.session(self.sid)
        self.parents: list[str] = []
        self.server = Server(tmp)
        self.outbox = Outbox(tmp / "outbox")
        self.agent = self.new_agent()
        self.n = 0
        self.must: list[str] = []  # texts stored exactly once
        self.at_most: list[str] = []  # texts stored once or never (a failed run's ask)
        self.never: list[str] = []  # texts never stored
        self.alike = ""  # a reply given to several reports, each stored
        self.alike_uses = 0

    def new_agent(self) -> TurnUploader:
        ledger = Ledger(self.home / "synapse" / "shipped")
        uploader = Uploader(self.server, self.outbox)  # type: ignore[arg-type]
        return TurnUploader(
            uploader, client="hermes", source="hermes-sync", lookback=10, ledger=ledger
        )

    def tok(self, kind: str) -> str:
        self.n += 1
        return f"{kind}-{self.seed}-{self.n}-text"

    def add(self, role: str, content: Any = "", **kw: Any) -> dict[str, Any]:
        return self.store.add(self.sid, role, content, **kw)

    def sync(self) -> None:
        """A run ended: the agent reads the store up to the last row and uploads."""
        last = self.store.conn.execute(
            "SELECT message_uid FROM messages WHERE session_id=? ORDER BY id DESC LIMIT 1",
            (self.sid,),
        ).fetchone()
        got = source.read(self.home, self.sid, last[0] if last else None, self.parents)
        assert got.records is not None, got.reason
        self.agent.note_copies(self.sid, got.copied)
        self.agent.upload(got.records, self.sid, timeout=5, finished=True)
        if not self.server.down:
            self.agent.uploader.drain(5)

    # -- what can happen between two syncs ---------------------------------------------

    def turn(self, tools: bool = False) -> None:
        user, final = self.tok("USER"), self.tok("FINAL")
        self.must += [user, final]
        self.add("user", user)
        if tools:
            one = call()
            self.add("assistant", "", tool_calls=[one])
            self.add(
                "tool", self.tok("RESULT") + " output", tool_name="terminal", tool_call_id=one["id"]
            )
        self.add("assistant", final)

    def report(self, reply: str | None = None) -> None:
        text = self.tok("REPORT")
        self.never.append(text)
        self.add(
            "user", f"[ASYNC DELEGATION COMPLETE]\n{text}", display_kind="async_delegation_complete"
        )
        if reply is None:
            reply = self.tok("FINAL")
            self.must.append(reply)
        self.add("assistant", reply)

    def nudged(self) -> None:
        user, final = self.tok("USER"), self.tok("FINAL")
        self.must += [user, final]
        self.add("user", user)
        self.add("assistant", "INTERIM " + self.tok("INTERIM"))
        self.add("user", "[System: Continue now. Execute the required tool calls.]")
        self.add("assistant", final)

    def marker(self) -> None:
        self.add("user", MODEL, display_kind="model_switch")
        self.turn()

    def note(self) -> None:
        text = self.tok("NOTE")
        self.never.append(text)
        self.add(
            "user",
            f"[IMPORTANT: Background process exited.]\n{text}",
            display_kind="process_complete",
        )
        self.turn()

    def picture(self) -> None:
        final = self.tok("FINAL")
        self.must.append(final)
        self.add("user", "[screenshot]", live=[{"type": "image_url", "image_url": {"url": "x"}}])
        self.add("assistant", final)

    def failed(self) -> None:
        user = self.tok("USER")
        self.at_most.append(user)
        self.add("user", user)
        self.add(
            "assistant",
            "This turn did not complete. Some actions may already have run; verify their "
            "effects before resending.",
            display_kind="failed_turn",
        )

    def steer(self) -> None:
        user, final = self.tok("USER"), self.tok("FINAL")
        self.must += [user, final]
        self.add("user", user)
        self.add("assistant", "", tool_calls=[call()])
        self.add("tool", self.tok("RESULT") + " output", tool_name="terminal")
        self.add("user", "[OUT-OF-BAND USER MESSAGE]\nsteer", display_kind="steer")
        self.add("assistant", final)

    def branch(self) -> None:
        """``/branch``: the provider ships what the session left holds first (provider.py
        ``on_session_switch``), then the child goes on with copies of its rows."""
        self.sync()
        child = f"sess-{self.n}"
        self.n += 1
        self.store.branch(self.sid, child)
        self.parents = [self.sid, *self.parents]
        self.sid = child
        self.agent.start(child, 10, continues=self.parents)

    def play(self) -> None:
        self.turn()
        self.sync()  # the session's first upload: nothing before it to look back past
        self.alike = self.tok("ALIKE")
        for _ in range(self.rnd.randint(6, 26)):
            event = self.rnd.choice(
                [
                    "turn",
                    "tools",
                    "report",
                    "alike",
                    "nudged",
                    "marker",
                    "note",
                    "picture",
                    "failed",
                    "steer",
                    "compact",
                    "rebuild",
                    "skip",
                    "down",
                    "branch",
                ]
            )
            if event == "compact":
                self.store.compact(self.sid, keep=self.rnd.randint(0, 6))
                continue
            if event == "rebuild":
                self.agent = self.new_agent()
                continue
            if event == "branch" and not self.parents:
                self.branch()
                continue
            if event == "skip":  # the run ends while no upload runs: no sync for it
                self.turn()
                continue
            if event == "down":
                self.server.down = True
                self.turn()
                self.sync()
                self.server.down = False
                continue
            if event == "alike":
                self.report(self.alike)
                self.alike_uses += 1
            elif event in ("turn", "tools"):
                self.turn(tools=event == "tools")
            elif event != "branch":
                getattr(self, event)()
            else:
                self.turn()
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

        out = []
        for token in self.must:
            got = times(token)
            if got != 1:
                out.append(f"{token} stored {got} times")
        if times(self.alike) != self.alike_uses:
            out.append(f"{self.alike}: {times(self.alike)} of {self.alike_uses} stored")
        out += [f"{t} stored {times(t)} times" for t in self.at_most if times(t) > 1]
        out += [f"{t} stored" for t in self.never if times(t)]
        return out


@pytest.mark.parametrize("seed", range(SEEDS))
def test_every_message_and_reply_is_stored_once(seed: int, tmp_path: Path) -> None:
    run = Run(seed, tmp_path)
    try:
        run.play()
    finally:
        run.store.close()
    assert run.problems() == []
