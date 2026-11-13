"""The core's one way out for a turn (plugin/synapse_client/upload.py, and the private
rule ``Client.ingest`` runs before its request).

Pinned here:
  * ``Client.ingest`` sends nothing for a body that names a session in private mode,
    whichever way the body names it, and says whether it is sure;
  * ``Uploader`` queues before it sends, keeps what failed, and neither sends nor queues
    a private session's body;
  * a drain drops what went private since it was queued, leaves what it cannot tell
    about, and goes on to the rest;
  * ``TurnUploader`` sends each finished turn once, in the ``messages`` format;
  * a conversation that goes on under a new session id stays private when the id it
    leaves is (``Uploader.carry_private``).

No network: the client's transport is replaced by a recorder. Temp directories only.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.client_core import use_core

use_core()
from synapse_client import private, upload  # noqa: E402
from synapse_client.api import Client  # noqa: E402
from synapse_client.outbox import Outbox, short_reason  # noqa: E402
from synapse_client.settings import Settings  # noqa: E402


class _Wire:
    """In place of the client's Transport: records each ingest request, then answers it
    or raises ``fail``. ``requests`` are the others (the private-session routes), which
    raise ``fail_private`` or answer from ``rows``, the sessions the server holds off the
    record; ``answer``, when set, is what a GET is answered with whatever the rows."""

    def __init__(self) -> None:
        self.tries = 0
        self.posts: list[dict[str, Any]] = []
        self.fail: BaseException | None = None
        self.requests: list[tuple[str, str, float]] = []
        self.fail_private: BaseException | None = None
        self.rows: set[str] = set()
        self.answer: Any = None

    def request_json(self, method: str, path: str, timeout: float = 15) -> Any:
        self.requests.append((method, path, timeout))
        if self.fail_private is not None:
            raise self.fail_private
        session_id = path.rsplit("/", 1)[-1]
        if method == "PUT":
            self.rows.add(session_id)
        elif method == "DELETE":
            self.rows.discard(session_id)
        elif self.answer is not None:
            return self.answer
        return {"status": "ok", "private": session_id in self.rows}

    def post_json(self, url: str, body: Any, timeout: float = 30) -> Any:
        self.tries += 1
        if self.fail is not None:
            raise self.fail
        self.posts.append({"url": url, "body": body, "timeout": timeout})
        return {"status": "ok"}

    def sessions(self) -> list[str]:
        return [post["body"].get("session_id") for post in self.posts]


def _client(tmp_path: Path) -> tuple[Client, _Wire]:
    env = {"SYNAPSE_URL": "http://synapse.test", "SYNAPSE_PRIVATE_DIR": str(tmp_path / "private")}
    client = Client(Settings.load(env=env, home=tmp_path / "home"), "synapse-client/test")
    wire = _Wire()
    client.transport = wire  # type: ignore[assignment]
    return client, wire


def _mark(tmp_path: Path, session_id: str, hours: float = 0.0) -> Path:
    """A private marker for ``session_id``, written ``hours`` ago."""
    marker = tmp_path / "private" / session_id
    marker.parent.mkdir(exist_ok=True)
    marker.write_text("")
    if hours:
        old = time.time() - hours * 3600
        os.utime(marker, (old, old))
    return marker


def _unreadable(tmp_path: Path, session_id: str) -> Path:
    """A marker no look gets through: a link to itself."""
    marker = tmp_path / "private" / session_id
    marker.parent.mkdir(exist_ok=True)
    marker.symlink_to(session_id)
    return marker


def _body(session_id: str, **more: Any) -> dict[str, Any]:
    return {"format": "messages", "session_id": session_id, "records": [], **more}


def _turn(n: int, *, done: bool = True) -> list[dict[str, Any]]:
    rows = [{"role": "user", "content": f"q{n}", "id": f"u{n}"}]
    if done:
        rows.append({"role": "assistant", "content": f"a{n}", "id": f"a{n}"})
    return rows


def _queued(box: Outbox, suffix: str = ".json") -> list[Any]:
    """The bodies waiting in ``box``; with ``suffix`` ".sending", the ones written ahead
    of a send still under way."""
    names = sorted(n for n in os.listdir(box.directory) if n.endswith(suffix))
    return [json.loads((box.directory / n).read_text()) for n in names]


# ---------------------------------------------------------------------------
# Client.ingest: the rule at the request
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "sessions"),
    [
        ({"format": "messages", "session_id": "s-off", "records": [{"role": "user"}]}, ()),
        ({"source": "hook", "records": [{"sessionId": "s-on"}, {"sessionId": "s-off"}]}, ()),
        ({"source": "hook", "records": [{"sessionId": "s-on"}]}, ("s-off",)),
    ],
    ids=["its own session_id", "a record's sessionId", "a session it was said to belong to"],
)
def test_ingest_sends_nothing_for_a_private_session(tmp_path, payload, sessions):
    client, wire = _client(tmp_path)
    _mark(tmp_path, "s-off")
    with pytest.raises(private.PrivateSession) as refused:
        client.ingest(payload, sessions=sessions)
    assert refused.value.session_id == "s-off" and refused.value.sure
    assert wire.tries == 0


def test_ingest_sends_a_session_on_the_record(tmp_path):
    client, wire = _client(tmp_path)
    _mark(tmp_path, "s-other")
    payload = _body("s-on", records=[{"role": "user", "content": "q"}])
    assert client.ingest(payload, timeout=7) == {"status": "ok"}
    assert wire.posts == [{"url": client.settings.ingest_url, "body": payload, "timeout": 7}]


def test_a_marker_past_its_time_is_removed_and_the_body_goes(tmp_path):
    client, wire = _client(tmp_path)
    marker = _mark(tmp_path, "s-1", hours=13)
    client.ingest(_body("s-1"))
    assert wire.sessions() == ["s-1"] and not marker.exists()


def test_a_marker_that_cannot_be_read_refuses_without_being_sure(tmp_path):
    client, wire = _client(tmp_path)
    _unreadable(tmp_path, "s-blind")
    lines: list[str] = []
    with pytest.raises(private.PrivateSession) as refused:
        client.ingest(_body("s-blind"), log=lines.append)
    assert refused.value.session_id == "s-blind" and not refused.value.sure
    assert wire.tries == 0
    assert lines == ["PRIVATE unreadable s-blind: OSError — skipping ingestion"]


def test_a_private_session_is_named_before_one_that_cannot_be_read(tmp_path):
    client, _wire = _client(tmp_path)
    _unreadable(tmp_path, "s-blind")
    _mark(tmp_path, "s-off")
    payload = {"records": [{"sessionId": "s-blind"}, {"sessionId": "s-off"}]}
    with pytest.raises(private.PrivateSession) as refused:
        client.ingest(payload)
    assert refused.value.session_id == "s-off" and refused.value.sure


def test_the_sessions_a_body_names_are_read_from_whatever_it_holds():
    assert private.sessions_of({"session_id": "a", "records": [{"sessionId": "b"}]}) == ["a", "b"]
    payload = {"records": [{"sessionId": "b"}, "junk", {"sessionId": 7}, {"sessionId": "b"}, {}]}
    assert private.sessions_of(payload) == ["b"]
    assert private.sessions_of({"session_id": 7, "records": "none"}) == []
    assert private.sessions_of({}) == []


# ---------------------------------------------------------------------------
# Uploader.send and Uploader.queue
# ---------------------------------------------------------------------------


def test_a_delivered_body_leaves_nothing_queued(tmp_path):
    client, wire = _client(tmp_path)
    box = Outbox(tmp_path / "q")
    assert upload.Uploader(client, box).send(_body("s-1"), timeout=5) == (upload.SENT, "")
    assert wire.posts[0]["timeout"] == 5 and _queued(box) == []


def test_a_body_is_queued_before_its_request(tmp_path):
    """A process that exits mid-request must leave the body behind."""
    client, wire = _client(tmp_path)
    box = Outbox(tmp_path / "q")
    seen: list[list[Any]] = []
    answer = wire.post_json
    wire.post_json = lambda *a, **k: seen.append(_queued(box, ".sending")) or answer(*a, **k)  # type: ignore[method-assign]
    upload.Uploader(client, box).send(_body("s-1"), timeout=5)
    assert seen == [[_body("s-1")]]
    assert os.listdir(box.directory) == []


def test_a_failed_send_waits_in_the_outbox_with_its_reason(tmp_path):
    client, wire = _client(tmp_path)
    box = Outbox(tmp_path / "q")
    lines: list[str] = []
    wire.fail = urllib.error.URLError("refused")
    reason = short_reason(wire.fail)
    up = upload.Uploader(client, box, log=lines.append)
    assert up.send(_body("s-1"), timeout=5) == (upload.QUEUED, reason)
    assert _queued(box) == [_body("s-1")] and box.pending() == 1
    assert lines == [f"synapse: upload failed ({reason}); queued for retry"]
    wire.fail = None
    assert up.drain(timeout=5) == (1, 0) and wire.sessions() == ["s-1"]


def test_a_private_session_is_neither_sent_nor_queued(tmp_path):
    client, wire = _client(tmp_path)
    box = Outbox(tmp_path / "q")
    up = upload.Uploader(client, box)
    _mark(tmp_path, "s-off")
    assert up.is_private("s-off") and not up.is_private("s-on")
    assert up.send(_body("s-off"), timeout=5) == (upload.PRIVATE, "")
    assert up.queue(_body("s-off"), "timeout") == upload.PRIVATE
    assert wire.tries == 0 and not box.directory.exists()


def test_a_session_taken_off_the_record_mid_send_leaves_nothing_queued(tmp_path):
    """The marker lands after the uploader's own look and before the request: the
    request's look refuses it, and the copy queued for the send goes too."""
    client, wire = _client(tmp_path)

    class _MarkedMeanwhile(Outbox):
        def put(self, payload: Any, reason: str = "", *, sending: float = 0) -> Path:
            _mark(tmp_path, "s-1")
            return super().put(payload, reason, sending=sending)

    box = _MarkedMeanwhile(tmp_path / "q")
    assert upload.Uploader(client, box).send(_body("s-1"), timeout=5) == (upload.PRIVATE, "")
    assert wire.tries == 0 and os.listdir(box.directory) == []


def test_without_an_outbox_a_failed_send_is_lost(tmp_path):
    client, wire = _client(tmp_path)
    wire.fail = TimeoutError("timed out")
    up = upload.Uploader(client)
    assert up.send(_body("s-1"), timeout=5) == (upload.LOST, short_reason(wire.fail))
    assert up.queue(_body("s-1"), "timeout") == upload.LOST
    assert up.drain(timeout=5) == (0, 0)


def test_a_failed_send_that_cannot_be_queued_is_lost_and_said(tmp_path):
    client, wire = _client(tmp_path)
    (tmp_path / "q").write_text("a file where the outbox goes")
    lines: list[str] = []
    wire.fail = TimeoutError("timed out")
    up = upload.Uploader(client, Outbox(tmp_path / "q"), log=lines.append)
    assert up.send(_body("s-1"), timeout=5)[0] == upload.LOST
    assert len(lines) == 2 and lines[1].startswith("synapse: could not queue a failed upload (")
    assert lines[1].endswith("; it is lost")


def test_a_lost_upload_is_said_to_warn_when_there_is_one(tmp_path):
    """Nothing sends a lost batch again, so its line is the one a harness raises above
    its ordinary log."""
    client, wire = _client(tmp_path)
    (tmp_path / "q").write_text("a file where the outbox goes")
    lines: list[str] = []
    warned: list[str] = []
    wire.fail = TimeoutError("timed out")
    up = upload.Uploader(client, Outbox(tmp_path / "q"), log=lines.append, warn=warned.append)
    assert up.send(_body("s-1"), timeout=5)[0] == upload.LOST
    assert [line.split(" (")[0] for line in lines] == ["synapse: upload failed"]
    assert [line.split(" (")[0] for line in warned] == ["synapse: could not queue a failed upload"]


def test_a_raising_log_does_not_stop_an_upload(tmp_path):
    client, wire = _client(tmp_path)
    wire.fail = TimeoutError("timed out")

    def bad_log(line: str) -> None:
        raise RuntimeError(line)

    up = upload.Uploader(client, Outbox(tmp_path / "q"), log=bad_log)
    assert up.send(_body("s-1"), timeout=5)[0] == upload.QUEUED


# ---------------------------------------------------------------------------
# Uploader.drain
# ---------------------------------------------------------------------------


def test_a_drain_drops_what_went_private_and_goes_on_to_the_rest(tmp_path):
    client, wire = _client(tmp_path)
    box = Outbox(tmp_path / "q")
    up = upload.Uploader(client, box)
    for session_id in ("s-off", "s-blind", "s-on"):
        box.put(_body(session_id), reason="timeout")
    box.put({"n": 1}, reason="timeout").write_text("[1, 2]")  # not an ingest body
    _mark(tmp_path, "s-off")
    blind = _unreadable(tmp_path, "s-blind")

    assert up.drain(timeout=5) == (1, 1)
    assert wire.sessions() == ["s-on"]
    assert _queued(box) == [_body("s-blind")]  # cannot tell: it stays, and is not sent

    blind.unlink()  # readable again, and not private
    assert up.drain(timeout=5) == (1, 0)
    assert wire.sessions() == ["s-on", "s-blind"] and _queued(box) == []


def test_bodies_a_drain_must_leave_do_not_keep_it_from_the_rest(tmp_path):
    """Each one left waiting used to count toward the drain's allowance: enough of them
    at the head of the queue and no drain reached what was behind."""
    client, wire = _client(tmp_path)
    box = Outbox(tmp_path / "q")
    lines: list[str] = []
    up = upload.Uploader(client, box, log=lines.append)
    for _ in range(3):
        box.put(_body("s-blind"), reason="timeout")
    for session_id in ("s-1", "s-2", "s-3"):
        box.put(_body(session_id), reason="timeout")
    _unreadable(tmp_path, "s-blind")

    assert up.drain(timeout=5, max_items=2) == (2, 4)
    assert wire.sessions() == ["s-1", "s-2"]
    assert up.drain(timeout=5, max_items=2) == (1, 3)
    assert wire.sessions() == ["s-1", "s-2", "s-3"]
    assert _queued(box) == [_body("s-blind")] * 3
    assert any(line.startswith("PRIVATE unreadable s-blind") for line in lines)


def test_a_drain_stops_at_a_send_that_fails_and_keeps_it(tmp_path):
    client, wire = _client(tmp_path)
    box = Outbox(tmp_path / "q")
    up = upload.Uploader(client, box)
    for session_id in ("s-1", "s-2"):
        box.put(_body(session_id), reason="timeout")
    wire.fail = urllib.error.URLError("refused")
    assert up.drain(timeout=5) == (0, 2) and wire.tries == 1


# ---------------------------------------------------------------------------
# TurnUploader
# ---------------------------------------------------------------------------


def _turns(tmp_path: Path, box: Outbox | None = None, **kwargs: Any):
    client, wire = _client(tmp_path)
    uploader = upload.Uploader(client, box)
    return upload.TurnUploader(uploader, client="t", source="t-sync", **kwargs), wire


def test_each_finished_turn_is_sent_once_with_the_turn_before_it(tmp_path):
    turns, wire = _turns(tmp_path)
    records = _turn(1) + _turn(2) + _turn(3, done=False)
    assert turns.upload(records, "s-1", timeout=5, project="p", cwd="/w", model="m") is True
    assert wire.posts[0]["body"] == {
        "format": "messages",
        "client": "t",
        "session_id": "s-1",
        "project": "p",
        "cwd": "/w",
        "model": "m",
        "source": "t-sync",
        "records": _turn(1) + _turn(2),
    }
    assert turns.upload(records, "s-1", timeout=5) is False and wire.tries == 1
    assert turns.upload(_turn(1) + _turn(2) + _turn(3), "s-1", timeout=5) is True
    assert wire.posts[1]["body"]["records"] == _turn(2) + _turn(3)  # 2 is the context


def test_a_resumed_conversation_looks_back_only_so_far(tmp_path):
    turns, wire = _turns(tmp_path, lookback=1)
    records = _turn(1) + _turn(2) + _turn(3)
    assert turns.upload(records, "s-1", timeout=5) is True
    assert wire.posts[0]["body"]["records"] == _turn(2) + _turn(3)
    assert turns.upload(records, "s-1", timeout=5) is False  # 1 and 2 count as sent
    turns.forget(2)  # a new session id: look again, two turns back
    assert turns.upload(records, "s-2", timeout=5) is True
    assert wire.posts[1]["body"]["records"] == records
    assert wire.posts[1]["body"]["session_id"] == "s-2"


def test_nothing_is_sent_or_remembered_for_a_private_session(tmp_path):
    turns, wire = _turns(tmp_path, Outbox(tmp_path / "q"))
    marker = _mark(tmp_path, "s-1")
    assert turns.upload(_turn(1), "s-1", timeout=5) is False
    assert turns.upload(_turn(1), "", timeout=5) is False  # no session: nothing either
    assert wire.tries == 0 and not (tmp_path / "q").exists()
    marker.unlink()
    assert turns.upload(_turn(1), "s-1", timeout=5) is True  # it was not counted as sent


def test_a_turn_refused_at_the_request_is_not_counted_as_sent(tmp_path):
    """The marker lands after the turn was claimed for this upload. Nothing goes and
    nothing is queued; once the session is back on the record the turn is sent."""
    client, wire = _client(tmp_path)
    marked: list[Path] = []

    class _MarkedMeanwhile(Outbox):
        def put(self, payload: Any, reason: str = "", *, sending: float = 0) -> Path:
            if not marked:
                marked.append(_mark(tmp_path, "s-1"))
            return super().put(payload, reason, sending=sending)

    box = _MarkedMeanwhile(tmp_path / "q")
    turns = upload.TurnUploader(upload.Uploader(client, box), client="t", source="t-sync")
    assert turns.upload(_turn(1), "s-1", timeout=5) is False
    assert wire.tries == 0 and os.listdir(box.directory) == []
    marked[0].unlink()
    assert turns.upload(_turn(1), "s-1", timeout=5) is True
    assert wire.posts[0]["body"]["records"] == _turn(1)


def test_a_turn_that_fails_waits_in_the_outbox_and_is_not_sent_twice(tmp_path):
    box = Outbox(tmp_path / "q")
    turns, wire = _turns(tmp_path, box)
    wire.fail = urllib.error.URLError("refused")
    assert turns.upload(_turn(1), "s-1", timeout=5) is False
    wire.fail = None
    assert turns.upload(_turn(1), "s-1", timeout=5) is False and wire.tries == 1
    assert turns.uploader.drain(timeout=5) == (1, 0)
    assert wire.posts[0]["body"]["records"] == _turn(1)


def test_a_turn_that_could_not_be_queued_is_tried_again(tmp_path):
    turns, wire = _turns(tmp_path)  # no outbox: a failed send is lost
    wire.fail = urllib.error.URLError("refused")
    assert turns.upload(_turn(1), "s-1", timeout=5) is False
    wire.fail = None
    assert turns.upload(_turn(1), "s-1", timeout=5) is True
    assert wire.posts[0]["body"]["records"] == _turn(1)


def test_after_a_failed_batch_the_rest_are_queued_without_a_request(tmp_path):
    box = Outbox(tmp_path / "q")
    turns, wire = _turns(tmp_path, box)
    records = [row for n in range(1, 22) for row in _turn(n)]  # 21 turns: two batches
    wire.fail = urllib.error.URLError("refused")
    assert turns.upload(records, "s-1", timeout=5) is False
    assert wire.tries == 1 and box.pending() == 2
    wire.fail = None
    assert turns.uploader.drain(timeout=5) == (2, 0)
    sent = [row["id"] for post in wire.posts for row in post["body"]["records"]]
    assert sent.count("a20") == 2 and sent.count("a21") == 1  # 20 leads the second as context


# ---------------------------------------------------------------------------
# A conversation that goes on under a new session id
# ---------------------------------------------------------------------------

_CONTINUES = "synapse: session s-new continues s-old, which is in private mode"


def test_a_session_that_continues_a_private_one_is_private_too(tmp_path):
    client, wire = _client(tmp_path)
    lines: list[str] = []
    up = upload.Uploader(client, Outbox(tmp_path / "q"), log=lines.append)
    _mark(tmp_path, "s-old")
    assert up.carry_private(["s-parent", "", "s-old"], "s-new") is True
    assert (tmp_path / "private" / "s-new").is_file()
    assert wire.requests == [("PUT", "/private-sessions/s-new", 5)]
    assert lines == [f"{_CONTINUES}: it is off the record too"]
    # The marker is all there is to it: another uploader refuses the session too.
    assert upload.Uploader(client).send(_body("s-new"), timeout=5) == (upload.PRIVATE, "")
    assert wire.tries == 0


def _get(session_id: str) -> tuple[str, str, float]:
    return ("GET", f"/private-sessions/{session_id}", 5)


def _put(session_id: str) -> tuple[str, str, float]:
    return ("PUT", f"/private-sessions/{session_id}", 5)


def test_with_no_marker_to_go_by_the_server_is_asked_before_anything_is_uploaded(tmp_path):
    """No marker is not an answer: one lasts twelve hours and `private off` removes it,
    while the server's row stays. Until the server has said, the new session is held."""
    client, wire = _client(tmp_path)
    box = Outbox(tmp_path / "q")
    up = upload.Uploader(client, box)
    done: list[str] = []
    _mark(tmp_path, "s-other")
    assert (
        up.carry_private(["s-old", "", "s-parent"], "s-new", settled=lambda: done.append("s"))
        is False
    )
    # Nothing was written or asked yet: the question waits for the session's next upload.
    assert not (tmp_path / "private" / "s-new").exists() and wire.requests == [] and done == []
    assert up.is_private("s-new") and up.asking("s-new")
    assert up.send(_body("s-new"), timeout=5) == (upload.PRIVATE, "")
    assert up.queue(_body("s-new"), "timeout") == upload.PRIVATE
    assert wire.tries == 0 and not box.directory.exists()

    assert up.settle_private("s-new") is True
    assert wire.requests == [_get("s-old"), _get("s-parent")] and done == ["s"]
    assert not up.is_private("s-new") and not up.asking("s-new")
    assert not (tmp_path / "private" / "s-new").exists()
    assert up.send(_body("s-new"), timeout=5) == (upload.SENT, "")
    assert up.settle_private("s-new") is True and len(wire.requests) == 2 and done == ["s"]


@pytest.mark.parametrize("marker_hours", [None, 13])
def test_a_session_the_server_holds_private_is_carried_without_its_marker(tmp_path, marker_hours):
    """`private off` removed the marker (None), or it lapsed (13 hours old): the row on
    the server still drops that session's turns, and the session that continues it holds
    copies of them."""
    client, wire = _client(tmp_path)
    lines: list[str] = []
    up = upload.Uploader(client, log=lines.append)
    done: list[str] = []
    if marker_hours is not None:
        _mark(tmp_path, "s-old", hours=marker_hours)
    wire.rows.add("s-old")
    assert up.carry_private(["s-old"], "s-new", settled=lambda: done.append("s")) is False
    assert up.is_private("s-new") and not (tmp_path / "private" / "s-old").exists()
    del lines[:]  # the line of private.py for a marker past its time
    assert up.settle_private("s-new") is True
    assert wire.requests == [_get("s-old"), _put("s-new")] and done == ["s"]
    assert lines == [f"{_CONTINUES}: it is off the record too"]
    assert (tmp_path / "private" / "s-new").is_file() and not up.asking("s-new")
    # The marker is all there is to it now: another uploader refuses the session too.
    assert upload.Uploader(client).send(_body("s-new"), timeout=5) == (upload.PRIVATE, "")
    assert wire.tries == 0


@pytest.mark.parametrize(
    "answer",
    [{}, {"status": "ok"}, {"private": "no"}, {"private": 0}, {"private": None}],
)
def test_a_reply_that_does_not_say_keeps_the_session_held(tmp_path, answer):
    client, wire = _client(tmp_path)
    up = upload.Uploader(client)
    done: list[str] = []
    assert up.carry_private(["s-old"], "s-new", settled=lambda: done.append("s")) is False
    wire.answer = answer
    assert up.settle_private("s-new") is False
    assert up.is_private("s-new") and up.asking("s-new") and done == []
    assert up.send(_body("s-new"), timeout=5) == (upload.PRIVATE, "")


def test_a_server_that_does_not_answer_keeps_the_session_held(tmp_path):
    client, wire = _client(tmp_path)
    box = Outbox(tmp_path / "q")
    up = upload.Uploader(client, box)
    done: list[str] = []
    wire.rows.add("s-old")
    assert up.carry_private(["s-old"], "s-new", settled=lambda: done.append("s")) is False
    wire.fail_private = urllib.error.URLError("down")
    assert up.settle_private("s-new") is False and up.settle_private("s-new") is False
    assert up.is_private("s-new") and done == []
    assert up.send(_body("s-new"), timeout=5) == (upload.PRIVATE, "")
    assert wire.tries == 0 and not box.directory.exists()
    # Another session's upload is not held up by it.
    assert up.send(_body("s-on"), timeout=5) == (upload.SENT, "")
    wire.fail_private = None
    assert up.settle_private("s-new") is True and done == ["s"]
    assert wire.requests[-2:] == [_get("s-old"), _put("s-new")]
    assert (tmp_path / "private" / "s-new").is_file()


def test_a_session_still_waiting_passes_its_question_on(tmp_path):
    """A second new id before the server answered about the first: the first is not
    known to be private, so the third asks about it and about what it was asking."""
    client, wire = _client(tmp_path)
    up = upload.Uploader(client)
    wire.rows.add("s-1")
    assert up.carry_private(["s-1"], "s-2") is False
    assert up.carry_private(["s-2"], "s-3") is False
    assert wire.requests == [] and not (tmp_path / "private").exists()
    assert up.settle_private("s-3") is True
    assert wire.requests == [_get("s-2"), _get("s-1"), _put("s-3")]
    assert (tmp_path / "private" / "s-3").is_file()


def test_an_earlier_session_on_the_record_everywhere_lets_the_new_one_go(tmp_path):
    client, wire = _client(tmp_path)
    up = upload.Uploader(client)
    done: list[str] = []
    _mark(tmp_path, "s-stale", hours=13)
    assert (
        up.carry_private(["s-old", "s-stale"], "s-new", settled=lambda: done.append("s")) is False
    )
    assert up.settle_private("s-new") is True and done == ["s"]
    assert wire.requests == [_get("s-old"), _get("s-stale")]
    assert up.send(_body("s-new"), timeout=5) == (upload.SENT, "")
    assert not (tmp_path / "private" / "s-new").exists()


def test_with_nothing_earlier_there_is_nothing_to_ask(tmp_path):
    client, wire = _client(tmp_path)
    up = upload.Uploader(client)
    done: list[str] = []
    assert up.carry_private(["", "s-new"], "s-new", settled=lambda: done.append("s")) is False
    assert done == ["s"] and not up.is_private("s-new") and wire.requests == []


def test_settled_is_not_called_before_the_server_has_the_row(tmp_path):
    """The caller's note says the session need not be asked about again: with the row
    still owed, a later program would find a lapsed marker and nothing else."""
    client, wire = _client(tmp_path)
    up = upload.Uploader(client)
    done: list[str] = []
    _mark(tmp_path, "s-old")
    wire.fail_private = urllib.error.URLError("down")
    assert up.carry_private(["s-old"], "s-new", settled=lambda: done.append("s")) is True
    assert up.settle_private("s-new") is False and done == []
    wire.fail_private = None
    assert up.settle_private("s-new") is True and done == ["s"]


def test_a_settled_that_raises_changes_nothing(tmp_path):
    client, _ = _client(tmp_path)
    up = upload.Uploader(client)

    def broken() -> None:
        raise RuntimeError("no disk")

    assert up.carry_private(["s-old"], "s-new", settled=broken) is False
    assert up.settle_private("s-new") is True and not up.is_private("s-new")


def test_carrying_without_a_timeout_tells_the_server_at_the_next_upload(tmp_path):
    client, wire = _client(tmp_path)
    lines: list[str] = []
    warned: list[str] = []
    up = upload.Uploader(client, log=lines.append, warn=warned.append)
    turns = upload.TurnUploader(up, client="t", source="t-sync")
    _mark(tmp_path, "s-old")
    assert up.carry_private(["s-old"], "s-new", timeout=None) is True
    assert wire.requests == [] and (tmp_path / "private" / "s-new").is_file()
    assert lines == [
        f"{_CONTINUES}: nothing is uploaded for it, and the server is told at its next upload"
    ]
    assert turns.upload(_turn(1), "s-new", timeout=5) is False
    assert wire.requests == [_put("s-new")] and wire.tries == 0 and warned == []


def test_every_session_the_server_is_still_to_be_told_about_is_told(tmp_path):
    """A session left behind by a second switch never has an upload of its own."""
    client, wire = _client(tmp_path)
    up = upload.Uploader(client)
    _mark(tmp_path, "s-1")
    wire.fail_private = urllib.error.URLError("down")
    assert up.carry_private(["s-1"], "s-2") is True
    assert up.carry_private(["s-2"], "s-3") is True
    assert up.carry_private(["s-3"], "s-4") is True
    wire.fail_private = None
    wire.requests.clear()
    assert up.settle_private("s-4") is True
    assert wire.requests == [_put("s-4"), _put("s-2"), _put("s-3")]
    assert wire.rows == {"s-2", "s-3", "s-4"}


def test_a_delivered_upload_tells_the_server_what_it_is_still_to_be_told(tmp_path):
    client, wire = _client(tmp_path)
    up = upload.Uploader(client)
    _mark(tmp_path, "s-1")
    wire.fail_private = urllib.error.URLError("down")
    assert up.carry_private(["s-1"], "s-2") is True
    wire.fail = urllib.error.URLError("down")
    wire.requests.clear()
    assert up.send(_body("s-on"), timeout=5)[0] == upload.LOST and wire.requests == []
    wire.fail = wire.fail_private = None
    assert up.send(_body("s-on"), timeout=5) == (upload.SENT, "")
    assert wire.requests == [_put("s-2")] and "s-2" in wire.rows


def test_a_turn_goes_up_at_the_upload_the_server_says_on_the_record_at(tmp_path):
    turns, wire = _turns(tmp_path)
    assert turns.uploader.carry_private(["s-old"], "s-new") is False
    assert turns.upload(_turn(1), "s-new", timeout=5) is True
    assert wire.requests == [_get("s-old")] and wire.sessions() == ["s-new"]


def test_a_session_the_server_then_calls_private_is_not_looked_back_at(tmp_path):
    """As after a switch that knew: what was said off the record is not looked back at
    once the session can be uploaded again (`private off --forget`)."""
    turns, wire = _turns(tmp_path, lookback=10)
    wire.rows.add("s-old")
    assert turns.uploader.carry_private(["s-old"], "s-new") is False
    assert turns.upload(_turn(1) + _turn(2), "s-new", timeout=5) is False
    assert wire.tries == 0 and (tmp_path / "private" / "s-new").is_file()
    (tmp_path / "private" / "s-new").unlink()
    assert turns.upload(_turn(1) + _turn(2), "s-new", timeout=5) is False
    assert wire.tries == 0


def test_turns_finished_while_the_server_did_not_answer_are_sent_when_it_does(tmp_path):
    """Nothing is queued for a session that may be private, so the window of the first
    upload grows by the turns that finish meanwhile: three earlier turns, a look back of
    one, and three turns held back."""
    turns, wire = _turns(tmp_path, lookback=1)
    assert turns.uploader.carry_private(["s-old"], "s-new") is False
    wire.fail_private = urllib.error.URLError("down")
    said = _turn(1) + _turn(2) + _turn(3)
    for n in (4, 5):
        said = said + _turn(n)
        assert turns.upload(said, "s-new", timeout=5) is False
        assert turns.upload(said + _turn(n + 1, done=False), "s-new", timeout=5) is False
    assert wire.tries == 0
    wire.fail_private = None
    assert turns.upload(said + _turn(6), "s-new", timeout=5) is True
    sent = [r["id"] for post in wire.posts for r in post["body"]["records"] if r["role"] == "user"]
    assert sorted(set(sent)) == ["u3", "u4", "u5", "u6"]  # u3 as the context of u4


def test_a_marker_that_cannot_be_read_is_carried_as_private(tmp_path):
    client, wire = _client(tmp_path)
    up = upload.Uploader(client)
    _unreadable(tmp_path, "s-old")
    assert up.carry_private(["s-old"], "s-new") is True
    assert (tmp_path / "private" / "s-new").is_file() and len(wire.requests) == 1


def test_carrying_to_the_session_itself_or_to_none_writes_nothing(tmp_path):
    client, wire = _client(tmp_path)
    up = upload.Uploader(client)
    marker = _mark(tmp_path, "s-old", hours=2)
    age = marker.stat().st_mtime
    assert up.carry_private(["s-old"], "s-old") is True
    assert up.carry_private(["s-old"], "") is True
    assert wire.requests == [] and marker.stat().st_mtime == age
    assert os.listdir(tmp_path / "private") == ["s-old"]


def test_a_server_that_cannot_be_told_is_told_again_at_the_next_turn(tmp_path):
    """The marker alone lasts twelve hours and `private off` removes it; the session it
    continues may be private for good. So the uploader holds the new one itself until
    the server has its row, and asks again each time the session has a turn."""
    client, wire = _client(tmp_path)
    lines: list[str] = []
    warned: list[str] = []
    up = upload.Uploader(client, log=lines.append, warn=warned.append)
    turns = upload.TurnUploader(up, client="t", source="t-sync")
    _mark(tmp_path, "s-old")
    wire.fail_private = urllib.error.URLError("down")
    assert up.carry_private(["s-old"], "s-new") is True
    assert lines == [] and warned == [
        f"{_CONTINUES}; the server could not be told. Nothing is uploaded for it here, "
        "and the server is told again at the session's next turn"
    ]
    (tmp_path / "private" / "s-new").unlink()  # expired, or `private off`
    assert up.is_private("s-new") and up.settle_private("s-new") is False

    assert turns.upload(_turn(1), "s-new", timeout=5) is False  # asks again: still down
    assert wire.requests == [_put("s-new")] * 3 and wire.tries == 0
    wire.fail_private = None
    # The server is told. The row is there now, so the session is as private as one
    # `on` was run for: with its marker gone its turns are sent, and the server drops
    # them.
    assert turns.upload(_turn(1) + _turn(2), "s-new", timeout=5) is True
    assert wire.requests == [_put("s-new")] * 4 and wire.tries == 1
    assert not up.is_private("s-new")
    assert up.settle_private("s-new") is True and len(wire.requests) == 4
    assert up.settle_private("s-never-carried") is True and len(wire.requests) == 4
    assert warned == warned[:1]


def _no_marker_written(monkeypatch) -> None:
    def refuse(settings: Any, session_id: str) -> Path:
        raise PermissionError(13, "read-only", session_id)

    monkeypatch.setattr(private, "mark", refuse)


def test_a_session_whose_marker_cannot_be_written_is_held_by_the_uploader(tmp_path, monkeypatch):
    """No marker says the new session is private, so the uploader that knows it refuses
    the session itself: what it is handed, and what waits in its outbox."""
    client, wire = _client(tmp_path)
    box = Outbox(tmp_path / "q")
    lines: list[str] = []
    warned: list[str] = []
    up = upload.Uploader(client, box, log=lines.append, warn=warned.append)
    turns = upload.TurnUploader(up, client="t", source="t-sync")
    _mark(tmp_path, "s-old")
    box.put(_body("s-new"), "timeout")
    box.put(_body("s-on"), "timeout")
    _no_marker_written(monkeypatch)
    assert up.carry_private(["s-old"], "s-new") is True
    assert not (tmp_path / "private" / "s-new").exists()
    assert warned == [
        f"{_CONTINUES}; its marker could not be written (PermissionError), so nothing is "
        "uploaded for it until this program ends"
    ]
    assert lines == [f"{_CONTINUES}: it is off the record too"]  # the server was told
    assert up.is_private("s-new") and not up.is_private("s-on")
    assert up.settle_private("s-new") is True and len(wire.requests) == 1  # nothing owed
    assert up.send(_body("s-new"), timeout=5) == (upload.PRIVATE, "")
    assert up.send({"records": [{"sessionId": "s-new"}]}, timeout=5) == (upload.PRIVATE, "")
    assert up.queue(_body("s-new"), "timeout") == upload.PRIVATE
    assert turns.upload(_turn(1), "s-new", timeout=5) is False
    assert wire.tries == 0
    # Its queued body is left as it is: only a marker or the server can say drop it.
    assert up.drain(timeout=5) == (1, 1) and wire.sessions() == ["s-on"]
    assert _queued(box) == [_body("s-new")]


def test_a_held_session_is_carried_on_to_the_next(tmp_path, monkeypatch):
    client, wire = _client(tmp_path)
    up = upload.Uploader(client)
    _mark(tmp_path, "s-1")
    _no_marker_written(monkeypatch)
    assert up.carry_private(["s-1"], "s-2") is True
    assert up.carry_private(["s-2"], "s-3") is True
    assert up.is_private("s-3") and os.listdir(tmp_path / "private") == ["s-1"]
    assert [path for _, path, _ in wire.requests] == [
        "/private-sessions/s-2",
        "/private-sessions/s-3",
    ]


def test_a_session_id_no_marker_can_be_named_by_is_held(tmp_path):
    client, wire = _client(tmp_path)
    up = upload.Uploader(client)
    _mark(tmp_path, "s-old")
    assert up.carry_private(["s-old"], "../s-new") is True
    assert up.send(_body("../s-new"), timeout=5) == (upload.PRIVATE, "")
    assert wire.tries == 0 and not (tmp_path / "s-new").exists()
    assert os.listdir(tmp_path / "private") == ["s-old"]


def test_carrying_never_raises(tmp_path, monkeypatch):
    client, wire = _client(tmp_path)

    def broken(line: str) -> None:
        raise RuntimeError("log is gone")

    up = upload.Uploader(client, log=broken, warn=broken)
    _mark(tmp_path, "s-old")
    _no_marker_written(monkeypatch)
    wire.fail_private = RuntimeError("anything at all")
    assert up.carry_private(["s-old"], "s-new") is True and up.is_private("s-new")
