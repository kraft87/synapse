"""Payloads queued for their caller's own send (plugin/synapse_client/outbox.py):
``put(..., sending=seconds)`` before the request, then ``remove`` once delivered or
``keep`` when the send failed.

Such a payload must outlive its process, not the machine, so it is not synced; until its
deadline no drain takes it and the cap does not count it, so a healthy upload is sent
once and never drops an older payload. Past the deadline its sender died or hung, and a
drain sends it. Temp directories only.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import time

import pytest

from tests.helpers.client_core import use_core

use_core()
from synapse_client.outbox import Outbox  # noqa: E402

_PAYLOAD = {"format": "messages", "session_id": "s-1", "records": [{"role": "user"}]}


def _waiting(box: Outbox) -> list[str]:
    return sorted(n for n in os.listdir(box.directory) if n.endswith(".json"))


@pytest.fixture()
def fsyncs(monkeypatch) -> list[int]:
    calls: list[int] = []
    real = os.fsync

    def counting(fd: int) -> None:
        calls.append(fd)
        real(fd)

    monkeypatch.setattr(os, "fsync", counting)
    return calls


def test_a_payload_being_sent_is_queued_but_no_drain_takes_it(tmp_path):
    box = Outbox(tmp_path / "outbox")
    path = box.put(_PAYLOAD, sending=60)
    assert path.name.endswith(".sending") and _waiting(box) == []
    assert json.loads(path.read_text(encoding="utf-8")) == _PAYLOAD
    if sys.platform != "win32":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert box.pending() == 1
    sent: list[dict] = []
    assert Outbox(tmp_path / "outbox").drain(sent.append) == (0, 1)  # another session
    assert sent == [] and path.exists()


def test_it_is_not_synced_and_not_held_to_the_cap(tmp_path, fsyncs):
    box = Outbox(tmp_path / "outbox", cap=1)
    older = box.put({"n": 0}, reason="HTTP 503 Service Unavailable")
    synced = len(fsyncs)
    path = box.put(_PAYLOAD, sending=60)
    assert len(fsyncs) == synced
    newer = box.put({"n": 1})  # the cap counts the two waiting, not the one being sent
    assert path.exists() and newer.exists() and not older.exists()
    assert box.dropped() == 1


def test_remove_deletes_it_and_never_raises(tmp_path):
    box = Outbox(tmp_path / "outbox")
    path = box.put(_PAYLOAD, sending=60)
    box.remove(path)
    box.remove(path)
    assert box.pending() == 0


def test_keep_queues_it_as_put_would_have(tmp_path, fsyncs):
    box = Outbox(tmp_path / "outbox", cap=1)
    older = box.put({"n": 0})
    path = box.put(_PAYLOAD, sending=60)
    before = len(fsyncs)
    box.keep(path, "HTTP 503 Service Unavailable")
    assert len(fsyncs) > before  # the payload, and the reason beside it
    assert box.last_error() == "HTTP 503 Service Unavailable"
    assert not path.exists() and not older.exists()
    assert box.dropped() == 1
    sent: list[dict] = []
    assert box.drain(sent.append) == (1, 0)
    assert sent == [_PAYLOAD]


def test_past_its_deadline_a_drain_sends_it_once(tmp_path):
    box = Outbox(tmp_path / "outbox")
    box.put(_PAYLOAD, sending=0.05)  # its sender then died
    time.sleep(0.1)
    sent: list[dict] = []
    assert box.drain(sent.append) == (1, 0)
    assert box.drain(sent.append) == (0, 0)
    assert sent == [_PAYLOAD]


def test_a_sender_past_its_deadline_leaves_it_to_the_drain(tmp_path):
    box = Outbox(tmp_path / "outbox")
    path = box.put(_PAYLOAD, sending=0.05)
    time.sleep(0.1)
    sent: list[dict] = []
    assert box.drain(sent.append) == (1, 0)
    box.keep(path, "timed out")  # its own send failed after all
    box.remove(path)  # or succeeded: one request more, never a second copy
    assert sent == [_PAYLOAD]
    assert box.pending() == 0 and box.last_error() == ""


def test_a_payload_that_is_not_json_raises_and_leaves_nothing(tmp_path):
    box = Outbox(tmp_path / "outbox")
    with pytest.raises(TypeError):
        box.put({"x": object()}, sending=60)
    assert box.pending() == 0
