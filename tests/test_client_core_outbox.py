"""The core's outbox (plugin/synapse_client/outbox.py): a durable, cross-process queue
for uploads that failed.

Temp directories only. The concurrency test runs real processes against one directory
and checks every payload is delivered exactly once.
"""

from __future__ import annotations

import email.message
import json
import os
import socket
import ssl
import stat
import subprocess
import sys
import textwrap
import time
import urllib.error
from pathlib import Path

import pytest

from tests.helpers.client_core import PLUGIN, drop_scripts, load_script, scratch_user, use_core

use_core()
from synapse_client import outbox as outbox_mod  # noqa: E402
from synapse_client.outbox import Outbox, failure_warning, short_reason  # noqa: E402


def _http(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://s/ingest", code, "x", email.message.Message(), None)


def _payloads(box: Outbox) -> list[dict]:
    names = sorted(n for n in os.listdir(box.directory) if n.endswith(".json"))
    return [json.loads((box.directory / n).read_text()) for n in names]


def _age(box: Outbox, path: Path, hours: float) -> Path:
    """Re-stamp a queued payload as enqueued ``hours`` ago."""
    old = time.time_ns() - int(hours * 3600 * 1e9)
    new = box.directory / f"{old:020d}{path.name[20:]}"
    os.rename(path, new)
    return new


# ---------------------------------------------------------------------------
# put
# ---------------------------------------------------------------------------


def test_put_writes_one_owner_only_file_per_payload(tmp_path):
    box = Outbox(tmp_path / "q")
    path = box.put({"records": [1], "s": "é"})
    assert path.parent == tmp_path / "q" and path.name.endswith(".json")
    assert json.loads(path.read_text(encoding="utf-8")) == {"records": [1], "s": "é"}
    assert os.listdir(box.directory) == [path.name]  # no temp file left
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(box.directory.stat().st_mode) == 0o700


def test_names_sort_in_the_order_payloads_were_queued(tmp_path):
    box = Outbox(tmp_path)
    for n in range(20):
        box.put({"n": n})
    assert [p["n"] for p in _payloads(box)] == list(range(20))
    assert box.pending() == 20


def test_put_raises_when_it_cannot_write(tmp_path):
    blocker = tmp_path / "q"
    blocker.write_text("a file where the directory should be")
    with pytest.raises(OSError):
        Outbox(blocker).put({"n": 1})


def test_put_raises_for_a_payload_that_is_not_json(tmp_path):
    with pytest.raises(TypeError):
        Outbox(tmp_path).put({"x": object()})
    assert Outbox(tmp_path).pending() == 0


def test_past_the_cap_the_oldest_are_dropped(tmp_path):
    lines: list[str] = []
    box = Outbox(tmp_path, cap=5, log=lines.append)
    for n in range(8):
        box.put({"n": n})
    assert [p["n"] for p in _payloads(box)] == [3, 4, 5, 6, 7]
    assert len(lines) == 3 and all("outbox full (5)" in line for line in lines)


# ---------------------------------------------------------------------------
# drain
# ---------------------------------------------------------------------------


def test_drain_sends_oldest_first_and_deletes_what_was_delivered(tmp_path):
    box = Outbox(tmp_path)
    for n in range(5):
        box.put({"n": n})
    sent: list[int] = []
    assert box.drain(lambda p: sent.append(p["n"])) == (5, 0)
    assert sent == [0, 1, 2, 3, 4]
    assert os.listdir(tmp_path) == []


def test_drain_stops_at_the_first_transport_failure(tmp_path):
    box = Outbox(tmp_path)
    for n in range(5):
        box.put({"n": n})
    sent: list[int] = []

    def send(payload):
        if payload["n"] == 2:
            raise urllib.error.URLError(ConnectionRefusedError())
        sent.append(payload["n"])

    assert box.drain(send) == (2, 3)
    assert sent == [0, 1]
    assert [p["n"] for p in _payloads(box)] == [2, 3, 4]  # put back, in order
    assert box.last_error() == "connection refused"
    assert not any(n.endswith(".claim") for n in os.listdir(tmp_path))


@pytest.mark.parametrize("code", [400, 404, 413, 422])
def test_a_payload_the_server_refuses_is_dropped(tmp_path, code):
    lines: list[str] = []
    box = Outbox(tmp_path, log=lines.append)
    for n in range(3):
        box.put({"n": n})
    sent: list[int] = []

    def send(payload):
        if payload["n"] == 0:
            raise _http(code)
        sent.append(payload["n"])

    assert box.drain(send) == (2, 0)
    assert sent == [1, 2]
    assert len(lines) == 1 and f"HTTP {code}" in lines[0] and "dropped" in lines[0]


@pytest.mark.parametrize("code", [401, 403, 408, 429, 500, 502, 503])
def test_a_status_that_is_not_about_the_payload_keeps_it(tmp_path, code):
    box = Outbox(tmp_path)
    for n in range(3):
        box.put({"n": n})

    def send(payload):
        raise _http(code)

    assert box.drain(send) == (0, 3)
    assert box.last_error().startswith(f"HTTP {code}")


def test_drain_handles_at_most_max_items(tmp_path):
    box = Outbox(tmp_path)
    for n in range(30):
        box.put({"n": n})
    assert box.drain(lambda p: None) == (20, 10)
    assert box.drain(lambda p: None, max_items=4) == (4, 6)


def test_an_unreadable_payload_is_dropped(tmp_path):
    lines: list[str] = []
    box = Outbox(tmp_path, log=lines.append)
    bad = box.put({"n": 0})
    bad.write_text("{truncated")
    box.put({"n": 1})
    sent: list[int] = []
    assert box.drain(lambda p: sent.append(p["n"])) == (1, 0)
    assert sent == [1] and "unreadable" in lines[0]


def test_a_missing_directory_drains_nothing(tmp_path):
    box = Outbox(tmp_path / "never-created")
    assert box.drain(lambda p: None) == (0, 0)
    assert (box.pending(), box.oldest_age(), box.last_error()) == (0, None, "")
    assert failure_warning(box) is None


def test_the_last_error_goes_once_the_queue_is_empty(tmp_path):
    box = Outbox(tmp_path)
    box.put({"n": 0}, reason="timed out")
    assert box.last_error() == "timed out"
    box.drain(lambda p: None)
    assert box.last_error() == "" and os.listdir(tmp_path) == []


def test_a_reason_is_stored_as_one_short_line(tmp_path):
    box = Outbox(tmp_path)
    box.put({"n": 0}, reason="line one\nline two " + "x" * 200)
    assert box.last_error().startswith("line one line two x") and len(box.last_error()) == 80


# ---------------------------------------------------------------------------
# Claims: across processes
# ---------------------------------------------------------------------------


def test_a_claimed_payload_is_skipped_and_still_counted(tmp_path):
    box = Outbox(tmp_path)
    first = box.put({"n": 0})
    box.put({"n": 1})
    os.rename(first, tmp_path / f"{first.name}.{time.time_ns()}.claim")  # another drain
    sent: list[int] = []
    assert box.drain(lambda p: sent.append(p["n"])) == (1, 1)
    assert sent == [1] and box.pending() == 1


def test_a_stale_claim_is_put_back_and_sent(tmp_path):
    box = Outbox(tmp_path, stale_claim=60)
    first = box.put({"n": 0})
    old = time.time_ns() - int(120 * 1e9)
    os.rename(first, tmp_path / f"{first.name}.{old}.claim")  # its process died
    sent: list[int] = []
    assert box.drain(lambda p: sent.append(p["n"])) == (1, 0)
    assert sent == [0]


def test_a_dead_writers_temp_file_is_cleaned_up(tmp_path):
    box = Outbox(tmp_path, stale_claim=60)
    tmp = tmp_path / ".00000000000000000001-1-abcd.json.ffff.tmp"
    tmp.write_text("{")
    old = time.time() - 120
    os.utime(tmp, (old, old))
    fresh = tmp_path / ".00000000000000000002-1-abcd.json.eeee.tmp"
    fresh.write_text("{")
    box.drain(lambda p: None)
    assert not tmp.exists() and fresh.exists()
    assert box.pending() == 0


_WORKER = textwrap.dedent(
    """
    import os, sys, time
    sys.path.append(sys.argv[1])
    from synapse_client.outbox import Outbox

    box = Outbox(sys.argv[2])
    log = sys.argv[3]

    def send(payload):
        time.sleep(0.002)
        fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, (str(payload["n"]) + "\\n").encode())
        finally:
            os.close(fd)

    while sys.argv[4] == "go" and box.pending():
        box.drain(send, max_items=7)
    """
)


def test_concurrent_drains_deliver_each_payload_exactly_once(tmp_path):
    box = Outbox(tmp_path / "q")
    total = 120
    for n in range(total):
        box.put({"n": n})
    log = tmp_path / "sent.log"
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, str(PLUGIN), str(box.directory), str(log), "go"]
        )
        for _ in range(4)
    ]
    for proc in procs:
        assert proc.wait(timeout=60) == 0
    sent = [int(line) for line in log.read_text().split()]
    assert sorted(sent) == list(range(total))
    assert box.pending() == 0


# ---------------------------------------------------------------------------
# The session-start warning
# ---------------------------------------------------------------------------


def test_no_warning_while_the_queue_is_short_and_young(tmp_path):
    box = Outbox(tmp_path)
    assert failure_warning(box) is None
    box.put({"n": 0}, reason="timed out")
    box.put({"n": 1})
    assert failure_warning(box) is None


def test_three_waiting_payloads_warn(tmp_path):
    box = Outbox(tmp_path)
    for n in range(3):
        box.put({"n": n}, reason="HTTP 503 Service Unavailable")
    line = failure_warning(box)
    assert line and line.startswith("Synapse uploads have been failing since ")
    assert line.endswith(
        "(HTTP 503 Service Unavailable); transcripts will be retried automatically."
    )


def test_one_payload_older_than_a_day_warns(tmp_path):
    box = Outbox(tmp_path)
    _age(box, box.put({"n": 0}), hours=25)
    assert failure_warning(box) == (
        "Synapse uploads have been failing since "
        + time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 25 * 3600))
        + " (unknown error); transcripts will be retried automatically."
    )


@pytest.fixture()
def scratch(monkeypatch, tmp_path):
    yield scratch_user(monkeypatch, tmp_path)
    drop_scripts()


def test_the_warning_is_the_retry_modules_sentence(scratch, tmp_path):
    retry = load_script("synapse_ingest_retry")
    box = Outbox(tmp_path / "q")
    first = _age(box, box.put({"n": 0}), hours=30)
    box.put({"n": 1}, reason="connection refused")
    now = time.time()
    since = int(first.name[:20]) / 1e9
    state = {
        retry.HEALTH_KEY: {"fail_since": since, "fails": 2, "last_error": "connection refused"}
    }
    assert failure_warning(box, now) == retry.failure_warning(state, now)
    assert outbox_mod.WARN_AFTER_FAILS == retry.WARN_AFTER_FAILS
    assert outbox_mod.WARN_AFTER_HOURS == retry.WARN_AFTER_HOURS


@pytest.mark.parametrize(
    "err",
    [
        _http(503),
        _http(599),
        urllib.error.URLError(TimeoutError()),
        TimeoutError(),
        urllib.error.URLError(ConnectionRefusedError()),
        urllib.error.URLError(socket.gaierror()),
        urllib.error.URLError(ssl.SSLError()),
        ConnectionResetError(),
        urllib.error.URLError(OSError()),
        urllib.error.URLError("no host given"),
        ValueError("bad reply"),
    ],
)
def test_short_reason_matches_the_retry_module(scratch, err):
    retry = load_script("synapse_ingest_retry")
    assert short_reason(err) == retry.short_reason(err)
