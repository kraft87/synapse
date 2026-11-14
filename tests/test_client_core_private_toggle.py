"""The core's private-mode toggle (plugin/synapse_client/private_toggle.py) and the
marker functions it writes with (private.py).

The toggle's behaviour against a server is pinned through the two scripts that wrap it
(tests/test_private_mode.py runs every case for the Claude Code and the Codex copy).
Pinned here is what the wrappers cannot show: the toggle is the client's (its server, its
marker folder, no module state), it names itself as it was told to, and it prints where
it was told to.

No network: the client's transport is replaced by a recorder. Temp directories only.
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.client_core import use_core

use_core()
from synapse_client import private  # noqa: E402
from synapse_client.api import Client  # noqa: E402
from synapse_client.private_toggle import Toggle  # noqa: E402
from synapse_client.settings import Settings  # noqa: E402


class _Rows:
    """In place of the client's Transport: the server's private_sessions rows."""

    def __init__(self) -> None:
        self.rows: set[str] = set()
        self.calls: list[tuple[str, str]] = []
        self.fail: dict[str, BaseException] = {}

    def request_json(self, method: str, path: str, timeout: float = 15) -> Any:
        session_id = path.rsplit("/", 1)[-1]
        self.calls.append((method, session_id))
        if method in self.fail:
            raise self.fail[method]
        if method == "PUT":
            self.rows.add(session_id)
        elif method == "DELETE":
            self.rows.discard(session_id)
        return {"status": "ok", "private": session_id in self.rows}


def _toggle(tmp_path: Path, **kwargs: Any) -> tuple[Toggle, _Rows, io.StringIO, io.StringIO]:
    env = {"SYNAPSE_URL": "http://synapse.test", "SYNAPSE_PRIVATE_DIR": str(tmp_path / "private")}
    client = Client(Settings.load(env=env, home=tmp_path / "home"), "synapse-client/test")
    rows = _Rows()
    client.transport = rows  # type: ignore[assignment]
    out, err = io.StringIO(), io.StringIO()
    return Toggle(client, out=out, err=err, **kwargs), rows, out, err


# ---------------------------------------------------------------------------
# The marker
# ---------------------------------------------------------------------------


def test_mark_writes_the_marker_the_check_reads_and_unmark_removes_it(tmp_path):
    toggle, _, _, _ = _toggle(tmp_path)
    settings = toggle.client.settings
    path = private.mark(settings, "s-1")
    assert path == tmp_path / "private" / "s-1" == private.marker(settings, "s-1")
    assert path.read_text() == "" and private.state(settings, "s-1") == private.PRIVATE
    assert private.unmark(settings, "s-1") is True and not path.exists()
    assert private.state(settings, "s-1") == private.OPEN
    assert private.unmark(settings, "s-1") is True  # none there: nothing to remove


def test_marking_again_starts_the_markers_time_over(tmp_path):
    toggle, _, _, _ = _toggle(tmp_path)
    settings = toggle.client.settings
    path = private.mark(settings, "s-1")
    old = time.time() - 11 * 3600
    os.utime(path, (old, old))
    private.mark(settings, "s-1")
    assert time.time() - path.stat().st_mtime < 60


@pytest.mark.parametrize("session_id", ["", "..", "../x", "a/b", "a\\b", "x\0y"])
def test_no_marker_is_written_or_removed_by_a_name_that_is_not_plain(tmp_path, session_id):
    toggle, _, _, _ = _toggle(tmp_path)
    settings = toggle.client.settings
    outside = tmp_path / "x"
    outside.write_text("keep")
    for act in (private.marker, private.mark, private.unmark):
        with pytest.raises(ValueError):
            act(settings, session_id)
    assert outside.read_text() == "keep" and not (tmp_path / "private").exists()


def test_a_marker_that_will_not_go_is_said_not_raised(tmp_path):
    toggle, _, _, err = _toggle(tmp_path)
    stuck = tmp_path / "private" / "s-1"
    (stuck / "inside").mkdir(parents=True)  # a folder where the marker would be
    assert private.unmark(toggle.client.settings, "s-1") is False
    assert toggle.off("s-1") == 1
    assert "could not remove the marker for 's-1'" in err.getvalue()
    # Said as it is: the marker is what keeps the session's turns on this machine.
    assert "this session stays off the record" in err.getvalue()
    assert "NOT off the record" not in err.getvalue()


# ---------------------------------------------------------------------------
# The toggle is the client's
# ---------------------------------------------------------------------------


def test_on_and_off_use_the_clients_server_and_marker_folder(tmp_path):
    toggle, rows, out, err = _toggle(tmp_path)
    assert toggle.on("s-1") == 0
    assert (tmp_path / "private" / "s-1").is_file() and rows.rows == {"s-1"}
    assert rows.calls == [("PUT", "s-1"), ("GET", "s-1")]  # written, then read back
    assert out.getvalue().splitlines() == [
        "private mode ON for s-1",
        f"  marker: {tmp_path / 'private' / 's-1'}",
        "  server: private_sessions row written (permanent)",
    ]
    assert toggle.off("s-1") == 0
    assert not (tmp_path / "private" / "s-1").exists() and rows.rows == {"s-1"}
    assert toggle.off("s-1", forget=True) == 0 and rows.rows == set()
    assert err.getvalue() == ""


def test_on_leaves_the_marker_when_the_server_cannot_be_told(tmp_path):
    toggle, rows, out, err = _toggle(tmp_path)
    rows.fail["PUT"] = urllib.error.URLError("down")
    assert toggle.on("s-1") == 1
    assert (tmp_path / "private" / "s-1").is_file() and out.getvalue() == ""
    assert "private mode FAILED: server flag not set (URLError" in err.getvalue()
    assert "This session is NOT off the record." in err.getvalue()


def test_the_lines_that_name_a_command_use_the_programs_name(tmp_path):
    toggle, rows, _, err = _toggle(tmp_path, program="synapse private")
    rows.rows.add("s-1")
    rows.fail["DELETE"] = urllib.error.URLError("down")
    rows.fail["GET"] = urllib.error.URLError("down")
    assert toggle.off("s-1", forget=True) == 1
    assert "Check with: synapse private status s-1" in err.getvalue()

    other, _, _, err = _toggle(tmp_path, program="private_mode.py")
    assert other.main([]) == 2
    assert "usage: private_mode.py on|off|status <session_id> [--forget]" in err.getvalue()


def test_main_runs_the_action_named(tmp_path):
    toggle, rows, out, err = _toggle(tmp_path)
    assert toggle.main(["on", " s-1 "]) == 0 and rows.rows == {"s-1"}
    assert toggle.main(["status", "s-1"]) == 0
    assert out.getvalue().splitlines()[-1] == "s-1: marker=yes server=yes"
    assert toggle.main(["off", "s-1", "--forget"]) == 0 and rows.rows == set()
    assert toggle.main(["status", "s-1"]) == 0
    assert out.getvalue().splitlines()[-1] == "s-1: marker=no server=no"
    assert toggle.main(["sideways", "s-1"]) == 2 and toggle.main(["on"]) == 2
    assert toggle.main(["on", "../s-1"]) == 1 and rows.rows == set()
    assert err.getvalue().splitlines()[-1].startswith("This session is NOT off the record.")


def test_forget_is_read_wherever_it_stands(tmp_path):
    toggle, rows, _, err = _toggle(tmp_path)
    assert toggle.main(["on", "s-1"]) == 0 and toggle.main(["on", "s-2"]) == 0
    assert toggle.main(["off", "--forget", "s-1"]) == 0 and rows.rows == {"s-2"}
    assert toggle.main(["off", "s-2", "--forget"]) == 0 and rows.rows == set()
    assert toggle.main(["off", "--forget"]) == 2 and err.getvalue().count("usage:") == 1
    assert rows.calls[-4:] == [("DELETE", "s-1"), ("GET", "s-1"), ("DELETE", "s-2"), ("GET", "s-2")]


def test_a_refused_off_or_status_says_nothing_was_changed(tmp_path):
    """Only a failed `on` leaves a session on the record that was meant to be off it.
    Of an `off` that was not run, "NOT off the record" would be false for a private one."""
    toggle, rows, _, err = _toggle(tmp_path)
    assert toggle.main(["on", "s-1"]) == 0
    calls = list(rows.calls)
    for argv in (["off", "--forgte", "s-1"], ["off", "../s-1"], ["status", "-v", "s-1"]):
        assert toggle.main(argv) == 1
    lines = err.getvalue().splitlines()
    assert lines[1::2] == ["Nothing was changed."] * 3
    assert "looks like a flag" in lines[0] and "invalid session id" in lines[2]
    assert rows.calls == calls and (tmp_path / "private" / "s-1").is_file()
    assert toggle.off("../s-1") == 1  # past main's own check of the id
    assert err.getvalue().splitlines()[-1] == "Nothing was changed."


def test_a_session_end_removes_the_marker_and_never_fails(tmp_path):
    toggle, rows, _, err = _toggle(tmp_path)
    toggle.on("s-1")
    calls = list(rows.calls)
    assert toggle.main(["--session-end"], io.StringIO(json.dumps({"session_id": "s-1"}))) == 0
    assert not (tmp_path / "private" / "s-1").exists()
    assert rows.rows == {"s-1"} and rows.calls == calls  # the row stays; no request
    for payload in ("not json", "[]", json.dumps({"session_id": "../x"}), json.dumps({})):
        assert toggle.session_end(io.StringIO(payload)) == 0
    assert err.getvalue() == ""
