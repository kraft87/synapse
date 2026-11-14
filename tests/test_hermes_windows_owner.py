"""The owner subject on the machine itself: what the OS and Hermes say this user is.

``harnesses/hermes/sessions.py`` serves a gateway session only when its ``user_id`` is
this machine's owner as Hermes names it: ``uid:<n>`` on POSIX, ``sid:<sid>`` on Windows
(``harnesses/hermes/winsid.py``). The unit tests in test_hermes_gateway_sessions.py fake
the lookup; these run the real one.

- On Windows, the lookup is checked against ``whoami /user``.
- With ``SYNAPSE_HERMES_SOURCE`` naming a Hermes source tree (no virtualenv needed: the
  modules read are stdlib only), a driver imports Hermes's own identity code under this
  interpreter and prints the subjects it mints: ``owner_worker_subject`` and
  ``host_owner_subject`` (what every dashboard login acts as), and on Windows the
  subject the control pipe mints for a same-user peer (``NativeControlServer`` and
  ``query_runtime_control``, the path ``hermes chat`` and the TUI take for their
  ticket). Each must equal what the client computes. The source tree is only read
  (``-B``) and the driver runs with a temp HOME and HERMES_HOME.
"""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.helpers.client_core import PLUGIN, use_core

use_core()
from synapse_client.harnesses.hermes import sessions as hs  # noqa: E402
from synapse_client.harnesses.hermes import winsid  # noqa: E402

_SOURCE = os.environ.get("SYNAPSE_HERMES_SOURCE", "")
_WINDOWS = os.name == "nt"


@pytest.mark.skipif(not _WINDOWS, reason="Windows only")
def test_the_lookup_is_the_sid_whoami_names():
    done = subprocess.run(
        ["whoami", "/user", "/fo", "csv", "/nh"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    row = next(csv.reader(io.StringIO(done.stdout.strip())))
    sid = row[-1].strip()
    assert sid.startswith("S-1-"), done.stdout
    assert winsid.process_sid() == sid
    assert hs.owner_subject() == f"sid:{sid}"


@pytest.mark.skipif(not _WINDOWS, reason="Windows only")
def test_on_windows_there_is_no_uid_to_fall_back_on():
    """The SID is the only owner on Windows: Python has no ``os.getuid`` there."""
    assert not hasattr(os, "getuid")
    assert winsid.WINDOWS


@pytest.mark.skipif(_WINDOWS, reason="POSIX only")
def test_off_windows_the_lookup_answers_nothing():
    assert winsid.process_sid() is None
    assert hs.owner_subject() == f"uid:{os.getuid()}"


_DRIVER = textwrap.dedent(
    """
    import json, os, sys
    from pathlib import Path

    sys.path[:0] = [sys.argv[1], sys.argv[2]]
    from gateway.owner_identity import host_owner_subject
    from gateway.session_worker_reservation import owner_worker_subject
    from synapse_client.harnesses.hermes.sessions import owner_subject

    found = {"synapse": owner_subject(), "worker": owner_worker_subject(),
             "host": host_owner_subject()}
    if os.name == "nt":
        from gateway.runtime_bootstrap_windows import NativeControlServer, query_runtime_control

        def handler(raw, subject):
            return json.dumps({"subject": subject}).encode() + b"\\n"

        server = NativeControlServer(Path(sys.argv[3]), handler)
        server.start()
        try:
            answer = query_runtime_control(Path(sys.argv[3]), b"hello", 10)
        finally:
            server.close()
        found["pipe"] = json.loads(answer)["subject"]
    print(json.dumps(found))
    """
)


@pytest.mark.skipif(not _SOURCE, reason="set SYNAPSE_HERMES_SOURCE to a Hermes source tree")
def test_hermes_mints_the_subject_the_client_computes(tmp_path):
    source = Path(os.path.expanduser(_SOURCE)).resolve()
    if not (source / "gateway" / "session_worker_reservation.py").is_file():
        pytest.fail(f"{source} is not a Hermes source tree")
    home = tmp_path / "hermes"
    home.mkdir()
    scratch = str(tmp_path / "user")
    env = {
        **os.environ,
        "HOME": scratch,
        "USERPROFILE": scratch,
        "LOCALAPPDATA": scratch,
        "APPDATA": scratch,
        "HERMES_HOME": str(home),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    done = subprocess.run(
        [sys.executable, "-I", "-B", "-c", _DRIVER, str(source), str(PLUGIN), str(home)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=str(tmp_path),
        check=False,
    )
    assert done.returncode == 0, done.stderr
    found = json.loads(done.stdout.strip().splitlines()[-1])
    expected = hs.owner_subject()
    assert expected is not None
    assert found["synapse"] == expected
    assert found["worker"] == expected
    assert found["host"] == expected
    if _WINDOWS:
        assert expected.startswith("sid:S-1-")
        assert found["pipe"] == expected
    else:
        assert "pipe" not in found
