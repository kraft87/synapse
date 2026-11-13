"""The outbox's directories are owner-only on POSIX (plugin/synapse_client/outbox.py):
the queue's own, and every parent it creates on the way (a Hermes home's ``synapse``
folder), whatever the umask. Parents that already exist keep their mode.
"""

from __future__ import annotations

import os
import stat
import sys

import pytest

from tests.helpers.client_core import use_core

use_core()
from synapse_client.outbox import Outbox  # noqa: E402

posix = pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture()
def open_umask():
    old = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(old)


@posix
@pytest.mark.parametrize("ahead", [False, True])
def test_every_directory_the_outbox_creates_is_owner_only(tmp_path, open_umask, ahead):
    home = tmp_path / "hermes-home"
    home.mkdir(mode=0o755)
    os.chmod(home, 0o755)
    box = Outbox(home / "synapse" / "outbox")
    box.put({"n": 1}, sending=60 if ahead else 0)
    assert _mode(home / "synapse") == 0o700
    assert _mode(home / "synapse" / "outbox") == 0o700
    assert _mode(home) == 0o755  # not the outbox's to change


@posix
def test_a_directory_made_meanwhile_is_left_as_it_is(tmp_path, open_umask, monkeypatch):
    box = Outbox(tmp_path / "synapse" / "outbox")
    real = type(tmp_path).mkdir

    def racing(self, mode=0o777, parents=False, exist_ok=False):
        if self.name == "outbox":
            real(self, 0o755)  # another process got there first
            raise FileExistsError(str(self))
        return real(self, mode, parents, exist_ok)

    monkeypatch.setattr(type(tmp_path), "mkdir", racing)
    box.put({"n": 1})
    assert box.pending() == 1


def test_a_file_where_the_directory_should_be_fails_the_put(tmp_path):
    (tmp_path / "synapse").write_text("not a directory", encoding="utf-8")
    with pytest.raises(OSError):
        Outbox(tmp_path / "synapse" / "outbox").put({"n": 1})
