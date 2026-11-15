"""What a harness has shipped, per session, kept on disk.

A harness that reads its conversation from its own store (Hermes's ``state.db``) ships
each stored row once. ``Ledger`` remembers which: per session id, the ids of the rows
whose turns went to the server or to the outbox (``uids``), the keys of turns whose
rows carry no id (``keys``, messages.turn_key), and the ids of rows the live path chose
not to look back at the first time it met the session (``passed``): not shipped, so
``synapse import hermes`` still sends them, and not looked at again by the live path. It outlives the process, so a restart,
an agent rebuilt mid-session (Hermes does on a model switch) and an import all see what
was shipped before.

One small JSON file per session, ``<directory>/<sha256 of the session id>.json``: the
id may hold characters no file name can. A change is a read, a merge and a write, under
an exclusive lock on ``<name>.lock`` (``fcntl`` on POSIX, ``msvcrt`` on Windows) and
through a dot-prefixed temp file renamed into place, so two processes never lose each
other's additions and a reader never sees half a file. Nothing is ever taken out: a
row shipped stays shipped.

A file that cannot be read counts as nothing shipped for that session, said once to
``log``; the caller's look-back then applies, and the server's span dedup absorbs a
turn sent again. ``add`` RAISES OSError when it cannot write: the caller must know.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import IO, Any, NamedTuple


def _lock(f: IO[Any]) -> None:
    """An exclusive lock on ``f``'s first byte: the whole lock file (msvcrt locks byte
    ranges; LK_LOCK retries for about ten seconds, then raises OSError)."""
    if os.name == "nt":
        import msvcrt

        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined,unused-ignore]
    else:
        import fcntl

        fcntl.flock(f, fcntl.LOCK_EX)


def _unlock(f: IO[Any]) -> None:
    if os.name == "nt":
        import msvcrt

        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined,unused-ignore]
    else:
        import fcntl

        fcntl.flock(f, fcntl.LOCK_UN)


Log = Callable[[str], None]
_VERSION = 1
#: How often a rename onto a file another process holds open is tried (Windows).
_REPLACE_TRIES = 20


class Shipped(NamedTuple):
    """What was shipped for one session: row ids, keys of turns with no ids, and row
    ids passed over by a first look-back. ``known`` is False when there is no record of
    the session at all."""

    uids: frozenset[str]
    keys: frozenset[str]
    known: bool
    passed: frozenset[str] = frozenset()


def nothing() -> Shipped:
    """The record of a session nothing is known of."""
    return Shipped(frozenset(), frozenset(), False)


def _strings(value: Any) -> frozenset[str]:
    return frozenset(v for v in value or () if isinstance(v, str) and v)


class Ledger:
    """The rows shipped per session, under ``directory``. Safe across threads and
    processes."""

    def __init__(self, directory: str | Path, *, log: Log | None = None) -> None:
        self.directory = Path(directory)
        self._log = log
        self._said: set[str] = set()

    def _say(self, line: str) -> None:
        if self._log is None or line in self._said:
            return
        self._said.add(line)
        try:
            self._log(line)
        except Exception:
            pass

    def _file(self, session_id: str) -> Path:
        name = hashlib.sha256(session_id.encode("utf-8", "surrogatepass")).hexdigest()[:40]
        return self.directory / f"{name}.json"

    def _open_lock(self, file: Path) -> IO[Any]:
        """``file``'s lock file, open and locked: ``_close_lock`` it."""
        self.directory.mkdir(parents=True, exist_ok=True)
        handle = open(file.with_name(file.name + ".lock"), "a+b")
        try:
            _lock(handle)
        except BaseException:
            handle.close()
            raise
        return handle

    @staticmethod
    def _close_lock(handle: IO[Any]) -> None:
        try:
            _unlock(handle)
        finally:
            handle.close()

    def _read(self, file: Path, session_id: str) -> Shipped:
        try:
            raw = file.read_bytes()
        except FileNotFoundError:
            return nothing()
        except OSError as err:
            self._say(f"synapse: the shipped record of {session_id} could not be read ({err})")
            return nothing()
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:
            self._say(f"synapse: the shipped record of {session_id} is damaged; starting over")
            return nothing()
        if not isinstance(data, dict) or data.get("session_id") != session_id:
            return nothing()
        return Shipped(
            _strings(data.get("uids")),
            _strings(data.get("keys")),
            True,
            _strings(data.get("passed")),
        )

    def load(self, session_id: str) -> Shipped:
        """What was shipped for ``session_id``. Never raises."""
        if not session_id or not self.directory.is_dir():
            return nothing()
        file = self._file(session_id)
        try:
            handle = self._open_lock(file)
            try:
                return self._read(file, session_id)
            finally:
                self._close_lock(handle)
        except OSError as err:
            self._say(f"synapse: the shipped record of {session_id} could not be read ({err})")
            return nothing()

    def add(
        self,
        session_id: str,
        uids: Iterable[str] = (),
        keys: Iterable[str] = (),
        passed: Iterable[str] = (),
    ) -> Shipped:
        """Record ``uids`` and ``keys`` as shipped for ``session_id``, and ``passed`` as
        passed over, keeping what is there; returns the record now. A session with
        nothing added is recorded too, as known. RAISES OSError when the record cannot
        be written."""
        file = self._file(session_id)
        handle = self._open_lock(file)
        try:
            now = self._read(file, session_id)
            merged = Shipped(
                now.uids | _strings(uids),
                now.keys | _strings(keys),
                True,
                now.passed | _strings(passed),
            )
            if now.known and merged == now:
                return now
            body = {
                "version": _VERSION,
                "session_id": session_id,
                "uids": sorted(merged.uids),
                "keys": sorted(merged.keys),
                "passed": sorted(merged.passed),
            }
            self._write(file, json.dumps(body, ensure_ascii=False).encode("utf-8"))
            return merged
        finally:
            self._close_lock(handle)

    def _write(self, file: Path, data: bytes) -> None:
        tmp = file.with_name(f".{file.name}.{secrets.token_hex(4)}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            for attempt in range(_REPLACE_TRIES):
                try:
                    os.replace(tmp, file)
                    break
                except PermissionError:
                    # Windows: a reader outside the lock has the file open for a moment.
                    if attempt == _REPLACE_TRIES - 1:
                        raise
                    time.sleep(0.05)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
