"""What a harness has shipped, per session, kept on disk.

A harness that reads its conversation from its own store (Hermes's ``state.db``) ships
each stored row once. ``Ledger`` remembers which: per session id, the ids of the rows
whose turns went to the server or to the outbox (``uids``), the keys of turns whose
rows carry no id (``keys``, messages.turn_key), and the ids of rows the live path chose
not to look back at the first time it met the session (``passed``): not shipped, so
``synapse import hermes`` still sends them, and not looked at again by the live path. It outlives the process, so a restart,
an agent rebuilt mid-session (Hermes does on a model switch) and an import all see what
was shipped before.

One small JSON file per session, ``<directory>/<sha256 of the session id>.json`` (the
Hermes provider's directory is ``<synapse home>/shipped/hermes/<server key>``: one
ledger per server, ``server_key``): the id may hold
characters no file name can. A record a newer client wrote (a higher ``version``) is
read as far as this one understands it and never rewritten; a damaged one is moved
aside (``.damaged``) and counts as nothing shipped. ``forget`` removes a session's
record: ``synapse private off --forget`` does, for every ledger under the Synapse home. A change is a read, a merge and a write, under
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
    #: How many times the record was cleared (``forget``): a process that read it before
    #: knows what it remembers is gone.
    cleared: int = 0
    #: False when the record could not be read: what was shipped is unknown, and a
    #: caller must change nothing it remembers.
    readable: bool = True


def _body(session_id: str, record: Shipped) -> bytes:
    body = {
        "version": _VERSION,
        "session_id": session_id,
        "uids": sorted(record.uids),
        "keys": sorted(record.keys),
        "passed": sorted(record.passed),
        "cleared": record.cleared,
    }
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


def unreadable() -> Shipped:
    """The record of a session that could not be read."""
    return Shipped(frozenset(), frozenset(), False, readable=False)


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

    def _make_directory(self) -> None:
        """The ledger's directory and the ones above it it creates, owner-only where
        the platform has modes: a name derived from the server's URL and token is not
        anyone else's business."""
        missing = []
        path = self.directory
        while not path.exists() and path != path.parent:
            missing.append(path)
            path = path.parent
        for one in reversed(missing):
            try:
                one.mkdir(mode=0o700)
            except FileExistsError:
                continue
            if os.name != "nt":
                os.chmod(one, 0o700)

    def _open_lock(self, file: Path) -> IO[Any]:
        """``file``'s lock file, open and locked: ``_close_lock`` it."""
        self._make_directory()
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

    def _read(self, file: Path, session_id: str) -> tuple[Shipped, bool]:
        """The record in ``file`` and whether this client may write over it: not one a
        newer client wrote (it is read as far as this client understands it, and left
        as it is). A damaged record is moved aside (``<name>.damaged``) and counts as
        nothing shipped."""
        try:
            raw = file.read_bytes()
        except FileNotFoundError:
            return nothing(), True
        except OSError as err:
            self._say(f"synapse: the shipped record of {session_id} could not be read ({err})")
            return unreadable(), False
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:
            data = None
        if not isinstance(data, dict) or not isinstance(data.get("version"), int):
            self._say(f"synapse: the shipped record of {session_id} is damaged; it was moved aside")
            try:
                os.replace(file, file.with_name(file.name + ".damaged"))
            except OSError:
                return nothing(), False
            return nothing(), True
        if data.get("session_id") != session_id:
            return nothing(), False
        cleared = data.get("cleared")
        record = Shipped(
            _strings(data.get("uids")),
            _strings(data.get("keys")),
            True,
            _strings(data.get("passed")),
            cleared if isinstance(cleared, int) and not isinstance(cleared, bool) else 0,
        )
        if data["version"] > _VERSION:
            self._say(
                f"synapse: the shipped record of {session_id} was written by a newer client; "
                "it is read, not changed"
            )
            return record, False
        return record, True

    def load(self, session_id: str) -> Shipped:
        """What was shipped for ``session_id``. Never raises."""
        if not session_id or not self.directory.is_dir():
            return nothing()
        file = self._file(session_id)
        try:
            handle = self._open_lock(file)
            try:
                return self._read(file, session_id)[0]
            finally:
                self._close_lock(handle)
        except OSError as err:
            self._say(f"synapse: the shipped record of {session_id} could not be read ({err})")
            return unreadable()

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
            now, writable = self._read(file, session_id)
            if not now.readable:
                raise OSError(f"the shipped record of {session_id} could not be read")
            merged = Shipped(
                now.uids | _strings(uids),
                now.keys | _strings(keys),
                True,
                now.passed | _strings(passed),
                now.cleared,
            )
            if (now.known and merged == now) or not writable:
                return merged
            self._write(file, _body(session_id, merged))
            return merged
        finally:
            self._close_lock(handle)

    def forget(self, session_id: str) -> None:
        """Forget what was shipped for ``session_id``: ``synapse private off --forget``
        put it back on the record, and the turns the server dropped while it was private
        must go again. The record stays, empty, and notes that it was cleared
        (``cleared``), so a running provider knows. RAISES OSError when the record exists
        and cannot be written."""
        if not session_id or not self.directory.is_dir():
            return
        file = self._file(session_id)
        if not file.exists():
            return
        handle = self._open_lock(file)
        try:
            now, writable = self._read(file, session_id)
            if not writable:
                raise OSError(f"the shipped record of {session_id} cannot be changed")
            cleared = Shipped(frozenset(), frozenset(), True, frozenset(), now.cleared + 1)
            self._write(file, _body(session_id, cleared))
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


def server_key(settings: Any) -> str:
    """Which server a ledger belongs to: the configured URL and this machine's token.
    A new or rebuilt server needs a new token (the old one is not in its database), so
    it gets a ledger of its own, and what was shipped to the old one is sent again
    (within the look-back, as the released client did). A server restored from a backup
    at the same address keeps the old token: no route of the server says which episodes
    it holds or which database it is, so that case is not told apart (DESIGN.md)."""
    seed = f"{getattr(settings, 'base_url', '') or ''}\n{getattr(settings, 'token', '') or ''}"
    return hashlib.sha256(seed.encode("utf-8", "surrogatepass")).hexdigest()[:20]


def for_server(settings: Any, harness: str, *, log: Log | None = None) -> Ledger:
    """The ledger of ``harness`` for the server ``settings`` name, under the Synapse home."""
    return Ledger(Path(settings.home) / "shipped" / harness / server_key(settings), log=log)


def directories(settings: Any) -> list[Path]:
    """Every ledger directory under the Synapse home of ``settings``."""
    root = Path(settings.home) / "shipped"
    try:
        return sorted({p.parent for p in root.rglob("*.json*")})
    except OSError:
        return []


def forget_everywhere(settings: Any, session_id: str) -> list[tuple[Path, OSError]]:
    """``Ledger.forget`` in every ledger under the Synapse home, each on its own: one
    that cannot be cleared does not stop the others. Returns the ones that failed."""
    failed: list[tuple[Path, OSError]] = []
    for directory in directories(settings):
        try:
            Ledger(directory).forget(session_id)
        except OSError as err:
            failed.append((directory, err))
    return failed
