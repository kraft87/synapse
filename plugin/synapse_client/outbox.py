"""A durable queue for uploads that failed, safe across processes.

One file per payload in one directory, named ``<time_ns>-<pid>-<random>.json`` so that
names sort oldest first. A file is written to a dot-prefixed temp name and renamed into
place, so a reader never sees half a payload; files and the directory are owner-only on
POSIX.

Draining claims each file by renaming it to ``<name>.<claim_ns>.claim``: of several
processes draining at once, exactly one rename succeeds. A claim older than
``stale_claim`` seconds belongs to a process that died mid-send and is put back. No
locks and no fcntl, so it works the same on Windows.

``put`` RAISES OSError (the caller must know a payload was lost). ``drain``,
``pending``, ``oldest_age`` and ``failure_warning`` never raise over a missing or
unreadable directory, and ``drain`` never raises over a failed send: it stops there.
"""

from __future__ import annotations

import http
import http.client
import json
import os
import secrets
import socket
import ssl
import time
import urllib.error
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_CAP = 500
DEFAULT_STALE_CLAIM = 600.0
WARN_AFTER_FAILS = 3  # payloads waiting -> warn at session start
WARN_AFTER_HOURS = 24.0  # ...or an oldest payload this old
#: 4xx statuses that say nothing about the payload: the credential, a timeout, a rate
#: limit. Every other 4xx means the server will never take this payload, so it is dropped.
_RETRY_4XX = frozenset({401, 403, 408, 429})
_SUFFIX = ".json"
_CLAIM = ".claim"
_LAST_ERROR = ".last_error"

Log = Callable[[str], None]


def short_reason(err: BaseException) -> str:
    """A short reason for a failed upload, from a fixed vocabulary: built from the
    exception type and HTTP status only, never from message text, which can echo URLs,
    headers or response bodies. A port of synapse_ingest_retry.short_reason."""
    if isinstance(err, urllib.error.HTTPError):
        try:
            return f"HTTP {err.code} {http.HTTPStatus(err.code).phrase}"
        except ValueError:
            return f"HTTP {err.code}"
    cause = err.reason if isinstance(err, urllib.error.URLError) else err
    if isinstance(cause, TimeoutError):
        return "timed out"
    if isinstance(cause, ConnectionRefusedError):
        return "connection refused"
    if isinstance(cause, socket.gaierror):
        return "DNS lookup failed"
    if isinstance(cause, (ssl.SSLError, ssl.CertificateError)):
        return "TLS error"
    if isinstance(cause, ConnectionError):
        return "connection dropped"
    if isinstance(cause, http.client.HTTPException):
        return "bad HTTP response"
    if isinstance(cause, OSError):
        return "network error"
    if cause is not err:
        return "server unreachable"
    return type(err).__name__


def _one_line(text: str) -> str:
    return " ".join(str(text).split())[:80]


def _stamp(name: str) -> int | None:
    """The enqueue time (ns) a payload or claim name starts with."""
    head = name.split("-", 1)[0]
    return int(head) if head.isdigit() else None


class Outbox:
    """The queue in ``directory`` (created on the first ``put``).

    ``cap`` bounds the payloads kept: past it, the oldest waiting ones are dropped.
    ``log`` receives a line for each payload dropped (cap, rejected, unreadable)."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        cap: int = DEFAULT_CAP,
        stale_claim: float = DEFAULT_STALE_CLAIM,
        log: Log | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.cap = cap
        self.stale_claim = stale_claim
        self._log = log

    def __repr__(self) -> str:
        return f"Outbox({str(self.directory)!r})"

    def _say(self, line: str) -> None:
        if self._log is not None:
            try:
                self._log(line)
            except Exception:
                pass

    # -- listing -------------------------------------------------------------------

    def _names(self) -> list[str]:
        try:
            return sorted(n for n in os.listdir(self.directory) if not n.startswith("."))
        except OSError:
            return []

    def _waiting(self) -> list[str]:
        return [n for n in self._names() if n.endswith(_SUFFIX)]

    def pending(self) -> int:
        """Payloads not yet delivered: waiting, plus those a drain holds right now."""
        return sum(1 for n in self._names() if n.endswith((_SUFFIX, _CLAIM)))

    def oldest_age(self, now: float | None = None) -> float | None:
        """Seconds since the oldest undelivered payload was queued; None when empty."""
        stamps = [s for n in self._names() if (s := _stamp(n)) is not None]
        if not stamps:
            return None
        return max(0.0, (time.time() if now is None else now) - min(stamps) / 1e9)

    def last_error(self) -> str:
        """The reason the most recent upload failed, as recorded; "" when none is."""
        try:
            return _one_line((self.directory / _LAST_ERROR).read_text(encoding="utf-8"))
        except OSError:
            return ""

    def _set_last_error(self, reason: str) -> None:
        try:
            self._write(self.directory / _LAST_ERROR, _one_line(reason).encode("utf-8"))
        except OSError:
            pass

    # -- writing -------------------------------------------------------------------

    def _ensure_dir(self) -> None:
        if not self.directory.is_dir():
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            if os.name != "nt":
                os.chmod(self.directory, 0o700)

    def _write(self, path: Path, data: bytes) -> None:
        """``data`` to ``path`` atomically: a dot-prefixed temp file, then a rename."""
        tmp = path.parent / f".{path.name}.{secrets.token_hex(4)}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def put(self, payload: Mapping[str, Any], reason: str = "") -> Path:
        """Queue ``payload`` (a JSON object) and return its file. ``reason`` is why it is
        being queued (``short_reason`` of the failed send), kept for ``failure_warning``;
        it must not carry token material. Past the cap, the oldest waiting payloads are
        dropped. RAISES OSError when the payload could not be written, TypeError or
        ValueError when it is not JSON-serialisable."""
        data = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        self._ensure_dir()
        name = f"{time.time_ns():020d}-{os.getpid()}-{secrets.token_hex(4)}{_SUFFIX}"
        path = self.directory / name
        self._write(path, data)
        if reason:
            self._set_last_error(reason)
        self._enforce_cap()
        return path

    def _enforce_cap(self) -> None:
        waiting = self._waiting()
        excess = self.pending() - self.cap
        for name in waiting[: max(0, excess)]:
            try:
                os.unlink(self.directory / name)
            except OSError:
                continue
            self._say(f"outbox full ({self.cap}): dropped the oldest payload {name}")

    # -- draining ------------------------------------------------------------------

    def _recover(self, now_ns: int) -> None:
        """Put back claims, and remove temp files, left by a process that died."""
        limit = int(self.stale_claim * 1e9)
        try:
            names = os.listdir(self.directory)
        except OSError:
            return
        for name in names:
            path = self.directory / name
            if name.endswith(_CLAIM) and not name.startswith("."):
                base, _, claimed = name[: -len(_CLAIM)].rpartition(".")
                if claimed.isdigit() and now_ns - int(claimed) > limit:
                    try:
                        os.rename(path, self.directory / base)
                    except OSError:
                        pass
            elif name.startswith(".") and name.endswith(".tmp"):
                try:
                    if now_ns - path.stat().st_mtime_ns > limit:
                        os.unlink(path)
                except OSError:
                    pass

    def _claim(self, name: str) -> Path | None:
        claimed = self.directory / f"{name}.{time.time_ns()}{_CLAIM}"
        try:
            os.rename(self.directory / name, claimed)
        except OSError:  # another drain took it, or (Windows) it is open elsewhere
            return None
        return claimed

    def _release(self, claimed: Path, name: str) -> None:
        try:
            os.rename(claimed, self.directory / name)
        except OSError:
            pass  # left as a claim: recovered once stale

    def _drop(self, claimed: Path, why: str) -> None:
        try:
            os.unlink(claimed)
        except OSError:
            pass
        self._say(why)

    def drain(self, send: Callable[[Any], object], max_items: int = 20) -> tuple[int, int]:
        """Send up to ``max_items`` payloads, oldest first, through ``send`` (which
        raises on failure: ``Client.ingest``). Returns ``(sent, kept)``: how many were
        delivered, and how many are still waiting.

        A delivered payload is deleted. One the server refused with a 4xx other than
        401, 403, 408 or 429 is dropped (and logged): it will never be accepted, and
        keeping it would block everything behind it. Any other failure puts the payload
        back, records its reason and STOPS, since the next send would fail the same way.
        An unreadable payload file is dropped. Never raises over a failed send."""
        self._recover(time.time_ns())
        sent = done = 0
        for name in self._waiting():
            if done >= max_items:
                break
            claimed = self._claim(name)
            if claimed is None:
                continue
            done += 1
            try:
                payload = json.loads(claimed.read_text(encoding="utf-8"))
            except (OSError, ValueError) as err:
                self._drop(
                    claimed, f"outbox: dropped unreadable payload {name} ({type(err).__name__})"
                )
                continue
            try:
                send(payload)
            except urllib.error.HTTPError as err:
                if 400 <= err.code < 500 and err.code not in _RETRY_4XX:
                    self._drop(
                        claimed, f"outbox: server refused {name}: {short_reason(err)}; dropped"
                    )
                    continue
                self._release(claimed, name)
                self._set_last_error(short_reason(err))
                break
            except Exception as err:
                self._release(claimed, name)
                self._set_last_error(short_reason(err))
                break
            try:
                os.unlink(claimed)
            except OSError:
                pass
            sent += 1
        kept = self.pending()
        if not kept:
            try:
                os.unlink(self.directory / _LAST_ERROR)
            except OSError:
                pass
        return sent, kept


def failure_warning(outbox: Outbox, now: float | None = None) -> str | None:
    """The session-start line for uploads that keep failing, or None.

    Shown when at least WARN_AFTER_FAILS payloads are waiting, or the oldest has waited
    WARN_AFTER_HOURS; the same sentence as synapse_ingest_retry.failure_warning, dated
    from the oldest waiting payload. Never raises."""
    try:
        now = time.time() if now is None else now
        count = outbox.pending()
        age = outbox.oldest_age(now)
        if not count or age is None:
            return None
        if count < WARN_AFTER_FAILS and age < WARN_AFTER_HOURS * 3600:
            return None
        when = datetime.fromtimestamp(now - age).strftime("%Y-%m-%d %H:%M")
        reason = outbox.last_error() or "unknown error"
        return (
            f"Synapse uploads have been failing since {when} ({reason}); "
            "transcripts will be retried automatically."
        )
    except Exception:
        return None
