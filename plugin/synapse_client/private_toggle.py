"""The private-mode toggle: take a session off the record, and what follows from it.

    on  <session_id>            nothing from this session is ingested, ever
    off <session_id>            stop skipping locally (the server row stays)
    off <session_id> --forget   also delete the server row (see below); the flag
                                counts before the id too
    status <session_id>
    --session-end               a SessionEnd hook: reads the payload on stdin

Two writes, both required, because they cover different failures:

  1. a marker file at ``<private_dir>/<session_id>`` (private.mark). Every uploader
     reads it before a request, so the turns never leave this machine, even with the
     server down;
  2. a row in ``private_sessions`` on the server, the durable half. The transcript stays
     on disk long after the marker expires, so without the row a later catch-up sweep or
     import would ingest exactly what was skipped.

``on`` checks that BOTH landed (the server flag is read back, not assumed) and answers
nonzero, loudly, if either did not. Only 0 means "off the record" is true.

``off`` removes the marker and LEAVES the row. Turning private mode off does not make
the turns already spoken ingestable: they are still on disk and the row is the only
thing keeping them out. A session id is never reused, so the row also keeps the rest of
that session private. ``--forget`` is the explicit way to delete the row, which exposes
the session to later imports again; a session-end step never does that.

Every harness's toggle is this one: ``synapse private`` (the CLI), and the
``private_mode.py`` scripts of the Claude Code and Codex plugins, which only build the
client. A harness that continues a conversation under a new session id carries private
mode over with upload.Uploader.carry_private.
"""

from __future__ import annotations

import json
import sys
import urllib.error
from collections.abc import Sequence
from typing import Any, TextIO

from . import private
from .api import Client

#: What a request to the server raises: no answer, an HTTP error, or a reply that is not
#: the JSON object the route returns.
_FAILED = (urllib.error.URLError, OSError, ValueError)
_USAGE = "on|off|status <session_id> [--forget]"
_NOT_PRIVATE = "This session is NOT off the record. Do not tell the user it is."
_UNCHANGED = "Nothing was changed."


def _why(err: BaseException) -> str:
    return f"{type(err).__name__}: {str(err)[:160]}"


class Toggle:
    """The toggle for one server and one marker folder (``client.settings``).
    ``program`` is how the user runs it, for the lines that name a command to run next
    (``private_mode.py``, ``synapse private``). Each action prints what happened and
    returns the exit code."""

    def __init__(
        self,
        client: Client,
        *,
        program: str = "synapse private",
        out: TextIO | None = None,
        err: TextIO | None = None,
    ) -> None:
        self.client = client
        self.program = program
        self._out = out
        self._err = err

    # -- output --------------------------------------------------------------------

    def _say(self, line: str) -> None:
        print(line, file=self._out or sys.stdout)

    def _warn(self, line: str) -> None:
        print(line, file=self._err or sys.stderr)

    def _fail(self, message: str, closing: str = _NOT_PRIVATE) -> int:
        """Say what failed and where that leaves the session. Only a failed ``on`` leaves
        one that was meant to be off the record on it."""
        self._warn(f"private mode FAILED: {message}")
        self._warn(closing)
        return 1

    def _marker_removed_note(self) -> None:
        """Last line of every failed --forget. ``off`` removes the marker before it sends
        the DELETE, so a --forget that does not finish leaves the session as a plain
        ``off`` does: its turns are sent again, and the server drops them while the row
        is there."""
        self._warn(
            "The local marker was removed. While the server row is kept, this session is in "
            "the same state as after a plain `off`: its turns are sent and the server "
            "discards them."
        )

    def _forget_failed(
        self,
        message: str,
        closing: str = "The server row was not deleted; the session stays off the record.",
    ) -> int:
        """A --forget that left the server row in place: the session stays off the
        record, so ``_fail``'s closing line would say the opposite. ``closing`` is
        replaced when the DELETE may still be in flight and the row's fate is open."""
        self._warn(f"private mode FAILED: {message}")
        self._warn(closing)
        self._marker_removed_note()
        return 1

    # -- the server half -----------------------------------------------------------

    def _server_private(self, session_id: str) -> bool:
        """The durable flag, read back from the server. Raises ``_FAILED``."""
        return bool(self.client.private_get(session_id).get("private"))

    # -- actions -------------------------------------------------------------------

    def on(self, session_id: str) -> int:
        # Marker first: it is the half that works with no network, and it protects the
        # very next turn. If the server write then fails that is said loudly, rather
        # than relying on a marker that expires.
        try:
            path = private.mark(self.client.settings, session_id)
        except (OSError, ValueError) as err:
            return self._fail(f"could not write the marker for {session_id!r}: {err}")
        try:
            self.client.private_put(session_id)
            if not self._server_private(session_id):
                return self._fail(
                    "server did not record the session as private (read-back said false)"
                )
        except _FAILED as err:
            return self._fail(
                f"server flag not set ({_why(err)}). Local marker is in place, but a later "
                "backfill could still ingest this session."
            )
        self._say(f"private mode ON for {session_id}")
        self._say(f"  marker: {path}")
        self._say("  server: private_sessions row written (permanent)")
        return 0

    def off(self, session_id: str, forget: bool = False) -> int:
        try:
            removed = private.unmark(self.client.settings, session_id)
        except ValueError as err:
            return self._fail(str(err), _UNCHANGED)
        if not removed:
            return self._fail(f"could not remove the marker for {session_id!r}", _UNCHANGED)
        if not forget:
            self._say(f"private mode OFF for {session_id} (marker removed)")
            self._say("  server: row KEPT — those turns stay uningestable. --forget to delete it.")
            return 0
        delete_error = ""
        delete_answered = False  # the server replied with an HTTP status: it is done with it
        try:
            self.client.private_delete(session_id)
        except _FAILED as err:
            if isinstance(err, urllib.error.HTTPError) and err.code == 403:
                return self._forget_failed(
                    "taking a session back on the record must be done from a full-trust device."
                )
            # Not a clear refusal: a timeout or a 5xx may come after the server committed
            # the delete, so say nothing about the row until it has been read back.
            delete_error = _why(err)
            delete_answered = isinstance(err, urllib.error.HTTPError)
        check = f"Check with: {self.program} status {session_id}"
        try:
            still_private = self._server_private(session_id)
        except _FAILED as err:
            # The DELETE went out but its effect could not be read back: state unknown,
            # so neither "deleted" nor "stays off the record" can be claimed.
            self._warn(
                "private mode FAILED: could not read back the server row after DELETE "
                f"({_why(err)}). {check}"
            )
            self._marker_removed_note()
            return 1
        if still_private:
            if delete_error and not delete_answered:
                # The request may not have reached the server, or may still be working
                # through it: the row was there at read-back and the DELETE can yet
                # remove it.
                return self._forget_failed(
                    f"server DELETE did not finish ({delete_error}) and the row was still "
                    "present when read back.",
                    f"The DELETE may still be applying on the server. {check}",
                )
            return self._forget_failed(
                f"server DELETE failed ({delete_error})."
                if delete_error
                else "server still reports the session as private after DELETE."
            )
        if delete_error:
            self._warn(
                f"private mode: the DELETE reported an error ({delete_error}), but the "
                "server no longer lists the session as private."
            )
        self._say(f"private mode OFF for {session_id} (marker removed, server row deleted)")
        return 0

    def status(self, session_id: str) -> int:
        try:
            marked = private.marker(self.client.settings, session_id).exists()
        except (OSError, ValueError):
            marked = False
        try:
            server = "yes" if self._server_private(session_id) else "no"
        except Exception as err:
            server = f"unknown ({type(err).__name__})"
        self._say(f"{session_id}: marker={'yes' if marked else 'no'} server={server}")
        return 0

    def session_end(self, stdin: Any = None) -> int:
        """A SessionEnd hook: remove the marker of the session that is ending (its id is
        in the hook's JSON payload on ``stdin``), keep the server row.

        The row is the durable record that those turns must never be ingested, by an
        import months from now included, so the cleanup is local only. Never fails and
        always answers 0: a hook must not break a session's teardown, and a marker left
        behind expires on its own."""
        try:
            payload = json.load(stdin if stdin is not None else sys.stdin) or {}
            session_id = payload.get("session_id") or ""
        except Exception:
            return 0
        if isinstance(session_id, str) and session_id:
            try:
                private.unmark(self.client.settings, session_id)
            except (OSError, ValueError):
                pass
        return 0

    # -- the command ---------------------------------------------------------------

    def main(self, argv: Sequence[str], stdin: Any = None) -> int:
        """Run ``argv`` (the arguments after the program's name) and return the exit
        code: 0 done, 1 failed, 2 not a command."""
        if argv and argv[0] == "--session-end":
            return self.session_end(stdin)
        # `--forget` counts before or after the id: `off --forget <id>` is how it is
        # spoken of.
        names = [arg.strip() for arg in argv[1:] if arg != "--forget"]
        if not names or argv[0] not in ("on", "off", "status"):
            self._warn("Private mode: take a session off the record.")
            self._warn(f"usage: {self.program} {_USAGE}")
            return 2
        action, session_id = argv[0], names[0]
        closing = _NOT_PRIVATE if action == "on" else _UNCHANGED
        flag = next((name for name in names if name.startswith("-")), "")
        if flag:
            # A flag that is not known is not read as the id: the action would be done
            # to a session of that name while the real one stays as it was.
            return self._fail(
                f"session id {flag!r} looks like a flag — usage: {self.program} {_USAGE}",
                closing,
            )
        if len(names) > 1:
            # Run for the first alone, `on` would leave the others on the record unsaid.
            return self._fail(f"one session id at a time, got {len(names)}", closing)
        if not session_id or not private.plain_name(session_id):
            return self._fail(f"invalid session id {session_id!r}", closing)
        if action == "on":
            return self.on(session_id)
        if action == "off":
            return self.off(session_id, forget="--forget" in argv[1:])
        return self.status(session_id)
