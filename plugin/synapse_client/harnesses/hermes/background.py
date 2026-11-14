"""Work a hook hands over and does not wait for.

Hermes's guide says a provider's ``sync_turn`` must not block: work with latency goes on
a thread of the provider's own (website/docs/developer-guide/memory-provider-plugin.md,
"Threading Contract"). Hermes calls ``sync_turn`` on one worker per agent, which also
runs the agent's prefetches and the two hooks of a session boundary
(agent/memory_manager.py ``_submit_background``). It takes a call past 30 seconds for a
stuck one and, with four stuck, skips the provider for every session of the home
(``_StuckCallGate``). A turn upload can take longer than that: it asks the server about
each session this one continues, sends a batch for each few turns and drains the outbox
the home shares.

``Background`` runs such jobs one at a time, in the order they were given: the upload
that ends one turn must come before the next turn's (turn_upload.py). Its thread lives
while there is a job, so an idle session holds none. The provider passes what starts it
(Hermes's ``spawn_context_thread``), and this module imports nothing of the host. A job
runs under the contextvars of the call that started the thread, which can be an earlier
call than its own; the upload path reads none.

At most ``LIMIT`` jobs wait for the one running. One more is refused, and the caller
says so: with a server that answers no request, every upload takes its whole timeout,
and a conversation that goes on faster than that would otherwise keep one copy of
itself for each turn.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from typing import Any

#: How many jobs may wait for the one running.
LIMIT = 16


class Background:
    """Jobs run one at a time, in order, on a thread that lives while there is one.
    ``spawn`` is called with ``_run`` and ``name`` and returns a thread not started yet.
    Safe across threads."""

    def __init__(self, spawn: Callable[..., Any], name: str, limit: int = LIMIT) -> None:
        self._spawn = spawn
        self._name = name
        self._limit = limit
        self._idle = threading.Condition()
        self._jobs: deque[tuple[Callable[..., Any], tuple[Any, ...]]] = deque()
        self._running = False

    def put(self, job: Callable[..., Any], *args: Any) -> bool:
        """Call ``job(*args)`` after the jobs given before it, and return at once. False,
        and the job is not kept, when ``limit`` jobs wait already. RAISES what starting
        the thread raises; the job is kept then, and runs with the next one given."""
        with self._idle:
            if len(self._jobs) >= self._limit:
                return False
            self._jobs.append((job, args))
            if self._running:
                return True
            self._running = True
        try:
            self._spawn(self._run, name=self._name).start()
        except BaseException:
            self._stopped()
            raise
        return True

    def wait(self, seconds: float) -> bool:
        """Wait until no job is running or left, ``seconds`` at most. Whether that came."""
        with self._idle:
            return self._idle.wait_for(lambda: not self._running, max(0.0, seconds))

    def _stopped(self) -> None:
        with self._idle:
            self._running = False
            self._idle.notify_all()

    def _run(self) -> None:
        """Run the jobs until none is left. A job that raises stops nothing: it is
        logged, and the next one runs."""
        try:
            while True:
                with self._idle:
                    if not self._jobs:
                        # Under the lock ``put`` takes: a job given now starts a thread.
                        self._running = False
                        self._idle.notify_all()
                        return
                    job, args = self._jobs.popleft()
                try:
                    job(*args)
                except Exception:
                    logging.getLogger(__name__).warning(
                        "synapse: a background job failed", exc_info=True
                    )
        except BaseException:
            self._stopped()
            raise
