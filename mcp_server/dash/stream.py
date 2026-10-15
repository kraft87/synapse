"""The SSE live stream (phase 3): LISTEN/NOTIFY worker, ring buffer and the per-connection
frame loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import psycopg
from psycopg.rows import dict_row
from starlette.requests import Request
from starlette.responses import StreamingResponse

from ingestion import pg_pool
from mcp_server.dash.common import _flag_set
from mcp_server.dash.feed import (
    _EP_SELECT,
    _FACT_SELECT,
    _TL_SELECT,
    _episode_item,
    _fact_item,
    _timeline_item,
)
from mcp_server.http_helpers import route

logger = logging.getLogger(__name__)


# Phase 3 (SSE live stream). The LISTEN worker hydrates each NOTIFY into a full FeedItem and
# appends it to a ring buffer; the /dash/api/stream route replays from it on resume and streams
# live. Feed-item event names by type; every other tunable for the stream loop.
_STREAM_EVENT_NAMES = {
    "episode": "new_episode",
    "fact": "new_fact",
    "timeline_event": "new_timeline_event",
    "reset": "reset",  # the LISTEN worker's reconnect marker, not a feed item
}
_STREAM_CHANNEL = "dash_feed"  # matches schema/043_dash_notify.sql
_STREAM_BUFFER_CAP = 512  # ring-buffer slots (contract: 512-event replay window)
_STREAM_POLL_S = 0.25  # per-connection buffer poll cadence (new-event latency ceiling)
_STREAM_STATUS_S = 1.0  # processing_status emit cadence (contract: ~1/s)
_STREAM_HEARTBEAT_S = 15.0  # comment heartbeat so idle proxies don't kill the stream
_STREAM_READY_S = 2.0  # a new connection waits at most this long for LISTEN + a first status
_STREAM_RECONNECT_MIN_S = 1.0  # LISTEN reconnect backoff: first retry delay ...
_STREAM_RECONNECT_MAX_S = 30.0  # ... doubling up to this


# schema/043 arms AFTER INSERT triggers on episodes / kg_relationships / timeline_events
# that pg_notify('dash_feed', {type,id}). A single lazily-started LISTEN worker (one
# dedicated async psycopg connection, started on the first SSE subscriber and stopped when
# the last disconnects) hydrates each notification into a full FeedItem — via the SAME
# per-type SELECT + mapper the /feed endpoint uses — and appends it to a 512-slot ring
# buffer keyed by a monotonically increasing integer event id. The /dash/api/stream route
# opens each connection with `ready` (the client refetches page 1 on it), replays from the
# buffer on resume (Last-Event-ID) and streams live thereafter.


def _hydrate_feed_item(db_url: str, type_: Any, id_: Any) -> dict[str, Any] | None:
    """Re-SELECT one feed row by id and map it to the SAME FeedItem shape /feed emits
    (including `flagged`). Returns None for an unknown type or a row that vanished between
    the NOTIFY and this read. Reuses _EP_SELECT/_FACT_SELECT/_TL_SELECT + the mappers."""
    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
        if type_ == "episode":
            if not str(id_).isdigit():
                return None
            row = conn.execute(f"{_EP_SELECT} WHERE id = %s", (int(id_),)).fetchone()
            item = _episode_item(row) if row else None
        elif type_ == "fact":
            row = conn.execute(f"{_FACT_SELECT} WHERE r.uuid = %s", (str(id_),)).fetchone()
            item = _fact_item(row) if row else None
        elif type_ == "timeline_event":
            if not str(id_).isdigit():
                return None
            row = conn.execute(f"{_TL_SELECT} WHERE id = %s", (int(id_),)).fetchone()
            item = _timeline_item(row) if row else None
        else:
            return None
        if item is None:
            return None
        item["flagged"] = (item["type"], item["id"]) in _flag_set(conn)
        return item


def _processing_status(db_url: str) -> dict[str, Any]:
    """Extraction-queue health for the header live badge: pending (queue_depth, the name the
    wire has always used), processing, failed and active (anything processing). ONE cheap
    indexed count (extraction_queue_status_idx) — the status refresher runs it once/second,
    shared across all connected SSE clients, never per-connection."""
    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
        row = (
            conn.execute(
                "SELECT count(*) FILTER (WHERE status = 'pending') AS pending, "
                "       count(*) FILTER (WHERE status = 'processing') AS processing, "
                "       count(*) FILTER (WHERE status = 'failed') AS failed "
                "FROM extraction_queue WHERE status IN ('pending', 'processing', 'failed')"
            ).fetchone()
            or {}
        )
        processing = int(row.get("processing") or 0)
        return {
            "queue_depth": int(row.get("pending") or 0),
            "processing": processing,
            "failed": int(row.get("failed") or 0),
            "active": processing > 0,
        }


class _FeedEventBuffer:
    """A fixed-capacity ring of hydrated feed events with monotonically increasing integer
    ids. Standalone (no I/O) so the replay/reset logic is unit-testable without a DB or HTTP.

    Resume semantics for a client's Last-Event-ID L:
      * L == head            -> replay []      (already current; stream live from here)
      * L  > head            -> reset          (client is ahead of us; our buffer was reset,
                                                e.g. a server restart — client refetches page 1)
      * L  < oldest - 1      -> reset          (a gap: events L+1..oldest-1 were evicted)
      * else                 -> replay since L (contiguous; hand back what was missed)
    """

    def __init__(self, capacity: int = _STREAM_BUFFER_CAP) -> None:
        self._events: deque[tuple[int, dict[str, Any]]] = deque(maxlen=capacity)
        self._next_id = 1

    def append(self, item: dict[str, Any]) -> int:
        eid = self._next_id
        self._next_id += 1
        self._events.append((eid, item))
        return eid

    @property
    def head(self) -> int:
        """The last assigned event id (0 when nothing has ever been appended)."""
        return self._next_id - 1

    def oldest(self) -> int | None:
        return self._events[0][0] if self._events else None

    def since(self, after_id: int) -> list[tuple[int, dict[str, Any]]]:
        return [(eid, it) for eid, it in self._events if eid > after_id]

    def resume(self, last_event_id: int) -> tuple[str, list[tuple[int, dict[str, Any]]]]:
        head = self.head
        if last_event_id >= head:
            # Current, or ahead of us (buffer was reset under the client). Ahead -> resync.
            return ("reset", []) if last_event_id > head else ("replay", [])
        oldest = self.oldest()
        if oldest is not None and last_event_id < oldest - 1:
            return ("reset", [])  # evicted gap between what the client has and what we kept
        return ("replay", self.since(last_event_id))


class _StreamManager:
    """Owns the lazily-started LISTEN worker + status refresher and the shared ring buffer.
    Zero background cost with no clients: both tasks start on the first subscriber and are
    cancelled when the last one disconnects. Everything runs in the server's single event
    loop, so subscribe/unsubscribe are plain synchronous ref-count flips (no lock needed —
    asyncio has no preemption between awaits)."""

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        self.buffer = _FeedEventBuffer()
        # The shared processing_status snapshot; None while there is none to serve (no
        # subscriber yet, or the last refresh failed), so a stale count is never sent.
        self.status: dict[str, Any] | None = None
        # Set once the LISTEN is actually issued — lets a caller (and the tests) know a
        # NOTIFY inserted from now on will be delivered, not raced against worker startup.
        self.listening = asyncio.Event()
        # Set once the status loop has made its first refresh attempt (success or not).
        self.status_ready = asyncio.Event()
        self._subscribers = 0
        self._tasks: list[asyncio.Task[Any]] = []

    def subscribe(self) -> None:
        self._subscribers += 1
        if self._subscribers == 1:
            loop = asyncio.get_running_loop()
            self._tasks = [
                loop.create_task(self._listen_loop()),
                loop.create_task(self._status_loop()),
            ]

    def unsubscribe(self) -> None:
        self._subscribers -= 1
        if self._subscribers <= 0:
            self._subscribers = 0
            for t in self._tasks:
                t.cancel()
            self._tasks = []
            self.listening.clear()
            self.status = None  # the next first subscriber must not be handed this run's count
            self.status_ready.clear()

    async def wait_ready(self) -> None:
        """Block until the LISTEN is issued and the first status snapshot is in, so what a
        new connection sends first is current. Gives up after _STREAM_READY_S when the DB is
        unreachable: the connection streams anyway, and the worker's gap marker resyncs the
        client once the DB is back."""
        try:
            async with asyncio.timeout(_STREAM_READY_S):
                await self.listening.wait()
                await self.status_ready.wait()
        except TimeoutError:
            logger.warning("dash stream: not ready after %.0fs; streaming anyway", _STREAM_READY_S)

    async def _listen_loop(self) -> None:
        """One dedicated async connection LISTENing on the dash_feed channel; hydrates each
        notification off the event loop (a thread executor) and appends it to the ring
        buffer. run_in_executor (not starlette's run_in_threadpool) keeps this background
        task off anyio's request-scoped context.

        A dropped connection (a Postgres restart, a network blip) is retried with backoff
        for as long as the task lives, i.e. while a client is subscribed. Notifications sent
        during the gap are lost, so each reconnect appends a `reset` event: connected clients
        refetch page 1 instead of sitting on a feed that silently went stale."""
        loop = asyncio.get_running_loop()
        delay = _STREAM_RECONNECT_MIN_S
        gap = False
        while True:
            aconn = None
            try:
                aconn = await psycopg.AsyncConnection.connect(self._db_url, autocommit=True)
                await aconn.execute(f"LISTEN {_STREAM_CHANNEL}")
                delay = _STREAM_RECONNECT_MIN_S
                if gap:
                    self.buffer.append({"type": "reset"})
                    gap = False
                self.listening.set()
                async for notify in aconn.notifies():
                    try:
                        payload = json.loads(notify.payload)
                        item = await loop.run_in_executor(
                            None,
                            _hydrate_feed_item,
                            self._db_url,
                            payload.get("type"),
                            payload.get("id"),
                        )
                        if item is not None:
                            self.buffer.append(item)
                    except Exception as e:  # one bad notification must not kill the stream
                        logger.warning("dash stream: hydrate failed: %s", e)
                logger.warning("dash stream: listen connection closed; reconnecting")
            except asyncio.CancelledError:
                return  # intended shutdown when the last subscriber leaves
            except Exception as e:
                logger.warning("dash stream: listen connection lost, retrying: %s", e)
            finally:
                self.listening.clear()
                if aconn is not None:
                    try:
                        await aconn.close()
                    except Exception:  # pragma: no cover - best-effort close during cancel
                        pass
            gap = True
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return
            delay = min(delay * 2, _STREAM_RECONNECT_MAX_S)

    async def _status_loop(self) -> None:
        """Refresh the shared processing_status snapshot once/second (one query total, not
        per-connection). Connections read self.status and emit it on their own cadence."""
        loop = asyncio.get_running_loop()
        try:
            while True:
                try:
                    self.status = await loop.run_in_executor(None, _processing_status, self._db_url)
                except Exception as e:
                    self.status = None  # withdraw the count rather than serve it stale
                    logger.warning("dash stream: status refresh failed: %s", e)
                self.status_ready.set()
                await asyncio.sleep(_STREAM_STATUS_S)
        except asyncio.CancelledError:
            pass


def _sse_frame(event: str, data: str, event_id: int | None = None) -> str:
    """One SSE frame: an `event:` line, an optional `id:` line (feed events only — so the
    client's Last-Event-ID tracks the resumable feed stream, not status/heartbeat), a
    `data:` line, terminated by a blank line."""
    lines = [f"event: {event}"]
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"data: {data}")
    return "\n".join(lines) + "\n\n"


async def _event_stream(
    manager: _StreamManager,
    is_disconnected: Callable[[], Awaitable[bool]],
    last_id: int | None,
) -> AsyncIterator[str]:
    """One SSE connection's frames: `ready`, the resume replay, then live feed events, a
    processing_status snapshot ~1/s and a heartbeat comment every 15s.

    `ready` is sent once the LISTEN is up and this connection's position is fixed: every row
    committed from then on is delivered, so the client refetches page 1 on it and holds
    everything older. That fetch is what closes the gaps (rows written while the client was
    away, before the worker was listening, or between its first /feed and the connect); the
    replay is only the shortcut for rows still in the ring, so a Last-Event-ID the ring can
    no longer resume simply gets no replay."""
    manager.subscribe()
    try:
        await manager.wait_ready()
        buf = manager.buffer
        sent = buf.head
        replay: list[tuple[int, dict[str, Any]]] = []
        if last_id is not None:
            mode, events = buf.resume(last_id)
            if mode == "replay":
                sent, replay = last_id, events
        yield _sse_frame("ready", "{}")
        for eid, item in replay:
            yield _sse_frame(_STREAM_EVENT_NAMES[item["type"]], json.dumps(item), eid)
            sent = eid
        last_status = 0.0  # the first pass sends the status at once
        last_hb = time.monotonic()
        while True:
            if await is_disconnected():
                break
            oldest = buf.oldest()
            if oldest is not None and sent < oldest - 1:
                # The ring outran this connection (a slow reader through a burst): the
                # evicted events are gone, so the client refetches rather than skip them.
                sent = buf.head  # before the yield: events appended while we're suspended are new
                yield _sse_frame("reset", "{}")
                continue
            for eid, item in buf.since(sent):
                yield _sse_frame(_STREAM_EVENT_NAMES[item["type"]], json.dumps(item), eid)
                sent = eid
            now = time.monotonic()
            if now - last_status >= _STREAM_STATUS_S and manager.status is not None:
                yield _sse_frame("processing_status", json.dumps(manager.status))
                last_status = now
            if now - last_hb >= _STREAM_HEARTBEAT_S:
                yield ": heartbeat\n\n"
                last_hb = now
            await asyncio.sleep(_STREAM_POLL_S)
    finally:
        manager.unsubscribe()


def register(mcp: Any, db_url: str, authorized: Callable[[Request], bool]) -> None:
    """Mount the SSE route (admin gated)."""
    # Per-register SSE manager (LISTEN worker + ring buffer). One per server; a fresh
    # register() in tests gets an isolated manager (own buffer/worker).
    stream_manager = _StreamManager(db_url)

    @mcp.custom_route("/dash/api/stream", methods=["GET"])  # type: ignore[untyped-decorator]
    @route(authorized)
    async def dash_stream(request: Request) -> StreamingResponse:
        """SSE live feed (Phase 3). Admin-gated like every /dash/api/* route — the
        client uses fetch + a ReadableStream parser (not EventSource) so it CAN send the
        Authorization header. Frames: `ready` first (the client refetches page 1 on it), then
        the Last-Event-ID (header or ?last_event_id=) replay while that id is still inside the
        ring buffer, then live feed events (new_episode|new_fact|new_timeline_event, data =
        the FeedItem JSON), `reset` when the ring outran this connection or the LISTEN worker
        lost its connection (refetch page 1 again), a processing_status snapshot ~1/s and a
        heartbeat comment every 15s."""
        raw = request.headers.get("last-event-id") or request.query_params.get("last_event_id")
        last_id = int(raw) if (raw and raw.lstrip("-").isdigit()) else None

        return StreamingResponse(
            _event_stream(stream_manager, request.is_disconnected, last_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # tell nginx/proxies not to buffer the stream
            },
        )
