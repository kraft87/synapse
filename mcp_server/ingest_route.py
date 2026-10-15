"""ingest route registration with explicit runtime dependencies."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP
from starlette.requests import Request

from ingestion.project_label import clean_label
from ingestion.surfaces import SurfaceTrust
from mcp_server.http_helpers import RouteError, route

if TYPE_CHECKING:
    from ingestion.capture import ToolCall
    from ingestion.models import Episode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Post:
    """The body of a records POST (``/ingest`` and ``/capture/dry-run`` read the same one)."""

    records: list[Any]
    source: str
    #: the client's label for the whole batch; else each turn is labelled from its own cwd
    project: str | None
    project_by_cwd: dict[str, str]
    record_format: str
    session_id: str | None

    @classmethod
    def read(cls, body: dict[str, Any]) -> _Post:
        """Raises RouteError (the 400) for a body that is not a records POST."""
        records = body.get("records")
        if not isinstance(records, list):
            raise RouteError("body must contain a 'records' list")
        # A batch spanning several projects (a session that moved between repositories)
        # carries each cwd's label instead.
        raw_by_cwd = body.get("cwd_projects")
        record_format = body.get("format") or "claude_code"
        if record_format not in ("claude_code", "codex"):
            raise RouteError(f"unknown format {record_format!r}")
        return cls(
            records=records,
            source=body.get("source") or "hook",
            # The client resolves the label (marker file, git toplevel, folder name); without
            # one the parser labels from the transcript's cwd. Either way, never a path.
            project=clean_label(body.get("project")),
            project_by_cwd={
                c: lb
                for c, v in (raw_by_cwd.items() if isinstance(raw_by_cwd, dict) else ())
                if isinstance(c, str) and (lb := clean_label(v))
            },
            record_format=record_format,
            session_id=body.get("session_id"),
        )

    def parse(self, stamp: str | None, *, sidechain: bool) -> tuple[list[Episode], list[ToolCall]]:
        """The turns of the records, each stamped with ``stamp`` (the surface the BEARER
        resolves to, assigned unconditionally so nothing a parser might ever lift out of the
        records can survive as one), and, with ``sidechain``, the tool calls of the sub-agent
        records, which the conversation parser drops."""
        if self.record_format == "codex":
            from ingestion.codex_client import CodexRolloutParser

            episodes = CodexRolloutParser().parse_records(
                self.records,
                self.source,
                self.project,
                session_id_hint=self.session_id,
                with_records=True,
            )
        else:
            from ingestion.jsonl_client import JSONLParser

            episodes = JSONLParser().parse_records(
                self.records,
                self.source,
                self.project,
                project_by_cwd=self.project_by_cwd or None,
                with_records=True,
            )
        for ep in episodes:
            ep.surface_id = stamp
        return episodes, self._sidechain_calls(stamp) if sidechain else []

    def _sidechain_calls(self, stamp: str | None) -> list[ToolCall]:
        """Never raises: a sub-agent record capture cannot read costs the capture only."""
        from ingestion import capture
        from ingestion.jsonl_client import JSONLParser

        if self.record_format != "claude_code":
            return []
        try:
            return capture.sidechain_tool_calls(
                self.records,
                lambda cwd: JSONLParser.project_of(cwd, self.project, self.project_by_cwd or None),
                stamp,
            )
        except Exception:
            logger.exception("web capture could not read the sub-agent records")
            return []


def _capture_enabled() -> bool:
    """Whether capture is on: the sub-agent records are worth reading, and the reply says so.
    A bad switch is not this route's to report (it reads as off): the episode writer logs it
    and counts it, and the turns are stored regardless."""
    from ingestion import capture

    try:
        return capture.enabled()
    except ValueError:
        return False


def register(
    mcp: FastMCP,
    db_url: Callable[[], str],
    _machine_authorized: Callable[[Request], bool],
    _request_trust: Callable[[Request], SurfaceTrust] | None = None,
) -> Any:
    """``_request_trust`` resolves the caller from its BEARER (schema 054); its
    credential-bound surface id is stamped on every episode written (schema 057). Omitted,
    or for the root token / an unknown credential, episodes are stamped NULL."""

    @mcp.custom_route("/ingest", methods=["POST"])
    @route(_machine_authorized, body=True)
    def ingest_turns(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        """Direct-push ingest endpoint — replaces the Logfire poll for Claude Code.

        A Claude Code ``Stop`` hook POSTs a bounded TAIL of the session transcript
        (raw JSONL records) here on every turn. We parse with the SAME ``JSONLParser``
        the disk sweep uses, then dedup by span_id (a turn's stable last-record uuid):
        already-stored turns are skipped, new turns append at ``max(sequence)+1``. The
        parser's positional sequence is NOT used as the key — a tail would renumber it
        from 1 and collide — so identity rides on span_id and the sweep/push still
        converge idempotently. A full-transcript POST stays correct too (all old turns
        skip), so this is backward compatible with the pre-tail hook.

        Any client that runs the real ``claude`` CLI fires this same hook.
        Body: {"records": [...], "project": optional, "source": optional,
        "format": optional ("claude_code" default | "codex"), "session_id": optional}.

        ``format: "codex"`` parses the records as Codex CLI rollout lines via
        ``CodexRolloutParser`` instead. A pushed tail usually lacks the rollout's
        session_meta line, so the Codex hook passes ``session_id`` (from the
        rollout filename) as a fallback identity hint. Everything downstream — span_id
        dedup, contamination and scratch guards, private sessions, delete tombstones and
        the cross-session replay guard — is ``ingestion.episode_writer``, shared with
        every backfill.

        Provenance (schema 057): each stored turn is stamped with the surface the BEARER
        resolves to, which is what lets a restricted device read its own uploads back.
        Nothing in the body can set or influence that stamp.

        Web research in the records' tool calls (any tool: ``ingestion.capture`` recognises it
        by shape) is captured too when ``SYNAPSE_WEB_CAPTURE=on``, from stored turns, re-shipped
        turns and sub-agent records alike; the reply's ``captured`` counts the artifacts stored
        and ``capture`` says whether the switch is on (an older server has neither key).
        """
        from ingestion.db import Database
        from ingestion.episode_writer import StoreStats, store_episodes

        post = _Post.read(body)
        # Resolved from the credential only.
        stamp = _request_trust(request).stamp_surface_id if _request_trust else None
        episodes, extra_calls = post.parse(stamp, sidechain=_capture_enabled())
        stats = StoreStats()
        if episodes or extra_calls:
            db = Database(db_url())
            try:
                # Every guard, the span dedup, the append at max(sequence)+1, the extraction
                # enqueue and the web capture live in ingestion.episode_writer, shared with
                # the backfills; the POST is one transaction on one connection.
                stats = store_episodes(db, episodes, extra_calls=extra_calls)
            finally:
                db.close()
            stats.log_skipped()
        return {
            "status": "ok",
            "ingested": stats.written,
            "captured": stats.captured,
            # Whether capture is on: the client ships sub-agent transcripts and the large
            # tool outputs it saved to files only to a server that will use them.
            "capture": _capture_enabled(),
        }

    @mcp.custom_route("/capture/dry-run", methods=["POST"])
    @route(_machine_authorized, body=True)
    def capture_dry_run(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        """What web capture WOULD store from a batch of transcript records. Writes nothing.

        The body is ``/ingest``'s, behind the same client gate. The records go through the
        detector, the capture rules, the private-session check, the duplicate check against
        what is stored and the per-POST and per-day caps exactly as a live capture does
        (``ingestion.capture.capture_calls(dry_run=True)``), and the reply groups the outcome
        by tool (``ingestion.capture.report``). It answers whether or not
        ``SYNAPSE_WEB_CAPTURE`` is on, and says which in ``enabled``: ``synapse capture
        --backfill`` reads that before it posts anything for real; an empty ``records`` list
        is the question alone.

        It does not run the episode guards (a tombstoned, contaminated or scratch-workspace
        turn, a replay) that keep a turn's research from being captured, and the caps and
        duplicates it sees are per POST: a client posting a long history in batches counts a
        page fetched twice in two batches twice.
        """
        from ingestion import capture
        from ingestion.db import Database
        from ingestion.private_sessions import PrivateSessions

        try:
            capture.check_config()  # the switch and the rule files, named if invalid
        except ValueError as e:
            raise RouteError(f"web capture is misconfigured: {e}", 500) from e
        post = _Post.read(body)
        stamp = _request_trust(request).stamp_surface_id if _request_trust else None
        episodes, extra_calls = post.parse(stamp, sidechain=True)
        calls = [c for ep in episodes for c in capture.episode_tool_calls(ep)] + extra_calls
        db = Database(db_url())
        try:
            result = capture.capture_calls(
                db, calls, is_private=PrivateSessions(db).is_private, dry_run=True
            )
        finally:
            db.close()
        return {"status": "ok", "enabled": capture.enabled(), **capture.report(result)}

    return ingest_turns
