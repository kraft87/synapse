"""retrieval tools registration with explicit runtime dependencies."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import logfire
from fastmcp import FastMCP

from ingestion.scope import coerce_group
from ingestion.surfaces import SurfaceTrust

logger = logging.getLogger(__name__)


def register(
    mcp: FastMCP,
    _get_recall: Callable[[], Any],
    _caller_trust: Callable[[str | None], SurfaceTrust],
    _scope_doc: Callable[[Any], Any],
) -> tuple[Any, ...]:
    @mcp.tool()
    @_scope_doc
    def recall(
        query: str,
        project: str | None = None,
        session_focus: list[str] | None = None,
        group_id: str = "technical",
        self_session: str | None = None,
        surface: str | None = None,
    ) -> dict:
        """Search the user's long-term memory: tens of thousands of reranked
        past-conversation turns and the knowledge-graph facts extracted from them.

        BEFORE answering anything that references past work — a prior decision or
        discussion ("what did we decide", "last time", "have we tried"), any device,
        purchase, tool, project, or person the user names, their preferences, or
        history this session alone can't supply — call this first. WHEN the topic
        shifts to something plausibly discussed before, call it again. Assume memory
        has the context; the failure mode is not checking, not over-checking.

        Do NOT call for facts already visible in the current conversation or context
        (read those directly), nor for generic-knowledge questions with no
        user-history angle (definitions, math, general how-tos).

        This is the overview — compressed passages blended with the buckets below,
        the right first call. Its drill-down sibling recall_full_turns serves
        complete raw turns: the follow-up when a passage is truncated mid-thought,
        and the retry when this overview comes back thin.

        Query in plain language carrying the message's distinctive nouns; leave
        `project` unset unless results come back noisy from another domain. Served
        passages carry `role` (user / assistant / mixed) and a `date`: for "current
        state of X", weight newer user-stated content over older assistant-role text,
        which may be speculation or a plan that never happened.

        Every served item carries an `id` (e:N episode, n:N note, f:<uuid> fact — also
        the superseded_facts pairs, t:N timeline, w:N web) — copy it into
        recall_feedback to rate that result. Only e:/n: ids are fetch()-able; the
        rest are feedback-only.

        A `warnings` list appears when a retrieval leg degraded; empty results with a
        warning mean a config problem, not empty memory.

        Follow-ups: fetch(ids) expands a truncated passage or note body;

        Args:
            query: Natural language search query.
            project: Optional project slug to filter results (e.g. "synapse").
            session_focus: Entity names active in current conversation for KG bias.
            group_id: Knowledge graph scope — "technical" (default) or "personal".
            self_session: NEVER set this yourself. The client's PreToolUse hook injects
                the calling session's id so serving can suppress self-session domination;
                calls without it simply skip that suppression.
            surface: DEPRECATED and ignored — the server identifies the calling
                device from its own credential. Never set it.
        """
        # With the personal scope off, "personal" is an alias for the one graph that
        # exists; coerced here too so the telemetry span records what was searched.
        group_id = coerce_group(group_id) or "technical"
        with logfire.span(
            "mcp.recall {query!r}",
            query=query[:80],
            project=project,
            group_id=group_id,
        ):
            return _get_recall().recall(
                query=query,
                project=project,
                session_focus=session_focus or [],
                group_id=group_id,
                source="mcp-tool",
                self_session=self_session,
                trust=_caller_trust(surface),
            )

    @mcp.tool()
    def recall_full_turns(
        query: str,
        project: str | None = None,
        limit: int = 5,
        self_session: str | None = None,
        session_id: str | None = None,
        surface: str | None = None,
    ) -> dict:
        """Search the complete, unabridged text of past conversation turns — the
        drill-down and retry sibling of recall().

        recall() serves compressed ~1400-char passage slices blended with facts and
        timeline; this serves WHOLE turns and nothing else, ranked by relevance +
        recency (keyword + semantic search + rerank over the full archive). `limit`
        sizes the response: 1-2 turns for a pinpoint quote, the default 5 to
        reconstruct a discussion (weak trailing matches drop automatically).

        Three triggers —
        - Exact wording: "what exactly did we say about X", quoting the actual
          exchange, reconstructing how a specific conversation went.
        - The retry: an overview recall came back thin or off-topic for something
          plausibly discussed before. Re-query here with the message's distinctive
          keywords BEFORE falling back to files, grep, or the live system — if it
          was ever said in a session, this is the tool that finds the saying.
        - Full context: an overview passage hit but the ~1400-char slice cut off
          what you need and you want the surrounding discussion.

        Do NOT open with this for general past-work questions — recall() is the
        right first call (cheaper, blended, usually sufficient). Do NOT use it to
        expand an id you already hold — fetch(ids) does that directly. Do NOT
        query it for live system state (ports, configs, processes) — it returns
        what was SAID, which may be stale; the box is ground truth for that.

        Served turns carry e:N ids — fetch()-able and rateable in recall_feedback.

        A `warnings` list appears when a retrieval leg degraded; empty results with a
        warning mean a config problem, not empty memory.

        Args:
            query: Plain-language query carrying the distinctive nouns/keywords.
            project: Optional project slug to filter results (e.g. "synapse").
            limit: Max turns returned (1-10, default 5).
            self_session: NEVER set this yourself. The client's PreToolUse hook
                injects the calling session's id so your own session's turns are
                excluded; calls without it simply skip that exclusion.
            session_id: Scope to ONE conversation — the `session` field from a
                recall/fetch result, or "self" for the current session.
            surface: DEPRECATED and ignored — the server identifies the calling
                device from its own credential. Never set it.
        """
        with logfire.span("mcp.recall_full_turns {query!r}", query=query[:80], project=project):
            if session_id == "self":
                if not self_session:  # no hook injection — refuse loudly, don't search globally
                    return {
                        "error": "session_id='self' requires the client hook to inject the caller's id; none arrived"
                    }
                session_id = self_session
            # Same engine path as the retired recall_episodes tool and the interim
            # recall(mode="turns") — telemetry keeps kind='episodes' so historical
            # per-tool metrics stay comparable.
            return _get_recall().recall_episodes(
                query=query,
                project=project,
                limit=max(1, min(int(limit), 10)),
                source="mcp-tool",
                self_session=self_session,
                session_id=session_id,
                trust=_caller_trust(surface),
            )

    @mcp.tool()
    def fetch(ids: list[str], surface: str | None = None) -> dict:
        """Expand memory ids into full records: episode ids from recall results
        ("e:227168") and note ids from the board ("n:12"), mixed freely in one call.

        WHEN a recall() passage is relevant but truncated and you need the whole
        turn, or WHEN a hook on the session-start board block (its n:ID lines)
        matters and you need the note body — pass the ids here. Bare numeric ids
        are treated as episode ids.

        Do NOT search with this — it only expands ids you already hold; recall()
        finds things. Do NOT re-fetch ids already expanded this session. Unknown or
        unparseable ids are returned under "skipped"; at most 20 ids per call.

        Args:
            ids: Ids to expand — "e:N" episodes, "n:N" notes, bare N = episode.
            surface: DEPRECATED and ignored — the server identifies the calling
                device from its own credential. Never set it.
        """
        with logfire.span("mcp.fetch", n=len(ids)):
            return _get_recall().fetch(ids, source="mcp-tool", trust=_caller_trust(surface))

    @mcp.tool()
    def fetch_session(
        session_id: str,
        around: str | None = None,
        radius: int = 3,
        offset: int = 0,
        limit: int = 10,
        self_session: str | None = None,
        surface: str | None = None,
    ) -> dict:
        """Read one conversation sequentially — like opening the transcript file at
        a spot, instead of searching. Every recall/fetch episode carries a `session`
        field; pass it here to see what surrounded that turn.

        WHEN a recalled turn is the middle of a discussion and you need how it
        started or what was decided after — pass its e:N id as `around`: that anchor
        comes back FULL, ±`radius` neighbors as 500-char heads with `full_chars`
        (expand interesting ones via fetch()). WHEN you want to skim a whole
        session, page it with offset/limit (heads only). Use "self" as session_id
        for the current conversation.

        Do NOT search with this — recall_full_turns(query, session_id=...) greps
        within a session; this reads it in order. An unindexed session returns an
        explicit error — THAT is the signal to read the on-disk transcript instead.

        Args:
            session_id: Session to read — the `session` field from a recall/fetch
                episode, or "self" for the current conversation.
            around: Anchor episode id ("e:N") to center the window on.
            radius: Neighbors per side around the anchor (0-10, default 3).
            offset: Anchorless paging — turn index to start from (default 0).
            limit: Anchorless paging — turns per page (1-25, default 10).
            self_session: NEVER set this yourself; the client hook injects it to
                resolve session_id="self".
            surface: DEPRECATED and ignored — the server identifies the calling
                device from its own credential. Never set it.
        """
        with logfire.span("mcp.fetch_session {sid}", sid=session_id[:40], around=around):
            if session_id == "self":
                if not self_session:  # no hook injection — refuse loudly
                    return {
                        "error": "session_id='self' requires the client hook to inject the caller's id; none arrived"
                    }
                session_id = self_session
            return _get_recall().fetch_session(
                session_id=session_id,
                around=around,
                radius=radius,
                offset=offset,
                limit=limit,
                source="mcp-tool",
                trust=_caller_trust(surface),
            )

    return recall, recall_full_turns, fetch, fetch_session
