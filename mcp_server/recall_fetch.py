"""Full-record and sequential-session reads with audience filtering."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ingestion.surfaces import SurfaceTrust
from mcp_server.recall_presentation import parse_fetch_ids as _parse_fetch_ids
from mcp_server.recall_presentation import to_recall_item as _to_recall_item
from mcp_server.recall_ranking import served_chars as _served_chars

logger = logging.getLogger(__name__)

_SESSION_RADIUS_DEFAULT = 3
_SESSION_RADIUS_MAX = 10
_SESSION_PAGE_DEFAULT = 10
_SESSION_PAGE_MAX = 25
_SESSION_HEAD_CHARS = 500


class RecallFetchMixin:
    """Read paths using the engine's connection, trust resolver, and metrics writer."""

    _db_url: str

    if TYPE_CHECKING:

        def _ensure_pg(self) -> Any: ...
        def _resolve_trust(
            self, surface: str | None, trust: SurfaceTrust | None
        ) -> SurfaceTrust: ...
        def _record_metrics(self, metrics: dict[str, Any]) -> None: ...
        def _increment_retrieval_counts(self, episode_ids: list[int]) -> None: ...

    def fetch(
        self,
        ids: list[Any],
        source: str | None = None,
        surface: str | None = None,
        trust: SurfaceTrust | None = None,
    ) -> dict[str, Any]:
        """Drill-down by id: expand recall()'s compact serves into full records.

        Accepts mixed prefixed ids — "e:N" episodes (bare N / bare int also accepted,
        the old fetch_episode back-compat) and "n:N" notes (the board's n:ID lines).
        Episodes come back in the recall-item shape ({id, content, project, date}),
        notes as {id, hook, body, type, project, updated}; both ordered to match the
        request. Unknown prefixes / unparseable ids are reported under ``skipped``;
        the total expanded is capped at _FETCH_MAX across both kinds.

        ``surface`` applies the calling host's filters (schema 053). This is not
        belt-and-braces: e:N / n:N ids are sequential integers, so an unfiltered fetch
        would let a restricted caller enumerate exactly what the board and recall were
        built to withhold."""
        t_start = time.perf_counter()
        st = self._resolve_trust(surface, trust)
        ep_ids, note_ids, skipped, normalized = _parse_fetch_ids(ids)
        out: dict[str, Any] = {"episodes": [], "notes": [], "skipped": skipped}
        if not normalized:
            return out
        out["episodes"] = self._fetch_episode_records(ep_ids, st.project_filter) if ep_ids else []
        out["notes"] = (
            self._fetch_note_records(note_ids, audience=st.audience_filter) if note_ids else []
        )
        chars = _served_chars({"episodes": out["episodes"]}) + sum(
            len(n.get("hook") or "") + len(n.get("body") or "") for n in out["notes"]
        )
        # served_ids carries the per-kind counts plus the served note ids ("n:N").
        # Notes are the one bucket with NO retrieval_count column (episodes bump
        # theirs in _fetch_episode_records), so this envelope is the only record of
        # WHICH notes get expanded — the item-6 measurability fix, zero new DDL.
        self._record_metrics(
            {
                "kind": "fetch",
                "source": source or "mcp",
                "query": ",".join(normalized)[:200],
                "ms_total": round((time.perf_counter() - t_start) * 1000.0, 1),
                "n_episodes": len(out["episodes"]),
                "chars": chars,
                "served_ids": {
                    "kinds": {"e": len(out["episodes"]), "n": len(out["notes"])},
                    "notes": [n["id"] for n in out["notes"]],
                },
            }
        )
        return out

    def fetch_session(
        self,
        session_id: str,
        around: str | None = None,
        radius: int = _SESSION_RADIUS_DEFAULT,
        offset: int = 0,
        limit: int = _SESSION_PAGE_DEFAULT,
        source: str | None = None,
        surface: str | None = None,
        trust: SurfaceTrust | None = None,
    ) -> dict[str, Any]:
        """Sequential read of one session's turns — the Read analog of the
        session drill-down (recall_episodes(session_id=...) is the Grep analog).

        Two modes:
          * ``around="e:N"``: the anchor turn full, ±``radius`` neighbors as
            _SESSION_HEAD_CHARS-char heads (cap _SESSION_RADIUS_MAX/side).
          * anchorless: ``offset``/``limit`` paging over the whole session,
            heads only (cap _SESSION_PAGE_MAX/page) — skim, then fetch() ids.

        Pure indexed read (episodes_session_idx + the (session_id, sequence)
        unique pair): no embedding, no rerank. An unknown session returns an
        explicit ``error`` — never a silent empty — so the caller knows to fall
        back to the on-disk transcript rather than concluding "nothing there".

        ``surface`` applies the restricted project allowlist (schema 053) to every read
        below, including the metadata probe — so a session outside the allowlist reports
        the same "not indexed" answer an unknown id does. Deliberately indistinguishable:
        a distinct "exists but forbidden" reply would itself disclose the project map.
        """
        t_start = time.perf_counter()
        radius = max(0, min(int(radius), _SESSION_RADIUS_MAX))
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), _SESSION_PAGE_MAX))
        out: dict[str, Any] = {"session_id": session_id}
        allowed = self._resolve_trust(surface, trust).project_filter
        # Appended to every query in this method; empty string on a full-trust surface.
        proj_sql = " AND project = ANY(%s)" if allowed is not None else ""
        proj_args: tuple[Any, ...] = (allowed,) if allowed is not None else ()

        try:
            pg = self._ensure_pg()
            meta = pg.execute(
                "SELECT count(*) AS n, min(created_at) AS first, max(created_at) AS last,"  # nosec B608 — proj_sql is a literal, its value is bound
                f" min(project) AS project FROM episodes WHERE session_id = %s{proj_sql}",
                (session_id, *proj_args),
            ).fetchone()
            if not meta or not meta["n"]:
                out["error"] = (
                    "session not indexed — no ingested turns under this id. If the session is"
                    " recent or ran headless, its transcript may still be on disk"
                    " (~/.claude/projects/*/<session_id>.jsonl); read it there."
                )
                return out
            out["project"] = meta["project"]
            out["total_turns"] = meta["n"]
            out["first_date"] = str(meta["first"])[:10]
            out["last_date"] = str(meta["last"])[:10]

            anchor_id: int | None = None
            if around:
                try:
                    anchor_id = int(str(around).split(":")[-1])
                except ValueError:
                    out["error"] = f"unparseable anchor id {around!r} — expected 'e:N'"
                    return out
                anchor = pg.execute(
                    "SELECT sequence FROM episodes "  # nosec B608 — proj_sql is a literal, its value is bound
                    f"WHERE id = %s AND session_id = %s{proj_sql}",
                    (anchor_id, session_id, *proj_args),
                ).fetchone()
                if anchor is None:
                    out["error"] = f"episode {around} is not in session {session_id}"
                    return out
                rows = pg.execute(
                    "SELECT id, sequence, content, created_at,"  # nosec B608 — proj_sql is a literal, its value is bound
                    " (human_turn IS NOT NULL) AS has_h, (assistant_turn IS NOT NULL) AS has_a"
                    f" FROM episodes WHERE session_id = %s AND sequence BETWEEN %s AND %s{proj_sql}"
                    " ORDER BY sequence",
                    (
                        session_id,
                        anchor["sequence"] - radius,
                        anchor["sequence"] + radius,
                        *proj_args,
                    ),
                ).fetchall()
            else:
                rows = pg.execute(
                    "SELECT id, sequence, content, created_at,"  # nosec B608 — proj_sql is a literal, its value is bound
                    " (human_turn IS NOT NULL) AS has_h, (assistant_turn IS NOT NULL) AS has_a"
                    f" FROM episodes WHERE session_id = %s{proj_sql}"
                    " ORDER BY sequence LIMIT %s OFFSET %s",
                    (session_id, *proj_args, limit, offset),
                ).fetchall()

            turns: list[dict[str, Any]] = []
            for r in rows:
                content = r["content"] or ""
                item: dict[str, Any] = {
                    "id": f"e:{r['id']}",
                    "seq": r["sequence"],
                    "date": str(r["created_at"])[:10],
                    "role": (
                        "mixed"
                        if r["has_h"] and r["has_a"]
                        else "user"
                        if r["has_h"]
                        else "assistant"
                    ),
                }
                if anchor_id is not None and r["id"] == anchor_id:
                    item["content"] = content  # the anchor — served whole
                else:
                    item["head"] = content[:_SESSION_HEAD_CHARS]
                    item["full_chars"] = len(content)  # what fetch(e:N) would expand to
                turns.append(item)
            out["turns"] = turns
            if anchor_id is not None:
                self._increment_retrieval_counts([anchor_id])
            return out
        except Exception as e:
            logger.warning("fetch_session failed: %s", e)
            out["error"] = f"fetch_session failed: {type(e).__name__}"
            return out
        finally:
            served = out.get("turns") or []
            chars = sum(len(t.get("content") or t.get("head") or "") for t in served)
            self._record_metrics(
                {
                    "kind": "fetch_session",
                    "source": source or "mcp",
                    "query": f"{session_id} around={around} r={radius} off={offset}"[:200],
                    "ms_total": round((time.perf_counter() - t_start) * 1000.0, 1),
                    "n_episodes": len(served),
                    "chars": chars,
                    "est_tokens": chars // 4,
                    "served_ids": {
                        "episodes": [t["id"] for t in served],
                        "error": out.get("error"),
                    },
                }
            )

    def _fetch_episode_records(
        self, parsed: list[int], allowed_projects: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """The episodes leg of fetch(): full untruncated turns by int id. Fail-soft —
        a read error serves an empty leg, never breaks the call.

        ``allowed_projects`` carries the restricted surface's allowlist. Ids are small
        integers and therefore guessable, so drill-down enforces the SAME predicate the
        overview does — otherwise fetch() would be a trivial bypass of every filter
        recall() applies."""
        try:
            conn = self._ensure_pg()
            sql = (
                "SELECT id, content, project, created_at, session_id"
                " FROM episodes WHERE id = ANY(%s)"
            )
            params: list[Any] = [parsed]
            if allowed_projects is not None:
                sql += " AND project = ANY(%s)"
                params.append(allowed_projects)
            rows = conn.execute(sql, params).fetchall()
        except Exception as e:
            logger.warning("fetch episodes leg failed: %s", e)
            return []
        by_id = {r["id"]: r for r in rows}
        found = [n for n in parsed if n in by_id]
        if found:
            self._increment_retrieval_counts(found)
        return [_to_recall_item({**by_id[n], "id": f"e:{n}"}) for n in found]

    def _fetch_note_records(
        self, parsed: list[int], audience: str | None = None
    ) -> list[dict[str, Any]]:
        """The notes leg of fetch(): board-note bodies by int id (the on-demand half of
        the board — hook on the board, body behind the id). Uses a short-lived Database
        like the other notes-store paths. Fail-soft like the episodes leg.

        ``audience`` applies the restricted surface's tier filter — same reason as the
        episodes leg: n:N ids are guessable, so the drill-down cannot be looser than the
        board that served them."""
        from ingestion.db import Database

        try:
            db = Database(self._db_url)
            try:
                rows = db.get_notes_by_ids(parsed, audience=audience)
            finally:
                db.close()
        except Exception as e:
            logger.warning("fetch notes leg failed: %s", e)
            return []
        by_id = {r["id"]: r for r in rows}
        return [
            {
                "id": f"n:{n}",
                "hook": r["hook"],
                "body": r["body"],
                "type": r["type"],
                "project": r["project"],
                "updated": str(r["updated_at"])[:10],
            }
            for n in parsed
            if (r := by_id.get(n)) is not None
        ]
