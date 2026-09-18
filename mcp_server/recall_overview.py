"""RecallOverviewMixin operations."""

from __future__ import annotations

import logging
import time
from typing import Any

from ingestion import embedding as _embedding
from ingestion.scope import coerce_group
from ingestion.surfaces import SurfaceTrust
from mcp_server.recall_presentation import apply_supersessions as _apply_supersessions
from mcp_server.recall_presentation import parse_episode_ids as _parse_episode_ids
from mcp_server.recall_presentation import to_web_recall_item as _to_web_recall_item
from mcp_server.recall_ranking import merge_rrf as _merge_rrf
from mcp_server.recall_ranking import served_chars as _served_chars
from mcp_server.recall_ranking import timed as _timed
from mcp_server.recall_warnings import config_hint as _config_hint
from mcp_server.recall_warnings import error_brief as _err_brief
from mcp_server.recall_warnings import submit_with_context as _submit_ctx
from mcp_server.recall_warnings import warn as _warn
from mcp_server.recall_warnings import warn_sink as _warn_sink

logger = logging.getLogger(__name__)


class RecallOverviewMixin:
    def recall(
        self,
        query: str,
        project: str | None = None,
        session_focus: list[str] | None = None,
        group_id: str = "technical",
        write_feedback: bool = True,
        source: str | None = None,
        debug: bool = False,
        self_session: str | None = None,
        surface: str | None = None,
        trust: SurfaceTrust | None = None,
    ) -> dict[str, Any]:
        """Overview retrieval: reranked episodes + KG facts (+ entities, web).

        Best for session start and 'what's the state of X?' queries. Serves a wide-pool
        reranked episode leg (the broad/needle workhorse) plus knowledge-graph facts for
        entity-level precision. The summary layer was retired (task #63).

        write_feedback=False suppresses the retrieval_count bump on surfaced facts — used
        by the auto-recall memory hook so automatic (non-agentic) recalls don't pollute the
        frequency-feedback signal that ranks future results. History fetch is unaffected.

        ``source`` tags the call origin (e.g. "mcp-tool", "http", "recall-hook:session")
        on the recall_metrics row so per-origin metrics are filterable in SQL.

        The episode bucket serves compact passages (markdown chunks) of the top reranked episodes
        instead of whole turns (Stage 2 — see _RECALL_PASSAGE_N). For raw full-episode drill-down,
        use recall_episodes() (the recall tool's mode="turns"). Facts carry their t_valid "as-of"
        date for currency weighting.

        ``debug`` (phase-2 dashboard console) attaches a ``debug`` key to the response SURFACING
        the SAME numbers already measured for the recall_metrics telemetry row — no extra
        instrumentation, no extra work. Off by default so every non-dashboard call is byte-identical
        in behavior AND in the telemetry it records (a live call-rate A/B depends on that). See the
        debug-dict assembly just below the metrics write for the exact shape.

        ``surface`` is the calling host's id (schema 053). Anything but a surface
        registered ``trust='full'`` — including a missing one — is RESTRICTED: episodes
        are filtered to the surface's project allowlist, notes to
        ``audience='work-safe'``, and the KG facts leg is skipped entirely (v1:
        kg_relationships has no project column, so serving zero facts is the only
        fail-closed answer available). Bare calls with no surface therefore serve a
        narrow result on purpose; that is the design, not a regression.

        ``group_id`` is coerced through ``coerce_group``: with the personal scope
        off (SYNAPSE_PERSONAL_SCOPE=0) a request for the personal graph is served
        from the technical one, because that is where every fact was written. A
        model that asks for a scope this deployment does not run gets answers
        instead of an empty graph.

        A ``warnings`` list is attached when a leg degraded (embedding or rerank
        backend down, KG or notes leg errored). The key is ABSENT on a healthy call,
        so the response shape is unchanged for every existing consumer. Empty buckets
        WITH a warning mean broken retrieval, not empty memory.
        """
        warnings: list[str] = []
        with _warn_sink(warnings):
            return self._recall_inner(
                query=query,
                warnings=warnings,
                project=project,
                session_focus=session_focus,
                group_id=group_id,
                write_feedback=write_feedback,
                source=source,
                debug=debug,
                self_session=self_session,
                surface=surface,
                trust=trust,
            )

    def _recall_inner(
        self,
        query: str,
        warnings: list[str],
        project: str | None = None,
        session_focus: list[str] | None = None,
        group_id: str = "technical",
        write_feedback: bool = True,
        source: str | None = None,
        debug: bool = False,
        self_session: str | None = None,
        surface: str | None = None,
        trust: SurfaceTrust | None = None,
    ) -> dict[str, Any]:
        """recall()'s body, run with ``warnings`` bound as the active degradation sink."""
        settings = self._settings()
        group_id = coerce_group(group_id) or "technical"
        t_start = time.perf_counter()
        ex = self._leg_executor
        st = self._resolve_trust(surface, trust)
        allowed = st.project_filter

        # BM25 is pure text search — it does NOT need the query embedding. Start it
        # FIRST so its ~165ms fetch overlaps the ~170ms Voyage query-embedding call
        # below, instead of running after it (the embed gates the vector/KG/web legs,
        # but not BM25). Each leg owns a thread-local PG connection, so concurrent
        # psycopg use is safe. Legs run through _timed for per-leg latency telemetry.
        f_bm25 = _submit_ctx(
            ex,
            _timed,
            self._search_bm25_episodes,
            query,
            project,
            settings._EPISODE_FETCH,
            None,
            allowed,
        )

        t_emb = time.perf_counter()
        try:
            query_emb = self._ensure_embedder().embed([query], task="query")[0]
        except Exception as e:
            logger.error("Embedding query failed: %s", e)
            detail = _err_brief(e)
            backend = _embedding.embed_provider()
            _warn(
                f"embedding failed ({backend}: {detail}): vector legs skipped, "
                f"results are BM25-only."
                + _config_hint(detail, backend=backend, env_prefix="SYNAPSE_EMBED")
            )
            query_emb = None
        ms_embed = (time.perf_counter() - t_emb) * 1000.0

        # The remaining legs all need the embedding; fan them out concurrently (they
        # no-op when it's unavailable). Summaries were retired (task #63): the KG owns
        # facts and the wide episode leg owns broad/needle, so no synth_documents leg.
        def _web_leg() -> list[dict[str, Any]]:
            return self._search_web_reranked(query, query_emb) if query_emb is not None else []

        def _kg_leg() -> tuple[list[Any], list[Any]]:
            # v1 KG posture on a restricted surface: SKIP. kg_relationships carries no
            # project column, and joining back through source episodes to derive one
            # isn't worth the per-query cost yet — so there is no way to filter facts,
            # and serving none is the only fail-closed option.
            if query_emb is None:
                _warn("KG facts leg skipped: no query embedding, so the facts bucket is empty.")
                return [], []
            if st.restricted:
                return [], []
            return self._search_kg(
                query, query_emb, group_id, session_focus or [], fact_limit=settings._FACT_LIMIT
            )

        f_vec = (
            _submit_ctx(
                ex,
                _timed,
                self._search_vector_episodes,
                query_emb,
                project,
                settings._EPISODE_FETCH,
                None,
                allowed,
            )
            if query_emb is not None
            else None
        )
        f_web = _submit_ctx(ex, _timed, _web_leg)
        f_kg = _submit_ctx(ex, _timed, _kg_leg)

        # Notes leg: hook-KNN + rerank floor over the curated notes store (the board's
        # searchable other half). Reuses this call's query embedding; no-ops without it.
        def _notes_leg() -> list[dict[str, Any]]:
            if query_emb is None:
                return []
            return self._search_notes(query, query_emb, project, audience=st.audience_filter)

        f_notes = _submit_ctx(ex, _timed, _notes_leg) if settings._NOTES_IN_RECALL else None

        # Fuse BM25 + vector into the rerank pool — identical to _episode_pool's output
        # (used by recall_episodes), just with BM25 hoisted ahead of the embed.
        bm25_eps, ms_bm25 = f_bm25.result()
        vec_eps, ms_vec = f_vec.result() if f_vec is not None else ([], 0.0)
        ep_pool = _merge_rrf(bm25_eps, vec_eps, id_key="id")[: settings._EPISODE_RERANK_POOL]
        web_ranked, ms_web = f_web.result()
        (kg_results, _seed_entities), ms_kg = f_kg.result()  # entities display retired

        facts_internal = kg_results[
            : settings._FACT_LIMIT
        ]  # carry _uuid for bump + superseded pairs
        # Feedback loop: bump retrieval_count on every surfaced edge so frequent hits
        # float higher next time. Already fire-and-forget — never blocks the response.
        surfaced_edge_uuids = [f["_uuid"] for f in facts_internal if f.get("_uuid")]
        if surfaced_edge_uuids and write_feedback:
            self._increment_fact_retrieval_counts(surfaced_edge_uuids, group_id)

        # Second wave, also concurrent: the cross-encoder rerank (Voyage HTTP) and the
        # bi-temporal superseded-pairs fetch hit different backends. Use the SCORED rerank — same
        # ordering as _rerank_pool, but it also yields the top relevance score (a recall-
        # confidence signal, and the basis for an eventual inject-only-if-relevant gate).
        f_rerank = _submit_ctx(ex, _timed, self._rerank_pool_scored, query, ep_pool)
        f_superseded = _submit_ctx(
            ex,
            self._fetch_superseded_pairs_pg,
            group_id,
            surfaced_edge_uuids,
            settings._SUPERSEDED_LIMIT,
        )
        scored, ms_rerank = f_rerank.result()
        superseded_facts = f_superseded.result()
        rerank_top = scored[0][1] if scored else 0.0  # RAW top score (telemetry) — pre-recency
        # Post-rerank recency re-injection: the cross-encoder is recency-blind, so an old
        # *definitive* claim out-ranks a newer *correction* when both make the pool. Re-weight
        # the FINAL ordering only; the rerank call, the pool, and rerank_top above are untouched.
        scored = self._apply_rerank_recency(scored, ep_pool)
        ranked = [ep_pool[i] for i, _ in scored]

        # Episodes: rerank order, then RRF-fused with the pool's BM25 order for lexical recovery
        # (see _RECALL_BM25_FUSE). Skipped on the degraded path (rerank_top <= 0): the pool is
        # already RRF(bm25, vector) there, so re-fusing BM25 would double-count it.
        ranked_eps = [x for x in ranked if x.get("doc_type") == "episode"]
        n_self_excluded = 0
        if settings._RECALL_SELF_EXCLUDE and self_session:
            pre_excl = len(ranked_eps)
            ranked_eps = self._exclude_self(ranked_eps, self_session)
            n_self_excluded = pre_excl - len(ranked_eps)
        n_bm25_lifted = 0  # telemetry: episodes fusion pulled INTO the src_k serving window
        if settings._RECALL_BM25_FUSE and rerank_top > 0.0:
            pre_fuse = {e.get("id") for e in ranked_eps[: settings._RECALL_PASSAGE_SRC_K]}
            ranked_eps = self._fuse_bm25_order(ranked_eps)
            n_bm25_lifted = len(
                {e.get("id") for e in ranked_eps[: settings._RECALL_PASSAGE_SRC_K]} - pre_fuse
            )
        # Query-echo suppression: drop episodes that are the prompt quoting itself (compaction
        # copies / re-ingested repeats); the slices below backfill freed slots from next-ranked.
        # Passage mining reads the top _RECALL_PASSAGE_SRC_K, so that bounds the lazy scan.
        keep, n_echo_suppressed = self._filter_query_echo(
            query, ranked_eps, max(settings._RECALL_EPISODE_LIMIT, settings._RECALL_PASSAGE_SRC_K)
        )
        if n_echo_suppressed:
            ranked_eps = [ranked_eps[i] for i in keep]
        # Stage 2: serve compact passages of the top reranked episodes instead of whole
        # turns. NO fallback: compaction yielding nothing means no passage cleared the
        # bar — LOW relevance — and low relevance must cost FEWER tokens, not more. The
        # old fallback dumped whole turns here, so a no-match query circular-matched the
        # current session's own freshly-ingested turns and blew est_tokens up to ~25k.
        # When compaction is empty the bucket is simply omitted (empty container); the
        # drill-down paths (recall_episodes / fetch) still return full turns on demand.
        ep_items: list[dict[str, Any]] | None = None
        if ranked_eps:
            ep_items = (
                self._compact_to_passages(
                    query, ranked_eps[: settings._RECALL_PASSAGE_SRC_K], settings._RECALL_PASSAGE_N
                )
                or None
            )
        # Floor enforcement: when even the top episode passage is below the floor — a weak
        # match under WORKING retrieval (query_emb present, a real score in (0, floor)) —
        # drop the episode bucket. Complements the compaction gate above at a harder
        # threshold and keeps low relevance cheap. Facts/timeline stay (their own
        # relevance gates). The same condition is still shadow-marked in served_ids below.
        # recall_episodes() (drill-down) does NOT enforce — the caller asked for turns.
        if (
            settings._RECALL_FLOOR_ENFORCE
            and query_emb is not None
            and 0.0 < rerank_top < settings._RECALL_FLOOR
        ):
            if settings._RECALL_FLOOR_KEEP_MIN > 0 and ep_items:
                ep_items = ep_items[: settings._RECALL_FLOOR_KEEP_MIN] or None
            else:
                ep_items = None
        # Web bucket: fused (BM25+vector) -> cross-encoder rerank -> _WEB_FLOOR -> dedupe, all done
        # in _web_leg (_search_web_reranked). Floor self-gates on intent, so this is often empty.
        web_chunks = web_ranked

        # Surface the internal _uuid to the caller as a "f:<uuid>" id (below) so facts
        # are citable in recall_feedback, same as episodes carry "e:N".
        # Optional relevance gate (SYNAPSE_RECALL_FACT_FLOOR > 0): drop off-topic facts —
        # the one place the "recall returns irrelevant stuff" lever measurably works. OFF by
        # default (adds one rerank of the served facts), so this is a no-op until enabled.
        served_facts = facts_internal
        if settings._RECALL_FACT_FLOOR > 0 and len(served_facts) > 1:
            served_facts = self._floor_facts(query, served_facts)
        # Supersession surface: if the query matched a now-invalid fact, pull in its CURRENT successor
        # (deduped) so a query about something that changed still gets today's answer, not nothing.
        sup_extras = self._surface_supersessions(
            query_emb, group_id, {f.get("_uuid") for f in served_facts if f.get("_uuid")}
        )
        if sup_extras:
            served_facts = list(served_facts) + sup_extras
        # Slim facts to {fact, date} — date = t_valid (when the fact became true), so the reader
        # can weight currency. Served facts are already live (invalidated edges filtered upstream).
        facts: list[dict[str, Any]] = []
        for f in served_facts:
            item: dict[str, Any] = {"fact": f["fact"]}
            if (uid := f.get("_uuid")) is not None:
                item["id"] = f"f:{uid}"  # KG edge uuid — cite in recall_feedback (not fetch())
            if (d := f.get("_date")) is not None:
                item["date"] = str(d)[:10]
            facts.append(item)

        # Episode-validity overlay: if a served episode/passage asserted a claim the KG has since
        # superseded, attach the CURRENT fact (via the invalidated_by link). Augments, never replaces
        # — the turn is immutable history and usually carries more than the stale claim. Deduped
        # against the facts bucket above. Cheap (partial GIN, fail-open); usually a no-op.
        if ep_items:
            sup = self._episode_supersessions(
                _parse_episode_ids([it.get("id") for it in ep_items if it.get("id")]),
                group_id,
            )
            if sup:
                _apply_supersessions(ep_items, sup, {f["fact"] for f in facts})

        # Entity DISPLAY bucket retired 2026-07-23: seed entities are name-keyed
        # side summaries (often just filenames), uninstrumented and uncitable, and
        # only ever displayed — the KG leg already consumed them internally to seed
        # fact retrieval before returning. Dropping the display costs no recall
        # quality and reclaims the tokens. seed_entities is now unused (see above).
        out: dict[str, Any] = {
            "query": query,
            "facts": facts,  # slim {fact: ...}
        }
        if ep_items:
            out["episodes"] = ep_items
        if web_chunks:
            out["web"] = [_to_web_recall_item(r) for r in web_chunks]
        if superseded_facts:
            # Renamed from "history" 2026-07-18 — the displaced version of each served
            # fact, keyed like the superseded_by columns elsewhere in the system.
            out["superseded_facts"] = superseded_facts  # {fact: old, superseded_by: current}
        note_items: list[dict[str, Any]] = []
        ms_notes = 0.0
        if f_notes is not None:
            try:
                note_items, ms_notes = f_notes.result()
            except Exception as e:
                logger.warning("notes leg failed: %s", e)
                _warn(f"notes leg failed ({_err_brief(e)}): no notes served.")
        if note_items:
            out["notes"] = note_items

        # Degradation notices (see the _WARN_SINK block up top). Present ONLY when a leg
        # actually degraded: a healthy recall carries no `warnings` key at all, so nothing
        # downstream has to learn a new field to keep working. `_served_chars` deliberately
        # does not count these: they are diagnostics, not served memory, and counting them
        # would move est_tokens telemetry on exactly the calls that are already anomalous.
        if warnings:
            out["warnings"] = list(warnings)

        # Fire-and-forget telemetry to recall_metrics (NOT logfire) — same background-write
        # pattern as the retrieval_count bump, so zero read-path latency.
        # served_ids (issue #10): WHICH results were served, per bucket. Episodes dedupe
        # because passages share a parent id.
        served_ids: dict[str, Any] = {
            "episodes": list(dict.fromkeys(it["id"] for it in (ep_items or []) if it.get("id"))),
            "facts": [f["_uuid"] for f in served_facts if f.get("_uuid")],
            "web": [c["id"] for c in web_chunks if c.get("id")],
            "notes": [it["id"] for it in note_items if it.get("id")],
            "n_echo_suppressed": n_echo_suppressed,
            "n_bm25_lifted": n_bm25_lifted,  # BM25 fusion recovered these into the served window
            # Trust verdict (schema 053): a restricted serve is narrower by design, so
            # the metrics have to say which regime produced these numbers.
            "trust": st.trust,
        }
        # Self-exclusion observability: keys present ONLY when the hook delivered a
        # session id, so hookless traffic (bench, dashboard) keeps the lean envelope
        # and "exclusion active but dropped nothing" (count 0) stays distinguishable
        # from "exclusion never ran" (keys absent).
        if self_session:
            served_ids["self_session"] = self_session
            served_ids["n_self_excluded"] = n_self_excluded
        # Shadow abstention floor (telemetry only): mark when an enforced floor WOULD have
        # abstained. Compares the RAW pre-recency rerank_top recorded below — NOT the
        # recency-adjusted ordering — so the marker and rerank_top_score always agree.
        self._floor_shadow(served_ids, float(rerank_top), emb_ok=query_emb is not None)
        chars = _served_chars(out)
        metrics: dict[str, Any] = {
            "kind": "recall",
            "source": source or "mcp",
            "query": query[:200],
            "group_id": group_id,
            "write_feedback": write_feedback,
            "ms_total": round((time.perf_counter() - t_start) * 1000.0, 1),
            "ms_embed": round(ms_embed, 1),
            "ms_bm25": round(ms_bm25, 1),
            "ms_vector": round(ms_vec, 1),
            "ms_kg": round(ms_kg, 1),
            "ms_web": round(ms_web, 1),
            "ms_rerank": round(ms_rerank, 1),
            "n_facts": len(facts),
            "n_episodes": len(out.get("episodes", [])),
            "n_web": len(web_chunks),
            "n_history": len(superseded_facts),
            "n_notes": len(note_items),
            "ms_notes": round(ms_notes, 1),
            "chars": chars,
            "est_tokens": chars // 4,
            "pool_bm25": len(bm25_eps),
            "pool_vector": len(vec_eps),
            "pool_fused": len(ep_pool),
            "kg_candidates": len(kg_results),
            "rerank_model": _embedding._RERANK_MODEL,
            "rerank_top_score": round(float(rerank_top), 4),
            "emb_ok": query_emb is not None,
            "served_ids": served_ids,
        }
        self._record_metrics(metrics)
        # Phase-2 dashboard debug envelope: surface the SAME numbers just recorded (no
        # re-instrumentation). Only the timed legs are exposed; the timeline key is
        # OMITTED when its leg is disabled (f_timeline is None), so the console renders
        # it as untimed/skipped rather than a spurious 0ms. Byte-identical when off.
        if debug:
            legs_ms: dict[str, Any] = {
                "embed": metrics["ms_embed"],
                "bm25": metrics["ms_bm25"],
                "vector": metrics["ms_vector"],
                "kg": metrics["ms_kg"],
                "web": metrics["ms_web"],
                "rerank": metrics["ms_rerank"],
            }
            if f_notes is not None:
                legs_ms["notes"] = metrics["ms_notes"]
            out["debug"] = {
                "total_ms": metrics["ms_total"],
                "legs_ms": legs_ms,
                "pool_sizes": {
                    "bm25": metrics["pool_bm25"],
                    "vector": metrics["pool_vector"],
                    "fused": metrics["pool_fused"],
                    "kg_candidates": metrics["kg_candidates"],
                },
                "rerank": {
                    "model": metrics["rerank_model"],
                    "top_score": metrics["rerank_top_score"],
                },
                "est_tokens": metrics["est_tokens"],
            }
        return out
