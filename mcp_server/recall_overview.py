"""RecallOverviewMixin operations."""

from __future__ import annotations

import functools
import logging
import time
from typing import Any

from ingestion import embedding as _embedding
from ingestion.scope import coerce_group
from ingestion.surfaces import SurfaceTrust
from mcp_server.recall_presentation import apply_supersessions as _apply_supersessions
from mcp_server.recall_presentation import fact_day as _fact_day
from mcp_server.recall_presentation import parse_episode_ids as _parse_episode_ids
from mcp_server.recall_presentation import to_web_recall_item as _to_web_recall_item
from mcp_server.recall_ranking import served_chars as _served_chars
from mcp_server.recall_ranking import timed as _timed
from mcp_server.recall_warnings import error_brief as _err_brief
from mcp_server.recall_warnings import submit_with_context as _submit_ctx
from mcp_server.recall_warnings import warn as _warn
from mcp_server.recall_warnings import warn_sink as _warn_sink

logger = logging.getLogger(__name__)

#: Extra facts asked of the KG leg when the call carries ``self_session``, so facts sourced
#: only from the calling session (2.6% of served facts before exclusion) can drop out
#: without leaving the bucket short.
_SELF_FACT_HEADROOM = 5


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
        by the HTTP route so non-agentic recalls (the dashboard console) don't count as
        served in the dashboard. That counter ranks nothing: only the episode counter,
        bumped by the drill-down tools, feeds ranking (``feedback_multiplier``).

        ``source`` tags the call origin (e.g. "mcp-tool", "http", "recall-hook:session")
        on the recall_metrics row so per-origin metrics are filterable in SQL.

        The episode bucket serves compact passages (markdown chunks) of the top reranked episodes
        instead of whole turns (see RecallConfig.recall_passage_n). For raw full-episode
        drill-down, use recall_episodes() (the recall_full_turns tool). Facts carry their t_valid
        "as-of" date for currency weighting.

        ``debug`` (phase-2 dashboard console) attaches a ``debug`` key to the response SURFACING
        the SAME numbers already measured for the recall_metrics telemetry row — no extra
        instrumentation, no extra work. Off by default so every non-dashboard call is byte-identical
        in behavior AND in the telemetry it records (a live call-rate A/B depends on that). See the
        debug-dict assembly just below the metrics write for the exact shape.

        ``surface`` is the calling host's id (schema 053). Anything but a surface
        registered ``trust='full'`` — including a missing one — is RESTRICTED: episodes
        are filtered to the surface's project allowlist (plus, for a credential-bound
        surface, the episodes it ingested itself: schema 057 ``surface_id``), notes to
        ``audience='work-safe'``, no web chunks are served (web rows carry no project
        provenance), and KG facts to those whose provenance is known and
        lies wholly inside the allowlist (schema 056 ``source_projects``: every source
        episode exists, has a project, and that project is allowed). The same rule
        covers every KG-derived extra (superseded pairs, the supersession surface, the
        episode-validity overlay). An empty allowlist, or a database without schema 056,
        serves no facts at all. Bare calls with no surface therefore serve a narrow
        result on purpose; that is the design, not a regression.

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
        cfg = self._config
        group_id = coerce_group(group_id) or "technical"
        t_start = time.perf_counter()
        ex = self._leg_executor
        st = self._resolve_trust(surface, trust)
        allowed = st.project_filter
        # Own-surface widening of the episode legs (schema 057): None unless the caller is
        # a credential-bound restricted surface AND the provenance column exists.
        own = self._own_surface(st)
        # KG provenance scope (schema 056). Full trust: no kwargs, so every KG call below
        # is exactly the call it always was. Restricted: each KG read gets the allowlist,
        # and an EMPTY allowlist skips them outright, because no fact can be a non-empty
        # subset of nothing.
        kg_scope: dict[str, Any] = {"allowed_projects": allowed} if st.restricted else {}
        kg_off = st.restricted and not allowed

        # The episode legs: BM25 starts before the query embedding so its fetch overlaps
        # that Voyage call, and the vector leg follows the embedding (see
        # _start_episode_legs). Each leg owns a pooled PG connection, so concurrent psycopg
        # use is safe. Legs run through _timed for per-leg latency telemetry.
        legs = self._start_episode_legs(query, project, allowed_projects=allowed, own_surface=own)
        query_emb = legs.query_emb
        ms_embed = legs.ms_embed

        # Self-exclusion asks the KG for a few extra facts, so those the calling session
        # sourced alone can drop out and the bucket still fills to fact_limit.
        fact_limit = cfg.fact_limit + (_SELF_FACT_HEADROOM if self_session else 0)

        # The remaining legs all need the embedding; fan them out concurrently (they
        # no-op when it's unavailable). Summaries were retired (task #63): the KG owns
        # facts and the wide episode leg owns broad/needle, so no synth_documents leg.
        def _kg_leg() -> tuple[list[Any], list[Any]]:
            # Restricted surface: serve only facts whose cached provenance set
            # (kg_relationships.source_projects, schema 056) is known, non-empty and a
            # subset of the allowlist. Mixed or unknown provenance is never served. With
            # an empty allowlist, or before 056 is applied, the leg serves nothing,
            # exactly the pre-056 skip (see _search_kg / _kg_scope_ready).
            if query_emb is None:
                _warn("KG facts leg skipped: no query embedding, so the facts bucket is empty.")
                return [], []
            if kg_off:
                return [], []
            return self._search_kg(
                query,
                query_emb,
                group_id,
                session_focus or [],
                fact_limit=fact_limit,
                **kg_scope,
            )

        # Web rows carry no project provenance and the web SQL has no scope predicate, so a
        # restricted caller (an unknown one included) is served no web bucket at all.
        f_web = (
            _submit_ctx(ex, _timed, self._search_web_reranked, query, query_emb)
            if query_emb is not None and not st.restricted
            else None
        )
        f_kg = _submit_ctx(ex, _timed, _kg_leg)
        # Supersession surface: if the query matched a now-invalid fact, pull in its CURRENT
        # successor so a query about something that changed still gets today's answer. It
        # needs only the embedding, so it runs beside the KG leg; dedupe against the served
        # facts happens once both are in.
        f_sup = (
            _submit_ctx(
                ex,
                functools.partial(self._surface_supersessions, **kg_scope),
                query_emb,
                group_id,
            )
            if query_emb is not None and not kg_off
            else None
        )

        # Notes leg: hook-KNN + rerank floor over the curated notes store (the board's
        # searchable other half). Reuses this call's query embedding; no-ops without it.
        def _notes_leg() -> list[dict[str, Any]]:
            if query_emb is None:
                return []
            return self._search_notes(query, query_emb, project, audience=st.audience_filter)

        f_notes = _submit_ctx(ex, _timed, _notes_leg)

        # Join the episode legs and start the cross-encoder rerank (Voyage HTTP, the largest
        # call) at once: it needs only this pool, so it runs while the web and KG legs and
        # the fact bookkeeping below are still in flight. Use the SCORED rerank: it also
        # yields the top relevance score (a recall-confidence signal, and the basis for the
        # abstention floor). The calling session's own turns are already out of the pool.
        pool = self._join_episode_pool(legs, self_session)
        ep_pool = pool.items
        f_rerank = _submit_ctx(ex, _timed, self._rerank_pool_scored, query, ep_pool)

        (kg_results, _seed_entities), ms_kg = f_kg.result()  # entities display retired
        sup_candidates = f_sup.result() if f_sup is not None else []
        # Self-exclusion, facts half: drop facts sourced only from the calling session, in
        # ONE provenance lookup over the facts and the supersession successors. Runs before
        # the retrieval_count bump so an unserved fact is not reinforced.
        n_self_facts_excluded = 0
        if self_session:
            own_facts = self._self_session_facts(
                [f["_uuid"] for f in (*kg_results, *sup_candidates) if f.get("_uuid")],
                self_session,
            )
            if own_facts:
                kg_results = [f for f in kg_results if f.get("_uuid") not in own_facts]
                sup_candidates = [f for f in sup_candidates if f.get("_uuid") not in own_facts]
                n_self_facts_excluded = len(own_facts)
        facts_internal = kg_results[: cfg.fact_limit]  # carry _uuid for bump + superseded
        # Bump retrieval_count on every surfaced edge: the dashboard's "served" count.
        # Nothing in ranking reads the fact counter. Fire-and-forget, never blocks the
        # response.
        surfaced_edge_uuids = [f["_uuid"] for f in facts_internal if f.get("_uuid")]
        if surfaced_edge_uuids and write_feedback:
            self._increment_fact_retrieval_counts(surfaced_edge_uuids, group_id)

        # The bi-temporal superseded-pairs fetch hits a different backend than the rerank.
        f_superseded = (
            None
            if kg_off
            else _submit_ctx(
                ex,
                functools.partial(self._fetch_superseded_pairs_pg, **kg_scope),
                group_id,
                surfaced_edge_uuids,
                cfg.superseded_limit,
            )
        )
        scored, ms_rerank = f_rerank.result()
        superseded_facts = f_superseded.result() if f_superseded is not None else []
        # Post-rerank recency re-injection (the cross-encoder is recency-blind, so an old
        # *definitive* claim out-ranks a newer *correction* when both make the pool), then
        # query-echo suppression, in _rank_episodes: the same ranking contract
        # recall_episodes() serves. The slices below backfill freed slots from next-ranked.
        # Passage mining reads the top recall_passage_src_k, so that bounds the lazy scan.
        # ``rerank_top`` is the RAW (pre-recency) top score among the episodes that survive
        # echo suppression, i.e. of what gets mined; ``rerank_top_all`` is the pool's own
        # top, recorded when the two differ so SYNAPSE_RECALL_FLOOR can be recalibrated.
        ranked_eps, rerank_top, n_echo_suppressed = self._rank_episodes(
            query,
            ep_pool,
            scored,
            max(cfg.recall_episode_limit, cfg.recall_passage_src_k),
        )
        rerank_top_all = scored[0][1] if scored else 0.0
        # Stage 2: serve compact passages of the top reranked episodes instead of whole
        # turns. NO fallback: compaction yielding nothing means no passage cleared the
        # bar — LOW relevance — and low relevance must cost FEWER tokens, not more. The
        # old fallback dumped whole turns here, so a no-match query circular-matched the
        # current session's own freshly-ingested turns and blew est_tokens up to ~25k.
        # When compaction is empty the bucket is simply omitted (empty container); the
        # drill-down paths (recall_episodes / fetch) still return full turns on demand.
        ep_items: list[dict[str, Any]] | None = None
        passage_stats: dict[str, int] = {}
        if ranked_eps:
            ep_items = (
                self._compact_to_passages(
                    query,
                    ranked_eps[: cfg.recall_passage_src_k],
                    cfg.recall_passage_n,
                    stats=passage_stats,
                )
                or None
            )
        # Abstention floor: when even the top episode passage is below the floor — a weak
        # match under WORKING retrieval (query_emb present, a real score in (0, floor)) —
        # drop the episode bucket. Complements the compaction gate above at a harder
        # threshold and keeps low relevance cheap. Facts stay (they have their own gating);
        # the same condition is marked in served_ids below. recall_episodes() (drill-down)
        # does NOT enforce — the caller asked for turns.
        if query_emb is not None and 0.0 < rerank_top < cfg.recall_floor:
            ep_items = None
        # Web bucket: fused (BM25+vector) -> cross-encoder rerank -> web_floor -> dedupe, all done
        # in _search_web_reranked. Floor self-gates on intent, so this is often empty.
        web_chunks, ms_web = f_web.result() if f_web is not None else ([], 0.0)

        # Facts are cited in recall_feedback as "f:<kg_relationships.id>" (below), the way
        # episodes carry "e:N".
        served_facts = facts_internal
        # Supersession extras: the successors not already served, up to sup_limit, added
        # beyond fact_limit. Go-forward coverage only; the stale fact itself is never served.
        served_uuids = {f.get("_uuid") for f in served_facts if f.get("_uuid")}
        sup_extras = []
        for f in sup_candidates:
            if f["_uuid"] in served_uuids:
                continue
            served_uuids.add(f["_uuid"])
            sup_extras.append(f)
            if len(sup_extras) >= cfg.sup_limit:
                break
        if sup_extras:
            served_facts = list(served_facts) + sup_extras
        # Slim facts to {fact, date, ongoing?, supported?}. date = t_valid: when the fact became
        # true AS STORED (an in-text date, else the conversation it came from) — it is NOT the
        # last time anyone confirmed it. Schema 058 adds the currency signals: `ongoing` marks
        # a claim that can lapse without being contradicted (a habit, a tool in use), and
        # `supported` is the date the user themself last asserted it; absent = never
        # user-confirmed or pre-058. Served facts are already live (invalidated edges
        # filtered upstream).
        facts: list[dict[str, Any]] = []
        for f in served_facts:
            item: dict[str, Any] = {"fact": f["fact"]}
            if (fact_id := f.get("_id")) is not None:
                item["id"] = f"f:{fact_id}"  # KG edge id — cite in recall_feedback (not fetch())
            if (d := f.get("_date")) is not None:
                item["date"] = _fact_day(d)
            if f.get("_ongoing"):
                item["ongoing"] = True
            if (sup := f.get("_supported")) is not None:
                item["supported"] = _fact_day(sup)
            facts.append(item)

        # Episode-validity overlay: if a served episode/passage asserted a claim the KG has since
        # superseded, attach the CURRENT fact (via the invalidated_by link). Augments, never replaces
        # — the turn is immutable history and usually carries more than the stale claim. Deduped
        # against the facts bucket above. Cheap (partial GIN, fail-open); usually a no-op.
        if ep_items and not kg_off:
            served_text: dict[int, str] = {}
            for it in ep_items:
                for eid in _parse_episode_ids([it.get("id")]):
                    served_text[eid] = served_text.get(eid, "") + (it.get("content") or "")
            sup = self._episode_supersessions(
                list(served_text),
                group_id,
                self_session=self_session,
                served_text=served_text,
                **kg_scope,
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
            "facts": [f"f:{f['_id']}" for f in served_facts if f.get("_id") is not None],
            "web": [c["id"] for c in web_chunks if c.get("id")],
            "notes": [it["id"] for it in note_items if it.get("id")],
            "n_echo_suppressed": n_echo_suppressed,
            # Passages skipped as repeats of another episode's chosen passage.
            "n_dup_passages": passage_stats.get("n_dup_passages", 0),
            # Chunks the rerank never saw because the episodes overran recall_passage_cand.
            "n_chunks_cut": passage_stats.get("n_chunks_cut", 0),
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
            served_ids["n_self_excluded"] = pool.n_self_excluded
            served_ids["n_self_facts_excluded"] = n_self_facts_excluded
        if rerank_top_all != rerank_top:
            served_ids["rerank_top_all"] = round(float(rerank_top_all), 4)
        # Abstention-floor marker (telemetry): the floor fired and dropped the episode bucket.
        # Compares the RAW pre-recency rerank_top recorded below — NOT the recency-adjusted
        # ordering — so the marker and rerank_top_score always agree.
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
            "ms_bm25": round(pool.ms_bm25, 1),
            "ms_vector": round(pool.ms_vector, 1),
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
            "pool_bm25": pool.n_bm25,
            "pool_vector": pool.n_vector,
            "pool_fused": len(ep_pool),
            "kg_candidates": len(kg_results),
            "rerank_model": _embedding._RERANK_MODEL,
            "rerank_top_score": round(float(rerank_top), 4),
            "emb_ok": query_emb is not None,
            "served_ids": served_ids,
        }
        self._record_metrics(metrics)
        # Phase-2 dashboard debug envelope: surface the SAME numbers just recorded (no
        # re-instrumentation). Only the timed legs are exposed. Byte-identical when off.
        if debug:
            legs_ms: dict[str, Any] = {
                "embed": metrics["ms_embed"],
                "bm25": metrics["ms_bm25"],
                "vector": metrics["ms_vector"],
                "kg": metrics["ms_kg"],
                "web": metrics["ms_web"],
                "rerank": metrics["ms_rerank"],
                "notes": metrics["ms_notes"],
            }
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
