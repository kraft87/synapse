"""recall() — the core retrieval function for Synapse.

Document types served by recall():
  1. Episodes  — individual user-turn exchanges, via a deep wide-pool rerank (WIN1)
  2. KG facts  — entity/relationship triples (extracted from chunks)

Search strategy:
  - BM25 (ParadeDB) + ANN cosine over episodes
  - KG vector seed + graph traversal for facts / entities
  - ONE cross-encoder rerank over the episode pool

KG legs serve from Postgres (kg_entities / kg_relationships, task #67).
The cutover was judged quality-EQUAL to FalkorDB on the relational golden set
(delta -0.017, CI spans zero; scripts/ab_kg_pg_quality.py) and fixed FalkorDB's
silently-dead BM25 leg (AND/phrase semantics returned 0 results on real
multi-word queries). FalkorDB itself was decommissioned in #67 PR 3, so the
``SYNAPSE_KG_READ`` rollback seam is gone; a KG-leg failure degrades to an
empty facts bucket rather than failing the whole recall.

The communities bucket was retired with the cutover: never measured as
contributing (absent from every layer ablation), stale since the Stage-4
community refresh was shelved, and it was the last FalkorDB read in recall.

recall_episodes() exposes the raw episode drill-down. The summary layer and the
chunk bucket were both retired (task #63): the KG owns facts and the direct
episode leg owns broad/needle, so neither earned its serving slot.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import psycopg
from psycopg.rows import dict_row

from ingestion import embedding as _embedding
from ingestion.surfaces import SurfaceTrust, lookup_surface
from mcp_server.recall_episode_search import RecallEpisodeSearchMixin
from mcp_server.recall_episodes import RecallEpisodesMixin
from mcp_server.recall_fetch import RecallFetchMixin
from mcp_server.recall_metrics import RecallMetricsMixin
from mcp_server.recall_overview import RecallOverviewMixin
from mcp_server.recall_passages import RecallPassagesMixin
from mcp_server.recall_presentation import apply_supersessions as _apply_supersessions
from mcp_server.recall_presentation import parse_episode_ids as _parse_episode_ids
from mcp_server.recall_presentation import passage_role as _passage_role
from mcp_server.recall_presentation import role_spans as _role_spans
from mcp_server.recall_presentation import to_recall_item as _to_recall_item
from mcp_server.recall_presentation import to_web_recall_item as _to_web_recall_item
from mcp_server.recall_query import bm25_best_window as _bm25_best_window
from mcp_server.recall_query import bm25_tokenize as _bm25_tokenize
from mcp_server.recall_query import echo_lcs_len as _echo_lcs_len
from mcp_server.recall_query import normalize_whitespace as _norm_ws
from mcp_server.recall_query import query_shingles as _query_shingles
from mcp_server.recall_ranking import cutoff_k as _cutoff_k
from mcp_server.recall_ranking import merge_rrf as _merge_rrf
from mcp_server.recall_ranking import recency_multiplier as _recency_multiplier
from mcp_server.recall_ranking import served_chars as _served_chars
from mcp_server.recall_ranking import timed as _timed
from mcp_server.recall_rerank import RecallRerankMixin
from mcp_server.recall_settings import RecallSettings
from mcp_server.recall_sources import RecallSourcesMixin
from mcp_server.recall_warnings import config_hint as _config_hint
from mcp_server.recall_warnings import error_brief as _err_brief
from mcp_server.recall_warnings import submit_with_context as _submit_ctx
from mcp_server.recall_warnings import warn as _warn
from mcp_server.recall_warnings import warn_sink as _warn_sink

_KG_OWNER = os.environ.get("SYNAPSE_KG_OWNER_ID", "default")

# Embedding width for every vector/halfvec SQL cast below. Resolved once from
# SYNAPSE_EMBED_DIMS (default 2048 — Voyage prod, SQL byte-identical); must match
# the width the schema was provisioned with or the halfvec HNSW index expressions
# don't apply. See ingestion/embedding.py (dims are validated against synapse_meta).
_EMBED_DIMS = _embedding.embed_dims()

# Sentinel for the lazily-built reranker: ``None`` is a VALID resolved value
# (SYNAPSE_RERANK_PROVIDER=none — fusion-only serving), so "unset" needs its own marker.
_RERANKER_UNSET = object()

_KG_CANDIDATE_LIMIT = 30
_RRF_K = 60
# The inline timeline leg (schema 033) was REMOVED from recall() on 2026-08-07 along
# with the standalone recall_timeline tool. Prod ran the leg off (SYNAPSE_RECALL_TIMELINE=0)
# from 2026-07-28: 0 helpful citations in 100 rated recalls while eating 524 served slots,
# and no chronology gaps appeared in feedback missing-reports during the off-window.
# Lifetime t: citations: 39 helpful / 92 noise (30% precision, below facts). The
# timeline_events store, its ingestion gate, and the board's "Last 7 days" block
# (timeline_routes._recent_events) are untouched — only retrieval is gone.
# The preferences leg (schema 035) was REMOVED from recall on 2026-07-27. Measured on
# recall_feedback + recall_metrics since b6dcaf3 (2026-07-22, when all six buckets became
# citable): prefs served 380 rows across rated queries for 6 helpful citations — 1.6% of
# served, 11.1% of the rated subset, against episodes 39.0% / facts 16.8% / timeline 8.2%.
# It cost ~2.68 items on essentially every recall and bought nothing. The `preferences`
# table and its other consumers (the ingestion gate, /preferences/top + the SessionStart
# block, the dashboard) are untouched — only the recall serving path is gone.
_RECENCY_HALF_LIFE_DAYS = 30  # content this old scores ~50% of today's content
# Recency re-injection AFTER the cross-encoder rerank (see _apply_rerank_recency). The
# reranker is recency-blind and an old *definitive* statement ("X is canonical") out-scores
# a newer *transition* statement ("switched off X to Y"), so upstream RRF-fusion recency is
# discarded by the rerank and stale content eats the limited slots. A shorter half-life here
# (the corpus churns within a single month) lets a fresh doc reclaim a slot from a marginally-
# more-relevant stale one. Tunable knob; validate on the staleness probe before changing.
_RERANK_RECENCY_HALF_LIFE_DAYS = 14
# Kill switch for the post-rerank recency re-weighting (validated ON; env var disables it).
_RERANK_RECENCY = os.getenv("SYNAPSE_RERANK_RECENCY", "1") != "0"
# Floor on the recency multiplier: a 14-day half-life drives a ~3-month-old episode to ~0.01x,
# which would BURY old content a query explicitly asks for. Clamp the multiplier to >= floor so
# old content is dampened at most 1/floor (4x at 0.25), never annihilated — recency breaks ties
# between comparably-relevant docs without overriding a strong old match. Env-tunable.
_RERANK_RECENCY_FLOOR = float(os.getenv("SYNAPSE_RERANK_RECENCY_FLOOR", "0.25") or "0.25")

# Query-echo suppression (backtest 2026-07-08, 100 real prompts): 74% served the prompt's own
# source episode in top-5 (65% at rank 1) — the "memory" was the prompt quoting itself (compaction
# copies, re-ingested repeats). The server can't know the caller's session id (MCP tools carry no
# session context), but a served episode whose content contains a long verbatim run of the query is
# echo, not memory. Drop those and backfill from the next-ranked candidates. Validated ON; env
# disables. Only meaningful once the query is long enough for the overlap threshold to mean anything.
_SUPPRESS_QUERY_ECHO = os.getenv("SYNAPSE_SUPPRESS_QUERY_ECHO", "1") != "0"
_ECHO_MIN_QUERY_LEN = 40  # skip suppression below this normalized-query length (heuristic is noise)
_ECHO_CONTENT_CAP = 8000  # per-doc chars scanned for the overlap (bounds SequenceMatcher work)

_RERANK_POOL = 6  # candidates per type fed to the reranker
_RERANK_DOC_CAP = 4000  # per-doc char cap (~1k tokens) sent to the reranker (bounds tokens).
# Keeps each doc FAR under Voyage rerank-2.5's ~32k-token PER-(query+doc) context limit, so
# no single doc is ever silently truncated. That 32k limit is per query+doc PAIR, NOT a
# total-request budget: verified 2026-06-18 that a 55.6k-token pool reranks in full (a gold
# planted at index 99 returns to rank 1), so _EPISODE_RERANK_POOL total size is unbounded by
# Voyage — pool size is a quality/latency knob, not a truncation risk. Don't re-litigate.
# Long-episode rerank windowing. The cross-encoder only sees each doc's first _RERANK_DOC_CAP
# chars, so an answer in the truncated tail of a long episode is invisible to ranking — measured
# on a back-half golden as retr@5 0.139 vs a 0.917 pool ceiling (scripts/passage_bench_v*.py,
# 2026-06-26). Fix: also feed the reranker the BM25-relevant window of long episodes and score the
# episode by max(head, window). BM25 selection is pure-Python (no embeddings, ~ms) so the read path
# pays only ~14 extra short rerank docs. max() is conservative — it can only RAISE an episode's
# score, never lower the head's — so short episodes are unaffected: validated as a STRICT win
# (tail retr@5 0.139→0.889; natural 0.929→0.929 retr@5, 0.952→0.976 retr@10). Disable with
# SYNAPSE_RERANK_WINDOW=0.
_RERANK_WINDOW = os.getenv("SYNAPSE_RERANK_WINDOW", "1") != "0"
_FACT_LIMIT = 12  # KG facts returned by recall()
# Raised 5→12 (2026-06-03). The KG is the STRONGEST layer on relational/multi-hop
# questions — its real job — where it beats summaries+chunks (kg_only 0.715 vs 0.489,
# marginal +0.307; scripts/kg_relational_baseline.py on a 40-Q relational golden set).
# Serving more of the graph's EXISTING facts (no new nodes/edges, zero graph bloat)
# lifts relational answer quality 0.746→0.826 and DOUBLES exact-fact keyword recall
# 0.119→0.214, while nudging broad +0.015 — safe on all query types (kg_factcount_safety.py).
# A wide-pool fact reranker added only +0.01-0.03 more for a per-recall API call — not
# worth it; just serving more is the win. Costs ~+200 fact-tokens/recall (facts are short).
_EPISODE_LIMIT = 5  # episodes returned by recall_episodes()

# fetch_session — sequential session read (the Read analog; spec 2026-08-07,
# shapes from the July-08 primitives draft: anchor full, neighbors as heads).
_SESSION_RADIUS_DEFAULT = 3  # neighbors per side around the anchor
_SESSION_RADIUS_MAX = 10
_SESSION_PAGE_DEFAULT = 10  # anchorless paging: turns per page
_SESSION_PAGE_MAX = 25
_SESSION_HEAD_CHARS = 500  # neighbor preview size; full_chars says what fetch() expands to
# WIN1 (2026-06-03, episode_pool_rerank): exact-fact misses were pure truncation —
# golds rank 16-88 in a leg but prod fetched only _EPISODE_LIMIT*3=15, so they never
# entered the served pool. Deep-fetch + WIDE-pool cross-encoder recovers them:
# answerability 90.5%→95.2%, exact-gold hit +7pts, served tokens -20%. Plain deeper
# RRF does nothing (can't lift a rank-40 gold) — the reranker is the active ingredient.
_EPISODE_FETCH = 100  # per-leg fetch depth for recall_episodes (was 15)
_EPISODE_RERANK_POOL = 100  # fused candidates fed to the cross-encoder, served down to limit
# (pool 50 → 0.929 answerability at half the rerank payload; 100 → 0.952. Dial down if
#  drill-down latency bites; exact-gold hit is identical 0.881 either way.)
_RECALL_EPISODE_LIMIT = 5  # direct episode turns served by recall()
# 2026-06-04 (episode_count_sweep): with summaries retired, episodes are the primary
# served layer — re-swept n now that they own the budget. Broad synthesis PEAKS at 5
# (0.658) and declines on both sides (n=3 0.566, n=6 0.599) — more turns dilute the
# synthesis; needle hits its early plateau at 4-5 (0.714); relational is n-indifferent
# (facts own it). 3→5 buys broad +0.092 / needle +0.047 for +444 ctx tokens; past 5 is
# strictly worse on broad. Was 3 (chosen when summaries competed for the slots).
# 2026-06-03 (recall_episode_blend): recall()'s chunk-derived bucket was redundant
# (marginal ~0 on needle/broad/relational per layer_contribution.py). Swapping it for the
# SAME direct wide-pool episode leg recall_episodes() uses lifts broad +0.127 and needle
# +0.262 while HOLDING relational (-0.015, noise), at +247 tokens (episodes paid for by
# dropping chunks). Adding episodes ON TOP of chunks hurt relational and cost ~2.7x the
# tokens; the swap is the win. Summaries (not starved: 2.58/3 served) now rank solo.
# recall() co-reranks summaries + the episode pool in ONE cross-encoder call and partitions
# by type (the reranker scores docs independently, so per-type order is identical to ranking
# each alone, in one round-trip). recall() keeps the full pool: profile_recall shows the
# rerank is cheap (~0.6s for 106 docs) and pool size barely moves it; the episode cost is the
# scan-bound BM25+vector fetch (~2.5s). Latency lever is parallelizing the legs, not the pool.
# Stage 2 (passage compaction). recall()'s episode bucket serves the most query-relevant PASSAGES
# (markdown chunks) of the top reranked episodes instead of whole episode turns: ~1/4 the tokens at
# near-baseline answer survival (kw-survival ~0.94 tail / ~0.95 natural at ~1050 served tokens vs
# ~5000 for full episodes; scripts/passage_bench_v*, 2026-06-26). recall_episodes() (drill-down) stays
# on FULL episodes. Passages are picked by a SECOND cross-encoder rerank over the markdown chunks of
# the top _RECALL_PASSAGE_SRC_K episodes. The bench's hybrid->rerank cascade collapses to a DIRECT
# rerank here because the chunk count is bounded (~20-80 from 20 episodes, capped at
# _RECALL_PASSAGE_CAND) and a direct rerank IS that cascade's quality ceiling — and rerank-only
# (no live passage embedding) keeps the second pass ~0.2s, not the ~2.2s a cosine leg would add.
_RECALL_PASSAGE_N = int(os.getenv("SYNAPSE_RECALL_PASSAGE_N", "3") or "3")  # passages served
# 10→20 (2026-07-23): a funnel decomposition on the _seglen golden found the answer episode is
# in the RRF pool 92% of the time but the reranker ranks ~11% of answers at position 11-20 — just
# outside the mining window. Mining the top 20 recovers them: served answer-recall 0.583→0.667,
# no full-episode/token cost (still serves _RECALL_PASSAGE_N passages, capped at _RECALL_PASSAGE_CAND).
_RECALL_PASSAGE_SRC_K = 20  # top reranked episodes to mine passages from
_RECALL_PASSAGE_CAND = 80  # cap on chunks fed to the passage reranker (bounds the extra call)
# Lexical-fusion of the FINAL episode order. The web-search-trained cross-encoder systematically
# under-ranks answer episodes that are exact lexical matches (query keyword/entity verbatim in the
# turn) — measured 2026-07-23 as ~31% of pooled answers stuck at rank 21-50. BM25 nails those, so
# RRF-fusing the rerank order with the pool's BM25 order for the SERVING order recovers them:
# answer-episode hit@10 0.636->0.909 (exact-fact golden) / 0.762->0.905 (natural), no regression,
# and it is BM25 specifically (rerank+vector was WORSE — vector shares the reranker's semantic bias).
# Reorders ranked_eps only; the rerank call, rerank_top telemetry, and the abstention floor are all
# untouched. Off with SYNAPSE_RECALL_BM25_FUSE=0.
_RECALL_BM25_FUSE = os.getenv("SYNAPSE_RECALL_BM25_FUSE", "1") != "0"
# Fusion shaping (LME 2026-07-25: unweighted full-order fusion cost multi-session -4.6pts —
# lexical hits colonize the served window on queries whose terms recur across many sessions).
# _W scales the BM25 term's RRF contribution (1.0 = original equal-weight fusion).
# _LIFT_CAP bounds how MANY episodes get a lexical lift (0 = uncapped): displacement of the
# cross-session semantic spread is a count problem, not a magnitude one.
_RECALL_BM25_FUSE_W = float(os.getenv("SYNAPSE_RECALL_BM25_FUSE_W", "1.0") or "1.0")
_RECALL_BM25_LIFT_CAP = int(os.getenv("SYNAPSE_RECALL_BM25_LIFT_CAP", "0") or "0")
# Reserved-slot fusion (LME sweep 2026-07-25): weight/cap shaping failed — RRF's flat
# 1/(k+pos) curve lifts a strong lexical hit past the head even at w=0.25, and the top 1-2
# lifts displace the most load-bearing passages. Fusion's validated win was getting the
# lexical hit INTO the mining window at all (hit@10 0.64->0.91), not ranking it first: with
# _RESERVE=N>0, the window head stays pure rerank order and the last N window slots are
# guaranteed to the best BM25 hits not already inside. Replaces RRF reordering when set.
_RECALL_BM25_RESERVE = int(os.getenv("SYNAPSE_RECALL_BM25_RESERVE", "0") or "0")

# Self-exclusion: when the recall call carries the calling session's id (injected by
# the client's PreToolUse hook as self_session), drop that session's episodes from the
# served pool entirely. The caller's own turns are already in its context window, and
# they were the top measured recall_feedback noise driver (2026-07-23). Full exclusion
# replaces the per-session serving cap for this purpose; drill-down (mode="turns",
# fetch) is unaffected. OFF by default.
_RECALL_SELF_EXCLUDE = int(os.getenv("SYNAPSE_RECALL_SELF_EXCLUDE", "0") or "0")

_ENTITY_LIMIT = 3  # seed entities (with summaries) returned by recall()
_SUPERSEDED_LIMIT = 2  # superseded-fact pairs returned by recall()
_WEB_LIMIT = 3  # web_chunks (deduped by parent page) returned by recall()
_WEB_FETCH = int(os.getenv("SYNAPSE_RECALL_WEB_FETCH", "50") or "50")  # per-leg (bm25+vector) depth
_WEB_RERANK_POOL = int(os.getenv("SYNAPSE_RECALL_WEB_POOL", "60") or "60")  # fused cands reranked
# Absolute cross-encoder floor on the WEB bucket (2026-07-24). Web chunks are external and topical,
# so an off-topic embedding/token collision scores genuinely low under the cross-encoder even when
# the bi-encoder/BM25 ranked it high on surface tokens. UNLIKE the fact floor there is NO keep->=1
# backstop: the bucket may serve ZERO, which SELF-GATES web on intent — it surfaces only when a
# scraped page is genuinely on-query. Validated on 42 feedback-labeled web ids: all 41 noise scored
# <=0.559 and the lone helpful id 0.914, so 0.60 gave 100% noise suppression at 100% helpful
# retention, collapsing web from 3.0 to ~0.07 chunks/query (n_helpful=1 — revisit as labels grow).
# 0 disables the floor (still fuses BM25+vector and reranks, just serves the top-_WEB_LIMIT).
_WEB_FLOOR = float(os.getenv("SYNAPSE_RECALL_WEB_FLOOR", "0.60") or "0.60")

# Notes leg (2026-08-05). The board injects ~39 hooks a session; every other live note
# (414 at ship time, growing ~17/day) was unreachable by search — hook embeddings have
# been written since schema 041 but nothing read them except the remember() dedup KNN.
# This serves the query-relevant notes as their own bucket: hook-embedding KNN over the
# live set (board scoping: global types + the caller's project) -> cross-encoder floor
# on "hook — body" text -> top _NOTES_LIMIT. The floor self-gates like web/timeline (an
# all-subfloor result serves NO bucket), so notes only appear on genuine hits. A note
# already on the board may be served again — accepted 2026-07-26; board-dedup needs the
# session hook to pass served ids and is a later, measured step. n: ids flow through
# served_ids/recall_feedback (#121), so helpful/noise labels accumulate from day one.
_NOTES_IN_RECALL = os.getenv("SYNAPSE_RECALL_NOTES", "1") != "0"
_NOTES_FLOOR = float(os.getenv("SYNAPSE_RECALL_NOTES_FLOOR", "0.60") or "0.60")
_NOTES_LIMIT = int(os.getenv("SYNAPSE_RECALL_NOTES_LIMIT", "3") or "3")
_NOTES_FETCH = int(os.getenv("SYNAPSE_RECALL_NOTES_FETCH", "24") or "24")
_NOTES_BODY_CAP = 700  # serve capped bodies; fetch("n:N") returns the full note

# Adaptive episode serving (variable-k) for recall_episodes() — OFF by default.
# When SYNAPSE_EPISODE_CUTOFF_TAU > 0, recall_episodes() serves the reranked turns
# scoring >= tau*top_score instead of a fixed top-`limit`, clamped to [MIN_K, MAX_K]:
# fewer turns when the top result dominates (focused/needle queries), more when many
# turns are comparably relevant (broad/multi-session). Validated on LongMemEval-S
# (tau=0.50 -> 81.8% vs fixed k=12 80.0%). PROD DEFAULT STAYS OFF: that win is bench-
# specific (LME's synthetic personas have no KG facts, so episodes carry everything),
# whereas on real data the KG owns multi-hop and the episode_count_sweep above measured
# broad synthesis PEAKING at 5 — so bench-tuned params do NOT transfer. Enabling needs a
# prod-data answerability sweep; conservative MAX_K=8 reflects that plateau. Applies to
# the drill-down path only; recall()'s overview bucket stays fixed at _RECALL_EPISODE_LIMIT
# (serving past 5 is "strictly worse on broad" per the sweep).
_EPISODE_CUTOFF_TAU = float(os.getenv("SYNAPSE_EPISODE_CUTOFF_TAU", "0") or "0")
_EPISODE_CUTOFF_MIN_K = int(os.getenv("SYNAPSE_EPISODE_CUTOFF_MIN_K", "3") or "3")
_EPISODE_CUTOFF_MAX_K = int(os.getenv("SYNAPSE_EPISODE_CUTOFF_MAX_K", "8") or "8")

# ── Relevance gates map ────────────────────────────────────────────────────────────────────
# recall() has SEVERAL independent relevance gates. They fall into two shapes; before adding a
# sixth, reuse the right shape rather than copy-pasting:
#   PER-ITEM rerank floors (drop individual served items below a score) — share _floor_by_rerank:
#     • FACTS    _RECALL_FACT_FLOOR (below, default 0/off, keep_min=1 — never blanks)
#     • WEB      _WEB_FLOOR         (line ~297, 0.60) — applied inline in _search_web_reranked on
#                                    the pool it ALREADY reranked for ordering, so it does not call
#                                    _floor_by_rerank (that would double-rerank); same idea, keep_min=0.
#   WHOLE-BUCKET abstention gate (drop the bucket when its TOP score is weak, not per item):
#     • EPISODES _RECALL_FLOOR / _RECALL_FLOOR_ENFORCE (line ~336) — episodes score flat-high so a
#                                    per-item floor is a no-op; the confidence gate is a top-score cut.
# (_RERANK_RECENCY_FLOOR at line ~102 is NOT a relevance gate — it floors the recency multiplier.)
# ───────────────────────────────────────────────────────────────────────────────────────────
# Absolute relevance gate on KG FACTS (default 0 = OFF). When SYNAPSE_RECALL_FACT_FLOOR > 0,
# cross-encoder-score the served facts against the query and DROP those below the floor — the
# genuine off-topic facts the vector/BM25/graph legs surface (e.g. a "Mattermost decommission"
# fact, scored 0.39, for a "FalkorDB decommission" query; a wholly-unrelated 0.28 fact). Probe
# (2026-06-18) on real prod facts: off-topic ~0.28-0.39, relevant >=0.44, so ~0.40 separates —
# but CALIBRATED ON ONLY 3 QUERIES with a thin gap, so default OFF until validated on more.
# This is where the "recall returns irrelevant stuff" lever actually lives: episodes score
# flat-high (0.68-0.94 at full length) so an episode floor is a no-op, but facts are short and
# fully scored, so off-topic ones genuinely score low. Costs ONE extra rerank of ~12 short facts
# per recall(), ONLY when enabled. Keeps >=1 fact so a query can't lose its facts bucket.
_RECALL_FACT_FLOOR = float(os.getenv("SYNAPSE_RECALL_FACT_FLOOR", "0") or "0")

# Abstention-floor SHADOW logging. An 84-run benchmark measured ZERO abstentions and ~30%
# confidently-wrong answers: when nothing clears relevance, recall serves the least-bad six
# and the model runs with them. Before ENFORCING a floor, log when one WOULD have fired so
# the threshold is picked from real data: when real rerank scores exist and the RAW
# pre-recency-reweight top score (the same value recall_metrics.rerank_top_score records)
# is strictly below _RECALL_FLOOR, the served_ids telemetry envelope gains
# {"would_abstain": true, "floor": <float>} — see _floor_shadow(). The marker is recorded
# regardless of enforcement. The threshold is a property of the RERANKER in use, not of
# recall: rerank scores are not comparable across models, and a floor calibrated on one
# backend replayed against another blanked 39-56% of episode buckets on the LME A/B set
# (2026-09-10). So the code default is 0 (off) and each examples/env preset sets the value
# calibrated for its reranker (0.58 for the default stack ~= p10 of 30 days of prod
# rerank_top_score: p05=0.5039 p10=0.5781 p25=0.6914 p50=0.7891, n=687).
_RECALL_FLOOR = float(os.getenv("SYNAPSE_RECALL_FLOOR", "0") or "0")
# Enforcement gate — ON by default (2026-07-23): the shadow phase validated the floor fires
# on ~9% of real recalls (the bottom ~p10 by episode-rerank strength). When enforced, recall()
# drops the EPISODE bucket if the RAW top rerank score is in (0, floor) under working
# retrieval — low relevance costs fewer tokens instead of serving the least-bad passages.
# Facts/timeline are unaffected (own gates); recall_episodes() drill-down never enforces.
# SYNAPSE_RECALL_FLOOR_ENFORCE=0 disables. SYNAPSE_RECALL_FLOOR unset or 0 disables both
# marker + enforce.
_RECALL_FLOOR_ENFORCE = os.getenv("SYNAPSE_RECALL_FLOOR_ENFORCE", "1") != "0"
# Keep-min under enforcement (LME 2026-07-25: blanking the bucket cost multi-session -8pts at
# the enforce commit — synthesis questions have flat score spreads, so a low TOP score doesn't
# mean no evidence). >0 serves that many top passages instead of none when the floor fires,
# mirroring the facts gate's keep_min=1 discipline; 0 preserves the blanking behavior.
_RECALL_FLOOR_KEEP_MIN = int(os.getenv("SYNAPSE_RECALL_FLOOR_KEEP_MIN", "0") or "0")

# Supersession surface (2026-06-27): a query that matches a now-INVALID fact should still
# return the CURRENT answer. When a superseded edge near the query carries a precise successor link
# (invalidated_by, schema 028), surface that successor's fact in the facts bucket (deduped) instead
# of silently dropping the stale match. Distance-gated so only ON-TOPIC superseded facts pull their
# correction in; go-forward coverage only (no link => skip); never serves the stale fact itself.
_SUP_CANDIDATES = 10  # nearest invalid-with-link edges to consider per recall
_SUP_LIMIT = 3  # max successor facts added per recall (additive, beyond _FACT_LIMIT)
_SUP_MAX_DIST = float(os.getenv("SYNAPSE_SUPERSEDE_MAX_DIST", "0.45") or "0.45")  # cosine-dist gate

logger = logging.getLogger(__name__)


def _resolve(db_url: str, surface: str | None, trust: SurfaceTrust | None) -> SurfaceTrust:
    """The serving verdict for one call: a pre-resolved one wins, else look up the id.

    ``trust`` is what the MCP server passes now that a caller is identified by its
    DEVICE TOKEN (schema 054) — the credential resolved at authentication time, so the
    engine must not go re-derive a verdict from a self-reported string. ``surface``
    remains for the id lanes that still exist: the ``oauth:<login>`` identity and, for
    one release, a legacy hostname param. Both fail closed the same way.
    """
    return trust if trust is not None else lookup_surface(db_url, surface)


def _floor_shadow(served_ids: dict[str, Any], rerank_top: float, emb_ok: bool) -> None:
    """Shadow abstention-floor marker — telemetry only, never touches the served payload.

    Mutates the recall_metrics ``served_ids`` envelope (same pattern as the existing
    n_echo_suppressed key) when an enforced floor WOULD have abstained: real rerank scores
    exist and the RAW pre-recency top score is strictly below _RECALL_FLOOR. No marker when:
    the floor is disabled (<= 0); the rerank is disabled or degraded (the all-0.0 sentinel —
    no real scores), which also covers the empty candidate pool (rerank_top 0.0); or the
    query embedding failed (emb_ok False — a weak top under crippled retrieval says nothing
    about whether relevant memory exists). _RECALL_FLOOR_ENFORCE is read but inert this
    release (see the knob's comment); enforcement ships separately."""
    if emb_ok and 0.0 < rerank_top < _RECALL_FLOOR:
        served_ids["would_abstain"] = True
        served_ids["floor"] = _RECALL_FLOOR


class Recall(
    RecallEpisodeSearchMixin,
    RecallRerankMixin,
    RecallPassagesMixin,
    RecallSourcesMixin,
    RecallOverviewMixin,
    RecallEpisodesMixin,
    RecallFetchMixin,
    RecallMetricsMixin,
):
    """Stateful retrieval engine. One instance per MCP server process."""

    def _settings(self) -> RecallSettings:
        """Read live knobs so benchmarks and operator overrides retain their behavior."""
        return RecallSettings(
            _RERANK_RECENCY_FLOOR=_RERANK_RECENCY_FLOOR,
            _RERANK_WINDOW=_RERANK_WINDOW,
            _ECHO_MIN_QUERY_LEN=_ECHO_MIN_QUERY_LEN,
            _RECALL_FACT_FLOOR=_RECALL_FACT_FLOOR,
            _RECALL_PASSAGE_CAND=_RECALL_PASSAGE_CAND,
            _EPISODE_CUTOFF_TAU=_EPISODE_CUTOFF_TAU,
            _EPISODE_CUTOFF_MIN_K=_EPISODE_CUTOFF_MIN_K,
            _EPISODE_CUTOFF_MAX_K=_EPISODE_CUTOFF_MAX_K,
            _WEB_FETCH=_WEB_FETCH,
            _WEB_LIMIT=_WEB_LIMIT,
            _NOTES_FLOOR=_NOTES_FLOOR,
            _EPISODE_FETCH=_EPISODE_FETCH,
            _NOTES_IN_RECALL=_NOTES_IN_RECALL,
            _SUPERSEDED_LIMIT=_SUPERSEDED_LIMIT,
            _RECALL_SELF_EXCLUDE=_RECALL_SELF_EXCLUDE,
            _RECALL_BM25_FUSE=_RECALL_BM25_FUSE,
            _RECALL_FLOOR_ENFORCE=_RECALL_FLOOR_ENFORCE,
            _RERANK_RECENCY=_RERANK_RECENCY,
            _SUPPRESS_QUERY_ECHO=_SUPPRESS_QUERY_ECHO,
            _WEB_RERANK_POOL=_WEB_RERANK_POOL,
            _WEB_FLOOR=_WEB_FLOOR,
            _NOTES_LIMIT=_NOTES_LIMIT,
            _NOTES_BODY_CAP=_NOTES_BODY_CAP,
            _EPISODE_RERANK_POOL=_EPISODE_RERANK_POOL,
            _FACT_LIMIT=_FACT_LIMIT,
            _RECALL_EPISODE_LIMIT=_RECALL_EPISODE_LIMIT,
            _RECALL_PASSAGE_SRC_K=_RECALL_PASSAGE_SRC_K,
            _RECALL_FLOOR=_RECALL_FLOOR,
            _KG_OWNER=_KG_OWNER,
            _RERANK_DOC_CAP=_RERANK_DOC_CAP,
            _ECHO_CONTENT_CAP=_ECHO_CONTENT_CAP,
            _RECALL_PASSAGE_N=_RECALL_PASSAGE_N,
            _RECALL_FLOOR_KEEP_MIN=_RECALL_FLOOR_KEEP_MIN,
            _SUP_MAX_DIST=_SUP_MAX_DIST,
            _NOTES_FETCH=_NOTES_FETCH,
            _RERANK_RECENCY_HALF_LIFE_DAYS=_RERANK_RECENCY_HALF_LIFE_DAYS,
            _SUP_CANDIDATES=_SUP_CANDIDATES,
            _EMBED_DIMS=_EMBED_DIMS,
        )

    @staticmethod
    def _floor_shadow(served_ids: dict[str, Any], rerank_top: float, emb_ok: bool) -> None:
        _floor_shadow(served_ids, rerank_top, emb_ok)

    @staticmethod
    def _echo_overlap(content: str, query: str) -> int:
        return _echo_lcs_len(content, query)

    def _resolve_trust(self, surface: str | None, trust: SurfaceTrust | None) -> SurfaceTrust:
        return _resolve(self._db_url, surface, trust)

    def __init__(
        self,
        db_url: str,
        voyage_api_key: str,
    ) -> None:
        self._db_url = db_url
        self._voyage_key = voyage_api_key
        self._kg_owner = _KG_OWNER
        self._embedder: Any = None
        self._reranker: Any = _RERANKER_UNSET
        # Postgres connections are THREAD-LOCAL: recall() fans its independent search
        # legs out across the leg executor, and a single psycopg connection can't be
        # used by two threads at once. Each worker lazily opens its own connection.
        self._pg_local = threading.local()
        # Background executor for fire-and-forget feedback writes (retrieval
        # count bumps). One worker is enough for single-user scale; queueing
        # on a hot recall burst is cheaper than blocking the response.
        self._async_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="recall-feedback"
        )
        # Leg executor: runs recall()'s independent search legs (summaries, episodes,
        # web, KG) concurrently. Persistent so each worker's thread-local PG connection
        # is reused across calls. Legs never submit here, so concurrent recalls queue
        # rather than deadlock.
        self._leg_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="recall-leg")

    def _ensure_pg(self) -> Any:
        # Thread-local: each thread (the caller + every leg-executor worker) owns its
        # own connection, so parallel search legs never share one psycopg handle.
        conn = getattr(self._pg_local, "conn", None)
        if conn is not None and not conn.closed:
            # Probe for half-open TCP connections (closed=False but server dropped us)
            try:
                conn.execute("SELECT 1")
                return conn
            except Exception:
                conn = None
        conn = psycopg.connect(self._db_url, row_factory=dict_row, autocommit=True)
        # HNSW search breadth for the halfvec vector indexes. Default 40 under-recalls a
        # 100-deep fetch; 200 gives recall@100 0.981 / recall@10 1.000 vs exact (validated).
        # Harmless on tables/queries that don't use an HNSW index.
        conn.execute("SET hnsw.ef_search = 200")
        self._pg_local.conn = conn
        return conn

    def _ensure_embedder(self) -> Any:
        if self._embedder is None:
            self._embedder = _embedding.create_embedder(
                voyage_api_key=self._voyage_key, db_url=self._db_url
            )
        return self._embedder

    def _ensure_reranker(self) -> Any:
        """Lazily-built rerank backend. ``None`` = rerank disabled
        (SYNAPSE_RERANK_PROVIDER=none) — callers serve the fusion (RRF) order."""
        if self._reranker is _RERANKER_UNSET:
            self._reranker = _embedding.create_reranker(voyage_api_key=self._voyage_key)
        return self._reranker

    @staticmethod
    def _exclude_self(ranked_eps: list[dict[str, Any]], self_session: str) -> list[dict[str, Any]]:
        """Drop the calling session's own episodes from the serving pool.

        The caller already holds its own turns in context; serving them back both
        wastes tokens and crowds out older real history (see _RECALL_SELF_EXCLUDE).
        """
        return [e for e in ranked_eps if e.get("session_id") != self_session]

    @staticmethod
    def _fuse_bm25_order(ranked_eps: list[dict[str, Any]], k: int = 60) -> list[dict[str, Any]]:
        """RRF-fuse the rerank order of ``ranked_eps`` with the pool's BM25 order.

        The web-trained cross-encoder under-ranks exact lexical matches; BM25 (already scored on
        the pool items as ``bm25_score``) ranks them high. Reciprocal-rank-fuse the two orders
        (k=60) so a strong lexical hit the reranker buried is lifted back into the served window.
        Only episodes that were BM25 hits (carry a ``bm25_score``) contribute a lexical term —
        vector-only episodes keep their rerank position. Validated 2026-07-23; see _RECALL_BM25_FUSE."""
        if len(ranked_eps) < 2:
            return ranked_eps
        bm = sorted(
            (i for i, e in enumerate(ranked_eps) if e.get("bm25_score") is not None),
            key=lambda i: ranked_eps[i]["bm25_score"],
            reverse=True,
        )
        if _RECALL_BM25_RESERVE > 0:
            # Reserved-slot mode: keep the mining-window head in pure rerank order and
            # guarantee the last _RESERVE window slots to the best BM25 hits not already
            # inside the window. Lexical recovery without displacing the semantic head.
            window = _RECALL_PASSAGE_SRC_K
            head = min(max(window - _RECALL_BM25_RESERVE, 0), len(ranked_eps))
            in_head = set(range(head))
            lifted = [i for i in bm if i not in in_head][:_RECALL_BM25_RESERVE]
            rest = [i for i in range(len(ranked_eps)) if i >= head and i not in lifted]
            order = list(range(head)) + lifted + rest
            return [ranked_eps[i] for i in order]
        fused: dict[int, float] = {i: 1.0 / (k + i + 1) for i in range(len(ranked_eps))}
        if _RECALL_BM25_LIFT_CAP > 0:
            bm = bm[:_RECALL_BM25_LIFT_CAP]
        for pos, i in enumerate(bm):
            fused[i] += _RECALL_BM25_FUSE_W / (k + pos + 1)
        return [ranked_eps[i] for i in sorted(fused, key=lambda i: fused[i], reverse=True)]


__all__ = [
    "Recall",
    "_apply_supersessions",
    "_bm25_best_window",
    "_bm25_tokenize",
    "_config_hint",
    "_cutoff_k",
    "_echo_lcs_len",
    "_err_brief",
    "_merge_rrf",
    "_norm_ws",
    "_parse_episode_ids",
    "_passage_role",
    "_query_shingles",
    "_recency_multiplier",
    "_role_spans",
    "_served_chars",
    "_submit_ctx",
    "_timed",
    "_to_recall_item",
    "_to_web_recall_item",
    "_warn",
    "_warn_sink",
]
