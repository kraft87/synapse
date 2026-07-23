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

import difflib
import json
import logging
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import psycopg
from psycopg.rows import dict_row, tuple_row
from psycopg.types.json import Json as PgJson

from ingestion import embedding as _embedding
from mcp_server.kg_pg import _vec_literal, search_kg_postgres

logger = logging.getLogger(__name__)

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
# Timeline leg (schema 033). recall() fuses a compact chronological event list into the
# payload on EVERY query. It originally fired behind a temporal-intent regex, but the
# regex missed 41% of dated questions on LongMemEval (ordering phrasings like "which
# came first, X or Y" carry no temporal keyword), and the leg is one cheap parallel DB
# read serving <=8 events (~0.5KB; empty result = no payload change) — a gate buys
# nothing. Kill switch SYNAPSE_RECALL_TIMELINE=0.
_TIMELINE_IN_RECALL = os.environ.get("SYNAPSE_RECALL_TIMELINE", "1") != "0"
_TIMELINE_LIMIT = 8
# Preferences leg (schema 035): the standing USER-preference bucket. Like the timeline
# leg, one cheap parallel DB read on every query — top-5 live prefs by cosine to the
# query embedding — reusing this call's query_emb (no extra Voyage round-trip). Kept out
# of the KG so preferences don't rebuild the User-supernode. Kill switch SYNAPSE_RECALL_PREFS=0.
_PREFS_IN_RECALL = os.environ.get("SYNAPSE_RECALL_PREFS", "1") != "0"
_PREFS_LIMIT = 5
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
_RERANK_WINDOW_SIZE = 1024  # window granularity for locating the relevant region of a long episode


_WS_RE = re.compile(r"\s+")


def _norm_ws(s: str) -> str:
    """Whitespace-collapsed lowercase — the normal form for query-echo overlap matching."""
    return _WS_RE.sub(" ", s).strip().lower()


def _echo_lcs_len(content: str, q: str) -> int:
    """Longest common substring length between a doc and the query (both pre-normalized).

    The exact-but-quadratic CONFIRM step of echo suppression — only reached for docs the
    shingle pre-filter flags (echoes are rare in the pool, so this almost never runs).
    Module-level so tests can spy on invocation count."""
    sm = difflib.SequenceMatcher(autojunk=False)
    sm.set_seq2(q)
    sm.set_seq1(content)
    return sm.find_longest_match(0, len(content), 0, len(q)).size


_ECHO_SHINGLE_WORDS = 5
_ECHO_SHINGLE_MIN_CHARS = 25  # shorter shingles are too match-prone to prune anything
_ECHO_SHINGLE_CAP = 2000  # bounds the pre-filter cost on a pathological query


def _query_shingles(q: str) -> list[str]:
    """Word shingles of the normalized query — the C-speed pre-filter for echo detection.

    A qualifying verbatim run (>= the 40-60 char echo threshold) spans several consecutive
    words, so after whitespace normalization it contains a whole _ECHO_SHINGLE_WORDS-word
    shingle of the query (step 1 word — every alignment is covered; only a pathological
    all-long-token run can slip through, and a miss just KEEPS the doc, fail-open).
    ``shingle in content`` is a C-level substring scan, so docs with no hit — the common
    case — skip the quadratic SequenceMatcher confirm entirely."""
    words = q.split(" ")
    out: list[str] = []
    for i in range(len(words) - _ECHO_SHINGLE_WORDS + 1):
        sh = " ".join(words[i : i + _ECHO_SHINGLE_WORDS])
        if len(sh) >= _ECHO_SHINGLE_MIN_CHARS:
            out.append(sh)
            if len(out) >= _ECHO_SHINGLE_CAP:
                break
    return out


def _bm25_tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z0-9_./#+-]+", s.lower())


def _bm25_best_window(content: str, q_tokens: list[str], cap: int) -> str:
    """Return the cap-sized slice of ``content`` centered on the window with the highest BM25 score
    vs the query tokens. Pure-Python (no embeddings) — cheap enough for the read path. Lets the
    reranker see the query-relevant region of a long episode instead of just its first ``cap`` chars."""
    win = _RERANK_WINDOW_SIZE
    starts = list(range(0, len(content), win))
    win_tokens = [_bm25_tokenize(content[s : s + win]) for s in starts]
    n = len(win_tokens)
    if n <= 1:
        return content[:cap]
    avgdl = sum(len(d) for d in win_tokens) / n
    df: dict[str, int] = {}
    for d in win_tokens:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    k1, b = 1.5, 0.75
    qset = set(q_tokens)
    best_i, best_s = 0, -1.0
    for i, d in enumerate(win_tokens):
        if not d:
            continue
        tf: dict[str, int] = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        dl = len(d)
        s = 0.0
        for t in qset:
            f = tf.get(t)
            if not f:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        if s > best_s:
            best_s, best_i = s, i
    lo = max(0, starts[best_i] + win // 2 - cap // 2)
    return content[lo : lo + cap]


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
# rerank here because the chunk count is bounded (~10-40 from 10 episodes) and a direct rerank IS that
# cascade's quality ceiling — and rerank-only (no live passage embedding) keeps the second pass ~0.2s,
# not the ~2.2s a cosine leg would add.
_RECALL_PASSAGE_N = int(os.getenv("SYNAPSE_RECALL_PASSAGE_N", "3") or "3")  # passages served
_RECALL_PASSAGE_SRC_K = 10  # top reranked episodes to mine passages from
_RECALL_PASSAGE_CAND = 80  # cap on chunks fed to the passage reranker (bounds the extra call)

_ENTITY_LIMIT = 3  # seed entities (with summaries) returned by recall()
_SUPERSEDED_LIMIT = 2  # superseded-fact pairs returned by recall()
_WEB_LIMIT = 3  # web_chunks (deduped by parent page) returned by recall()

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
# {"would_abstain": true, "floor": <float>} — see _floor_shadow(). The served payload is
# byte-identical either way; this is telemetry only. Default 0.58 ~= p10 of the last 30
# days' rerank_top_score distribution (p05=0.5039 p10=0.5781 p25=0.6914 p50=0.7891, n=687).
# 0 disables the marker.
_RECALL_FLOOR = float(os.getenv("SYNAPSE_RECALL_FLOOR", "0.58") or "0.58")
# Enforcement gate — read but deliberately INERT this release: the shadow distribution must
# first show the floor abstains on the right calls. Flipping it to 1 today changes nothing;
# serving-side enforcement ships in a later release once a validated floor exists.
_RECALL_FLOOR_ENFORCE = os.getenv("SYNAPSE_RECALL_FLOOR_ENFORCE", "0") != "0"

# Supersession surface (2026-06-27): a query that matches a now-INVALID fact should still
# return the CURRENT answer. When a superseded edge near the query carries a precise successor link
# (invalidated_by, schema 028), surface that successor's fact in the facts bucket (deduped) instead
# of silently dropping the stale match. Distance-gated so only ON-TOPIC superseded facts pull their
# correction in; go-forward coverage only (no link => skip); never serves the stale fact itself.
_SUP_CANDIDATES = 10  # nearest invalid-with-link edges to consider per recall
_SUP_LIMIT = 3  # max successor facts added per recall (additive, beyond _FACT_LIMIT)
_SUP_MAX_DIST = float(os.getenv("SYNAPSE_SUPERSEDE_MAX_DIST", "0.45") or "0.45")  # cosine-dist gate


def _to_web_recall_item(row: dict[str, Any]) -> dict[str, Any]:
    """Web chunk → slim recall shape.

    Returns the LLM-written Contextual Retrieval `context_prefix` as the
    user-facing description when available — ~80 tokens of "where this
    chunk fits in the parent doc" rather than ~375 tokens of raw chunk
    content. The caller has the URL if they want the full page; this
    keeps recall token-light.

    Falls back to a 200-char excerpt of raw content for chunks that
    haven't been contextualized yet.
    """
    out: dict[str, Any] = {}
    if (rid := row.get("id")) is not None:
        # Web rows already carry the served "w:N" form (assigned at query time,
        # _search_web) — pass it through; re-wrapping would double the prefix to
        # "w:w:N" and fail recall_feedback's validator. Guard covers a bare id too.
        out["id"] = str(rid) if str(rid).startswith("w:") else f"w:{rid}"
    context = (row.get("context_prefix") or "").strip()
    if context:
        out["context"] = context
    else:
        excerpt = (row.get("content") or "")[:200].strip()
        if excerpt:
            out["excerpt"] = excerpt
    if url := row.get("url"):
        out["url"] = url
    if title := row.get("title"):
        out["title"] = title
    if (ts := row.get("created_at")) is not None:
        out["date"] = str(ts)[:10]
    return out


def _to_recall_item(row: dict[str, Any]) -> dict[str, Any]:
    """Slim a SQL row into the minimum shape an LLM caller can act on.

    Keeps the episode id (for fetch() drill-down) + content + project (only when
    non-null) + date (truncated from full timestamp). Drops everything used only for ranking
    or debug: session_id, doc_type, retrieval_count, vec_distance, bm25_score, etc.
    """
    out: dict[str, Any] = {}
    if (rid := row.get("id")) is not None:
        out["id"] = rid  # "e:N" — pass to fetch() to expand the full turn
    out["content"] = row.get("content", "")
    if (project := row.get("project")) is not None:
        out["project"] = project
    if (ts := row.get("created_at")) is not None:
        out["date"] = str(ts)[:10]
    return out


# Provenance labeling (issue #17). Episode content is assembled from parts whose first
# line carries a role marker ([user] .., [assistant] .., [tool:X] .., [result] ..,
# [context] .., plus [title]/[attachments] on the claude.ai lane). Passage compaction
# slices that content, so a served passage can lose its marker — and with it the only
# signal that the text was the assistant's own past output (possibly speculation)
# rather than something the user stated. _role_spans() recovers the marker layout;
# _passage_role() maps a served char span back to who produced it: "user"
# (human-stated, incl. attachments), "assistant" (agent-side: assistant text, tool
# calls/results, carried [context]), or "mixed". Advisory metadata, fail-soft: text
# before the first marker or in a marker-free episode gets no label, and chunk-offset
# drift can only blur a label toward "mixed" — never an error.
_ROLE_MARKER_RE = re.compile(
    r"(?m)^\[(user|attachments|assistant|context|result|title|tool:[^\]\n]{0,80})\]"
)
_USER_MARKERS = {"user", "attachments"}


def _role_spans(content: str) -> list[tuple[int, str]]:
    """(offset, side) per recognized role marker, in order; side is "user"/"assistant".

    [title] closes the previous region without opening an attributed one (it's a
    neutral conversation-name header, not speech)."""
    spans: list[tuple[int, str]] = []
    for m in _ROLE_MARKER_RE.finditer(content):
        tag = m.group(1)
        side = "" if tag == "title" else ("user" if tag in _USER_MARKERS else "assistant")
        spans.append((m.start(), side))
    return spans


def _passage_role(spans: list[tuple[int, str]], start: int, end: int) -> str | None:
    """Which side(s) produced content[start:end]. None when unattributable."""
    if not spans or end <= start:
        return None
    seen: set[str] = set()
    for i, (off, side) in enumerate(spans):
        nxt = spans[i + 1][0] if i + 1 < len(spans) else math.inf
        if off < end and nxt > start and side:
            seen.add(side)
    if not seen:
        return None
    return seen.pop() if len(seen) == 1 else "mixed"


_FETCH_MAX = 20  # max records fetch() will expand in one call, across kinds (bounds the read)


def _parse_episode_ids(ids: list[Any]) -> list[int]:
    """Parse recall()'s episode ids ("e:227168" strings or bare ints) into int episode ids,
    deduped, order-preserving, capped at _FETCH_MAX. Skips anything unparseable."""
    out: list[int] = []
    seen: set[int] = set()
    for x in ids:
        n: int | None = None
        if isinstance(x, bool):
            continue
        if isinstance(x, int):
            n = x
        elif isinstance(x, str):
            s = x.split(":", 1)[1] if x.startswith("e:") else x
            if s.isdigit():
                n = int(s)
        if n is not None and n not in seen:
            seen.add(n)
            out.append(n)
    return out[:_FETCH_MAX]


def _parse_fetch_ids(ids: list[Any]) -> tuple[list[int], list[int], list[str], list[str]]:
    """Split fetch()'s mixed ids into episode ids ("e:N", or bare N / bare int for
    back-compat) and note ids ("n:N"), each deduped and order-preserving, capped at
    _FETCH_MAX across BOTH kinds (over-cap ids are silently dropped, matching the old
    episode-only path). Unknown prefixes and unparseable ids land verbatim in
    ``skipped``. Also returns the accepted ids, normalized ("e:5" / "n:3"), in request
    order — the telemetry query field."""
    eps: list[int] = []
    notes: list[int] = []
    skipped: list[str] = []
    normalized: list[str] = []
    seen: set[str] = set()
    for x in ids:
        kind: str | None = None
        n: int | None = None
        if isinstance(x, int) and not isinstance(x, bool):
            kind, n = "e", x
        elif isinstance(x, str):
            s = x.strip()
            prefix, _, rest = s.partition(":")
            if not _:  # no colon: bare N is an episode id (back-compat)
                prefix, rest = "e", s
            if prefix in ("e", "n") and rest.isdigit():
                kind, n = prefix, int(rest)
        if kind is None or n is None:
            skipped.append(str(x))
            continue
        key = f"{kind}:{n}"
        if key in seen or len(normalized) >= _FETCH_MAX:
            continue
        seen.add(key)
        normalized.append(key)
        (eps if kind == "e" else notes).append(n)
    return eps, notes, skipped, normalized


def _apply_supersessions(
    items: list[dict[str, Any]], sup: dict[int, list[str]], served_facts: set[str]
) -> None:
    """Attach ``superseded_by`` (current facts that superseded a claim the item made) to each served
    episode/passage, keyed by its parsed "e:N" id. Skips now-facts already in the served facts bucket
    (no duplicate tokens) — this is a top-up for the case the correction didn't rank into facts.
    Mutates ``items`` in place."""
    for it in items:
        rid = it.get("id")
        if not isinstance(rid, str) or not rid.startswith("e:"):
            continue
        s = rid.split(":", 1)[1]
        if not s.isdigit():
            continue
        nows = [f for f in sup.get(int(s), []) if f and f not in served_facts]
        if nows:
            it["superseded_by"] = nows


def _rrf_score(rank: int, k: int = _RRF_K) -> float:
    return 1.0 / (k + rank + 1)


def _recency_multiplier(created_at: Any, half_life_days: float = _RECENCY_HALF_LIFE_DAYS) -> float:
    """Exponential decay: 1.0 today → 0.5 at half_life → approaches 0."""
    if created_at is None:
        return 1.0
    from datetime import UTC, datetime

    try:
        if isinstance(created_at, str):
            ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        else:
            ts = created_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        age_days = (datetime.now(UTC) - ts).total_seconds() / 86400
        return math.exp(-age_days * math.log(2) / half_life_days)
    except Exception:
        return 1.0


def _feedback_multiplier(retrieval_count: Any) -> float:
    """Boost frequently retrieved items: log(1 + count), capped at 2x."""
    try:
        count = int(retrieval_count or 0)
        return min(2.0, 1.0 + math.log1p(count) * 0.3)
    except Exception:
        return 1.0


# Reciprocal Rank Fusion constant (Graphiti uses 1). Score for an item at
# 0-based rank r in a list is 1/(r + _RRF_KG_K + 1).
_RRF_KG_K = 1


def _rrf_fuse(ranked_lists: list[list[str]]) -> dict[str, float]:
    """Fuse several ranked uuid lists into one {uuid: score} via RRF.

    Each list contributes 1/(rank + _RRF_KG_K + 1) to an item's score; scores
    sum across lists so items found by multiple methods rise. This replaces
    the old "fact-embedding fills slots, traversal on scraps" selection — every
    method competes in one pool, which both improves relevance (benchmarked
    S1, +12% vs the fallback model) and removes the slot-boundary instability.
    """
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, uuid_ in enumerate(lst):
            scores[uuid_] = scores.get(uuid_, 0.0) + 1.0 / (rank + _RRF_KG_K + 1)
    return scores


def _merge_rrf(
    *ranked_lists: list[dict[str, Any]],
    id_key: str = "id",
    apply_recency: bool = True,
) -> list[dict[str, Any]]:
    """Merge ranked lists with RRF + recency decay + feedback boost."""
    scores: dict[Any, float] = {}
    items: dict[Any, dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            item_id = item[id_key]
            base = _rrf_score(rank)
            if apply_recency:
                base *= _recency_multiplier(item.get("created_at"))
            base *= _feedback_multiplier(item.get("retrieval_count"))
            scores[item_id] = scores.get(item_id, 0.0) + base
            if item_id not in items:
                items[item_id] = item

    return sorted(items.values(), key=lambda x: scores[x[id_key]], reverse=True)


def _timed(fn: Any, *args: Any) -> tuple[Any, float]:
    """Run ``fn(*args)`` and return ``(result, elapsed_ms)`` — per-leg latency
    telemetry. Submitted through the leg executor so each leg reports its own
    wall time (vs. submit-to-result, which would include queue wait)."""
    t = time.perf_counter()
    r = fn(*args)
    return r, (time.perf_counter() - t) * 1000.0


def _served_chars(out: dict[str, Any]) -> int:
    """Total characters of the SERVED payload — the context cost recall() imposes
    on the caller's window. ``est_tokens`` ~= chars/4. Counts every answer-bearing
    bucket (facts/episodes/entities/web/superseded_facts), not the ranking-only fields."""
    n = len(out.get("query") or "")
    for f in out.get("facts", []):
        n += len(f.get("fact") or "")
    for e in out.get("episodes", []):
        n += len(e.get("content") or "") + len(str(e.get("date") or ""))
    for e in out.get("entities", []):
        n += len(e.get("name") or "") + len(e.get("summary") or "")
    for w in out.get("web", []):
        n += len(w.get("context") or "") + len(w.get("excerpt") or "") + len(w.get("title") or "")
    for t in out.get("timeline", []):
        n += len(t.get("fact") or "") + len(str(t.get("date") or ""))
    for p in out.get("preferences", []):
        n += len(p.get("pref") or "") + len(str(p.get("polarity") or ""))
    for h in out.get("superseded_facts", []):
        n += len(str(h.get("fact") or "")) + len(str(h.get("superseded_by") or ""))
    return n


def _cutoff_k(scores: list[float], tau: float, min_k: int, max_k: int) -> int:
    """Adaptive serving size from a RELATIVE rerank-score cutoff.

    Keeps the leading docs whose score >= tau*top_score, then clamps the count
    to [min_k, max_k] (and to len(scores)). `scores` must be in rerank order
    (descending). The top doc always qualifies, so the result is >= 1."""
    if not scores:
        return 0
    top = scores[0]
    if top <= 0:  # rerank degraded to RRF order (all 0.0) — caller handles fallback
        return min(len(scores), max_k)
    keep = sum(1 for s in scores if s >= tau * top)
    return min(max(keep, min_k), max_k, len(scores))


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


class Recall:
    """Stateful retrieval engine. One instance per MCP server process."""

    def __init__(
        self,
        db_url: str,
        voyage_api_key: str,
    ) -> None:
        self._db_url = db_url
        self._voyage_key = voyage_api_key
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
        self._leg_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="recall-leg")
        self._timeline_engine: Any = None  # lazy TimelineRecall (timeline leg)

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

    def _ensure_timeline(self) -> Any:
        if self._timeline_engine is None:
            from mcp_server.timeline import TimelineRecall

            self._timeline_engine = TimelineRecall(
                db_url=self._db_url, voyage_api_key=self._voyage_key
            )
        return self._timeline_engine

    def _search_preferences(
        self, query_emb: list[float] | None, group_id: str, limit: int
    ) -> list[dict[str, Any]]:
        """The preferences leg (schema 035): top-N LIVE user preferences for this
        owner/group, nearest to the query embedding by cosine. Reuses the query_emb
        recall already computed — no extra Voyage call. Thread-local PG (leg executor).
        Degrades to [] if the table isn't present or the embedding is unavailable."""
        if query_emb is None:
            return []
        conn = self._ensure_pg()
        vlit = _vec_literal(query_emb)
        try:
            rows: list[dict[str, Any]] = conn.execute(
                # nosec B608 — _EMBED_DIMS is a validated int, not user input
                "SELECT id, pref, polarity, left(first_seen::text, 10) AS since, assert_count "
                "FROM preferences "
                "WHERE owner_id = %s AND group_id = %s AND t_invalid IS NULL "
                "AND embedding IS NOT NULL "
                f"ORDER BY embedding <=> %s::vector({_EMBED_DIMS}) ASC LIMIT %s",
                (_KG_OWNER, group_id, vlit, limit),
            ).fetchall()
        except psycopg.errors.UndefinedTable:
            return []  # migration 035 not applied on this deployment yet — degrade
        except Exception as e:
            logger.warning("preferences leg failed: %s", e)
            return []
        return rows

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

    # ------------------------------------------------------------------
    # BM25 search (ParadeDB) — episodes + chunks
    # ------------------------------------------------------------------

    @staticmethod
    def _ts_col(table: str) -> str:
        """Timestamp column name varies by table."""
        return "generated_at" if table == "synth_documents" else "created_at"

    @staticmethod
    def _extra_cols(table: str) -> str:
        # Slim SELECT: only fields read by the ranking pipeline. Drill-down
        # columns (source_ids, sequence ranges, synth_type) are dropped — caller
        # never sees them, so don't waste DB→Python wire bytes.
        # retrieval_count IS kept for episodes because _feedback_multiplier reads it.
        if table == "episodes":
            return ", retrieval_count"
        if table == "chunks":
            return ", episode_ids"  # chunk = retrieval signal; serve its episodes
        return ""

    def _bm25_table(
        self, table: str, query: str, project: str | None, limit: int, doc_type: str
    ) -> list[dict[str, Any]]:
        pg = self._ensure_pg()
        ts = self._ts_col(table)
        extra = self._extra_cols(table)
        try:
            if project:
                rows = pg.execute(
                    f"""
                    SELECT id, content, project,
                           {ts} AS created_at{extra},
                           paradedb.score(id) AS bm25_score
                    FROM {table}
                    WHERE id @@@ paradedb.match('content', %s) AND project = %s
                    ORDER BY bm25_score DESC LIMIT %s
                    """,
                    (query, project, limit),
                ).fetchall()
            else:
                rows = pg.execute(
                    f"""
                    SELECT id, content, project,
                           {ts} AS created_at{extra},
                           paradedb.score(id) AS bm25_score
                    FROM {table}
                    WHERE id @@@ paradedb.match('content', %s)
                    ORDER BY bm25_score DESC LIMIT %s
                    """,
                    (query, limit),
                ).fetchall()
            return [
                {**dict(r), "doc_type": doc_type, "id": f"{doc_type[0]}:{r['id']}"} for r in rows
            ]
        except Exception as e:
            logger.warning("BM25 %s search failed: %s", table, e)
            return []

    def _search_bm25_episodes(
        self, query: str, project: str | None, limit: int
    ) -> list[dict[str, Any]]:
        return self._bm25_table("episodes", query, project, limit, "episode")

    # ------------------------------------------------------------------
    # Vector search — episodes
    # ------------------------------------------------------------------

    def _vector_table(
        self, table: str, emb_literal: str, project: str | None, limit: int, doc_type: str
    ) -> list[dict[str, Any]]:
        pg = self._ensure_pg()
        ts = self._ts_col(table)
        extra = self._extra_cols(table)
        # Distance is computed over halfvec(N), NOT vector(N). The default embeddings are
        # 2048-dim (voyage-4-large), which exceeds pgvector's 2000-dim limit for HNSW on
        # the full `vector` type — so the HNSW indexes are built on `embedding::halfvec(N)`
        # (half-precision, index limit 4000 dims), with N = _EMBED_DIMS as provisioned.
        # The ORDER BY expression must match that index expression verbatim to be served
        # from it; the alias form is NOT index-eligible.
        # Half precision is loss-free for what's served: recall@10 vs exact scan = 1.000,
        # recall@100 = 0.981 (and the reranker re-scores the pool). 878ms -> 23ms on episodes.
        try:
            if project:
                rows = pg.execute(
                    f"""
                    SELECT id, content, project,
                           {ts} AS created_at{extra},
                           (embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS})) AS vec_distance
                    FROM {table}
                    WHERE is_embedded = TRUE AND project = %s
                    ORDER BY embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS}) ASC LIMIT %s
                    """,
                    (emb_literal, project, emb_literal, limit),
                ).fetchall()
            else:
                rows = pg.execute(
                    f"""
                    SELECT id, content, project,
                           {ts} AS created_at{extra},
                           (embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS})) AS vec_distance
                    FROM {table}
                    WHERE is_embedded = TRUE
                    ORDER BY embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS}) ASC LIMIT %s
                    """,
                    (emb_literal, emb_literal, limit),
                ).fetchall()
            return [
                {**dict(r), "doc_type": doc_type, "id": f"{doc_type[0]}:{r['id']}"} for r in rows
            ]
        except Exception as e:
            logger.warning("Vector %s search failed: %s", table, e)
            return []

    def _search_vector_episodes(
        self, query_emb: list[float], project: str | None, limit: int
    ) -> list[dict[str, Any]]:
        emb_literal = "[" + ",".join(str(x) for x in query_emb) + "]"
        return self._vector_table("episodes", emb_literal, project, limit, "episode")

    # ------------------------------------------------------------------
    # Reranking + episode leg
    # ------------------------------------------------------------------

    def _rerank_docs(self, query: str, pool: list[dict[str, Any]]) -> tuple[list[str], list[int]]:
        """Build the docs fed to the reranker + their pool owner index.

        Every item contributes its head (first _RERANK_DOC_CAP chars). With _RERANK_WINDOW on,
        items longer than the cap ALSO contribute their BM25-relevant window, so the caller can
        score the episode by max(head, window) and recover answers in the truncated tail (see the
        _RERANK_WINDOW note). ``owner[k]`` is the pool index doc ``k`` belongs to."""
        docs: list[str] = []
        owner: list[int] = []
        q_tokens = _bm25_tokenize(query) if _RERANK_WINDOW else []
        for i, c in enumerate(pool):
            content = c.get("content") or ""
            docs.append(content[:_RERANK_DOC_CAP])
            owner.append(i)
            if _RERANK_WINDOW and len(content) > _RERANK_DOC_CAP:
                docs.append(_bm25_best_window(content, q_tokens, _RERANK_DOC_CAP))
                owner.append(i)
        return docs, owner

    def _rerank_pool(self, query: str, pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reorder a pooled candidate list by a Voyage cross-encoder. Degrades
        gracefully to the incoming RRF order if the reranker errors (rate limit,
        outage) — recall must never hard-fail on the rerank leg."""
        if len(pool) <= 1:
            return pool
        scored = self._rerank_pool_scored(query, pool)
        return [pool[i] for i, _ in scored] if scored else pool

    def _rerank_pool_scored(
        self, query: str, pool: list[dict[str, Any]]
    ) -> list[tuple[int, float]]:
        """Like _rerank_pool but returns (pool_index, relevance_score) pairs in
        rerank order, so callers can apply a relative score cutoff. Each episode is scored by the
        MAX over its rerank docs (head + optional BM25 window — see _rerank_docs). Degrades to the
        incoming RRF order with score 0.0 if the reranker errors — recall must never hard-fail on
        the rerank leg. A 0.0 top score signals the caller to fall back to fixed-k serving."""
        if not pool:
            return []
        if len(pool) == 1:
            return [(0, 1.0)]
        reranker = self._ensure_reranker()
        if reranker is None:
            # Rerank disabled (SYNAPSE_RERANK_PROVIDER=none): serve the incoming
            # fusion (RRF) order. Score 0.0 = the same "fixed-k" signal as the
            # degraded path below. Logged once at startup by create_reranker().
            return [(i, 0.0) for i in range(len(pool))]
        docs, owner = self._rerank_docs(query, pool)
        try:
            scored = reranker.rerank_scored(query, docs)
        except Exception as e:
            logger.warning("Scored rerank failed, using RRF order: %s", e)
            return [(i, 0.0) for i in range(len(pool))]
        best: dict[int, float] = {}
        for di, s in scored:
            if 0 <= di < len(owner):
                oi = owner[di]
                if oi not in best or s > best[oi]:
                    best[oi] = s
        return sorted(best.items(), key=lambda x: x[1], reverse=True)

    def _apply_rerank_recency(
        self, scored: list[tuple[int, float]], pool: list[dict[str, Any]]
    ) -> list[tuple[int, float]]:
        """Re-inject recency into the post-rerank ordering (see _RERANK_RECENCY_HALF_LIFE_DAYS).

        Multiplies each candidate's rerank score by _recency_multiplier(created_at) at the
        14-day half-life, FLOORED at _RERANK_RECENCY_FLOOR so old-but-relevant content is
        dampened at most 1/floor rather than annihilated, then re-sorts descending. Any
        downstream tau cutoff then operates on THESE recency-adjusted scores (that is the
        intended contract — recency is part of the final relevance, not a post-cut tweak).
        No-op (returns ``scored`` unchanged) when disabled (SYNAPSE_RERANK_RECENCY=0) or when
        the reranker degraded to RRF order (top score 0.0 — leave that fallback ordering
        untouched; a 0.0 top is the same signal _cutoff_k reads)."""
        if not _RERANK_RECENCY or not scored or scored[0][1] <= 0.0:
            return scored
        floor = _RERANK_RECENCY_FLOOR
        adjusted = [
            (
                i,
                s
                * max(
                    floor,
                    _recency_multiplier(pool[i].get("created_at"), _RERANK_RECENCY_HALF_LIFE_DAYS),
                ),
            )
            for i, s in scored
        ]
        adjusted.sort(key=lambda x: x[1], reverse=True)
        return adjusted

    def _filter_query_echo(
        self, query: str, items: list[dict[str, Any]], need: int
    ) -> tuple[list[int], int]:
        """Echo suppression over a ranked list: (indices to keep, echoes dropped).

        A served episode whose content shares a long verbatim run with the query is an echo
        (compaction copy / re-ingested repeat), not recalled memory — drop it. Threshold:
        longest common substring >= min(60, max(40, len(query_norm)//2)) chars (the heuristic
        validated in the 2026-07-08 backtest), via difflib.SequenceMatcher (autojunk=False)
        over whitespace-collapsed lowercase, scanning at most _ECHO_CONTENT_CAP chars per doc.

        Cost containment — the pool is ~90 docs and SequenceMatcher is quadratic, so a naive
        full-pool scan costs seconds on long (800+ char) prompt-sized queries:
          - LAZY: walk in rank order and stop scanning once ``need`` survivors accumulate —
            items past that point can never be served, so they're kept unscanned.
          - Shingle pre-filter (_query_shingles): only docs containing one of the query's
            word shingles (C-level ``in``) reach the SequenceMatcher confirm (_echo_lcs_len);
            echoes are rare in the pool, so the confirm almost never runs.
        Keeps everything when disabled, when the query is too short for the threshold to be
        meaningful (< _ECHO_MIN_QUERY_LEN), or when the query yields no usable shingles."""
        keep_all = list(range(len(items)))
        if not _SUPPRESS_QUERY_ECHO or not items or need <= 0:
            return keep_all, 0
        q = _norm_ws(query)
        if len(q) < _ECHO_MIN_QUERY_LEN:
            return keep_all, 0
        shingles = _query_shingles(q)
        if not shingles:  # all-short-word query — the pre-filter can't attest, fail open
            return keep_all, 0
        thr = min(60, max(40, len(q) // 2))
        keep: list[int] = []
        dropped = 0
        for i, it in enumerate(items):
            if len(keep) >= need:
                keep.extend(range(i, len(items)))  # unscanned tail — can't be served
                break
            content = _norm_ws(it.get("content") or "")[:_ECHO_CONTENT_CAP]
            if (
                content
                and any(sh in content for sh in shingles)
                and _echo_lcs_len(content, q) >= thr
            ):
                dropped += 1
                continue
            keep.append(i)
        return keep, dropped

    def _floor_facts(self, query: str, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop served facts the cross-encoder scores below _RECALL_FACT_FLOOR (off-topic).

        Facts are short, so they're scored at full length — unlike episodes (flat-high at
        full length), off-topic facts genuinely score low here, so an absolute floor works.
        Degrades to keeping all facts on rerank failure; keeps >=1 so the bucket is never
        blanked. Caller gates on _RECALL_FACT_FLOOR > 0 so this runs only when enabled."""
        reranker = self._ensure_reranker()
        if reranker is None:  # rerank disabled — no floor, keep all facts
            return facts
        texts = [(f.get("fact") or "")[:_RERANK_DOC_CAP] for f in facts]
        try:
            scored = reranker.rerank_scored(query, texts)
        except Exception as e:
            logger.warning("Fact-floor rerank failed, keeping all facts: %s", e)
            return facts
        kept = [facts[i] for i, s in scored if s >= _RECALL_FACT_FLOOR]
        return kept or [facts[i] for i, _ in scored[:1]]

    def _compact_to_passages(
        self, query: str, episodes: list[dict[str, Any]], n: int
    ) -> list[dict[str, Any]]:
        """Compact the top reranked episodes into the n most query-relevant PASSAGES (Stage 2).

        Splits each episode into markdown chunks (ingestion.web_chunker.chunk_markdown), reranks the
        pooled chunks with the SAME cross-encoder, and serves the top-n with their parent episode's
        project/date. ~1/4 the tokens of full-episode serving at near-baseline answer survival
        (scripts/passage_bench_v*, 2026-06-26). A direct rerank over the chunks is the bench's
        hybrid->rerank cascade's ceiling at this bounded chunk count, and rerank-only avoids a live
        passage embed. Returns [] on chunk/rerank failure so the caller falls back to full episodes;
        gated by _RECALL_PASSAGES so this runs only when enabled."""
        from ingestion.web_chunker import chunk_markdown

        passages: list[str] = []
        owner: list[dict[str, Any]] = []
        bounds: list[tuple[int, int]] = []  # passage char span in its parent's content
        role_spans: dict[int, list[tuple[int, str]]] = {}  # id(episode) -> marker layout
        for e in episodes:
            content = e.get("content") or ""
            if not content.strip():
                continue
            role_spans[id(e)] = _role_spans(content)
            try:
                chunks = [
                    (c.content, c.char_start, c.char_end)
                    for c in chunk_markdown(content)
                    if c.content.strip()
                ]
            except Exception:
                chunks = [(content, 0, len(content))]
            for ch, lo, hi in chunks:
                passages.append(ch)
                owner.append(e)
                bounds.append((lo, hi))
        if not passages:
            return []
        # Cap the rerank input so a pathologically long episode can't blow up the call.
        if len(passages) > _RECALL_PASSAGE_CAND:
            passages = passages[:_RECALL_PASSAGE_CAND]
            owner = owner[:_RECALL_PASSAGE_CAND]
            bounds = bounds[:_RECALL_PASSAGE_CAND]
        # EXPERIMENT (env-gated): structural compaction v2.
        # SYNAPSE_PASSAGE_QUOTA=k caps served chunks per parent episode (slot allocation —
        # one loud session can't eat every slot). SYNAPSE_PASSAGE_WINDOW=w merges each
        # winning chunk with up to w adjacent chunks of the same episode (restores the
        # connective context that makes fragments summable). Both 0/off by default.
        _quota = int(os.environ.get("SYNAPSE_PASSAGE_QUOTA", "0") or "0")
        _window = int(os.environ.get("SYNAPSE_PASSAGE_WINDOW", "0") or "0")
        if len(passages) <= n:
            chosen = list(range(len(passages)))  # already in episode-rerank order
        else:
            reranker = self._ensure_reranker()
            if reranker is None:  # rerank disabled — no selection signal, serve full episodes
                return []
            try:
                scored = reranker.rerank_scored(query, passages, top_k=None if _quota else n)
            except Exception as e:
                logger.warning("Passage rerank failed, serving full episodes: %s", e)
                return []
            if _quota:
                per: dict[int, int] = {}
                chosen = []
                for i, _s in scored:
                    k = id(owner[i])
                    if per.get(k, 0) >= _quota:
                        continue
                    per[k] = per.get(k, 0) + 1
                    chosen.append(i)
                    if len(chosen) >= n:
                        break
            else:
                chosen = [i for i, _ in scored[:n]]
        out: list[dict[str, Any]] = []
        used: set[int] = set()
        for i in chosen:
            if i in used:
                continue
            ep = owner[i]
            lo = hi = i
            for _ in range(_window):
                if lo - 1 >= 0 and owner[lo - 1] is ep:
                    lo -= 1
                if hi + 1 < len(passages) and owner[hi + 1] is ep:
                    hi += 1
            span = [j for j in range(lo, hi + 1) if j not in used]
            used.update(span)
            item: dict[str, Any] = {}
            if (rid := ep.get("id")) is not None:
                item["id"] = rid  # parent episode — pass to fetch() to expand the full turn
            item["content"] = "\n".join(passages[j] for j in span)
            if (project := ep.get("project")) is not None:
                item["project"] = project
            if (ts := ep.get("created_at")) is not None:
                item["date"] = str(ts)[:10]
            # Provenance label (issue #17): who produced this slice of the turn —
            # "user" / "assistant" / "mixed"; omitted when unattributable.
            if role := _passage_role(role_spans[id(ep)], bounds[span[0]][0], bounds[span[-1]][1]):
                item["role"] = role
            out.append(item)
        return out

    def _select_episodes(
        self, query: str, pool: list[dict[str, Any]], limit: int
    ) -> tuple[list[dict[str, Any]], int, float]:
        """Rerank `pool`, re-inject recency + suppress query echo, and pick what to serve.

        Returns ``(episodes, n_echo_suppressed, rerank_top)``. ``rerank_top`` is the RAW
        pre-recency top rerank score — the value recall_metrics.rerank_top_score records and
        the shadow abstention floor compares against — 0.0 when the pool is empty or the
        rerank degraded/was disabled. Default (SYNAPSE_EPISODE_CUTOFF_TAU <= 0):
        fixed top-`limit` in the recency-adjusted rerank order. With tau > 0: adaptive relative
        score-cutoff (keep score >= tau*top, clamped [_EPISODE_CUTOFF_MIN_K, _EPISODE_CUTOFF_MAX_K])
        over the recency-adjusted scores. Echoed episodes (the query quoting itself) are dropped
        BEFORE the final slice/cutoff so the next-ranked candidates backfill the freed slots. Always
        degrades to RRF order on rerank failure; never hard-fails."""
        if not pool:
            return [], 0, 0.0
        scored = self._rerank_pool_scored(query, pool)
        if not scored:
            return [], 0, 0.0
        rerank_top = scored[0][1]  # RAW top score (telemetry + shadow floor) — pre-recency
        degraded = rerank_top <= 0.0  # reranker down/disabled -> RRF order, fixed-k
        scored = self._apply_rerank_recency(scored, pool)  # no-op when disabled/degraded
        ranked = [pool[i] for i, _ in scored]
        ranked_scores = [s for _, s in scored]
        fixed_k = _EPISODE_CUTOFF_TAU <= 0 or degraded
        # Echo suppression's lazy scan only needs enough survivors to cover the serve
        # window: `limit` on the fixed-k path, at most _EPISODE_CUTOFF_MAX_K on the tau
        # path (k is clamped there, so ranked[:k] never reaches past the scanned prefix).
        need = limit if fixed_k else max(limit, _EPISODE_CUTOFF_MAX_K)
        keep, n_echo = self._filter_query_echo(query, ranked, need)
        if n_echo:
            ranked = [ranked[i] for i in keep]
            ranked_scores = [ranked_scores[i] for i in keep]
        if fixed_k:
            return ranked[:limit], n_echo, rerank_top
        k = _cutoff_k(
            ranked_scores,
            _EPISODE_CUTOFF_TAU,
            _EPISODE_CUTOFF_MIN_K,
            _EPISODE_CUTOFF_MAX_K,
        )
        return ranked[:k], n_echo, rerank_top

    def _episode_pool(
        self,
        query: str,
        query_emb: list[float] | None,
        project: str | None,
        fetch: int = _EPISODE_FETCH,
        pool_size: int = _EPISODE_RERANK_POOL,
    ) -> list[dict[str, Any]]:
        """Fused BM25+vector episode candidate pool, PRE-rerank (WIN1 deep-fetch).

        Shared primitive: recall_episodes() reranks it and serves top-k (deep
        drill-down); recall() merges it with the summary candidates and co-reranks
        in a single cross-encoder pass, then partitions by doc_type. The deep fetch
        is what lets the reranker recover a rank-16..88 gold; plain RRF can't.
        BM25-only when the query embedding is unavailable."""
        bm25_eps = self._search_bm25_episodes(query, project, fetch)
        vec_eps: list[dict[str, Any]] = []
        if query_emb is not None:
            vec_eps = self._search_vector_episodes(query_emb, project, fetch)
        return _merge_rrf(bm25_eps, vec_eps, id_key="id")[:pool_size]

    # ------------------------------------------------------------------
    # web_chunks search — BM25 + vector over scraped web pages
    # ------------------------------------------------------------------
    #
    # web_chunks is a sidecar of web_artifacts (1:N). Each scrape produces
    # ~1500-char chunks with 20% overlap. Both search paths JOIN web_artifacts
    # to surface url + title in the result row. Dedup by web_artifact_id at
    # the merge layer so a single page never takes more than one slot.

    def _search_bm25_web(self, query: str, limit: int) -> list[dict[str, Any]]:
        pg = self._ensure_pg()
        try:
            rows = pg.execute(
                """
                SELECT c.id, c.content, c.context_prefix, c.web_artifact_id,
                       c.content_ts AS created_at,
                       a.url, a.title, a.tool_name,
                       paradedb.score(c.id) AS bm25_score
                FROM web_chunks c
                JOIN web_artifacts a ON a.id = c.web_artifact_id
                WHERE c.id @@@ paradedb.match('content', %s)
                ORDER BY bm25_score DESC LIMIT %s
                """,
                (query, limit),
            ).fetchall()
            return [{**dict(r), "doc_type": "web", "id": f"w:{r['id']}"} for r in rows]
        except Exception as e:
            logger.warning("BM25 web search failed: %s", e)
            return []

    def _search_vector_web(self, query_emb: list[float], limit: int) -> list[dict[str, Any]]:
        pg = self._ensure_pg()
        emb_literal = "[" + ",".join(str(x) for x in query_emb) + "]"
        try:
            rows = pg.execute(
                """
                SELECT c.id, c.content, c.context_prefix, c.web_artifact_id,
                       c.content_ts AS created_at,
                       a.url, a.title, a.tool_name,
                       (c.embedding <=> %s::vector) AS vec_distance
                FROM web_chunks c
                JOIN web_artifacts a ON a.id = c.web_artifact_id
                WHERE c.is_embedded = TRUE
                ORDER BY vec_distance ASC LIMIT %s
                """,
                (emb_literal, limit),
            ).fetchall()
            return [{**dict(r), "doc_type": "web", "id": f"w:{r['id']}"} for r in rows]
        except Exception as e:
            logger.warning("Vector web search failed: %s", e)
            return []

    @staticmethod
    def _dedupe_by_artifact(chunks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        """Keep only the highest-ranked chunk per parent web_artifact_id.

        Input is assumed already RRF-ordered. First occurrence of an
        artifact_id wins; subsequent chunks from the same page drop.
        """
        seen: set[int] = set()
        out: list[dict[str, Any]] = []
        for c in chunks:
            aid = c.get("web_artifact_id")
            if aid in seen:
                continue
            if aid is not None:
                seen.add(aid)
            out.append(c)
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------------
    # KG traversal (Postgres)
    # ------------------------------------------------------------------

    def _search_kg(
        self,
        query: str,
        query_emb: list[float],
        group_id: str,
        session_focus: list[str],
        fact_limit: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """KG facts leg over kg_entities / kg_relationships (task #67).

        Runs inside one transaction so the planner GUCs are SET LOCAL — scoped
        to this query, not the thread-local connection that other legs may
        reuse later (enable_seqscan=off session-wide would be a foot-gun for
        any future unindexed query). hnsw.ef_search is already set
        session-wide by _ensure_pg. A failure degrades to an empty facts
        bucket rather than failing the whole recall.
        """
        try:
            conn = self._ensure_pg()
            with conn.transaction():
                # kg_pg indexes rows positionally; recall's thread-local conns
                # default to dict_row, so override at the cursor.
                cur = conn.cursor(row_factory=tuple_row)
                cur.execute("SET LOCAL enable_seqscan = off")
                cur.execute("SET LOCAL max_parallel_workers_per_gather = 0")
                return search_kg_postgres(
                    cur, query, query_emb, _KG_OWNER, group_id, session_focus, fact_limit
                )
        except Exception as e:
            logger.warning("KG search failed: %s", e)
            return [], []

    def _fetch_superseded_pairs_pg(
        self,
        group_id: str,
        active_edge_uuids: list[str],
        cap: int,
    ) -> list[dict[str, Any]]:
        """Superseded-fact pairs for served edges (Postgres port of the old
        FalkorDB _fetch_history_pairs).

        DISTINCT ON picks the most recently invalidated predecessor per active
        edge in SQL (the FalkorDB path does this dedup in Python).
        """
        if not active_edge_uuids or cap <= 0:
            return []
        try:
            conn = self._ensure_pg()
            rows = conn.execute(
                """
                SELECT DISTINCT ON (a.uuid) a.uuid AS uid, a.fact AS now_fact,
                       o.uuid AS old_uid, o.fact AS old_fact
                FROM kg_relationships a
                JOIN kg_relationships o
                  ON o.src_uuid = a.src_uuid AND o.tgt_uuid = a.tgt_uuid
                 AND o.owner_id = a.owner_id AND o.group_id = a.group_id
                WHERE a.owner_id = %s AND a.group_id = %s
                  AND a.uuid = ANY(%s) AND a.t_invalid IS NULL
                  AND o.t_invalid IS NOT NULL AND o.uuid <> a.uuid
                  AND o.fact IS NOT NULL AND a.fact IS NOT NULL
                ORDER BY a.uuid, o.t_invalid DESC
                """,
                (_KG_OWNER, group_id, active_edge_uuids),
            ).fetchall()
        except Exception as e:
            logger.debug("PG superseded-pairs query failed: %s", e)
            return []
        by_uid = {r["uid"]: r for r in rows}
        out: list[dict[str, Any]] = []
        for uid in active_edge_uuids:
            r = by_uid.get(uid)
            if r is None:
                continue
            # id is the invalidated (old) edge's uuid, "f:<uuid>" like a regular fact.
            # Collision-free: the facts bucket serves the CURRENT edge (a.uuid), never
            # the superseded predecessor (o.uuid, enforced <> a.uuid above). Lets the
            # stale pair be cited in recall_feedback.
            out.append(
                {
                    "id": f"f:{r['old_uid']}",
                    "fact": r["old_fact"],
                    "superseded_by": r["now_fact"],
                }
            )
            if len(out) >= cap:
                break
        return out

    # ------------------------------------------------------------------
    # Feedback weights
    # ------------------------------------------------------------------

    def _increment_retrieval_counts(self, episode_ids: list[int]) -> None:
        if not episode_ids:
            return
        pg = self._ensure_pg()
        try:
            placeholders = ",".join(["%s"] * len(episode_ids))
            pg.execute(
                f"UPDATE episodes SET retrieval_count = retrieval_count + 1 WHERE id IN ({placeholders})",
                episode_ids,
            )
        except Exception as e:
            logger.warning("Failed to increment retrieval counts: %s", e)

    def _increment_fact_retrieval_counts(self, edge_uuids: list[str], group_id: str) -> None:
        """Bump retrieval_count on each surfaced RELATES_TO edge — fire-and-forget.

        Submitted to a background thread so recall() returns immediately. The
        bump itself is single-Cypher-batch via UNWIND (one round-trip instead
        of N). Best-effort: a failure here must not affect the response that's
        already on its way back to the caller.

        Uses COALESCE because pre-existing edges (Graphiti era) have no
        retrieval_count property — only edges Synapse created from round 2
        onward initialize it explicitly.
        """
        if not edge_uuids:
            return
        # Submit to background pool; do NOT wait for the future. The recall
        # response is already returning by the time this runs.
        self._async_executor.submit(self._do_increment, list(edge_uuids), group_id)

    def _do_increment(self, edge_uuids: list[str], group_id: str) -> None:
        """Worker that runs in the background thread. Uses its own connection
        to avoid contention with a follow-up recall already mid-flight."""
        try:
            with psycopg.connect(self._db_url, autocommit=True) as conn:
                conn.execute(
                    "UPDATE kg_relationships "
                    "SET retrieval_count = COALESCE(retrieval_count, 0) + 1 "
                    "WHERE owner_id = %s AND group_id = %s AND uuid = ANY(%s)",
                    (_KG_OWNER, group_id, edge_uuids),
                )
        except Exception as e:
            logger.debug("Background fact-bump (PG) failed: %s", e)

    # Telemetry: every recall()/recall_episodes() call records one row to
    # recall_metrics (schema/021) — timing per leg, served-payload tokens, pool
    # sizes, rerank model + top score, origin. NOT logfire: this is local,
    # SQL-queryable, and reuses the existing background-write pattern. A recall_
    # episodes row leaves the recall-only columns NULL.
    _METRIC_COLS = (
        "kind",
        "source",
        "query",
        "group_id",
        "write_feedback",
        "ms_total",
        "ms_embed",
        "ms_bm25",
        "ms_vector",
        "ms_kg",
        "ms_web",
        "ms_rerank",
        "n_facts",
        "n_episodes",
        "n_entities",
        "n_web",
        "n_history",
        "chars",
        "est_tokens",
        "pool_bm25",
        "pool_vector",
        "pool_fused",
        "kg_candidates",
        "rerank_model",
        "rerank_top_score",
        "emb_ok",
        "n_timeline",
        "ms_timeline",
        "n_prefs",
        "ms_prefs",
        "served_ids",
    )

    def _record_metrics(self, m: dict[str, Any]) -> None:
        """Fire-and-forget: submit the metrics row to the background pool so the
        recall response returns immediately (same discipline as the fact bump)."""
        self._async_executor.submit(self._do_record, m)

    def _do_record(self, m: dict[str, Any]) -> None:
        """Background worker: insert one recall_metrics row. Best-effort — a
        failure (table missing pre-migration, DB hiccup) must never surface to
        the caller, whose response already returned."""
        try:
            cols = self._METRIC_COLS
            placeholders = ",".join(["%s"] * len(cols))
            vals = [m.get(c) for c in cols]
            sid_idx = cols.index("served_ids")
            if vals[sid_idx] is not None:
                vals[sid_idx] = PgJson(vals[sid_idx])
            with psycopg.connect(self._db_url, autocommit=True) as conn:
                conn.execute(
                    f"INSERT INTO recall_metrics ({','.join(cols)}) VALUES ({placeholders})",
                    vals,
                )
        except Exception as e:
            logger.debug("recall_metrics write failed: %s", e)

    def record_event(
        self,
        kind: str,
        *,
        source: str | None = None,
        query: str | None = None,
        group_id: str | None = None,
        ms_total: float | None = None,
        chars: int | None = None,
        est_tokens: int | None = None,
        served_ids: dict[str, Any] | None = None,
    ) -> None:
        """Public passthrough onto the fire-and-forget recall_metrics writer for
        non-recall callers (the remember tool's kind='remember' rows, the board's
        kind='board' rows): one telemetry row, background insert, never surfaces
        a failure. Columns not passed stay NULL."""
        self._record_metrics(
            {
                "kind": kind,
                "source": source,
                "query": query,
                "group_id": group_id,
                "ms_total": ms_total,
                "chars": chars,
                "est_tokens": est_tokens,
                "served_ids": served_ids,
            }
        )

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        project: str | None = None,
        session_focus: list[str] | None = None,
        group_id: str = "technical",
        write_feedback: bool = True,
        source: str | None = None,
        debug: bool = False,
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
        use recall_episodes(). Facts carry their t_valid "as-of" date for currency weighting.

        ``debug`` (phase-2 dashboard console) attaches a ``debug`` key to the response SURFACING
        the SAME numbers already measured for the recall_metrics telemetry row — no extra
        instrumentation, no extra work. Off by default so every non-dashboard call is byte-identical
        in behavior AND in the telemetry it records (a live call-rate A/B depends on that). See the
        debug-dict assembly just below the metrics write for the exact shape.
        """
        t_start = time.perf_counter()
        web_n = _WEB_LIMIT * 4  # depth for dedup-by-artifact to have headroom
        ex = self._leg_executor

        # BM25 is pure text search — it does NOT need the query embedding. Start it
        # FIRST so its ~165ms fetch overlaps the ~170ms Voyage query-embedding call
        # below, instead of running after it (the embed gates the vector/KG/web legs,
        # but not BM25). Each leg owns a thread-local PG connection, so concurrent
        # psycopg use is safe. Legs run through _timed for per-leg latency telemetry.
        f_bm25 = ex.submit(_timed, self._search_bm25_episodes, query, project, _EPISODE_FETCH)

        t_emb = time.perf_counter()
        try:
            query_emb = self._ensure_embedder().embed([query], task="query")[0]
        except Exception as e:
            logger.error("Embedding query failed: %s", e)
            query_emb = None
        ms_embed = (time.perf_counter() - t_emb) * 1000.0

        # The remaining legs all need the embedding; fan them out concurrently (they
        # no-op when it's unavailable). Summaries were retired (task #63): the KG owns
        # facts and the wide episode leg owns broad/needle, so no synth_documents leg.
        def _web_leg() -> list[dict[str, Any]]:
            return self._search_vector_web(query_emb, web_n) if query_emb is not None else []

        def _kg_leg() -> tuple[list[Any], list[Any]]:
            if query_emb is None:
                return [], []
            return self._search_kg(
                query, query_emb, group_id, session_focus or [], fact_limit=_FACT_LIMIT
            )

        f_vec = (
            ex.submit(_timed, self._search_vector_episodes, query_emb, project, _EPISODE_FETCH)
            if query_emb is not None
            else None
        )
        f_web = ex.submit(_timed, _web_leg)
        f_kg = ex.submit(_timed, _kg_leg)

        # Timeline leg: only on temporal intent. Reuses this call's query embedding
        # (no second Voyage call); the engine owns its own thread-local PG conns.
        def _timeline_leg() -> list[dict[str, Any]]:
            res = self._ensure_timeline().recall_timeline(
                query=query, project=project, query_emb=query_emb, group_id=group_id
            )
            return list(res.get("items") or [])[:_TIMELINE_LIMIT]

        f_timeline = ex.submit(_timed, _timeline_leg) if _TIMELINE_IN_RECALL else None

        # Preferences leg: top-5 live user preferences by cosine to this query. One cheap
        # parallel read reusing query_emb; empty result = no payload change (kill switch
        # SYNAPSE_RECALL_PREFS=0). Group-scoped like the KG; owner is the single-user const.
        f_prefs = (
            ex.submit(_timed, self._search_preferences, query_emb, group_id, _PREFS_LIMIT)
            if _PREFS_IN_RECALL
            else None
        )

        # Fuse BM25 + vector into the rerank pool — identical to _episode_pool's output
        # (used by recall_episodes), just with BM25 hoisted ahead of the embed.
        bm25_eps, ms_bm25 = f_bm25.result()
        vec_eps, ms_vec = f_vec.result() if f_vec is not None else ([], 0.0)
        ep_pool = _merge_rrf(bm25_eps, vec_eps, id_key="id")[:_EPISODE_RERANK_POOL]
        vec_web, ms_web = f_web.result()
        (kg_results, seed_entities), ms_kg = f_kg.result()

        facts_internal = kg_results[:_FACT_LIMIT]  # carry _uuid for bump + superseded pairs
        # Feedback loop: bump retrieval_count on every surfaced edge so frequent hits
        # float higher next time. Already fire-and-forget — never blocks the response.
        surfaced_edge_uuids = [f["_uuid"] for f in facts_internal if f.get("_uuid")]
        if surfaced_edge_uuids and write_feedback:
            self._increment_fact_retrieval_counts(surfaced_edge_uuids, group_id)

        # Second wave, also concurrent: the cross-encoder rerank (Voyage HTTP) and the
        # bi-temporal superseded-pairs fetch hit different backends. Use the SCORED rerank — same
        # ordering as _rerank_pool, but it also yields the top relevance score (a recall-
        # confidence signal, and the basis for an eventual inject-only-if-relevant gate).
        f_rerank = ex.submit(_timed, self._rerank_pool_scored, query, ep_pool)
        f_superseded = ex.submit(
            self._fetch_superseded_pairs_pg, group_id, surfaced_edge_uuids, _SUPERSEDED_LIMIT
        )
        scored, ms_rerank = f_rerank.result()
        superseded_facts = f_superseded.result()
        rerank_top = scored[0][1] if scored else 0.0  # RAW top score (telemetry) — pre-recency
        # Post-rerank recency re-injection: the cross-encoder is recency-blind, so an old
        # *definitive* claim out-ranks a newer *correction* when both make the pool. Re-weight
        # the FINAL ordering only; the rerank call, the pool, and rerank_top above are untouched.
        scored = self._apply_rerank_recency(scored, ep_pool)
        ranked = [ep_pool[i] for i, _ in scored]

        # Episodes: pure rerank order (matches recall_episodes / the measured swap).
        ranked_eps = [x for x in ranked if x.get("doc_type") == "episode"]
        # Query-echo suppression: drop episodes that are the prompt quoting itself (compaction
        # copies / re-ingested repeats); the slices below backfill freed slots from next-ranked.
        # Passage mining reads the top _RECALL_PASSAGE_SRC_K, so that bounds the lazy scan.
        keep, n_echo_suppressed = self._filter_query_echo(
            query, ranked_eps, max(_RECALL_EPISODE_LIMIT, _RECALL_PASSAGE_SRC_K)
        )
        if n_echo_suppressed:
            ranked_eps = [ranked_eps[i] for i in keep]
        episodes_served = ranked_eps[:_RECALL_EPISODE_LIMIT]
        # Stage 2: serve compact passages of the top reranked episodes instead of whole turns.
        # Falls back to full episodes if compaction yields nothing. Drill-down stays full.
        ep_items: list[dict[str, Any]] | None = None
        if ranked_eps:
            ep_items = (
                self._compact_to_passages(
                    query, ranked_eps[:_RECALL_PASSAGE_SRC_K], _RECALL_PASSAGE_N
                )
                or None
            )
        if ep_items is None and episodes_served:
            ep_items = [_to_recall_item(r) for r in episodes_served]
        # Web bucket: vector-only, dedupe by parent page. BM25 over this corpus produces
        # cross-domain token collisions; the bi-encoder captures topic over surface tokens.
        web_chunks = self._dedupe_by_artifact(vec_web, _WEB_LIMIT)

        # Surface the internal _uuid to the caller as a "f:<uuid>" id (below) so facts
        # are citable in recall_feedback, same as episodes carry "e:N".
        # Optional relevance gate (SYNAPSE_RECALL_FACT_FLOOR > 0): drop off-topic facts —
        # the one place the "recall returns irrelevant stuff" lever measurably works. OFF by
        # default (adds one rerank of the served facts), so this is a no-op until enabled.
        served_facts = facts_internal
        if _RECALL_FACT_FLOOR > 0 and len(served_facts) > 1:
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

        # Entity bucket: top-N seed entities with non-trivial summaries.
        # Skip when summary is missing or just echoes the entity name.
        entities_bucket: list[dict[str, Any]] = []
        for s in seed_entities[:_ENTITY_LIMIT]:
            summary = (s.get("summary") or "").strip()
            name = (s.get("name") or "").strip()
            if not summary or summary.lower() == name.lower():
                continue
            entities_bucket.append({"name": name, "summary": summary})

        out: dict[str, Any] = {
            "query": query,
            "facts": facts,  # slim {fact: ...}
        }
        if ep_items:
            out["episodes"] = ep_items
        if entities_bucket:
            out["entities"] = entities_bucket  # {name, summary}
        if web_chunks:
            out["web"] = [_to_web_recall_item(r) for r in web_chunks]
        if superseded_facts:
            # Renamed from "history" 2026-07-18 — the displaced version of each served
            # fact, keyed like the superseded_by columns elsewhere in the system.
            out["superseded_facts"] = superseded_facts  # {fact: old, superseded_by: current}
        timeline_items: list[dict[str, Any]] = []
        ms_timeline = 0.0
        if f_timeline is not None:
            try:
                timeline_items, ms_timeline = f_timeline.result()
            except Exception as e:
                logger.warning("timeline leg failed: %s", e)
        if timeline_items:
            # Slim chronological bucket: date + fact (+type/salience). The dates make
            # interval questions answerable AND auditable (anchor events, never a bare N).
            tl: list[dict[str, Any]] = []
            for t in timeline_items:
                if t.get("kind", "event") != "event":
                    continue
                item = {
                    "date": str(t.get("t_valid"))[:10],
                    "fact": t.get("fact"),
                    "type": t.get("event_type"),
                    "salience": t.get("salience"),
                }
                if (tid := t.get("_id")) is not None:
                    item = {"id": f"t:{tid}", **item}  # cite in recall_feedback (not fetch())
                tl.append(item)
            out["timeline"] = tl

        prefs_items: list[dict[str, Any]] = []
        ms_prefs = 0.0
        if f_prefs is not None:
            try:
                prefs_items, ms_prefs = f_prefs.result()
            except Exception as e:
                logger.warning("preferences leg failed: %s", e)
        if prefs_items:
            # Slim preference bucket: the pref text + polarity, plus since/asserted so the
            # reader can weight a long-standing, oft-repeated preference over a one-off.
            out["preferences"] = [
                {
                    **({"id": f"p:{p['id']}"} if p.get("id") is not None else {}),
                    "pref": p.get("pref"),
                    "polarity": p.get("polarity"),
                    "since": p.get("since"),
                    "asserted": p.get("assert_count"),
                }
                for p in prefs_items
            ]

        # Fire-and-forget telemetry to recall_metrics (NOT logfire) — same background-write
        # pattern as the retrieval_count bump, so zero read-path latency.
        # served_ids (issue #10): WHICH results were served, per bucket. Episodes dedupe
        # because passages share a parent id; timeline mirrors the bucket's event filter.
        served_ids: dict[str, Any] = {
            "episodes": list(dict.fromkeys(it["id"] for it in (ep_items or []) if it.get("id"))),
            "facts": [f["_uuid"] for f in served_facts if f.get("_uuid")],
            "timeline": [
                t["_id"]
                for t in timeline_items
                if t.get("_id") is not None and t.get("kind", "event") == "event"
            ],
            "prefs": [p["id"] for p in prefs_items if p.get("id") is not None],
            "n_echo_suppressed": n_echo_suppressed,
        }
        # Shadow abstention floor (telemetry only): mark when an enforced floor WOULD have
        # abstained. Compares the RAW pre-recency rerank_top recorded below — NOT the
        # recency-adjusted ordering — so the marker and rerank_top_score always agree.
        _floor_shadow(served_ids, float(rerank_top), emb_ok=query_emb is not None)
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
            "n_entities": len(entities_bucket),
            "n_web": len(web_chunks),
            "n_history": len(superseded_facts),
            "n_timeline": len(out.get("timeline", [])),
            "ms_timeline": round(ms_timeline, 1),
            "n_prefs": len(out.get("preferences", [])),
            "ms_prefs": round(ms_prefs, 1),
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
        # re-instrumentation). Only the timed legs are exposed; timeline/prefs keys are
        # OMITTED when their leg is disabled (future f_* is None), so the console renders
        # those as untimed/skipped rather than a spurious 0ms. Byte-identical when off.
        if debug:
            legs_ms: dict[str, Any] = {
                "embed": metrics["ms_embed"],
                "bm25": metrics["ms_bm25"],
                "vector": metrics["ms_vector"],
                "kg": metrics["ms_kg"],
                "web": metrics["ms_web"],
                "rerank": metrics["ms_rerank"],
            }
            if f_timeline is not None:
                legs_ms["timeline"] = metrics["ms_timeline"]
            if f_prefs is not None:
                legs_ms["prefs"] = metrics["ms_prefs"]
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

    def recall_episodes(
        self,
        query: str,
        project: str | None = None,
        limit: int = _EPISODE_LIMIT,
        source: str | None = None,
    ) -> dict[str, Any]:
        """Raw episode drill-down: individual conversation turns.

        Best for 'show me exactly what was said about X' queries.
        Returns full episode content ranked by relevance + recency.
        """
        t_start = time.perf_counter()
        try:
            query_emb = self._ensure_embedder().embed([query], task="query")[0]
        except Exception as e:
            logger.error("Embedding query failed: %s", e)
            query_emb = None

        # Deep-fetch both legs (golds rank up to ~88 in a single leg), fuse, then
        # rerank the WIDE pool and select what to serve (WIN1 — see _episode_pool).
        # Fixed top-`limit` by default; adaptive score-cutoff when enabled (see
        # _select_episodes / _EPISODE_CUTOFF_TAU).
        episodes, n_echo_suppressed, rerank_top = self._select_episodes(
            query, self._episode_pool(query, query_emb, project), limit
        )

        # Increment feedback counts BEFORE slimming (we lose the parseable id afterwards)
        ep_ids = [
            int(r["id"].split(":")[1])
            for r in episodes
            if isinstance(r.get("id"), str) and r["id"].startswith("e:")
        ]
        if ep_ids:
            self._increment_retrieval_counts(ep_ids)

        out = {
            "query": query,
            "episodes": [_to_recall_item(r) for r in episodes],
        }
        served_ids: dict[str, Any] = {
            "episodes": [r["id"] for r in episodes if r.get("id")],
            "n_echo_suppressed": n_echo_suppressed,
        }
        # Shadow abstention floor (telemetry only) — same RAW pre-recency score contract
        # as recall(); the served episodes above are untouched.
        _floor_shadow(served_ids, float(rerank_top), emb_ok=query_emb is not None)
        chars = _served_chars(out)
        self._record_metrics(
            {
                "kind": "episodes",
                "source": source or "mcp",
                "query": query[:200],
                "ms_total": round((time.perf_counter() - t_start) * 1000.0, 1),
                "n_episodes": len(episodes),
                "chars": chars,
                "est_tokens": chars // 4,
                "rerank_top_score": round(float(rerank_top), 4),
                "emb_ok": query_emb is not None,
                "served_ids": served_ids,
            }
        )
        return out

    def fetch(self, ids: list[Any], source: str | None = None) -> dict[str, Any]:
        """Drill-down by id: expand recall()'s compact serves into full records.

        Accepts mixed prefixed ids — "e:N" episodes (bare N / bare int also accepted,
        the old fetch_episode back-compat) and "n:N" notes (the board's n:ID lines).
        Episodes come back in the recall-item shape ({id, content, project, date}),
        notes as {id, hook, body, type, project, updated}; both ordered to match the
        request. Unknown prefixes / unparseable ids are reported under ``skipped``;
        the total expanded is capped at _FETCH_MAX across both kinds."""
        t_start = time.perf_counter()
        ep_ids, note_ids, skipped, normalized = _parse_fetch_ids(ids)
        out: dict[str, Any] = {"episodes": [], "notes": [], "skipped": skipped}
        if not normalized:
            return out
        out["episodes"] = self._fetch_episode_records(ep_ids) if ep_ids else []
        out["notes"] = self._fetch_note_records(note_ids) if note_ids else []
        chars = _served_chars({"episodes": out["episodes"]}) + sum(
            len(n.get("hook") or "") + len(n.get("body") or "") for n in out["notes"]
        )
        self._record_metrics(
            {
                "kind": "fetch",
                "source": source or "mcp",
                "query": ",".join(normalized)[:200],
                "ms_total": round((time.perf_counter() - t_start) * 1000.0, 1),
                "n_episodes": len(out["episodes"]),
                "chars": chars,
                "served_ids": {"kinds": {"e": len(out["episodes"]), "n": len(out["notes"])}},
            }
        )
        return out

    def _fetch_episode_records(self, parsed: list[int]) -> list[dict[str, Any]]:
        """The episodes leg of fetch(): full untruncated turns by int id. Fail-soft —
        a read error serves an empty leg, never breaks the call."""
        try:
            conn = self._ensure_pg()
            rows = conn.execute(
                "SELECT id, content, project, created_at FROM episodes WHERE id = ANY(%s)",
                (parsed,),
            ).fetchall()
        except Exception as e:
            logger.warning("fetch episodes leg failed: %s", e)
            return []
        by_id = {r["id"]: r for r in rows}
        found = [n for n in parsed if n in by_id]
        if found:
            self._increment_retrieval_counts(found)
        return [_to_recall_item({**by_id[n], "id": f"e:{n}"}) for n in found]

    def _fetch_note_records(self, parsed: list[int]) -> list[dict[str, Any]]:
        """The notes leg of fetch(): board-note bodies by int id (the on-demand half of
        the board — hook on the board, body behind the id). Uses a short-lived Database
        like the other notes-store paths. Fail-soft like the episodes leg."""
        from ingestion.db import Database

        try:
            db = Database(self._db_url)
            try:
                rows = db.get_notes_by_ids(parsed)
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

    def _episode_supersessions(
        self, episode_ids: list[int], group_id: str, cap: int = 6
    ) -> dict[int, list[str]]:
        """Map served episode ids -> the CURRENT facts that superseded a claim each made.

        A retired edge P citing the episode (episodes @> [id]) links to its superseding live edge N
        via P.invalidated_by (schema 028 + backfill); N.fact is the "now" value. Hits the partial GIN
        (schema 029, WHERE invalidated_by IS NOT NULL) so the per-recall lookup is cheap. Fail-open —
        a lookup error just yields no annotations, never breaks recall."""
        if not episode_ids:
            return {}
        ors = " OR ".join(["p.episodes @> %s::jsonb"] * len(episode_ids))
        params: list[Any] = [json.dumps([i]) for i in episode_ids] + [_KG_OWNER, group_id, cap]
        try:
            conn = self._ensure_pg()
            rows = conn.execute(
                "SELECT p.episodes, n.fact FROM kg_relationships p "
                "JOIN kg_relationships n ON n.uuid = p.invalidated_by "
                f"WHERE p.invalidated_by IS NOT NULL AND ({ors}) "
                "  AND p.owner_id = %s AND p.group_id = %s LIMIT %s",
                params,
            ).fetchall()
        except Exception as e:
            logger.warning("episode supersession lookup failed: %s", e)
            return {}
        idset = set(episode_ids)
        out: dict[int, list[str]] = {}
        for r in rows:
            fact = r.get("fact")
            if not fact:
                continue
            for eid in r.get("episodes") or []:
                if eid in idset:
                    out.setdefault(int(eid), []).append(fact)
        return out

    def _surface_supersessions(
        self,
        query_emb: list[float] | None,
        group_id: str,
        served_uuids: set[str],
        cap: int = _SUP_LIMIT,
    ) -> list[dict[str, Any]]:
        """A query that matches a now-INVALID fact should still return the CURRENT answer.

        Finds superseded edges near the query that carry a precise successor link (invalidated_by,
        schema 028), resolves the live successor, and returns those not already served — deduped by
        uuid and distance-gated (_SUP_MAX_DIST) so only on-topic superseded facts pull their
        correction in. Go-forward coverage only (no link => skipped); never returns the stale fact
        itself. Shape matches the KG fact leg ({fact, _uuid, _date}) so it flows through fact serving.
        Fail-open — a lookup error just yields no extras."""
        if query_emb is None:
            return []
        vec = _vec_literal(query_emb)
        try:
            conn = self._ensure_pg()
            rows = conn.execute(
                "SELECT n.uuid, n.fact, n.t_valid, "
                f"  (p.fact_embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS})) AS d "
                "FROM kg_relationships p "
                "JOIN kg_relationships n ON n.uuid = p.invalidated_by AND n.t_invalid IS NULL "
                "WHERE p.t_invalid IS NOT NULL AND p.invalidated_by IS NOT NULL "
                "  AND p.fact_embedding IS NOT NULL AND p.owner_id = %s AND p.group_id = %s "
                f"ORDER BY p.fact_embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS}) LIMIT %s",
                (vec, _KG_OWNER, group_id, vec, _SUP_CANDIDATES),
            ).fetchall()
        except Exception as e:
            logger.warning("supersession surface failed: %s", e)
            return []
        out: list[dict[str, Any]] = []
        for r in rows:
            d = r.get("d")
            if d is None or d > _SUP_MAX_DIST:
                continue
            u, fact = r.get("uuid"), r.get("fact")
            if u and fact and u not in served_uuids:
                out.append({"fact": fact, "_uuid": u, "_date": r.get("t_valid")})
                served_uuids.add(u)
                if len(out) >= cap:
                    break
        return out
