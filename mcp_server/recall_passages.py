"""RecallPassagesMixin operations."""

from __future__ import annotations

import logging
import os
from typing import Any

from ingestion import embedding as _embedding
from mcp_server.recall_presentation import passage_role as _passage_role
from mcp_server.recall_presentation import role_spans as _role_spans
from mcp_server.recall_warnings import config_hint as _config_hint
from mcp_server.recall_warnings import error_brief as _err_brief
from mcp_server.recall_warnings import warn as _warn

logger = logging.getLogger(__name__)


class RecallPassagesMixin:
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
        settings = self._settings()
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
        if len(passages) > settings._RECALL_PASSAGE_CAND:
            passages = passages[: settings._RECALL_PASSAGE_CAND]
            owner = owner[: settings._RECALL_PASSAGE_CAND]
            bounds = bounds[: settings._RECALL_PASSAGE_CAND]
        # EXPERIMENT (env-gated): structural compaction v2.
        # SYNAPSE_PASSAGE_QUOTA=k caps served chunks per parent episode (slot allocation —
        # one loud session can't eat every slot). SYNAPSE_PASSAGE_WINDOW=w merges each
        # winning chunk with up to w adjacent chunks of the same episode (restores the
        # connective context that makes fragments summable). Both 0/off by default.
        _quota = int(os.environ.get("SYNAPSE_PASSAGE_QUOTA", "0") or "0")
        _window = int(os.environ.get("SYNAPSE_PASSAGE_WINDOW", "0") or "0")
        # Session-diversity cap on the SERVED passages: at most _sess_cap of the n served may
        # share a session_id. Fixes self/recency domination — the live session's freshly ingested
        # turns are topically dense AND the recency leg boosts them, so uncapped all n served come
        # from the current session, crowding out older real history (the top recall_feedback noise
        # driver, measured 2026-07-23). Backfills from lower-ranked passages so a genuinely
        # single-session result still serves n — the cap trims domination, never costs recall.
        # ON by default (=2); disable with SYNAPSE_RECALL_SESSION_CAP=0.
        _sess_cap = int(os.environ.get("SYNAPSE_RECALL_SESSION_CAP", "2") or "2")
        # Slack gate on the cap: a capped session's passage still serves when every
        # alternative from an under-served session scores more than _sess_slack below it —
        # diversity acts as a near-tie tiebreak instead of a hard constraint. The hard cap
        # (slack=0) measurably starves multi-hop questions whose evidence lives in 1-2
        # sessions: LME multi-session served-precision drops 0.88->0.72 (2026-07-25).
        # Rerank scores are 0-1 relevance; 0 (default) keeps the hard-cap behavior.
        _sess_slack = float(os.environ.get("SYNAPSE_RECALL_SESSION_CAP_SLACK", "0") or "0")
        # Freshness scope on the cap: >0 restricts the cap to sessions whose newest pooled
        # episode is within this many hours of now. The measured noise pattern the cap fixes
        # is specifically the LIVE session's turns crowding the bucket (prod replay 2026-07-25:
        # 11 of 27 dominations were <6h-old sessions at query time); an OLD session serving
        # multiple slots usually means the evidence genuinely lives there (LME multi-session:
        # capping those drops served-precision 0.88->0.72). Unparseable timestamps count as
        # fresh (cap applies — the conservative, shipped-behavior side). 0 = cap all sessions.
        _sess_fresh_h = float(os.environ.get("SYNAPSE_RECALL_SESSION_CAP_FRESH_H", "0") or "0")
        _capped = bool(_quota or _sess_cap)  # either cap walks the full ranking + backfills
        if len(passages) <= n:
            chosen = list(range(len(passages)))  # already in episode-rerank order
        else:
            reranker = self._ensure_reranker()
            if reranker is None:  # rerank disabled — no selection signal, serve full episodes
                return []
            try:
                scored = reranker.rerank_scored(query, passages, top_k=None if _capped else n)
            except Exception as e:
                logger.warning("Passage rerank failed, serving full episodes: %s", e)
                detail = _err_brief(e)
                backend = _embedding.rerank_provider()
                _warn(
                    f"passage rerank failed ({backend}: {detail}): episode passages dropped "
                    f"from this result, retry with recall_full_turns for raw turns."
                    + _config_hint(detail, backend=backend, env_prefix="SYNAPSE_RERANK")
                )
                return []
            if _capped:
                sess_fresh: dict[Any, bool] = {}
                if _sess_fresh_h > 0:
                    from datetime import UTC, datetime

                    _now = datetime.now(UTC)
                    for f_ep in episodes:
                        e_sid = f_ep.get("session_id")
                        if e_sid is None:
                            continue
                        try:
                            ts = f_ep.get("created_at")
                            dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=UTC)
                            fresh = (_now - dt).total_seconds() <= _sess_fresh_h * 3600
                        except Exception:
                            fresh = True  # unknown age -> treat as fresh, cap applies
                        sess_fresh[e_sid] = sess_fresh.get(e_sid, False) or fresh
                per_ep: dict[int, int] = {}
                per_sess: dict[Any, int] = {}
                chosen = []
                for pos, (i, _s) in enumerate(scored):
                    ep = owner[i]
                    if _quota and per_ep.get(id(ep), 0) >= _quota:
                        continue
                    sid = ep.get("session_id")
                    if (
                        _sess_cap
                        and sid is not None
                        and per_sess.get(sid, 0) >= _sess_cap
                        and (_sess_fresh_h <= 0 or sess_fresh.get(sid, True))
                    ):
                        if _sess_slack <= 0:
                            continue
                        alt = next(
                            (
                                s2
                                for i2, s2 in scored[pos + 1 :]
                                if owner[i2].get("session_id") != sid
                                and not (_quota and per_ep.get(id(owner[i2]), 0) >= _quota)
                                and not (
                                    owner[i2].get("session_id") is not None
                                    and per_sess.get(owner[i2].get("session_id"), 0) >= _sess_cap
                                )
                            ),
                            None,
                        )
                        if alt is not None and (_s - alt) <= _sess_slack:
                            continue  # near-tie: diversity wins the slot
                        # No alternative within slack — serving diversity here would cost
                        # real relevance, so the capped session keeps the slot.
                    per_ep[id(ep)] = per_ep.get(id(ep), 0) + 1
                    if sid is not None:
                        per_sess[sid] = per_sess.get(sid, 0) + 1
                    chosen.append(i)
                    if len(chosen) >= n:
                        break
                # Backfill: caps starved us below n (pool is genuinely one session / one episode) —
                # relax and take the next-best passages in score order so diversity-trimming never
                # reduces the served count when nothing more diverse exists to serve.
                if len(chosen) < n:
                    seen = set(chosen)
                    for i, _s in scored:
                        if i not in seen:
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
            if (sid := ep.get("session_id")) is not None:
                item["session"] = sid  # pivot key for fetch_session drill-down
            # Provenance label (issue #17): who produced this slice of the turn —
            # "user" / "assistant" / "mixed"; omitted when unattributable.
            if role := _passage_role(role_spans[id(ep)], bounds[span[0]][0], bounds[span[-1]][1]):
                item["role"] = role
            out.append(item)
        return out
