"""RecallPassagesMixin operations."""

from __future__ import annotations

import logging
import re
from typing import Any

from ingestion import embedding as _embedding
from ingestion.web_chunker import CHUNK_OVERLAP, chunk_markdown
from mcp_server.recall_presentation import passage_role as _passage_role
from mcp_server.recall_presentation import role_spans as _role_spans
from mcp_server.recall_warnings import config_hint as _config_hint
from mcp_server.recall_warnings import error_brief as _err_brief
from mcp_server.recall_warnings import warn as _warn

logger = logging.getLogger(__name__)


class RecallPassagesMixin:
    def _compact_to_passages(
        self,
        query: str,
        episodes: list[dict[str, Any]],
        n: int,
        stats: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Compact the top reranked episodes into the n most query-relevant PASSAGES (Stage 2).

        Splits each episode into markdown chunks (ingestion.web_chunker.chunk_markdown), reranks the
        pooled chunks with the SAME cross-encoder, and serves the top-n with their parent episode's
        project/date. ~1/4 the tokens of full-episode serving at near-baseline answer survival
        (scripts/passage_bench_v*, 2026-06-26). A direct rerank over the chunks is the bench's
        hybrid->rerank cascade's ceiling at this bounded chunk count, and rerank-only avoids a live
        passage embed. Serves at most n chunks as one item per parent episode, skipping chunks that
        repeat another episode's chosen chunk; ``stats`` (if given) gets ``n_dup_passages``.
        Returns [] on chunk/rerank failure so the caller falls back to full episodes."""
        settings = self._settings()
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
        # Every path walks the full ranking: a passage redundant with one already chosen is
        # skipped and its slot backfills from the next-ranked passage.
        redundant = _Redundancy(passages, owner)
        dup_ix: set[int] = set()

        def _fresh(i: int) -> bool:
            if redundant.covers(i):
                dup_ix.add(i)
                return False
            return True

        chosen: list[int] = []
        if len(passages) <= n:
            for i in range(len(passages)):  # already in episode-rerank order
                if _fresh(i):
                    chosen.append(i)
                    redundant.keep(i)
        else:
            reranker = self._ensure_reranker()
            if reranker is None:  # rerank disabled — no selection signal, serve full episodes
                return []
            try:
                scored = reranker.rerank_scored(query, passages)
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
            per_sess: dict[Any, int] = {}
            for i, _s in scored:
                if not _fresh(i):
                    continue
                sid = owner[i].get("session_id")
                if sid is not None and per_sess.get(sid, 0) >= _SESSION_CAP:
                    continue
                if sid is not None:
                    per_sess[sid] = per_sess.get(sid, 0) + 1
                chosen.append(i)
                redundant.keep(i)
                if len(chosen) >= n:
                    break
            # Backfill: the cap starved us below n (the pool is genuinely one session) —
            # relax and take the next-best passages in score order so diversity-trimming never
            # reduces the served count when nothing more diverse exists to serve. Redundant
            # passages stay out: a repeat adds tokens, not information.
            if len(chosen) < n:
                seen = set(chosen)
                for i, _s in scored:
                    if i not in seen and _fresh(i):
                        chosen.append(i)
                        redundant.keep(i)
                        if len(chosen) >= n:
                            break
        if stats is not None:
            stats["n_dup_passages"] = len(dup_ix)
        # One item per parent episode, placed at its best chunk's rank. Chunks of one turn used
        # to serve as separate items under the same id, adjacent ones repeating CHUNK_OVERLAP
        # chars, which read as the same episode served twice (recall_feedback, Sep-Oct 2026).
        # Adjacent chunks join with their overlap removed; gaps between runs show as [...].
        groups: dict[int, tuple[dict[str, Any], set[int]]] = {}
        for i in chosen:
            ep = owner[i]
            groups.setdefault(id(ep), (ep, set()))[1].add(i)
        out: list[dict[str, Any]] = []
        for ep, members in groups.values():
            runs: list[list[int]] = []
            for j in sorted(members):
                if runs and j == runs[-1][-1] + 1:
                    runs[-1].append(j)
                else:
                    runs.append([j])
            spans = role_spans[id(ep)]
            run_roles = [_passage_role(spans, bounds[r[0]][0], bounds[r[-1]][1]) for r in runs]
            sides = {s for s in run_roles if s}
            role = (sides.pop() if len(sides) == 1 else "mixed") if sides else None
            texts: list[str] = []
            for run in runs:
                text = passages[run[0]]
                for j in run[1:]:
                    text = _join_overlap(text, passages[j])
                # Separate runs merged into one mixed item lose their own role labels, so a run
                # that starts mid-turn gets its speaker's marker back.
                if len(runs) > 1 and role == "mixed" and not _starts_with_marker(text):
                    if side := _side_at(spans, bounds[run[0]][0]):
                        text = f"[{side}] {text}"
                texts.append(text)
            item: dict[str, Any] = {}
            if (rid := ep.get("id")) is not None:
                item["id"] = rid  # parent episode — pass to fetch() to expand the full turn
            item["content"] = _RUN_GAP.join(texts)
            if (project := ep.get("project")) is not None:
                item["project"] = project
            if (ts := ep.get("created_at")) is not None:
                item["date"] = str(ts)[:10]
            if (sid := ep.get("session_id")) is not None:
                item["session"] = sid  # pivot key for fetch_session drill-down
            # Provenance label (issue #17): who produced this slice of the turn —
            # "user" / "assistant" / "mixed"; omitted when unattributable.
            if role:
                item["role"] = role
            out.append(item)
        return out


#: Session-diversity cap on the served passages: at most this many of the n served chunks may
#: share a session_id, backfilled from lower-ranked chunks when nothing more diverse exists.
#: The live session's freshly ingested turns are topically dense and recency-boosted, so
#: uncapped they took every slot (the top recall_feedback noise driver, 2026-07-23). The
#: 2026-07-25 alternatives (a score-slack tiebreak, a freshness-scoped cap) and the
#: per-episode quota/window experiments were never enabled and were removed 2026-10-10;
#: chunks of one episode now merge into one item instead.
_SESSION_CAP = 2

#: Separator between non-adjacent chunks of one episode inside a merged passage item.
_RUN_GAP = "\n[…]\n"

#: Redundancy test for passages of DIFFERENT episodes: the candidate's word 5-shingles are at
#: least _DUP_CONTAINMENT contained in a chosen passage AND every number it mentions appears
#: there too. Catches the same turn ingested twice (e.g. one session captured by two sources),
#: multi-episode turns that repeat the user prompt, and re-run cron prompts. The digit guard
#: keeps near-copies that differ in a dose, version, date or PR number: those are corrections,
#: not repeats, and embeddings and shingles both barely register the change.
_DUP_SHINGLE = 5
_DUP_CONTAINMENT = 0.8
_WORD_RE = re.compile(r"\w+")
_DIGITS_RE = re.compile(r"\d+")


class _Redundancy:
    """Shingle/digit index over the chosen passages (see _DUP_CONTAINMENT)."""

    def __init__(self, passages: list[str], owner: list[dict[str, Any]]) -> None:
        self._passages = passages
        self._owner = owner
        self._keys: dict[int, tuple[frozenset[tuple[str, ...]], frozenset[str]]] = {}
        self._kept: list[tuple[int, frozenset[tuple[str, ...]], frozenset[str]]] = []

    def _key(self, i: int) -> tuple[frozenset[tuple[str, ...]], frozenset[str]]:
        if i not in self._keys:
            text = self._passages[i]
            words = _WORD_RE.findall(text.lower())
            k = _DUP_SHINGLE
            shingles = frozenset(tuple(words[j : j + k]) for j in range(len(words) - k + 1))
            self._keys[i] = (shingles, frozenset(_DIGITS_RE.findall(text)))
        return self._keys[i]

    def covers(self, i: int) -> bool:
        """True when a chosen passage of another episode already says what passage i says."""
        shingles, digits = self._key(i)
        if not shingles:
            return False
        ep = id(self._owner[i])
        return any(
            ep != kept_ep
            and len(shingles & kept_sh) >= _DUP_CONTAINMENT * len(shingles)
            and digits <= kept_digits
            for kept_ep, kept_sh, kept_digits in self._kept
        )

    def keep(self, i: int) -> None:
        self._kept.append((id(self._owner[i]), *self._key(i)))


def _join_overlap(a: str, b: str) -> str:
    """Concatenate adjacent chunks of one document, dropping the tail of ``a`` that
    chunk_markdown prepended to ``b`` as overlap."""
    for k in range(min(len(a), len(b), 2 * CHUNK_OVERLAP), 0, -1):
        if a.endswith(b[:k]):
            return a + b[k:]
    return a + "\n" + b


def _starts_with_marker(text: str) -> bool:
    spans = _role_spans(text)
    return bool(spans) and spans[0][0] == 0


def _side_at(spans: list[tuple[int, str]], pos: int) -> str | None:
    """The speaker ("user"/"assistant") whose section contains char offset ``pos``."""
    side = None
    for offset, s in spans:
        if offset > pos:
            break
        side = s or None
    return side
