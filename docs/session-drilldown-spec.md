# Session drill-down — spec draft (2026-08-07)

Goal: from a surfaced episode, the model pivots into that session the same way
it works a jsonl file on disk — sequential read (Read) and scoped search (Grep)
— with one uniform interface whether the caller is drilling into its own
session or one from months ago.

Motivating incident (2026-08-05): recall surfaced an episode, but the session
around it carried the real context; the model had to fall back to grepping the
raw transcript on disk. Ingestion-side fixes shipped in 0.12.0; this is the
retrieval side.

## 1. Surface `session` on served episodes (prerequisite)

`_to_recall_item` (mcp_server/recall.py:447) currently drops `session_id` as a
ranking-only field. Add it back as `session` on every served episode item in
`recall()`, `recall_full_turns()`, and `fetch()`. Without this the model has no
key to pivot with. The pool SQL already selects it (`_extra_cols`).

Cost: ~40 chars/episode served. Re-run passage_bench to confirm the token
budget doesn't move.

## 2. `fetch_session` — the Read analog

New MCP tool. Pure indexed read over `episodes` (session_id + sequence,
`get_session_episodes` already exists), no embedding, no rerank, target <20ms.
This is the July-08 primitives-api fetch_session shape, unchanged in spirit.

```
fetch_session(
    session_id: str,
    around: str | None = None,   # anchor episode id "e:N" from a recall result
    radius: int = 3,             # neighbors per side, hard cap 10
    offset: int = 0,             # anchorless mode: sequential page start (turn index)
    limit: int = 10,             # anchorless mode: turns per page, cap 25
) -> {
    session_id, project, total_turns, first_date, last_date,
    turns: [{id: "e:N", seq, role, date,
             content        # anchor turn only: full text
             head,          # neighbors/paged turns: first 500 chars
             full_chars}]   # so the caller knows what fetch(e:N) expands to
}
```

- `around` mode: anchor served full, ±`radius` neighbors as 500-char heads.
- Anchorless mode: paged sequential read, heads only — skim the session like
  Read with offset/limit, expand interesting turns via `fetch()`.
- Unknown session → explicit `{"error": "session not indexed", ...}`, never a
  silent empty list. The error is the model's signal to fall back to the
  on-disk transcript; the server (docker VM) can't read cortex's files, so the
  filesystem fallback stays with the caller — the tool's job is honesty.

## 3. `session_id` filter on `recall_full_turns` — the Grep analog

```
recall_full_turns(query, ..., session_id: str | None = None)
```

Restricts the episode pool to one session before the existing BM25 + vector +
rerank. The pool is tiny (tens to hundreds of turns) so the cross-encoder stays
— it's cheap at that size and keeps ranking behavior identical to the global
path. `recall()` (the blended overview) does NOT get the filter: drill-down
belongs in the drill-down tool, and the benchmarked overview serving path stays
untouched.

## Expected flow

1. `recall("postponement of X")` → episode e:231910, `session: "8e92..."`.
2. `fetch_session("8e92...", around="e:231910")` → the surrounding exchange.
3. `recall_full_turns("the exact deadline we agreed", session_id="8e92...")`
   → targeted search within that one conversation.

Same three moves the model makes on disk (find file, read around the hit, grep
the file), so no new mental model.

## Non-goals

- No server-side filesystem fallback (cross-host; ingestion now owns
  completeness, the 0.12.0 sweep is the backstop).
- No changes to recall()'s serving/ranking path beyond the `session` field.

## Telemetry

`fetch_session` logs to recall_metrics as kind='fetch_session' so usage shows
up next to the episodes/turns legs.

## Open design calls (Kyle)

1. `session` field format: full session UUID, or shortened prefix with
   server-side disambiguation?
2. `fetch_session` as a new tool vs overloading `fetch()` with "s:" ids —
   spec assumes a new tool (the around/radius params don't fit fetch()).
3. Neighbor head size: 500 chars per the July-08 primitives draft — still
   right, or align with recall()'s ~1400-char passage cap?
4. Should the current-session hook (`self_session_inject`) auto-fill
   `session_id` when the model passes `session_id="self"`?

## Estimate

Server + tool + tests roughly half a day; passage_bench re-run for item 1 adds
an hour. No schema change (episodes already indexed on session_id + sequence —
verify index during implementation).
