# Audience scoping — keep personal memory off untrusted hosts

Status: APPROVED for build 2026-08-31. Written 2026-08-13; revised 2026-08-31 against
the code map (writer inventory, serving-route inventory, existing timeline `domain`
column).

## Problem

Private mode (schema 050, shipped 2026-08-12) controls what gets **captured**. Nothing
controls what gets **served**. The board (`GET /context`) injects the full User/feedback
note set plus the last-7-days timeline digest into *every* session on *every* machine
that has the plugin installed, and `recall()` searches the whole corpus regardless of
where the query comes from.

Concrete failure: a work laptop ingests into the same Synapse (it already does — work
project episodes exist in the corpus). Opening any session on that laptop injects
personal notes — family, health, job search — into a transcript sitting on
employer-owned, employer-monitorable hardware. The board's recency weighting makes it
worse: whatever is most personally urgent is exactly what floats to the top.

Serving is the leak, so serving is where the filter goes. (Same rationale as private
mode's drop-at-ingest decision inverted: one enforcement chokepoint, not N consumers
remembering to filter.)

## Shape

Server-side only; the plugin stays a thin client and never sees what it isn't served.

### 1. Audience tier on notes

Add `audience TEXT NOT NULL DEFAULT 'personal' CHECK (audience IN ('personal','work-safe'))`
to `notes`. Two tiers, not N: every extra tier is a classification decision the user
has to make per note, and the failure mode of a rich taxonomy is nobody maintains it.
Default `personal` — **fail-closed**: an unclassified note never leaves trusted hosts.

Write-time tagging. There is no dream note-writer; the single write chokepoint is
`ingestion/notes.py:reconcile_note` → `db.insert_note`. Rules, in precedence order:

1. Explicit `audience` param on `remember()` wins.
2. A `remember()` from a restricted surface defaults `work-safe` — symmetric with what
   that host is allowed to read (else notes written at work vanish from the work board
   next session).
3. Otherwise: `work-safe` iff the note's `project` is in the union of restricted
   surfaces' `allowed_projects`, else `personal`. This keeps the work board from
   decaying to empty (new work-project notes keep arriving) without inventing new
   classification machinery — same provenance rule as the episode boundary.

The dream notes judge (retype / re-project / supersede) must preserve `audience` on
update, and re-derive it by rule 3 if it changes a note's `project`.

### 2. Timeline

**No** `audience` column on `timeline_events`. It already has `domain
IN ('personal','technical')` (038), which fails open (NULL) — adding a second
overlapping label with opposite fail semantics invites drift. Instead: the board's
last-7-days digest for a restricted surface filters events by the surface's project
allowlist, excluding NULL project (fail-closed). Recall has had no timeline leg since
2026-08-07, so the digest is the only timeline serving path that matters.

### 3. Host trust on the serving routes

- New table
  `surfaces(surface_id TEXT PRIMARY KEY, trust TEXT NOT NULL DEFAULT 'restricted'
  CHECK (trust IN ('full','restricted')), allowed_projects TEXT[] NOT NULL DEFAULT '{}',
  created_at, updated_at)`. Unknown surface (no row) ⇒ `restricted` with empty
  allowlist. Column default is `restricted` too — fail-closed in the schema, not just
  the no-row path; promotion to `full` is always an explicit act. Small machine-token
  CRUD route for registration/promotion (precedent: private-sessions routes).
- Surface identity: reuse the existing `SYNAPSE_SURFACE`/hostname constant
  (`plugin/scripts/config.py`). The plugin sends it on `GET /context`
  (`board_block.py`), and the PreToolUse inject hook (`self_session_inject.py`) adds a
  `surface` param to `recall`, `recall_full_turns`, `fetch`, `fetch_session`, and
  `remember` tool calls — matcher extended to cover all five; `plugin-codex` mirrored.
  A missing `surface` on any serving path ⇒ treated `restricted`, empty allowlist.
- Enforcement on a `restricted` surface:
  - **Board**: notes filtered to `audience='work-safe'`; timeline digest and the
    episodes banner filtered by project allowlist (banner goes count-only if that's
    simpler).
  - **recall / recall_full_turns**: notes by `audience`; episodes (BM25 + vector legs)
    by `project = ANY(allowed_projects)`. **KG facts leg: skipped entirely in v1** —
    `kg_relationships` has no `project` column, and the join back through source
    episodes isn't worth the cost yet; serving zero facts is fail-closed.
  - **fetch(ids) / fetch_session**: same predicates — ids are guessable, drill-down
    must not bypass the overview filter. `fetch_session` was missing from the original
    spec and is model-reachable today.
  - **Plain-HTTP `POST /recall`**: gains the `surface` param, same enforcement.
- Fail-closed on error: if the `surfaces` lookup fails, serving routes treat the
  request as restricted. The board's existing degrade-to-empty-section behavior must
  not degrade to *unfiltered*.

## Rollout order

1. Merge + deploy (enforcement ships on, no feature flag).
2. Immediately register trusted installs (`trust='full'`) and the work surface
   (`restricted` + its `allowed_projects`). Unknown surfaces are restricted from the
   moment of deploy, so untrusted hosts are protected even before their plugin
   updates — the cost of a stale/unregistered trusted host is a filtered board, not a
   leak.
3. Backfill: heuristic pass (type=project + technical projects ⇒ `work-safe`) writes
   proposed tags per live note to a **local review file — never committed, the repo is
   public**; user reviews, apply-script sets tags. Until applied, all notes are
   `personal`: restricted boards run empty but safe. Feedback/rules notes ride the
   same review pass.

## What this does NOT do (v1)

- No per-note ACLs, no multi-user sharing model. One owner, two audiences, per-host
  trust. Anything richer waits for a real second user.
- No retroactive classification of 40k+ episodes. Project allowlist is the episode
  boundary; accept the coarseness.
- No client-side filtering. The plugin never sees what it isn't served.
- No KG facts on restricted surfaces (see above).
- No dashboard gating: `/dash/*` still serves everything under the machine token.
  Mitigation is operational — don't open the dashboard from untrusted machines.
  Revisit if that constraint ever bites.

## Decisions (resolved 2026-08-31)

1. Scope: full filtered access on the work surface — work-safe notes and feedback
   rules serve there, not just work-project episodes. Project-scope-only would gut the
   board's value on a host that gets daily use.
2. Backfill: heuristic + manual review (local file, not a dashboard UI — the dashboard
   has no notes editor and building one isn't warranted for a one-time pass).
3. `remember()` from a restricted surface defaults `work-safe`.
4. Surface spoofing: hostname self-reported under the shared machine token is accepted
   for the current threat model (employer monitoring, not employer attacking the API).
   Per-surface tokens via the device-flow lane if that ever changes.
