# Audience scoping — keep personal memory off untrusted hosts

Status: v1 SHIPPED 2026-08-31 (schema 053, PRs #172/#174). Revised 2026-09-01 for
**credential-bound trust** (schema 054), which replaces the hostname anchor.
Written 2026-08-13.

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

`notes.audience TEXT NOT NULL DEFAULT 'personal' CHECK (audience IN ('personal','work-safe'))`.
Two tiers, not N: every extra tier is a classification decision the user has to make per
note, and the failure mode of a rich taxonomy is nobody maintains it. Default `personal`
— **fail-closed**: an unclassified note never leaves trusted hosts.

Write-time tagging at the single chokepoint, `ingestion/notes.py:reconcile_note` →
`db.insert_note`. Rules, in precedence order:

1. Explicit `audience` param on `remember()` wins.
2. A `remember()` from a **registered, approved** restricted surface defaults
   `work-safe` — symmetric with what that surface is allowed to read (else notes written
   at work vanish from the work board next session).
3. Otherwise: `work-safe` iff the note's `project` is in the union of **approved**
   restricted surfaces' `allowed_projects`, else `personal`. This keeps the work board
   from decaying to empty without inventing new classification machinery.

The dream notes judge (retype / re-project / supersede) preserves `audience` on update,
and re-derives it by rule 3 if it changes a note's `project`.

### 2. Timeline

**No** `audience` column on `timeline_events`. It already has `domain
IN ('personal','technical')` (038), which fails open (NULL) — adding a second
overlapping label with opposite fail semantics invites drift. Instead: the board's
last-7-days digest for a restricted surface filters events by the surface's project
allowlist, excluding NULL project (fail-closed).

### 3. Trust — a surface is a CREDENTIAL, not a hostname (schema 054)

**What v1 got wrong.** Schema 053 keyed trust on the plugin's self-reported
`SYNAPSE_SURFACE`/hostname constant, accepted under the one shared machine token. Its
own threat model said so out loud: "an employer READING a transcript, not attacking the
API." But the untrusted machine has to hold the machine token in order to ingest at all,
so anything on that machine could send `surface=<trusted-host>` and be served the whole
corpus. The trust boundary was a string chosen by the side being bounded. That is not a
threat model, it is a hope.

**The fix.** One token per device. The credential IS the surface identity, so there is
nothing left to claim.

- `surfaces` extends with `token_hash` (sha256 hex, unique where non-null), `status IN
  ('pending','approved','revoked')` default `pending`, `label` (self-reported hostname,
  **display only**), `pair_code`, `last_seen_at`, and `requested_trust` /
  `requested_projects`. `trust` and `allowed_projects` keep their meaning and their
  fail-closed defaults.
- **Only `status='approved'` serves anything.** A pending device holds a real token that
  authenticates as nothing; a revoked one holds a token that matches no row.
- Three caller kinds resolve through `ingestion/surfaces.resolve_caller`:
  1. **device token** — sha256 lookup; the row's trust applies and any `surface` param
     is ignored;
  2. **`oauth:<login>`** — the claude.ai connector lane, unchanged from PR #174: an
     identity the server verified, resolved by id;
  3. **legacy hostname `surface` param** — accepted from ROOT-token callers for one
     release, so 0.16.x plugins keep working during the migration.

**Enrollment (TOFU).** `POST /surfaces/enroll` mints a device token. If this Synapse has
no approved device yet, the first one in is approved at full trust — nobody exists who
could approve it, and the branch is unreachable ever after. Every later enrollment lands
`pending` with a 6-character pair code and is served nothing until approved.

**The machine role is a REQUEST.** The plugin's install prompt asks personal or work
(`SYNAPSE_MACHINE_ROLE`, default personal) and sends it as `requested_trust`. It is
recorded on the pending row, shown next to the pair code on **both** ends, and used as
the DEFAULT that `approve` fills in — so approving is a one-word confirmation of a role
the operator can already read. It grants nothing on its own: a machine declaring itself
personal is served exactly as much as one declaring nothing, which is nothing. An
explicit `trust`/`allowed_projects` on approve overrides the request in either
direction; a restricted approve with no stated projects inherits the union of existing
approved restricted surfaces' allowlists, so a second work machine sees the same work
projects as the first. **Edge case, deliberate:** a *first-ever* device that requests
`restricted` does NOT bootstrap — auto-approving a device that cannot reach the admin
routes would leave the deployment unadministrable, so it lands pending and the operator
uses `scripts/surface_admin.py bootstrap`.

**Two gates, and their asymmetry is the design.**

| Gate | Admits | Guards |
| --- | --- | --- |
| client (`_machine_authorized`) | root token, or any APPROVED device token | `/ingest`, `/recall`, `/context`, skills, config, timeline, preferences, private mode, remember-spool, `/surfaces/enroll` |
| admin (`_admin_authorized`) | APPROVED **full-trust device** token only — **never root** | `/surfaces/{approve,mint,list,PUT,DELETE}`, all of `/dash/api/*` |

Every machine holds the root token long enough to enroll. If root could also approve, a
machine could self-approve and TOFU would be decoration. The dashboard had to move for
the same reason: `/dash/api` serves the whole corpus unfiltered, so leaving it on the
root token would be a bypass around the entire filter. The dashboard login now mints a
full-trust device token for the OAuth-allowlisted identity (`dash:<login>`) instead of
handing the browser the root token. `issue_machine_token` refuses device tokens: leaves
do not get to fetch the root.

**Break-glass** is deliberately not a token: `scripts/surface_admin.py` talks to Postgres
directly, which requires shell access on the DB host — a strictly higher bar than holding
a bearer, and the right shape for a recovery tool.

**Root token verification never touches the database.** The services on the Docker host
authenticate with it to reach `/ingest` and the internal write paths; a verifier that had
to read PG to admit root would turn every PG blip into a total auth outage on the one
lane that repairs things.

`resolve_caller` is the single resolution point and **never fails open**: no credential,
no row, a non-approved row, missing table, unreachable database, malformed row — all
return restricted with an empty allowlist. `project = ANY('{}')` is false for every row,
NULL project included, so an unknown caller serves *nothing* rather than everything.

**Enforcement on a restricted surface** (unchanged from v1):

- **Board**: notes filtered to `audience='work-safe'`; timeline digest and episodes
  banner filtered by project allowlist. The pending-device nudge is full-trust-only — a
  restricted surface must not learn other machines exist, let alone hold a code that
  could approve one.
- **recall / recall_full_turns**: notes by `audience`; episodes (BM25 + vector legs) by
  `project = ANY(allowed_projects)`. **KG facts leg skipped entirely** —
  `kg_relationships` has no `project` column, so serving zero facts is fail-closed.
- **fetch(ids) / fetch_session**: same predicates — ids are guessable, drill-down must
  not bypass the overview filter.
- **Plain-HTTP `POST /recall`**: same enforcement, resolved from the bearer.

## Rollout order (schema 054)

The order matters: 054 tightens `/dash/api` and moves every client onto a new credential,
so an operator who runs it out of order locks themselves out of their own dashboard.

1. **Apply `schema/054_credential_bound_trust.sql` on prod.** Existing rows are stamped
   `approved` (they were owner-registered), so nothing that works today stops working.
2. **Deploy the server.** Root-token clients keep working through the legacy `surface`
   param; the dashboard's *existing* fragment token (the root token) now 401s on
   `/dash/api` — that is expected, see step 4.
3. **Update the plugin to 0.17.0 on the machine you trust most, and open a session.**
   It enrolls; since prod already has approved (credential-less) rows but none with a
   token, that first enrollment takes the bootstrap branch and comes back approved at
   full trust. If it comes back pending instead, use break-glass:
   `scripts/surface_admin.py bootstrap "<label>"` on the DB host and set the printed
   token as `SYNAPSE_INGEST_TOKEN` there.
4. **Log into the dashboard again.** The login flow mints its own full-trust device
   token; the old fragment token is dead.
5. **Update the remaining machines.** Each enrolls pending and prints its pair code; the
   trusted machine's board lists the same code plus the role it asked for. Approve from
   there: `/synapse-devices approve <CODE>`.
6. **For a machine you will not put the enrollment token on:**
   `/synapse-devices mint "work-laptop" --projects work-thing` from the trusted machine,
   and carry only the minted token over.
7. **Next release:** drop the legacy `surface` param and the root token's serving lane
   entirely, once every machine is enrolled (`/surfaces` shows `has_token` per row).

## What this does NOT do

- No per-note ACLs, no multi-user sharing model. One owner, two audiences, per-device
  trust. Anything richer waits for a real second user.
- No retroactive classification of 40k+ episodes. Project allowlist is the episode
  boundary; accept the coarseness.
- No client-side filtering. The plugin never sees what it isn't served.
- No KG facts on restricted surfaces (see above).
- No device-token expiry or rotation schedule. Revocation is manual and immediate; a
  TTL would add a renewal path (and a renewal credential) for no threat this model has.
- Pending devices cannot ingest. Their transcripts are not lost — the ingest hook's
  `--catchup` sweep re-posts them after approval — but a long-pending machine can
  outrun that window.

## Decisions

1. Scope: full filtered access on a restricted surface — work-safe notes and feedback
   rules serve there, not just work-project episodes. Project-scope-only would gut the
   board's value on a machine that gets daily use. (2026-08-31)
2. Backfill: heuristic + manual review via `scripts/audience_backfill.py`, to a local
   file outside the repo — the repo is public and the review file lists every note hook.
   (2026-08-31)
3. `remember()` from a restricted surface defaults `work-safe`. (2026-08-31)
4. ~~Surface spoofing: hostname self-reported under the shared machine token is accepted
   for the current threat model.~~ **Superseded 2026-09-01**: the machine token has to
   live on the untrusted machine for ingest to work, so a self-reported hostname was
   never a boundary. Per-device tokens + TOFU approval (schema 054).
5. The self-declared machine role is a request, not a grant, and approve defaults to it.
   Rejected alternative: let a work-declared machine auto-approve itself restricted. It
   reads safe and is not — "restricted" is only narrow relative to an allowlist the
   machine would then also be choosing, and it would make the pending state mean
   "personal machines only", which is a rule nobody would remember. (2026-09-01)
