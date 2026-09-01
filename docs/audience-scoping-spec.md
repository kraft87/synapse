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
  ('approved','revoked')`, a display-only `label`, and `last_seen_at`. `trust` and
  `allowed_projects` keep their meaning and their fail-closed defaults. `status`
  defaults to `revoked` — it reads odd and is deliberate: every real creation path sets
  it explicitly, so the default only ever catches a hand-written row, and "nobody
  granted this" should behave exactly like "this grant was pulled".
- **Only `status='approved'` serves anything.** Revoking also clears the hash, so a
  revoked credential matches no row at all, and the row survives as the audit record.
- Three caller kinds resolve through `ingestion/surfaces.resolve_caller`:
  1. **device token** — sha256 lookup; the row's trust applies and any `surface` param
     is ignored;
  2. **`oauth:<login>`** — the claude.ai connector lane, unchanged from PR #174: an
     identity the server verified, resolved by id;
  3. **legacy hostname `surface` param** — accepted from ROOT-token callers for one
     release, so 0.16.x plugins keep working during the migration.

### Enrollment is anchored to the owner's identity

A new machine gets its device token by completing the IdP's device flow (RFC 8628 — the
same lane `synapse login` already uses: a short code approved in a browser on any
device, phone included) and passing the resulting `device_code` to
`POST /surfaces/enroll {device_code, label, trust, allowed_projects}`. The server polls
the IdP, reads the identity, enforces the same allowlist every other login clears, and
only then mints. The row lands `approved` immediately.

**TOFU was considered and rejected.** The first cut of this design used a pair code: a
new machine enrolled `pending`, printed a 6-character code, and a second already-trusted
machine approved it. That ceremony exists to compensate for an ANONYMOUS enrollment
credential — if all a new machine can prove is "I hold the shared token", something else
has to vouch for it. An OAuth-verified owner standing at the new machine has already
answered the only question the ceremony was asking, so it bought a second round trip, a
pending state in which a laptop silently has no memory, and a code to copy between
screens, in exchange for nothing. **Accepted in the threat model:** theft of the owner's
IdP credential, mitigated by the IdP's own 2FA (Authelia here) and in any case already
sufficient to reach the dashboard and read everything directly.

`trust` is the answer to the plugin's install-time "personal or work?" prompt
(`SYNAPSE_MACHINE_ROLE`), and it is authoritative because the person who answered it is
the person who just authenticated. The prompt defaults to `personal`, because the
single-user common case is a machine that should see everything and a default that makes
the normal path silently useless gets worked around rather than understood. The narrow
default lives one layer down: the SERVER treats an *unstated* role as `restricted`, so a
client that never asked the question cannot resolve it to full access. Human says
nothing ⇒ personal; software says nothing ⇒ restricted. A restricted enrollment that
names no projects inherits the union of what approved restricted surfaces already read
(a second work machine should see the same work projects as the first); an explicit `[]`
still means empty.

**`POST /surfaces/mint`** is the same operation for a machine that will never run a
browser flow — a headless box, a service, a container. It is `admin`-gated, so it needs
a machine that is already trusted, and the token is carried over by hand.

**The root machine token cannot enroll or mint.** It is the services' credential for
`/ingest` and the internal write paths, and every machine that ever ran the plugin has
held it. A credential that widespread must not be able to create a new credential.

**Two gates, and their asymmetry is the design.**

| Gate | Admits | Guards |
| --- | --- | --- |
| client (`_machine_authorized`) | root token, or any APPROVED device token | `/ingest`, `/recall`, `/context`, skills, config, timeline, preferences, private mode, remember-spool |
| admin (`_admin_authorized`) | APPROVED **full-trust device** token only — **never root** | `/surfaces/{mint,list,PUT,DELETE}`, all of `/dash/api/*` |

The dashboard had to move for the same reason: `/dash/api` serves the whole corpus
unfiltered, so leaving it on the root token would be a bypass around the entire filter.
The dashboard login now mints a full-trust device token for the OAuth-allowlisted
identity (`dash:<login>`, one row per identity, rotated per login) instead of handing
the browser the root token. `issue_machine_token` refuses device tokens: leaves do not
get to fetch the root.

**Break-glass** is deliberately not a token: `scripts/surface_admin.py` (list / mint /
revoke / bootstrap) talks to Postgres directly, which requires shell access on the DB
host — a strictly higher bar than holding a bearer. It is the answer for the three cases
where nothing else works: no IdP configured at all, the IdP down or the account locked,
or no full-trust device left.

**Root token verification never touches the database.** The services on the Docker host
authenticate with it to reach `/ingest` and the internal write paths; a verifier that had
to read PG to admit root would turn every PG blip into a total auth outage on the one
lane that repairs things.

`resolve_caller` is the single resolution point and **never fails open**: no credential,
no row, a revoked row, missing table, unreachable database, malformed row — all return
restricted with an empty allowlist. `project = ANY('{}')` is false for every row, NULL
project included, so an unknown caller serves *nothing* rather than everything.

**Enforcement on a restricted surface** (unchanged from v1):

- **Board**: notes filtered to `audience='work-safe'`; timeline digest and episodes
  banner filtered by project allowlist.
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
3. **Update the plugin to 0.17.0 and run `synapse-login` on each machine.** It signs in,
   then enrolls: a second device-flow approval that mints that machine's own token,
   scoped by its install-time personal/work answer, stored in the same
   `SYNAPSE_INGEST_TOKEN` slot. Until a machine enrolls it keeps working on the root
   token via the legacy lane, and its SessionStart block says it is not enrolled.
4. **Log into the dashboard again.** The login flow mints its own full-trust device
   token; the old fragment token is dead.
5. **For a machine that cannot run a browser flow anywhere** (headless, a service):
   `/synapse-devices mint "<label>" --projects work-thing` from an already-trusted
   machine, or `scripts/surface_admin.py mint` on the DB host, and carry the token over.
6. **Next release:** drop the legacy `surface` param and the root token's serving lane
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
- No enrollment without an identity provider. A bearer-only deployment has no identity
  to anchor an enrollment to, so `POST /surfaces/enroll` reports 503 and points at the
  break-glass CLI. Adding a weaker fallback would recreate exactly the hole 054 closes.

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
   never a boundary. Per-device tokens (schema 054).
5. Enrollment is OAuth-anchored, not trust-on-first-use. Rejected alternative: pending
   rows + a pair code approved from a second trusted machine. That ceremony compensates
   for an anonymous enrollment credential; with an allowlisted identity at the new
   machine there is nothing left for it to establish, and it costs a state in which a
   laptop silently has no memory. IdP-credential theft is accepted as out of scope
   (Authelia 2FA, and such an attacker could read the dashboard directly). (2026-09-01)
6. The install prompt defaults to `personal` while the server defaults an *unstated*
   role to `restricted`. A narrow prompt default would make the common single-user case
   silently useless and get worked around; a wide server default would let a client that
   never asked the question resolve it to full access. (2026-09-01)
