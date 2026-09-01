-- 054_credential_bound_trust.sql
-- Credential-bound surface trust — a surface is a CREDENTIAL, not a hostname.
--
-- Schema 053 keyed trust on the plugin's self-reported SYNAPSE_SURFACE/hostname,
-- accepted under the one shared machine token. That was explicitly scoped to a threat
-- model of "an employer READS a transcript", because any holder of the machine token
-- could send `surface=<trusted-host>` and be served the full corpus. The moment the
-- token is on the untrusted machine (it has to be — that machine ingests), the trust
-- boundary is a string the untrusted side chooses. This migration replaces that with a
-- per-device token: the credential IS the surface, so there is nothing to spoof.
--
-- Extends `surfaces` rather than adding a table: trust/allowed_projects already mean
-- exactly the right thing, and one row per surface keeps audience derivation
-- (restricted_surface_projects) reading a single place.
--
--   token_hash   sha256 hex of the device token. NULL for the two credential-less
--                surface kinds that survive: legacy hostname rows (migration window)
--                and `oauth:<login>` identity rows (the claude.ai connector lane,
--                which authenticates by OIDC/GitHub identity, not by a device token).
--   status       'pending' | 'approved' | 'revoked'. ONLY 'approved' serves anything.
--                A freshly enrolled device is 'pending': it holds a real token that
--                authenticates as nothing until a trusted device approves it (TOFU).
--   label        the device's self-reported hostname. DISPLAY ONLY — never resolved,
--                never matched, never used to pick a row. It exists so a human can
--                tell which pending row is the laptop they just installed on.
--   pair_code    short human-transferable code, shown on BOTH ends of an approval
--                (the enrolling device prints it, the trusted device's board lists
--                it). Cleared on approve so a code is single-use.
--   last_seen_at best-effort liveness for the operator's `GET /surfaces` view.
--
--   requested_trust / requested_projects
--                what the enrolling machine SAYS it is ("personal" or "work"), captured
--                at install time. A REQUEST, never a grant: it is stored on the pending
--                row and read only as the DEFAULT that `approve` fills in, so the
--                operator confirms a role instead of retyping one. Nothing serves off
--                these columns. A machine that declares itself personal is still served
--                nothing until an approved full-trust device approves it — the whole
--                security property would evaporate if a self-declaration could
--                activate itself.
--
-- The unique partial index on token_hash is the security-relevant constraint: two rows
-- can never share a credential, so "which surface is this" has exactly one answer.
--
-- Fail-closed defaults are unchanged and now doubled: trust DEFAULT 'restricted'
-- (053) AND status DEFAULT 'pending' (here). An INSERT that forgets both columns
-- produces a row that can read nothing.

ALTER TABLE surfaces
    ADD COLUMN IF NOT EXISTS token_hash         TEXT,
    ADD COLUMN IF NOT EXISTS status             TEXT NOT NULL DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS label              TEXT,
    ADD COLUMN IF NOT EXISTS pair_code          TEXT,
    ADD COLUMN IF NOT EXISTS last_seen_at       TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS requested_trust    TEXT,
    ADD COLUMN IF NOT EXISTS requested_projects TEXT[];

-- ADD CONSTRAINT has no IF NOT EXISTS; the DO blocks make the file re-runnable.
DO $$
BEGIN
    ALTER TABLE surfaces ADD CONSTRAINT surfaces_status_chk
        CHECK (status IN ('pending', 'approved', 'revoked'));
EXCEPTION
    WHEN duplicate_object THEN NULL;
END
$$;

-- Nullable: a client too old to send a role, or one that sends nonsense, must land as
-- "no request" rather than being rejected at enrollment. NULL reads as "unstated", and
-- approve then falls back to its own restricted default.
DO $$
BEGIN
    ALTER TABLE surfaces ADD CONSTRAINT surfaces_requested_trust_chk
        CHECK (requested_trust IS NULL OR requested_trust IN ('full', 'restricted'));
EXCEPTION
    WHEN duplicate_object THEN NULL;
END
$$;

-- One credential, one surface. Partial so the credential-less rows (legacy hostname,
-- oauth:<login>) can coexist without colliding on NULL.
CREATE UNIQUE INDEX IF NOT EXISTS surfaces_token_hash_idx
    ON surfaces (token_hash) WHERE token_hash IS NOT NULL;

-- A pending row is looked up by pair code exactly once, but the code has to be unique
-- while it is live or an approval could hit the wrong device.
CREATE UNIQUE INDEX IF NOT EXISTS surfaces_pair_code_idx
    ON surfaces (pair_code) WHERE pair_code IS NOT NULL;

-- Every pre-054 row was created by the owner through the machine-token CRUD, so it is
-- approved by construction. Without this the migration would silently revoke every
-- registered host (status defaults to 'pending') and every board would go empty.
UPDATE surfaces SET status = 'approved' WHERE status = 'pending';
