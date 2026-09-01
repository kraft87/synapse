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
--   status       'approved' | 'revoked'. Only 'approved' serves anything; revoking
--                clears the hash, so the credential itself stops matching any row and
--                the row stays behind as the audit record of what was once trusted.
--   label        the device's self-reported hostname. DISPLAY ONLY — never resolved,
--                never matched, never used to pick a row. It exists so a human can
--                tell which device a listing line refers to.
--   last_seen_at best-effort liveness for the operator's `GET /surfaces` view.
--
-- The unique partial index on token_hash is the security-relevant constraint: two rows
-- can never share a credential, so "which surface is this" has exactly one answer.
--
-- DEFAULT 'revoked' reads odd and is deliberate. Enrollment is owner-authenticated, so
-- every code path that creates a live device sets status explicitly; the default only
-- ever applies to a row someone wrote by hand. "Nobody granted this" and "this grant was
-- pulled" should behave identically, and the failure that matters is a hand-written row
-- silently authenticating. Fail-closed defaults are now doubled: trust DEFAULT
-- 'restricted' (053) AND status DEFAULT 'revoked' (here).

ALTER TABLE surfaces
    ADD COLUMN IF NOT EXISTS token_hash   TEXT,
    ADD COLUMN IF NOT EXISTS status       TEXT NOT NULL DEFAULT 'revoked',
    ADD COLUMN IF NOT EXISTS label        TEXT,
    ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMPTZ;

-- ADD CONSTRAINT has no IF NOT EXISTS; the DO block makes the file re-runnable.
DO $$
BEGIN
    ALTER TABLE surfaces ADD CONSTRAINT surfaces_status_chk
        CHECK (status IN ('approved', 'revoked'));
EXCEPTION
    WHEN duplicate_object THEN NULL;
END
$$;

-- One credential, one surface. Partial so the credential-less rows (legacy hostname,
-- oauth:<login>) can coexist without colliding on NULL.
CREATE UNIQUE INDEX IF NOT EXISTS surfaces_token_hash_idx
    ON surfaces (token_hash) WHERE token_hash IS NOT NULL;

-- Every pre-054 row was created by the owner through the machine-token CRUD, so it is
-- approved by construction. Without this the migration would silently revoke every
-- registered host and every board would go empty.
UPDATE surfaces SET status = 'approved' WHERE status = 'revoked' AND token_hash IS NULL;
