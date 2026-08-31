-- 053_audience_scoping.sql
-- Audience scoping — keep personal memory off untrusted hosts.
--
-- Private mode (050) controls what gets CAPTURED. Nothing controlled what gets
-- SERVED: the board injects the whole curated note set into every session on every
-- machine running the plugin, and recall() searches the whole corpus regardless of
-- where the query came from. A work laptop ingesting into the same Synapse therefore
-- got personal notes injected into transcripts on employer-owned hardware.
--
-- Serving is the leak, so the filter goes at serving. Two pieces:
--
--   notes.audience   — a two-tier classification of each curated note. Two tiers, not
--                      N: every extra tier is a per-note decision the user has to make,
--                      and a rich taxonomy nobody maintains fails silently. DEFAULT
--                      'personal' is the fail-closed half — an unclassified note never
--                      leaves a trusted host.
--
--   surfaces         — per-host trust. A row registers one surface id (the plugin's
--                      SYNAPSE_SURFACE / hostname constant). No row means restricted
--                      with an empty allowlist, and the COLUMN default is 'restricted'
--                      too, so promotion to 'full' is always an explicit act rather
--                      than something an INSERT can do by omission.
--
-- Deliberately NOT here: an audience column on timeline_events. That table already
-- carries `domain IN ('personal','technical')` (038) which fails OPEN (NULL); a second
-- overlapping label with the opposite fail semantics invites drift. The board's
-- last-7-days digest filters restricted surfaces by the project allowlist instead.

ALTER TABLE notes ADD COLUMN IF NOT EXISTS audience TEXT NOT NULL DEFAULT 'personal';

-- ADD CONSTRAINT has no IF NOT EXISTS; the DO block makes the file re-runnable.
DO $$
BEGIN
    ALTER TABLE notes ADD CONSTRAINT notes_audience_chk
        CHECK (audience IN ('personal', 'work-safe'));
EXCEPTION
    WHEN duplicate_object THEN NULL;
END
$$;

-- Restricted serving reads the live set filtered to one audience; the partial index
-- matches the shape every filtered read uses (notes_live_idx's WHERE clause + audience).
CREATE INDEX IF NOT EXISTS notes_audience_idx
    ON notes (owner_id, audience)
    WHERE superseded_by IS NULL;

CREATE TABLE IF NOT EXISTS surfaces (
    surface_id       TEXT        PRIMARY KEY,
    trust            TEXT        NOT NULL DEFAULT 'restricted',
    allowed_projects TEXT[]      NOT NULL DEFAULT '{}',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT surfaces_trust_chk CHECK (trust IN ('full', 'restricted'))
);
