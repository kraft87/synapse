-- 035_preferences.sql
-- preferences — the standing USER-PREFERENCE store: an append-only, dedup-by-reassertion
-- log of durable likes / dislikes / rules ("prefers bullet lists over tables", "never
-- suggest contract roles"). Kept DELIBERATELY OUT of the KG: every preference hangs off
-- the single User entity, so modelling them as edges rebuilds the exact User-supernode
-- the timeline store (033) was built to avoid (a "User prefers everything" star node has
-- nothing to traverse and drowns entity resolution). Like timeline_events, this is its own
-- flat time-log; unlike it, rows ARE reconciled — a restatement bumps assert_count and a
-- contradiction supersedes (t_invalid + superseded_by) rather than piling up duplicates.
--
-- Rationale: preferences are Synapse's worst LongMemEval category (56.7% vs Mastra 73.3),
-- and a June A/B showed injecting TYPED preference facts is worth +13 pts on that category.
-- Served two ways: a small recall() cosine bucket + a bounded session-start block.
CREATE TABLE IF NOT EXISTS preferences (
    id             BIGSERIAL PRIMARY KEY,
    owner_id       TEXT        NOT NULL,               -- single-owner constant today ('default'); SYNAPSE_KG_OWNER_ID axis
    group_id       TEXT        NOT NULL,               -- 'technical' | 'personal' (derived from project, like the KG)
    project        TEXT,                                -- originating project tag, if any (informational)
    pref           TEXT        NOT NULL,               -- self-contained, third-person-about-user ("User prefers bullet lists over tables")
    polarity       TEXT        NOT NULL,               -- 'like' | 'dislike' | 'rule' (see CHECK)
    first_seen     TIMESTAMPTZ NOT NULL DEFAULT now(), -- when this preference was first asserted
    last_asserted  TIMESTAMPTZ NOT NULL DEFAULT now(), -- most recent (re)assertion — recency for the session-start block
    assert_count   INT         NOT NULL DEFAULT 1,     -- re-assertion frequency — the confidence/strength signal
    t_invalid      TIMESTAMPTZ,                          -- set when SUPERSEDED (a later contradicting assertion won); NULL = live
    superseded_by  BIGINT      REFERENCES preferences(id),  -- the row that replaced this one
    source_ref     TEXT,                                -- provenance, e.g. 'ep:<episode_id>'
    embedding      vector(2048),                        -- Voyage over the bare pref text (aligns with the plain query embedding at serve time)
    embed_model    TEXT,                                -- model id, so a future re-embed is possible
    ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT preferences_polarity_chk CHECK (polarity IN ('like', 'dislike', 'rule'))
);

-- Live-set lookup: the gate's dedup KNN and both serving paths only ever read live rows
-- for one owner/group, so the partial index over exactly that predicate keeps them fast.
CREATE INDEX IF NOT EXISTS preferences_live_idx
    ON preferences (owner_id, group_id) WHERE t_invalid IS NULL;

-- Vector index DEFERRED, matching timeline_events (033): a single user accrues only dozens
-- of live preferences, so a filtered seqscan is instant; 2048-dim also exceeds vector-type
-- HNSW's 2000-dim cap, so an index later needs the halfvec expression pattern (018/#104).
-- Adding it is non-breaking.
