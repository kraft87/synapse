-- 030_config_lane.sql
-- Config lane: mirror a machine's opted-in config files (CLAUDE.md, rules/*.md, ...) into Postgres
-- so the dream pipeline can read them and PROPOSE edits, and so accepted edits can be applied back.
-- Mirrors the skills lane (schema 022/024): config_registry = the per-surface file mirror;
-- config_proposals = dream's proposed edits (the skill_gap_candidates analogue).
--
-- V1 scope = ingest (publish) + propose + LOCAL apply. Cross-surface fan-out of a 'general' edit and
-- continuous two-way file sync are later phases — clean adds because proposals are already diffs.
CREATE SCHEMA IF NOT EXISTS config_lane;

-- The mirror: one row per (surface, file). A "surface" is a machine/install (the plugin sends its
-- id, default hostname). file_key is the file's LOGICAL id = its path relative to the config root
-- (e.g. "rules/voice.md"), stable across machines so a future 'general' edit can target "the same
-- file" everywhere. abs_path is the real on-disk path on that surface, used to write an edit back.
CREATE TABLE IF NOT EXISTS config_lane.config_registry (
    surface_id    TEXT        NOT NULL,
    file_key      TEXT        NOT NULL,
    abs_path      TEXT        NOT NULL,
    content       TEXT        NOT NULL,
    content_hash  TEXT        NOT NULL,
    modified_at   TIMESTAMPTZ,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (surface_id, file_key)
);

-- Dream's proposed edits to a config file (mirrors skills_lane.skill_gap_candidates). A proposal is
-- a DIFF against a file_key. scope='local' applies to the originating surface; scope='general' fans
-- out to every surface sharing the file_key (deferred to the sync phase). status walks the same gate
-- as skills, ending at 'applied' once written back.
CREATE TABLE IF NOT EXISTS config_lane.config_proposals (
    id            BIGSERIAL PRIMARY KEY,
    kind          TEXT NOT NULL CHECK (kind IN ('add', 'edit', 'consolidate', 'remove')),
    file_key      TEXT NOT NULL,
    scope         TEXT NOT NULL DEFAULT 'general' CHECK (scope IN ('local', 'general')),
    surface_id    TEXT,                          -- set when scope='local' (the one target surface)
    diff          TEXT NOT NULL DEFAULT '',      -- unified diff / proposed change
    summary       TEXT NOT NULL DEFAULT '',
    evidence      JSONB NOT NULL DEFAULT '[]',   -- correction instances [{session_id, ts, signal, why}]
    status        TEXT NOT NULL DEFAULT 'proposed'
                    CHECK (status IN ('observe', 'proposed', 'accepted', 'rejected', 'applied')),
    reject_reason TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS config_proposals_status_idx
    ON config_lane.config_proposals (status, updated_at DESC);
